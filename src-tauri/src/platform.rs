//! knowledge_root 的平台解析 + Android 种子库释放。
//!
//! 规则（见 spec §5）：
//! - 开发模式（cwd 向上能找到 `resources/knowledge/`）→ 用它，改文件立刻生效
//! - 其余（已安装桌面端 / Android）→ `app_data_dir/knowledge`
//!
//! `$RESOURCE` 回落分支已随"知识库不再打进安装包"一起删除。
use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// 从 `start` 向上查找 `resources/knowledge/` 目录。
/// 解决 `cargo run`（cwd 在 src-tauri/）与 `tauri build`（cwd 在项目根）
/// 工作目录不一致的问题 —— 始终从 cwd 出发逐层往上找。
pub fn find_knowledge_root(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        let candidate = d.join("resources").join("knowledge");
        if candidate.is_dir() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

/// 知识库根目录查找优先级：开发目录 > 兜底路径（app_data）。
///
/// 老的"安装包资源目录"分支已随"知识库不再打进安装包"一起删除 —— 桌面端
/// 已装版本直接走 app_data_dir/knowledge，Android 由 materialize_seed 释放种子库。
pub fn pick_knowledge_root(dev_root: Option<PathBuf>, app_data: PathBuf) -> PathBuf {
    dev_root.unwrap_or(app_data)
}

#[derive(Debug, Clone, Deserialize)]
pub struct SeedEntry {
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Clone, Default)]
pub struct SeedStats {
    pub written: usize,
    pub skipped: usize,
}

/// 解析种子库清单（Android APK assets 里的 `seed-manifest.json`）。
/// 拒绝路径不合法（包含 `..`、绝对路径等）的条目 —— 复用
/// `crate::sync::normalize_rel_path` 的同一套规则，避免两套安全策略并行漂移。
pub fn parse_seed_manifest(json: &str) -> Result<Vec<SeedEntry>> {
    #[derive(Deserialize)]
    struct Raw {
        #[serde(default)]
        files: Vec<SeedEntry>,
    }
    let raw: Raw = serde_json::from_str(json)?;
    for e in &raw.files {
        // 与 sync::normalize_rel_path 同一套拒绝规则
        crate::sync::normalize_rel_path(&e.path)?;
    }
    Ok(raw.files)
}

/// 把种子库从（Android 上不可直接读的）asset URI 释放到真实目录。
/// `read` 由调用方注入：Android 传 `app.fs()`，测试传内存 map。
/// 已存在且大小相符 → 跳过（幂等）。
pub fn copy_seed_entries(
    entries: &[SeedEntry],
    read: &dyn Fn(&str) -> Result<Vec<u8>>,
    root: &Path,
) -> Result<SeedStats> {
    let mut stats = SeedStats::default();
    std::fs::create_dir_all(root)?;
    for e in entries {
        let dest = root.join(&e.path);
        if std::fs::metadata(&dest).map(|m| m.len() == e.size).unwrap_or(false) {
            stats.skipped += 1;
            continue;
        }
        let body = read(&e.path)?;
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&dest, body)
            .map_err(|err| anyhow::anyhow!("write seed {}: {err}", e.path))?;
        stats.written += 1;
    }
    Ok(stats)
}

/// 桌面端老版本升级迁移：知识库原先随安装包打在 $RESOURCE/knowledge。
/// 目标目录为空时才搬（只搬一次），搬完用户本地编辑过的文件不会被覆盖。
pub fn migrate_legacy_bundled(src: &Path, dest: &Path) -> Result<usize> {
    if !src.is_dir() {
        return Ok(0);
    }
    std::fs::create_dir_all(dest)?;
    if std::fs::read_dir(dest)?.next().is_some() {
        return Ok(0);
    }
    let mut copied = 0usize;
    for entry in WalkDir::new(src) {
        let entry = match entry { Ok(e) => e, Err(_) => continue };
        let rel = entry.path().strip_prefix(src)?;
        if rel.as_os_str().is_empty() {
            continue;
        }
        let target = dest.join(rel);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
            copied += 1;
        }
    }
    Ok(copied)
}

/// Android 启动时把 APK assets 里的种子库释放到 knowledge/。
/// 桌面端是 no-op —— 桌面不打包任何资源。
#[cfg(target_os = "android")]
pub fn materialize_seed(app: &tauri::AppHandle, root: &Path) -> Result<SeedStats> {
    use std::cell::Cell;
    use tauri::Emitter;
    use tauri_plugin_fs::FsExt;
    let manifest_path = app
        .path()
        .resolve("seed-manifest.json", tauri::path::BaseDirectory::Resource)?;
    let json = app
        .fs()
        .read_to_string(&manifest_path)
        .map_err(|e| anyhow::anyhow!("read seed-manifest: {e}"))?;
    let entries = parse_seed_manifest(&json)?;
    let total = entries.len();
    // 用 Cell 包计数器 —— 否则闭包只能写成 FnMut，但 copy_seed_entries 要求
    // &dyn Fn。Cell::set 只取 &self，让闭包保持 Fn 形态。
    let done = Cell::new(0usize);
    let emit_app = app.clone();
    let read = move |p: &str| -> Result<Vec<u8>> {
        let asset = format!("knowledge/{p}");
        let full = app.path().resolve(&asset, tauri::path::BaseDirectory::Resource)?;
        let bytes = app.fs().read(&full).map_err(|e| anyhow::anyhow!("read asset {asset}: {e}"))?;
        let cur = done.get() + 1;
        done.set(cur);
        let _ = emit_app.emit(
            "seed_progress",
            serde_json::json!({"done": cur, "total": total}),
        );
        Ok(bytes)
    };
    copy_seed_entries(&entries, &read, root)
}

#[cfg(not(target_os = "android"))]
pub fn materialize_seed(_app: &tauri::AppHandle, _root: &Path) -> Result<SeedStats> {
    Ok(SeedStats::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_seed_manifest_reads_paths_and_sizes() {
        let json = r#"{"files":[{"path":"01-基础/_index.md","size":12},{"path":"01-基础/a.md","size":34}]}"#;
        let e = parse_seed_manifest(json).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].path, "01-基础/_index.md");
        assert_eq!(e[1].size, 34);
    }

    #[test]
    fn parse_seed_manifest_rejects_traversal() {
        assert!(parse_seed_manifest(r#"{"files":[{"path":"../evil.md","size":1}]}"#).is_err());
    }

    #[test]
    fn copy_seed_entries_writes_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("knowledge");
        let body = |p: &str| -> Result<Vec<u8>> {
            if p == "01-基础/a.md" { Ok(b"hello".to_vec()) } else { anyhow::bail!("no asset {p}") }
        };
        let entries = vec![SeedEntry { path: "01-基础/a.md".into(), size: 5 }];
        let s1 = copy_seed_entries(&entries, &body, &root).unwrap();
        assert_eq!((s1.written, s1.skipped), (1, 0));
        let s2 = copy_seed_entries(&entries, &body, &root).unwrap();
        assert_eq!((s2.written, s2.skipped), (0, 1), "第二次应全部跳过");
        assert_eq!(std::fs::read(root.join("01-基础/a.md")).unwrap(), b"hello");
    }

    #[test]
    fn copy_seed_entries_propagates_read_error() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("knowledge");
        let body = |_p: &str| -> Result<Vec<u8>> { anyhow::bail!("asset 读取失败") };
        let err = copy_seed_entries(&[SeedEntry { path: "a/b.md".into(), size: 1 }], &body, &root);
        assert!(err.is_err());
    }

    #[test]
    fn pick_knowledge_root_prefers_dev_dir() {
        assert_eq!(
            pick_knowledge_root(Some(PathBuf::from("/dev/resources/knowledge")), PathBuf::from("/data/knowledge")),
            PathBuf::from("/dev/resources/knowledge")
        );
    }

    #[test]
    fn pick_knowledge_root_falls_back_to_app_data() {
        assert_eq!(
            pick_knowledge_root(None, PathBuf::from("/data/knowledge")),
            PathBuf::from("/data/knowledge")
        );
    }

    #[test]
    fn migrate_legacy_bundled_copies_tree_once() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("old");
        let dest = dir.path().join("new");
        std::fs::create_dir_all(src.join("a")).unwrap();
        std::fs::write(src.join("a/x.md"), b"x").unwrap();
        let n = migrate_legacy_bundled(&src, &dest).unwrap();
        assert_eq!(n, 1);
        assert_eq!(std::fs::read(dest.join("a/x.md")).unwrap(), b"x");
    }

    // ---- 从 commands.rs 迁来的 find_knowledge_root 测试 ----

    #[test]
    fn find_knowledge_root_walks_up_to_directory() {
        // 在 tmp 下构造 project_root/src-tauri/，把 project_root/resources/knowledge/ 建出来
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();
        let deep = project_root.join("src-tauri");
        std::fs::create_dir_all(&deep).unwrap();
        let knowledge = project_root.join("resources").join("knowledge");
        std::fs::create_dir_all(&knowledge).unwrap();
        // 从 src-tauri（cwd）开始查找，应向上找到 project_root/resources/knowledge
        let found = find_knowledge_root(&deep).unwrap();
        assert_eq!(found.canonicalize().unwrap(), knowledge.canonicalize().unwrap());
    }

    #[test]
    fn find_knowledge_root_returns_none_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(find_knowledge_root(tmp.path()).is_none());
    }
}