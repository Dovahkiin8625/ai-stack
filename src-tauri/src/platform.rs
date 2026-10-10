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

/// 桌面端老版本升级迁移：知识库原先随安装包打在 `$RESOURCE/knowledge`。
/// 把 bundled 目录搬进 `dest`（首次启动后是 `app_data_dir/knowledge`）。
///
/// 契约（**一次性，标记闸门**）：
/// - 由 `marker`（应放在 `dest` 旁边，如 `app_data_dir/.legacy-migrated`）
///   守门：标记已存在 → 立即返回 0，**不**走 `$RESOURCE`、**不**碰任何文件。
///   这是为了在用户编辑过文章（长度变化）后，下次启动不会被 bundled 原版
///   静默覆盖 —— `write_resource` 改的是 dest，不是 src；每次启动若不短路，
///   大小一变就被覆盖，用户数据丢失。
/// - 标记只能落在 `dest` **外面**：scanner 会 walk `dest/`（= `knowledge/`），
///   多出来一个文件就会被当成文章 / 多出来一个目录就会被当成新分类。
/// - 单次执行**可恢复**：迁移途中被打断（断电 / 进程崩）时，标记还没写，
///   下次启动会再跑。同一 src/dest 第一次跑时 dest 必然不存在，所以
///   "dest 文件存在但大小不一"只可能是"上次中断留下的半成品"，覆盖是
///   安全的；用户在此期间不可能凭空在 dest 里编辑文件 —— 那个目录是迁移
///   这一刻才被创建出来的。
/// - src 不存在（新装 / bundled.resources 已移除）直接返回 0，**不**写
///   标记 —— 让后续每次启动仍然跳过这步的"src 是否存在"判断（便宜），
///   同时也保留未来"bundled 重新引入 → 自动触发迁移"的退路。
pub fn migrate_legacy_bundled(src: &Path, dest: &Path, marker: &Path) -> Result<usize> {
    if marker.exists() {
        return Ok(0);
    }
    if !src.is_dir() {
        return Ok(0);
    }
    std::fs::create_dir_all(dest)?;
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
            continue;
        }
        // 单次执行内的"按字节比对"：第一次执行时 dest 不存在，所以任何已
        // 存在但大小不一的 dest 文件只能是上次中断留下的半成品 —— 覆盖无副作用。
        let src_size = match entry.metadata() {
            Ok(m) => m.len(),
            Err(_) => continue,
        };
        if std::fs::metadata(&target)
            .map(|m| m.len() == src_size)
            .unwrap_or(false)
        {
            continue;
        }
        std::fs::copy(entry.path(), &target)?;
        copied += 1;
    }
    // 整轮 walk 跑完（没中途 `?` 失败）才写标记 —— 下次启动看标记直接短路。
    // 用 create_new 而不是 write，避免覆盖一份可能已经记录了"上次部分完成"的旧标记。
    if let Some(parent) = marker.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(marker)
    {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // 并发启动：另一个进程抢先写了标记，认。
        }
        Err(e) => return Err(anyhow::anyhow!("write migration marker: {e}")),
    }
    Ok(copied)
}

/// Android 启动时把 APK assets 里的种子库释放到 knowledge/。
/// 桌面端是 no-op —— 桌面不打包任何资源。
#[cfg(target_os = "android")]
pub fn materialize_seed(app: &tauri::AppHandle, root: &Path) -> Result<SeedStats> {
    use std::cell::Cell;
    // tauri::Manager 必须在 scope —— `app.path()` 是 Manager trait 的方法，不是 inherent。
    use tauri::{Emitter, Manager};
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
        let marker = dir.path().join(".legacy-migrated");
        std::fs::create_dir_all(src.join("a")).unwrap();
        std::fs::write(src.join("a/x.md"), b"x").unwrap();
        let n = migrate_legacy_bundled(&src, &dest, &marker).unwrap();
        assert_eq!(n, 1);
        assert_eq!(std::fs::read(dest.join("a/x.md")).unwrap(), b"x");
        assert!(marker.exists(), "迁移完成后必须留下标记");
    }

    /// 标记守门：第一次迁移完成后用户编辑了 dest 里的文件（长度变了），
    /// 再启动一次 —— **绝不能**因为大小不一而覆盖用户编辑。
    /// 旧"按字节比对"的实现就是栽在这里；这里钉住标记闸门契约。
    #[test]
    fn migrate_legacy_bundled_does_not_overwrite_user_edits_after_marker() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("old");
        let dest = dir.path().join("new");
        let marker = dir.path().join(".legacy-migrated");
        // src: 一份原始内容
        std::fs::create_dir_all(src.join("a")).unwrap();
        std::fs::write(src.join("a/x.md"), b"original-content").unwrap();

        // 第一次跑：迁移 + 写标记
        let first = migrate_legacy_bundled(&src, &dest, &marker).unwrap();
        assert_eq!(first, 1);

        // 用户编辑（长度变了）
        std::fs::write(dest.join("a/x.md"), b"USER EDIT, totally different length now").unwrap();

        // 模拟"下次启动"：再调一次
        let second = migrate_legacy_bundled(&src, &dest, &marker).unwrap();
        assert_eq!(second, 0, "标记存在时整个迁移必须短路，不应覆盖");
        assert_eq!(
            std::fs::read(dest.join("a/x.md")).unwrap(),
            b"USER EDIT, totally different length now",
            "用户编辑必须原样保留"
        );
    }

    /// 模拟"上次迁移中断，只搬了一半"：标记还没写。再跑一次应只补齐缺失，
    /// 已就位且大小相同的文件不被覆盖。Marker 一旦被写 —— 下一轮就完全短路。
    #[test]
    fn migrate_legacy_bundled_resumes_after_partial_copy() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("old");
        let dest = dir.path().join("new");
        let marker = dir.path().join(".legacy-migrated");
        // src 准备两份文件
        std::fs::create_dir_all(src.join("a")).unwrap();
        std::fs::write(src.join("a/x.md"), b"x-content").unwrap();
        std::fs::write(src.join("a/y.md"), b"y-content").unwrap();
        // dest 模拟"上次中断"：x 已就位且大小相同，y 缺失；marker 不在
        std::fs::create_dir_all(dest.join("a")).unwrap();
        std::fs::write(dest.join("a/x.md"), b"x-content").unwrap();
        assert!(!marker.exists());

        let n = migrate_legacy_bundled(&src, &dest, &marker).unwrap();
        assert_eq!(n, 1, "只补一份 y.md");
        assert_eq!(std::fs::read(dest.join("a/x.md")).unwrap(), b"x-content");
        assert_eq!(std::fs::read(dest.join("a/y.md")).unwrap(), b"y-content");
        assert!(marker.exists(), "恢复跑完后也要写标记");
    }

    /// dest 里有一个空子目录（用户不小心建的），不能被当成"已经迁过了"卡死迁移。
    /// 新实现按文件是否就位判断 —— 应继续迁完整棵树。
    #[test]
    fn migrate_legacy_bundled_ignores_stray_empty_dir_in_dest() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("old");
        let dest = dir.path().join("new");
        let marker = dir.path().join(".legacy-migrated");
        std::fs::create_dir_all(src.join("a")).unwrap();
        std::fs::write(src.join("a/x.md"), b"x").unwrap();
        // dest 故意留一个空的子目录
        std::fs::create_dir_all(dest.join("orphan")).unwrap();

        let n = migrate_legacy_bundled(&src, &dest, &marker).unwrap();
        assert_eq!(n, 1, "空子目录不能阻挡迁移");
        assert_eq!(std::fs::read(dest.join("a/x.md")).unwrap(), b"x");
    }

    /// src 不存在（典型新装场景，bundled.resources 已不再打进去）→ 直接 0，
    /// 不创建空的 dest 浪费 I/O，**也不**写标记 —— 保留未来"bundled 重新
    /// 引入 → 自动触发迁移"的退路。
    #[test]
    fn migrate_legacy_bundled_noop_when_src_missing() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("does-not-exist");
        let dest = dir.path().join("new");
        let marker = dir.path().join(".legacy-migrated");
        let n = migrate_legacy_bundled(&src, &dest, &marker).unwrap();
        assert_eq!(n, 0);
        assert!(!dest.exists(), "src 不存在时不应创建 dest");
        assert!(!marker.exists(), "src 不存在时不应写标记");
    }

    /// 用户编辑之后再写 marker → 迁移**仍**是 0（已被 marker 拦下）。
    /// 双保险：即便 marker 真的没写，第二次 walk 也会按字节比对跳过（因为
    /// 用户编辑后的大小恰好与 src 不同 —— 这是"用户文件"覆盖的隐藏 bug）。
    /// 这个测试是给那条防御的最终兜底：只要 marker 写过，就绝不可能再 walk。
    #[test]
    fn migrate_legacy_bundled_short_circuits_on_marker_even_with_user_edits() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("old");
        let dest = dir.path().join("new");
        let marker = dir.path().join(".legacy-migrated");
        std::fs::create_dir_all(src.join("a")).unwrap();
        std::fs::write(src.join("a/x.md"), b"src-content").unwrap();

        // 模拟"之前的迁移"：dest 已被填好（用户用 bundled 版）
        std::fs::create_dir_all(dest.join("a")).unwrap();
        std::fs::write(dest.join("a/x.md"), b"src-content").unwrap();

        // 写好 marker
        std::fs::write(&marker, b"").unwrap();

        // 用户编辑 dest
        std::fs::write(dest.join("a/x.md"), b"user-edit").unwrap();

        // 再启动一次：marker 存在 → 短路 → 用户编辑保留
        let n = migrate_legacy_bundled(&src, &dest, &marker).unwrap();
        assert_eq!(n, 0);
        assert_eq!(std::fs::read(dest.join("a/x.md")).unwrap(), b"user-edit");
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
