//! 知识库远程同步：manifest 解析、路径规范化、下载与落盘。
//!
//! 设计要点（见 docs/superpowers/specs/2026-10-08-android-support-design.md）：
//! - manifest 里的 `path` 相对 `knowledge/`，必须先过 `normalize_rel_path`
//!   再拼绝对路径，杜绝 `..` 目录穿越。
//! - 单个条目的路径/分类不合法时**跳过**而不是让整份 manifest 失败 —— 一条坏数据
//!   不该让 380 篇文档全部同步不上。
use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub files: Vec<ManifestFile>,
    #[serde(default)]
    pub indexes: Vec<ManifestFile>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ManifestFile {
    pub path: String,
    pub size: u64,
    #[serde(default)]
    pub sha256: Option<String>,
}

pub fn parse_manifest(json: &str) -> Result<Manifest> {
    let raw: Manifest = serde_json::from_str(json)?;
    Ok(Manifest {
        version: raw.version,
        files: sanitize(raw.files),
        indexes: sanitize(raw.indexes),
    })
}

/// 过滤掉路径不合法 / 没有分类的条目，并把反斜杠归一成 `/`。
fn sanitize(mut entries: Vec<ManifestFile>) -> Vec<ManifestFile> {
    entries.retain_mut(|e| match normalize_rel_path(&e.path) {
        Ok(normalized) if split_category(&normalized).is_some() => {
            e.path = normalized;
            true
        }
        _ => false,
    });
    entries
}

/// 归一化相对路径：反斜杠 → `/`，去掉前导 `./` 与 `/`。
/// 拒绝绝对路径、Windows 盘符与任何 `..` 段。
pub fn normalize_rel_path(raw: &str) -> Result<String> {
    let mut s = raw.trim().replace('\\', "/");
    // 绝对路径必须在剥离前直接拒绝（避免 `/etc/passwd` 被归一成 `etc/passwd`）
    if s.starts_with('/') {
        anyhow::bail!("manifest path must not be absolute: {raw}");
    }
    while let Some(rest) = s.strip_prefix("./") {
        s = rest.to_string();
    }
    let s = s.trim_start_matches('/').to_string();
    if s.is_empty() {
        anyhow::bail!("manifest path is empty");
    }
    if s.contains(':') {
        anyhow::bail!("manifest path must not contain ':': {raw}");
    }
    if s.split('/').any(|seg| seg == ".." || seg == ".") {
        anyhow::bail!("manifest path must not traverse: {raw}");
    }
    Ok(s)
}

/// 拆成 (category_path, rel_path)。根目录下的文件返回 None。
pub fn split_category(path: &str) -> Option<(String, String)> {
    let idx = path.rfind('/')?;
    let category = &path[..idx];
    let rel = &path[idx + 1..];
    if category.is_empty() || rel.is_empty() {
        return None;
    }
    Some((category.to_string(), rel.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_rel_path_converts_backslash() {
        assert_eq!(normalize_rel_path("a\\b\\c.md").unwrap(), "a/b/c.md");
    }

    #[test]
    fn normalize_rel_path_strips_leading_slash_dot() {
        assert_eq!(normalize_rel_path("./a/b.md").unwrap(), "a/b.md");
    }

    #[test]
    fn normalize_rel_path_rejects_parent_traversal() {
        assert!(normalize_rel_path("../../etc/passwd").is_err());
        assert!(normalize_rel_path("a/../../b.md").is_err());
    }

    #[test]
    fn normalize_rel_path_rejects_absolute_and_empty() {
        assert!(normalize_rel_path("/etc/passwd").is_err());
        assert!(normalize_rel_path("C:/win.md").is_err());
        assert!(normalize_rel_path("").is_err());
    }

    #[test]
    fn split_category_splits_at_last_slash() {
        assert_eq!(
            split_category("01-基础/线性回归.md"),
            Some(("01-基础".into(), "线性回归.md".into()))
        );
        // 根目录下的文件没有分类 —— scanner 同样拒绝根目录资源
        assert_eq!(split_category("root.md"), None);
    }

    #[test]
    fn parse_manifest_reads_files_and_indexes() {
        let json = r#"{
            "version": "2026-10-08",
            "files": [{"path": "01-基础/a.md", "size": 12, "sha256": "abc"}],
            "indexes": [{"path": "01-基础/_index.md", "size": 30}]
        }"#;
        let m = parse_manifest(json).unwrap();
        assert_eq!(m.files.len(), 1);
        assert_eq!(m.files[0].size, 12);
        assert_eq!(m.files[0].sha256.as_deref(), Some("abc"));
        assert_eq!(m.indexes.len(), 1);
        // sha256 缺省时为 None，不报错（首版只用 size 校验）
        assert_eq!(m.indexes[0].sha256, None);
    }

    #[test]
    fn parse_manifest_skips_entries_with_bad_paths_instead_of_failing() {
        let json = r#"{
            "version": "v",
            "files": [
                {"path": "01-基础/good.md", "size": 3},
                {"path": "../../evil.md", "size": 3},
                {"path": "no-slash.md", "size": 3}
            ]
        }"#;
        let m = parse_manifest(json).unwrap();
        // 只有一条既路径合法、又有分类的条目被保留
        assert_eq!(m.files.len(), 1);
        assert_eq!(m.files[0].path, "01-基础/good.md");
    }

    #[test]
    fn parse_manifest_rejects_malformed_json() {
        assert!(parse_manifest("{ not json").is_err());
    }
}
