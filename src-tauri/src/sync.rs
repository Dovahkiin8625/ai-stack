//! 知识库远程同步：manifest 解析、路径规范化、下载与落盘。
//!
//! 设计要点（见 docs/superpowers/specs/2026-10-08-android-support-design.md）：
//! - manifest 里的 `path` 相对 `knowledge/`，必须先过 `normalize_rel_path`
//!   再拼绝对路径，杜绝 `..` 目录穿越。
//! - 单个条目的路径/分类不合法时**跳过**而不是让整份 manifest 失败 —— 一条坏数据
//!   不该让 380 篇文档全部同步不上。
use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

/// 下载来源抽象。生产用 HttpFetcher；单测用 FakeFetcher，不碰网络。
///
/// **线程约束（务必看）**：`get` 是同步阻塞调用，必须在没有任何 tokio
/// 运行时的线程上执行 —— 也就是说，**不能在 Tauri 命令线程或 `spawn_blocking`
/// 派生的 worker 上调用**。`reqwest::blocking` 内部依赖自己专属的 I/O reactor，
/// 跟 tokio 的 reactor 互相冲突；在 tokio runtime 内构造或使用会直接 panic。
/// 正确的派发方式见 `HttpFetcher::new` 的文档。
///
/// 之所以把 trait 设计成同步而非 async，是为了让单测能直接用 FakeFetcher 替代，
/// 不需要拉起任何运行时；命令层必须自己负责把调用搬到合适的线程上。
pub trait Fetcher: Send + Sync {
    fn get(&self, url: &str) -> Result<Vec<u8>>;
}

/// 当前线程是否在某个 tokio runtime 内（包括 `#[tokio::test]`、Tauri 命令线程、
/// `spawn_blocking` worker 等）。
///
/// `HttpFetcher::new` 借此提早 fail，避免到 `reqwest::blocking` 内部才 panic。
/// `pub(crate)` 仅为单测可见 —— 调用方只关心它返回的错误信息。
pub(crate) fn in_async_runtime() -> bool {
    tokio::runtime::Handle::try_current().is_ok()
}

pub struct HttpFetcher {
    client: reqwest::blocking::Client,
}

impl HttpFetcher {
    /// 构造一个 HTTP fetcher。**必须在没有 tokio 运行时的线程上调用**——
    /// 推荐用 `std::thread::spawn`，等结果用 `tokio::sync::oneshot` 传回，
    /// 不要用 `tauri::async_runtime::spawn_blocking` 也不要直接 await。
    pub fn new() -> Result<Self> {
        if in_async_runtime() {
            anyhow::bail!(
                "HttpFetcher 不能在 tokio 运行时的线程上构造或使用：\
                 reqwest::blocking 与 tokio 的 reactor 互斥，在此处会 panic。\
                 请改用 std::thread::spawn 派生下载线程，并通过 oneshot 把结果传回 Tauri 命令。"
            );
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?;
        Ok(Self { client })
    }
}

impl Fetcher for HttpFetcher {
    fn get(&self, url: &str) -> Result<Vec<u8>> {
        let bytes = self
            .client
            .get(url)
            .send()?
            .error_for_status()?
            .bytes()?;
        Ok(bytes.to_vec())
    }
}

pub fn remote_url(base_url: &str, path: &str) -> String {
    format!("{}/{}", base_url.trim_end_matches('/'), path)
}

/// 下载中的临时文件目录 —— 必须是 knowledge/ 的**同级**，绝不能落在 knowledge/ 里面：
/// scanner 会 walk 整个 knowledge/，多出来的目录会被当成一个新分类。
pub fn tmp_dir_for(root: &Path) -> PathBuf {
    root.parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".tmp")
}

#[derive(Debug, Clone)]
pub struct PendingFile {
    pub category_path: String,
    pub rel_path: String,
    pub size: u64,
}

fn local_path(root: &Path, entry: &PendingFile) -> PathBuf {
    // path 已在 upsert 阶段过 normalize_rel_path，这里再拼一次以防万一
    root.join(&entry.category_path).join(&entry.rel_path)
}

fn temp_name(entry: &PendingFile) -> String {
    format!(
        "{}.{}.part",
        entry.category_path.replace('/', "__"),
        entry.rel_path.replace('/', "__")
    )
}

/// 确保本地文件就位。`base_url` 为空表示纯本地模式，不允许触发任何下载。
/// 本地已有且大小与 manifest 一致 → 直接返回，完全不联网。
pub fn ensure_local_at(
    root: &Path,
    entry: &PendingFile,
    base_url: &str,
    fetcher: &dyn Fetcher,
) -> Result<()> {
    let dest = local_path(root, entry);
    if let Ok(meta) = std::fs::metadata(&dest) {
        if meta.len() == entry.size {
            return Ok(());
        }
    }
    if base_url.trim().is_empty() {
        anyhow::bail!(
            "{} 尚未缓存，且未配置知识库同步地址",
            entry.rel_path
        );
    }
    let tmp_dir = tmp_dir_for(root);
    std::fs::create_dir_all(&tmp_dir)?;
    let tmp = tmp_dir.join(temp_name(entry));
    let url = remote_url(
        base_url,
        &format!("{}/{}", entry.category_path, entry.rel_path),
    );
    let body = fetcher
        .get(&url)
        .map_err(|e| anyhow::anyhow!("下载 {url} 失败：{e}"))?;
    if body.len() as u64 != entry.size {
        let _ = std::fs::remove_file(&tmp);
        anyhow::bail!(
            "下载校验失败 {}：收到 {} 字节，服务器声明 {} 字节",
            entry.rel_path,
            body.len(),
            entry.size
        );
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&tmp, &body)?;
    std::fs::rename(&tmp, &dest)?;
    Ok(())
}

/// 并发（最多 3 路）下载全部待缓存文件。单个文件失败不中断整批，
/// 最后一个参数把 (已完成, 总数) 报给调用方用于进度条。
pub fn download_all<F: Fetcher + 'static>(
    root: &Path,
    base_url: &str,
    pending: Vec<PendingFile>,
    fetcher: Arc<F>,
    on_progress: &(dyn Fn(usize, usize) + Sync),
) -> Result<()> {
    let total = pending.len();
    if total == 0 {
        on_progress(0, 0);
        return Ok(());
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let root_ref = root.to_path_buf();
    let base = base_url.to_string();
    let progress: &(dyn Fn(usize, usize) + Sync) = on_progress;
    let queue = std::sync::Mutex::new(pending);
    std::thread::scope(|scope| {
        for _ in 0..3 {
            let next = &next;
            let done = &done;
            let queue = &queue;
            let root_ref = &root_ref;
            let base = &base;
            let fetcher = fetcher.clone();
            scope.spawn(move || loop {
                let item = {
                    let q = queue.lock().unwrap();
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    q.get(i).cloned()
                };
                let Some(item) = item else {
                    break;
                };
                // 单个文件失败不拖垮整批 —— 用户点「下载全部」时一两个 404 不该让整次失败
                let _ = ensure_local_at(&root_ref, &item, &base, fetcher.as_ref());
                let d = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                progress(d, total);
            });
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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

    // ---- 下载器 ----

    struct FakeFetcher {
        map: std::collections::HashMap<String, Vec<u8>>,
        calls: std::sync::Mutex<Vec<String>>,
    }

    impl FakeFetcher {
        fn new(pairs: &[(&str, &[u8])]) -> Self {
            Self {
                map: pairs.iter().map(|(k, v)| (k.to_string(), v.to_vec())).collect(),
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl Fetcher for FakeFetcher {
        fn get(&self, url: &str) -> Result<Vec<u8>> {
            self.calls.lock().unwrap().push(url.to_string());
            self.map
                .get(url)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("404 {url}"))
        }
    }

    #[test]
    fn remote_url_joins_base_and_path() {
        assert_eq!(
            remote_url("https://x.com/kb/", "a/b.md"),
            "https://x.com/kb/a/b.md"
        );
        assert_eq!(remote_url("https://x.com/kb", "a/b.md"), "https://x.com/kb/a/b.md");
    }

    #[test]
    fn ensure_local_skips_download_when_file_already_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(root.join("a/x.md"), b"hello").unwrap();
        let f = FakeFetcher::new(&[]);
        ensure_local_at(&root, &PendingFile { category_path: "a".into(), rel_path: "x.md".into(), size: 5 }, "https://x/kb", &f).unwrap();
        assert!(f.calls.lock().unwrap().is_empty(), "本地已有且大小相符，不应发请求");
    }

    #[test]
    fn ensure_local_redownloads_when_size_differs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(root.join("a/x.md"), b"stale").unwrap();
        let f = FakeFetcher::new(&[("https://x/kb/a/x.md", b"fresh-content")]);
        ensure_local_at(&root, &PendingFile { category_path: "a".into(), rel_path: "x.md".into(), size: 13 }, "https://x/kb", &f).unwrap();
        assert_eq!(std::fs::read(root.join("a/x.md")).unwrap(), b"fresh-content");
    }

    #[test]
    fn ensure_local_downloads_missing_file_via_tmp_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(&root).unwrap();
        let f = FakeFetcher::new(&[("https://x/kb/a/y.md", b"body")]);
        ensure_local_at(&root, &PendingFile { category_path: "a".into(), rel_path: "y.md".into(), size: 4 }, "https://x/kb", &f).unwrap();
        assert_eq!(std::fs::read(root.join("a/y.md")).unwrap(), b"body");
        // 临时文件不能留在 knowledge/ 里（否则 scanner 会把它扫成新分类）
        let leftovers: Vec<_> = walkdir::WalkDir::new(&root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && e.file_name() != "y.md")
            .collect();
        assert!(leftovers.is_empty(), "knowledge/ 里残留了临时文件");
    }

    #[test]
    fn ensure_local_leaves_no_partial_file_when_download_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(&root).unwrap();
        let f = FakeFetcher::new(&[]); // 一律 404
        let err = ensure_local_at(&root, &PendingFile { category_path: "a".into(), rel_path: "z.md".into(), size: 9 }, "https://x/kb", &f);
        assert!(err.is_err());
        assert!(!root.join("a/z.md").exists(), "失败后不能留下半个文件");
        // 整个 knowledge/ 里也不该有任何残留文件 —— 防住未来把 temp 写到
        // <root>/a/z.md.part 之类的实现回归（单看目的路径查不出来）。
        let leftovers: Vec<_> = walkdir::WalkDir::new(&root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .collect();
        assert!(
            leftovers.is_empty(),
            "失败后 knowledge/ 里残留了文件：{:?}",
            leftovers.iter().map(|e| e.path()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn ensure_local_rejects_size_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(&root).unwrap();
        let f = FakeFetcher::new(&[("https://x/kb/a/s.md", b"12345")]); // manifest 说 100 字节
        let err = ensure_local_at(&root, &PendingFile { category_path: "a".into(), rel_path: "s.md".into(), size: 100 }, "https://x/kb", &f);
        assert!(err.is_err());
        assert!(!root.join("a/s.md").exists(), "校验不过不能落盘");
    }

    #[test]
    fn download_all_processes_every_pending_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(&root).unwrap();
        let f = Arc::new(FakeFetcher::new(&[
            ("https://x/kb/a/1.md", b"1"),
            ("https://x/kb/a/2.md", b"2"),
            ("https://x/kb/b/3.md", b"3"),
        ]));
        let pending = vec![
            PendingFile { category_path: "a".into(), rel_path: "1.md".into(), size: 1 },
            PendingFile { category_path: "a".into(), rel_path: "2.md".into(), size: 1 },
            PendingFile { category_path: "b".into(), rel_path: "3.md".into(), size: 1 },
        ];
        let seen = std::sync::Mutex::new(Vec::new());
        download_all(&root, "https://x/kb", pending, f.clone(), &|d, t| {
            seen.lock().unwrap().push((d, t));
        }).unwrap();
        assert_eq!(f.calls.lock().unwrap().len(), 3);
        assert_eq!(std::fs::read(root.join("b/3.md")).unwrap(), b"3");
        assert_eq!(seen.lock().unwrap().last().copied(), Some((3, 3)));
    }

    // ---- HttpFetcher 在 tokio 运行时内的检测 ----
    //
    // reqwest::blocking 在 tokio runtime 内构造会 panic，所以我们在
    // HttpFetcher::new() 入口加了一道提前检查。这里测的是这道检查背后的
    // helper (`in_async_runtime`)，外加通过 HttpFetcher::new() 验证它真的接上了。

    #[test]
    fn in_async_runtime_is_false_on_an_ordinary_thread() {
        // cargo test 默认线程不在任何 tokio runtime 内
        assert!(!in_async_runtime());
    }

    #[tokio::test]
    async fn in_async_runtime_is_true_inside_a_tokio_runtime() {
        // #[tokio::test] 默认启用了 current-thread runtime
        assert!(in_async_runtime());
    }

    #[test]
    fn http_fetcher_new_bails_with_actionable_error_inside_runtime() {
        // 在 tokio runtime 内构造 HttpFetcher 必须返回错误而不是 panic，
        // 否则 Tauri 命令线程上误用会让窗口闪退。
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(async { HttpFetcher::new() });
        match result {
            Ok(_) => panic!("在 tokio runtime 内构造 HttpFetcher 必须失败"),
            Err(err) => {
                let msg = format!("{err:#}");
                assert!(
                    msg.contains("tokio") && msg.contains("std::thread::spawn"),
                    "错误信息应当点名 root cause 与正确派发方式，实际：{msg}",
                );
            }
        }
    }
}
