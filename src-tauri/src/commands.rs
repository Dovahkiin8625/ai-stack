//! Tauri 命令入口
use crate::db::{Db, NoteRow, ResourceRow};
use crate::reader::{self, ResourceContent};
use crate::scanner::{self, ScanConfig, ScanSummary};
use crate::sync::{Fetcher, HttpFetcher, Manifest, PendingFile};
use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

/// 跨命令共享的不可变路径配置。
/// 在 setup 阶段初始化一次；每个命令按需 `clone`。
pub struct AppState {
    pub db_path: PathBuf,
    pub knowledge_root: PathBuf,
    /// 共享的 HTTP fetcher。setup() 在没有 tokio runtime 的主线程上构造一次，
    /// 命令层 clone Arc 后放到 std::thread::spawn 派生线程上用 —— 绝不能在 tokio
    /// worker 线程上调用其 get()，否则 reqwest::blocking 与 tokio reactor 互斥会 panic。
    /// 测试场景下传 None，命中 base_url 为空路径时不接触 fetcher。
    pub http_fetcher: Option<Arc<HttpFetcher>>,
}

fn ensure_db_dir(p: &PathBuf) {
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).ok();
    }
}

fn resolve_db_path(app: &AppHandle) -> PathBuf {
    let mut p = app
        .path()
        .app_data_dir()
        .expect("app_data_dir resolvable");
    p.push("ai-stack");
    std::fs::create_dir_all(&p).ok();
    p.push("ai-stack.db");
    p
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CategoryDto {
    pub path: String,
    pub parent_path: Option<String>,
    pub title: String,
    pub sort_order: i64,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ResourceDto {
    pub id: i64,
    pub category_path: String,
    pub rel_path: String,
    pub r#type: String,
    pub title: String,
    pub size_bytes: i64,
    pub indexed_at: String,
    pub page_count: Option<i64>,
    pub word_count: Option<i64>,
    /// 文件是否已缓存到本地。前端据此显示「下载」/「已下载」徽标，
    /// 替代早期版本的「根据 path 是否存在推断」做法。
    pub present: bool,
}

impl From<ResourceRow> for ResourceDto {
    fn from(r: ResourceRow) -> Self {
        Self {
            id: r.id,
            category_path: r.category_path,
            rel_path: r.rel_path,
            r#type: r.r#type,
            title: r.title,
            size_bytes: r.size_bytes,
            indexed_at: r.indexed_at,
            page_count: r.page_count,
            word_count: r.word_count,
            present: r.present,
        }
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct NoteDto {
    pub id: i64,
    pub resource_id: i64,
    pub content: String,
    pub anchor_text: Option<String>,
    pub anchor_occurrence: i64,
    pub prompt: Option<String>,
    pub source: String,
    /// PDF 笔记用的 0-based 页码。markdown/DOCX 笔记为 None。
    pub page_idx: Option<i64>,
    pub created_at: String,
    pub updated_at: String,
}

impl From<NoteRow> for NoteDto {
    fn from(n: NoteRow) -> Self {
        Self {
            id: n.id,
            resource_id: n.resource_id,
            content: n.content,
            anchor_text: n.anchor_text,
            anchor_occurrence: n.anchor_occurrence,
            prompt: n.prompt,
            source: n.source,
            page_idx: n.page_idx,
            created_at: n.created_at,
            updated_at: n.updated_at,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotePayload {
    pub resource_id: i64,
    pub content: String,
    pub anchor_text: Option<String>,
    pub anchor_occurrence: i64,
    /// 笔记来源：'user' 人工添加 / 'ai' 大模型讲解。前端不传时默认 'user'。
    #[serde(default)]
    pub source: String,
    /// PDF 笔记用的 0-based 页码。markdown/DOCX 笔记不传或传 null。
    #[serde(default)]
    pub page_idx: Option<i64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NoteUpdatePayload {
    pub id: i64,
    pub content: String,
    pub anchor_text: Option<String>,
    pub anchor_occurrence: i64,
    /// 编辑 AI 笔记时降级为 'user'；不传或为空则保留原 source。
    #[serde(default)]
    pub source: Option<String>,
    /// 改 PDF 笔记页码时使用；不传或为 null 则保留原 page_idx（翻页跳转不失效）。
    #[serde(default)]
    pub page_idx: Option<i64>,
}

#[tauri::command]
pub async fn scan_library(
    force: bool,
    app: AppHandle,
) -> Result<ScanSummary, String> {
    let state: tauri::State<AppState> = app.state();
    let db_path = state.db_path.clone();
    let knowledge_root = state.knowledge_root.clone();
    ensure_db_dir(&db_path);
    let app2 = app.clone();
    let progress_emit = move |phase: &str| {
        let _ = app2.emit(
            "scan_progress",
            serde_json::json!({"phase": phase, "current": 0, "total": 0}),
        );
    };
    tauri::async_runtime::spawn_blocking(move || -> anyhow::Result<ScanSummary> {
        let mut db = Db::open(&db_path)
            .with_context(|| format!("open {}", db_path.display()))?;
        db.migrate().context("migrate")?;
        if force {
            db.conn.execute("DELETE FROM resources", []).ok();
            db.conn.execute("DELETE FROM categories", []).ok();
        }
        progress_emit("walking");
        let cfg = ScanConfig { knowledge_root: knowledge_root.clone() };
        let summary = scanner::scan(&db, &cfg)?;
        progress_emit("done");
        Ok(summary)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_categories(app: AppHandle) -> Result<Vec<CategoryDto>, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let cats = db.list_categories().map_err(|e| e.to_string())?;
    Ok(cats
        .into_iter()
        .map(|c| CategoryDto {
            path: c.path,
            parent_path: c.parent_path,
            title: c.title,
            sort_order: c.sort_order,
        })
        .collect())
}

#[tauri::command]
pub fn list_resources(
    category_path: String,
    app: AppHandle,
) -> Result<Vec<ResourceDto>, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let rows = db.list_resources(&category_path).map_err(|e| e.to_string())?;
    Ok(rows.into_iter().map(ResourceDto::from).collect())
}

#[tauri::command]
pub async fn read_resource(id: i64, app: AppHandle) -> Result<ResourceContent, String> {
    // 必须异步：ensure_cached 在文件缺失时可能走同步 HTTP 下载，绝对不能在
    // Tauri 命令线程（=tokio runtime 线程）上跑 —— reqwest::blocking 会 panic。
    // 派生 std::thread 跑全部工作（DB 查询 + 补齐 + 读文件），oneshot 把结果
    // 桥回 async 上下文，UI 不冻结。
    let state: tauri::State<AppState> = app.state();
    let db_path = state.db_path.clone();
    let knowledge_root = state.knowledge_root.clone();
    let http_fetcher = state.http_fetcher.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let result = (|| -> anyhow::Result<ResourceContent> {
            let db = Db::open(&db_path).context("open db")?;
            let (category_path, rel_path, kind) = lookup_resource_path_kind(&db, id)?;
            let local_state = AppState {
                db_path: db_path.clone(),
                knowledge_root: knowledge_root.clone(),
                http_fetcher: http_fetcher.clone(),
            };
            ensure_cached(&local_state, &db, id, &category_path, &rel_path)
                .map_err(|e| anyhow::anyhow!("{e:#}"))?;
            let abs = reader::resolve_absolute(&knowledge_root, &category_path, &rel_path);
            reader::read(&abs, &kind).map_err(|e| anyhow::anyhow!("read {}: {e:#}", abs.display()))
        })();
        let _ = tx.send(result.map_err(|e| e.to_string()));
    });
    rx.await.map_err(|e| format!("download worker join: {e}"))?
}

/// 编辑模式的核心 helper：把新 markdown 写回文件 → 重新过 markdown_extract 拿新
/// html/word 数 → 更新 resources 行的 size/word_count/indexed_at → 返回 (html, words)。
///
/// 拆成独立 pub 函数（而不是直接 inline 在 Tauri 命令里）有两个目的：
/// 1. 测试不依赖 AppHandle，可以裸调用 Db + 路径
/// 2. 命令端只剩"查 DB 找路径 + 调 helper"两件事，肉眼可审
///
/// 允许空 markdown —— 用户清空文件是合法操作；前端可以在保存前自检路径下是否还有
/// 其它展示用的数据，但服务端不替它做内容策略判断。
pub fn write_resource_markdown(
    db: &Db,
    resource_id: i64,
    abs: &Path,
    markdown: &str,
) -> Result<(String, usize)> {
    // 1. 写文件
    std::fs::write(abs, markdown)
        .with_context(|| format!("write {}", abs.display()))?;
    // 3. 重新过 markdown_extract —— 保证返回的 html 是新内容（不是旧缓存）
    let (html, words, _source) = crate::readers::markdown_extract(abs)?;
    // 4. 同步刷新 DB 行的 size/word_count/indexed_at，否则下次 scanner 会用
    //    旧 size 当差异判断依据，永远不再重新 extract 这条资源
    db.update_resource_after_write(resource_id, markdown.len() as i64, words as i64)
        .context("update resource row after write")?;
    Ok((html, words))
}

#[tauri::command]
pub fn write_resource(
    id: i64,
    markdown: String,
    app: AppHandle,
) -> Result<ResourceContent, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    // 复用 read_resource 的查 path + kind 逻辑：把 kind/路径 一次拿出来。
    // 走 lookup_resource_path_kind 让"找不到"错误也走中文 + 操作指引路径，
    // 否则用户编辑一篇文章时同步把它清掉，会看到英文 + 原始 rusqlite 字符串。
    let (category_path, rel_path, kind) =
        lookup_resource_path_kind(&db, id).map_err(|e| format!("{e:#}"))?;
    if kind != "markdown" {
        return Err(format!(
            "write_resource 仅支持 markdown 资源，resource {id} 是 {kind}"
        ));
    }
    let abs = reader::resolve_absolute(&state.knowledge_root, &category_path, &rel_path);
    let (html, word_count) =
        write_resource_markdown(&db, id, &abs, &markdown).map_err(|e| format!("{e:#}"))?;
    Ok(ResourceContent::Markdown {
        html,
        word_count,
        markdown,
    })
}

/// 返回资源的原始字节。
///
/// 三类 reader 都用它拿原文件自己渲染：
/// - PDF：前端 pdf.js `getDocument({data})`，自带 text layer 支持选中/复制
/// - DOCX：前端 mammoth.js 把 docx → semantic HTML（保留 bold/italic/headings/lists/tables/images）
/// - PPTX：前端 pptxviewjs 把 pptx → canvas slide（按 slide 翻页，完整保留版式）
///
/// markdown reader 不走这里（后端已经返回 html）。
///
/// 与 read_resource 同源考量：转 async，把潜在的网络下载放到 std::thread 上，
/// 避免 Tauri tokio runtime 线程与 reqwest::blocking reactor 互斥而 panic。
#[tauri::command]
pub async fn read_resource_bytes(id: i64, app: AppHandle) -> Result<Vec<u8>, String> {
    let state: tauri::State<AppState> = app.state();
    let db_path = state.db_path.clone();
    let knowledge_root = state.knowledge_root.clone();
    let http_fetcher = state.http_fetcher.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let result = (|| -> anyhow::Result<Vec<u8>> {
            let db = Db::open(&db_path).context("open db")?;
            let (category_path, rel_path, kind) = lookup_resource_path_kind(&db, id)?;
            if !matches!(kind.as_str(), "pdf" | "docx" | "pptx") {
                anyhow::bail!("read_resource_bytes 仅支持 PDF/DOCX/PPTX，resource {id} 是 {kind}");
            }
            let local_state = AppState {
                db_path: db_path.clone(),
                knowledge_root: knowledge_root.clone(),
                http_fetcher: http_fetcher.clone(),
            };
            ensure_cached(&local_state, &db, id, &category_path, &rel_path)
                .map_err(|e| anyhow::anyhow!("{e:#}"))?;
            let abs = reader::resolve_absolute(&knowledge_root, &category_path, &rel_path);
            std::fs::read(&abs).map_err(|e| anyhow::anyhow!("read {}: {e}", abs.display()))
        })();
        let _ = tx.send(result.map_err(|e| e.to_string()));
    });
    rx.await.map_err(|e| format!("download worker join: {e}"))?
}

/// 读取子分类目录下的 `_index.md` 并解析为 `[{ relPath, title, description }]`。
///
/// 给中间区目录列表的"每行附带描述"用。
/// 不入库（编辑 `_index.md` 不必触发重扫），每次调用现读现解析，路径 1KB 内的文件
/// 解析开销可以忽略；前端按 categoryPath 在 store 里缓存。
///
/// 异步实现：文件缺失时若 manifest 标记了 `_index.md` 且配置了 base_url，
/// 会从远端补齐；该 HTTP 下载必须在 std::thread 上跑以避开 tokio runtime。
/// 即使 base_url 缺失 / index_files 缺记录 / 下载失败，原 `parse_file` 仍走原本
/// 的"目录缺少 _index.md"错误路径，前端据此渲染提示 —— 不把网络失败变成硬错误。
#[tauri::command]
pub async fn read_subcategory_index(
    category_path: String,
    app: AppHandle,
) -> Result<crate::readers::index_md::ParsedIndex, String> {
    let state: tauri::State<AppState> = app.state();
    let db_path = state.db_path.clone();
    let knowledge_root = state.knowledge_root.clone();
    let http_fetcher = state.http_fetcher.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let result = (|| -> anyhow::Result<crate::readers::index_md::ParsedIndex> {
            // _index.md 在该子分类目录下，文件名固定。
            let abs = knowledge_root.join(&category_path).join("_index.md");
            if !abs.is_file() {
                // 打开 DB 查 index_files：仅当 manifest 标记了此 index 且未 present 时才尝试下载。
                let db = Db::open(&db_path).context("open db for index lookup")?;
                let index_path = format!("{category_path}/_index.md");
                if let Some((size, present)) = db.index_file(&index_path)? {
                    if !present {
                        let base_url = db
                            .get_config("sync_base_url")?
                            .unwrap_or_default();
                        if !base_url.trim().is_empty() {
                            if let Some((cat, rel)) = index_path
                                .split_once('/')
                                .map(|(a, b)| (a.to_string(), b.to_string()))
                            {
                                let local_state = AppState {
                                    db_path: db_path.clone(),
                                    knowledge_root: knowledge_root.clone(),
                                    http_fetcher: http_fetcher.clone(),
                                };
                                let entry = PendingFile {
                                    category_path: cat,
                                    rel_path: rel,
                                    size,
                                };
                                // 下载失败也吞掉 —— 仍让 parse_file 返回原来的
                                // "目录缺少 _index.md" 错误，前端据此提示用户。
                                if ensure_index_file(
                                    &local_state,
                                    &entry,
                                    &base_url,
                                )
                                .is_ok()
                                    && abs.is_file()
                                {
                                    db.mark_index_present(&index_path)?;
                                }
                            }
                        }
                    }
                }
            }
            crate::readers::index_md_parse_file(&abs)
                .map_err(|e| anyhow::anyhow!("parse _index.md for {category_path}: {e:#}"))
        })();
        let _ = tx.send(result);
    });
    rx.await
        .map_err(|e| format!("index download worker join: {e}"))?
        .map_err(|e| format!("{e:#}"))
}

// === Task 4: 同步命令族 ===

/// DB 里 `present=0` 的远端条目 —— 待下载队列。
/// 仅看 manifest 管理的行（remote_hash IS NOT NULL）；纯本地行不会出现在这里。
pub fn pending_rows(db: &Db) -> Result<Vec<PendingFile>> {
    let mut stmt = db.conn.prepare(
        "SELECT category_path, rel_path, size_bytes FROM resources
         WHERE remote_hash IS NOT NULL AND present = 0",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok(PendingFile {
                category_path: r.get(0)?,
                rel_path: r.get(1)?,
                size: r.get::<_, i64>(2)?.max(0) as u64,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// manifest 落库后把本地已经存在的文件标成 present=1
/// （用户可能在同步前就手动下载过；manifest 接管后也得反映"实际可用"状态）。
/// 同时清掉 manifest 已经不再列出的远端条目（`drop_resources_absent_from_manifest`）。
pub fn apply_manifest(db: &Db, knowledge_root: &Path, manifest: &Manifest) -> Result<()> {
    db.upsert_manifest(&manifest.files)?;
    db.upsert_index_files(&manifest.indexes)?;
    let keep: Vec<(String, String)> = manifest
        .files
        .iter()
        .filter_map(|f| crate::sync::split_category(&f.path))
        .collect();
    db.drop_resources_absent_from_manifest(&keep)?;
    for f in &manifest.files {
        let Some((cat, rel)) = crate::sync::split_category(&f.path) else {
            continue;
        };
        let abs = reader::resolve_absolute(knowledge_root, &cat, &rel);
        if std::fs::metadata(&abs)
            .map(|m| m.len() == f.size)
            .unwrap_or(false)
        {
            db.mark_resource_present(&cat, &rel)?;
        }
    }
    for f in &manifest.indexes {
        if std::fs::metadata(knowledge_root.join(&f.path))
            .map(|m| m.len() == f.size)
            .unwrap_or(false)
        {
            db.mark_index_present(&f.path)?;
        }
    }
    Ok(())
}

/// DB 查 资源的 (目录路径, 文件名, 类型)，找不到时返回**用户可读**的中文错误，
/// 不把 `rusqlite::Error::QueryReturnedNoRows` 的 `"Query returned no rows"` 原文
/// 透出到 UI —— 那是 RULING D 明令禁止的内部 jargon。
///
/// 抽出这个 helper 是为了：
/// 1. 三处调用（read_resource / read_resource_bytes / download_resource）共享
///    一份错误文案，避免日后再出现"某条路径漏改"的回归。
/// 2. 单测可以裸用 `&Db` 直接断言错误文本，不依赖 AppHandle。
pub fn lookup_resource_path_kind(
    db: &Db,
    id: i64,
) -> Result<(String, String, String)> {
    db.conn
        .query_row(
            "SELECT category_path, rel_path, type FROM resources WHERE id = ?1",
            [id],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)),
        )
        .map_err(|e| resource_lookup_error(e, id, false))
}

/// 同上，但 select 的是 size_bytes（供 download_resource 走 HTTP 时用）。
pub fn lookup_resource_for_download(
    db: &Db,
    id: i64,
) -> Result<(String, String, i64)> {
    db.conn
        .query_row(
            "SELECT category_path, rel_path, size_bytes FROM resources WHERE id = ?1",
            [id],
            |r| Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
            )),
        )
        .map_err(|e| resource_lookup_error(e, id, true))
}

/// 资源查表错误的统一翻译：找不到行 → 告诉用户去刷新；其它 → 套一层中文前缀
/// "读取知识库失败，请稍后重试"，再附上原始 Display —— 便于前端或后端日志排查
/// 时能看到底层错误，但**不会**让用户第一眼看到 raw SQL（"Query returned no
/// rows" / "database is locked" 等），也不会被错误地当成"资源不存在"提示。
fn resource_lookup_error(e: rusqlite::Error, id: i64, for_download: bool) -> anyhow::Error {
    match e {
        rusqlite::Error::QueryReturnedNoRows => {
            if for_download {
                anyhow::anyhow!(
                    "文章 id={id} 已不在知识库中（可能同步时被清掉），无法下载 —— 请返回列表刷新"
                )
            } else {
                anyhow::anyhow!(
                    "文章 id={id} 已不在知识库中（可能同步时被清掉），请返回列表刷新"
                )
            }
        }
        other => anyhow::anyhow!("读取知识库失败，请稍后重试：{other}"),
    }
}

/// `_index.md` 按需下载 —— 与 resources 共用 ensure_local_at，但 index_files 没有
/// `category_path` / `rel_path` 拆分（path 直接是分类相对路径），手工拼一下条目。
/// 失败时让上层静默忽略 —— 仍然走 "目录缺少 _index.md" 错误路径。
fn ensure_index_file(state: &AppState, entry: &PendingFile, base_url: &str) -> Result<()> {
    let fetcher = state
        .http_fetcher
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("HTTP fetcher 未初始化"))?;
    crate::sync::ensure_local_at(&state.knowledge_root, entry, base_url, fetcher.as_ref())
}

/// 文件本地缺失时自动从远端补齐再读。命中缓存或纯本地模式（无 base_url）时静默返回。
///
/// **必须从非 tokio runtime 线程调用**：内部使用共享的 `HttpFetcher`，而
/// reqwest::blocking 与 tokio reactor 互斥。async 命令的派生线程是 std::thread，
/// 满足该约束。
///
/// **错误信息必须是可读的中文**：用户在无网络下点开未缓存文章时，不能看到
/// "resource not found" 这种天书。`Review Focus #1` 单测钉死这条契约。
pub fn ensure_cached(
    state: &AppState,
    db: &Db,
    id: i64,
    category_path: &str,
    rel_path: &str,
) -> Result<()> {
    let abs = reader::resolve_absolute(&state.knowledge_root, category_path, rel_path);
    let size: i64 = db.conn.query_row(
        "SELECT size_bytes FROM resources WHERE id = ?1",
        [id],
        |r| r.get(0),
    )?;
    if std::fs::metadata(&abs)
        .map(|m| m.len() as i64 == size)
        .unwrap_or(false)
    {
        return Ok(());
    }
    let base_url = db.get_config("sync_base_url")?.unwrap_or_default();
    if base_url.trim().is_empty() {
        anyhow::bail!(
            "「{rel_path}」尚未缓存到本机，且未配置知识库同步地址（设置 → 知识库同步）"
        );
    }
    let fetcher = state
        .http_fetcher
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("HTTP fetcher 未初始化"))?;
    crate::sync::ensure_local_at(
        &state.knowledge_root,
        &PendingFile {
            category_path: category_path.to_string(),
            rel_path: rel_path.to_string(),
            size: size.max(0) as u64,
        },
        &base_url,
        fetcher.as_ref(),
    )?;
    db.mark_resource_present(category_path, rel_path)?;
    Ok(())
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SyncStatusDto {
    pub total: i64,
    pub present: i64,
    pub total_bytes: i64,
    pub cached_bytes: i64,
    /// 是否配置了远端地址 —— false 时前端不显示同步状态条。
    pub configured: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SyncManifestDto {
    pub files: i64,
    pub indexes: i64,
    /// 纯本地模式（base_url 为空）时为 true，前端据此不显示同步状态条，
    /// 但也不当成"失败"展示。
    pub skipped: bool,
}

/// 用户在设置页改的同步源地址。写入 app_config 表，立刻可被下一次 sync_* 读到。
#[tauri::command]
pub fn set_sync_base_url(url: String, app: AppHandle) -> Result<(), String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    db.set_config("sync_base_url", url.trim())
        .map_err(|e| e.to_string())
}

/// 读取当前同步统计。纯 DB 操作，可走普通同步 Tauri 命令。
#[tauri::command]
pub fn sync_status(app: AppHandle) -> Result<SyncStatusDto, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let st = db.sync_status().map_err(|e| e.to_string())?;
    let configured = db
        .get_config("sync_base_url")
        .map_err(|e| e.to_string())?
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    Ok(SyncStatusDto {
        total: st.total,
        present: st.present,
        total_bytes: st.total_bytes,
        cached_bytes: st.cached_bytes,
        configured,
    })
}

/// 拉远端 manifest.json → 解析 → 落库（upsert_manifest + upsert_index_files）。
///
/// 异步命令：fetch 部分用 std::thread 跑（HttpFetcher 不能在 tokio 上用）；
/// apply_manifest 部分 DB/文件系统操作用 spawn_blocking 派到工作线程，
/// 释放 tokio worker 给其它命令。
#[tauri::command]
pub async fn sync_manifest(app: AppHandle) -> Result<SyncManifestDto, String> {
    let state: tauri::State<AppState> = app.state();
    let db_path = state.db_path.clone();
    let knowledge_root = state.knowledge_root.clone();
    let http_fetcher = state.http_fetcher.clone();

    // 1. 读 base_url —— 失败要可读（无配置 = 纯本地模式 = skipped）
    let base_url = {
        let db = Db::open(&db_path).map_err(|e| e.to_string())?;
        db.get_config("sync_base_url")
            .map_err(|e| e.to_string())?
            .unwrap_or_default()
    };
    if base_url.trim().is_empty() {
        // 纯本地模式（开发）不报错，静默跳过 —— 前端据此不显示状态条
        return Ok(SyncManifestDto {
            files: 0,
            indexes: 0,
            skipped: true,
        });
    }
    let fetcher = http_fetcher
        .as_ref()
        .ok_or_else(|| "HTTP fetcher 未初始化，请重启应用".to_string())?;

    // 2. 拉 manifest —— 必须 std::thread（不能在 tokio 上跑 reqwest::blocking）
    let url = crate::sync::remote_url(&base_url, "manifest.json");
    let (tx, rx) = tokio::sync::oneshot::channel();
    let fetcher_clone = fetcher.clone();
    std::thread::spawn(move || {
        let result: anyhow::Result<Manifest> = (|| {
            let body = fetcher_clone
                .get(&url)
                .map_err(|e| anyhow::anyhow!("拉取 {url} 失败：{e}"))?;
            let text = String::from_utf8(body)
                .map_err(|e| anyhow::anyhow!("manifest 不是 UTF-8：{e}"))?;
            crate::sync::parse_manifest(&text)
                .map_err(|e| anyhow::anyhow!("manifest 解析失败：{e}"))
        })();
        let _ = tx.send(result);
    });
    let manifest = rx
        .await
        .map_err(|e| format!("manifest fetch worker join: {e}"))?
        .map_err(|e| format!("{e:#}"))?;

    // 3. 落库 —— DB + 文件大小比对，无网络，spawn_blocking 安全。
    let manifest_for_thread = Manifest {
        version: manifest.version.clone(),
        files: manifest.files.clone(),
        indexes: manifest.indexes.clone(),
    };
    tauri::async_runtime::spawn_blocking(move || -> anyhow::Result<()> {
        let mut db = Db::open(&db_path)?;
        db.migrate()?;
        apply_manifest(&db, &knowledge_root, &manifest_for_thread)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| format!("apply manifest: {e:#}"))?;

    Ok(SyncManifestDto {
        files: manifest.files.len() as i64,
        indexes: manifest.indexes.len() as i64,
        skipped: false,
    })
}

/// 显式下载单个资源（前端「下载」按钮触发）。
/// 异步命令：网络下载不能阻塞 UI 线程；走 std::thread + oneshot。
#[tauri::command]
pub async fn download_resource(id: i64, app: AppHandle) -> Result<(), String> {
    let state: tauri::State<AppState> = app.state();
    let db_path = state.db_path.clone();
    let knowledge_root = state.knowledge_root.clone();
    let http_fetcher = state.http_fetcher.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let result = (|| -> anyhow::Result<()> {
            let db = Db::open(&db_path)?;
            let (cat, rel, size) = lookup_resource_for_download(&db, id)?;
            let base_url = db
                .get_config("sync_base_url")?
                .unwrap_or_default();
            if base_url.trim().is_empty() {
                anyhow::bail!("未配置知识库同步地址，无法下载「{rel}」（设置 → 知识库同步）");
            }
            let fetcher = http_fetcher
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("HTTP fetcher 未初始化"))?;
            crate::sync::ensure_local_at(
                &knowledge_root,
                &PendingFile {
                    category_path: cat,
                    rel_path: rel.clone(),
                    size: size.max(0) as u64,
                },
                &base_url,
                fetcher.as_ref(),
            )
            .map_err(|e| anyhow::anyhow!("{e:#}"))?;
            db.mark_resource_present_unchecked(id)?;
            Ok(())
        })();
        let _ = tx.send(result.map_err(|e| e.to_string()));
    });
    rx.await.map_err(|e| format!("download worker join: {e}"))?
}

/// 后台跑批量下载，立即返回。进度通过 `sync_progress` 事件 emit：
/// - payload `{"done": N, "total": M}`，N 是已完成数（含失败的）
/// - 单个文件失败不中断整批（sync::download_all 内部已吞掉单条错误）
///
/// `std::thread::spawn` 而非 `spawn_blocking`：避免 reqwest::blocking 与
/// tokio reactor 互斥而 panic。
#[tauri::command]
pub fn download_all(app: AppHandle) -> Result<(), String> {
    let state: tauri::State<AppState> = app.state();
    let db_path = state.db_path.clone();
    let knowledge_root = state.knowledge_root.clone();
    let http_fetcher = state.http_fetcher.clone();

    // 1. 同步查 DB（要 pending_rows 列表，base_url 也同步读），Tauri 主线程上做没代价
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let base_url = db
        .get_config("sync_base_url")
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    if base_url.trim().is_empty() {
        // 没配地址就别折腾了，前端可以据此提示
        return Err("未配置知识库同步地址（设置 → 知识库同步）".into());
    }
    let pending = pending_rows(&db).map_err(|e| e.to_string())?;
    if pending.is_empty() {
        return Ok(());
    }
    let fetcher = http_fetcher
        .clone()
        .ok_or_else(|| "HTTP fetcher 未初始化，请重启应用".to_string())?;
    let app_emit = app.clone();

    // 2. 派生 std::thread 跑批量下载（不在 tokio 上，绝不会 panic）
    std::thread::spawn(move || {
        let result: anyhow::Result<()> = (|| {
            crate::sync::download_all(
                &knowledge_root,
                &base_url,
                pending,
                fetcher,
                &move |done, total| {
                    let _ = app_emit.emit(
                        "sync_progress",
                        serde_json::json!({"done": done, "total": total}),
                    );
                },
            )?;
            // 下载完统一刷新 present 标记（失败的由 download_all 内部的
            // ensure_local_at 自动保持 present=0）
            let db = Db::open(&db_path)?;
            mark_all_present(&db)
        })();
        if let Err(e) = result {
            // 整批失败（如 fetcher 构造错误）至少要让前端收到终结事件，
            // 避免 UI 进度条永远停在 (n-1, n)。
            let _ = app.emit(
                "sync_progress",
                serde_json::json!({"done": 0, "total": 0, "error": format!("{e:#}")}),
            );
        }
    });
    Ok(())
}

fn mark_all_present(db: &Db) -> Result<()> {
    for p in pending_rows(db)? {
        db.mark_resource_present(&p.category_path, &p.rel_path)?;
    }
    Ok(())
}

#[tauri::command]
pub fn list_notes(resource_id: i64, app: AppHandle) -> Result<Vec<NoteDto>, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    db.list_notes(resource_id)
        .map(|rows| rows.into_iter().map(NoteDto::from).collect())
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn create_note(payload: NotePayload, app: AppHandle) -> Result<NoteDto, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let source = match payload.source.as_str() {
        "" | "user" | "ai" => {
            if payload.source.is_empty() { "user" } else { payload.source.as_str() }
        }
        other => return Err(format!("非法 source: {other}")),
    };
    db.insert_note(
        payload.resource_id,
        payload.content.trim(),
        payload.anchor_text.as_deref(),
        payload.anchor_occurrence,
        source,
        None, // 普通 create_note 永远没有 prompt（用户从右侧 composer 输入，不走 AI 问答流）
        payload.page_idx,
    )
    .map(NoteDto::from)
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn update_note(payload: NoteUpdatePayload, app: AppHandle) -> Result<NoteDto, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    // source 只接受 "user" / "ai"；其它值（如 None、空串）一律不更新，
    // 这样普通编辑不会意外改变笔记类型，AI 编辑降级是显式行为。
    let source = payload
        .source
        .as_deref()
        .filter(|s| matches!(*s, "user" | "ai"));
    db.update_note(
        payload.id,
        payload.content.trim(),
        payload.anchor_text.as_deref(),
        payload.anchor_occurrence,
        source,
        payload.page_idx,
    )
    .map(NoteDto::from)
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_note(id: i64, app: AppHandle) -> Result<(), String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    db.delete_note(id).map_err(|e| e.to_string())
}

// === Phase 5: LLM AI 讲解（流式） ===

/// AI 讲解的输入：选中文本 + 文章局部上下文（前后窗 + 最近章节标题）。
/// 后端建占位笔记 → 立即返回 → 后台任务拉流并逐 chunk 通过事件回推前端 + 同步写 DB。
/// 这样用户点完菜单后笔记抽屉立刻展开占位笔记，文字边生成边流入，无 loading 气泡。
#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AiAnnotatePayload {
    /// 占位笔记挂在哪个资源下。前端用 selectedResource.id 传进来。
    pub resource_id: i64,
    pub selected_text: String,
    pub context_before: String,
    pub context_after: String,
    pub section_title: String,
    pub base_url: String,
    /// 高性能模型 ID。本命令会启用 extended thinking；
    /// 轻量模型不接收 thinking 字段，由前端路由保证不混用。
    pub performance_model: String,
    pub api_key: String,
    /// PDF 笔记用的 0-based 页码。markdown/DOCX 不传（AI 注会写入 None）。
    #[serde(default)]
    pub page_idx: Option<i64>,
}

/// 单次流式 chunk 的事件载荷。前端 listen "ai-annotate-chunk" 拿到后
/// append 到对应 note.content。
///
/// 重要：必须 rename_all = "camelCase"。前端 TypeScript 用 `e.payload.noteId`
/// 读字段；后端 Rust 字段是 `note_id`。一旦不写这条 attribute，序列化出来是
/// `{"note_id": ...}`，前端拿到 undefined，所有 chunk 都被 listener 的
/// `if (noteId !== activeNoteId) return` 早 return 过滤掉 —— 表现为"占位卡片
/// 永远空白但无任何控制台错误"。
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AiChunkPayload {
    note_id: i64,
    text: String,
}

/// 流式失败时的事件载荷。前端拿到后清理监听器；错误文本已由后端写进 note.content。
/// `final_content` 是后端实际写入 DB 的完整错误文本，避免前端重复拼前缀导致不一致。
///
/// 同 AiChunkPayload：必须 camelCase，否则 `e.payload.finalContent` 是 undefined。
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AiErrorPayload {
    note_id: i64,
    message: String,
    final_content: String,
}

#[tauri::command]
pub async fn start_ai_annotate(
    payload: AiAnnotatePayload,
    app: AppHandle,
) -> Result<NoteDto, String> {
    // 1. 校验
    let selected = payload.selected_text.trim();
    if selected.is_empty() {
        return Err("选中文本为空".to_string());
    }
    let api_key = payload.api_key.trim();
    if api_key.is_empty() {
        return Err("未配置 API Key，请在设置页填写".to_string());
    }
    let model = payload.performance_model.trim();
    if model.is_empty() {
        return Err("未配置高性能模型，请在设置页填写".to_string());
    }

    // 2. 创建占位笔记（content=""，source="ai"），让前端能立刻显示一个空卡片
    let state: tauri::State<AppState> = app.state();
    let db_path = state.db_path.clone();
    let db = Db::open(&db_path).map_err(|e| e.to_string())?;
    let anchor_text = if selected.is_empty() {
        None
    } else {
        Some(selected.to_string())
    };
    let placeholder = db
        .insert_note(
            payload.resource_id,
            "",
            anchor_text.as_deref(),
            0,
            "ai",
            None, // AI 讲解场景：用户没问问题，prompt 永远为 None；走"询问 AI"走 start_ai_qa 才有 prompt
            payload.page_idx,
        )
        .map_err(|e| e.to_string())?;
    let note_id = placeholder.id;
    let placeholder_anchor = placeholder.anchor_text.clone();
    let placeholder_occurrence = placeholder.anchor_occurrence;
    let placeholder_page_idx = placeholder.page_idx;

    // 3. 后台任务：拉 Anthropic SSE 流 → 逐 chunk emit 事件 + 同步写 DB
    let app_clone = app.clone();
    tokio::spawn(async move {
        let result = stream_ai_annotation(
            app_clone.clone(),
            note_id,
            placeholder_anchor.clone(),
            placeholder_occurrence,
            payload,
            db_path.clone(),
        )
        .await;

        match result {
            Ok(()) => {
                let _ = app_clone.emit("ai-annotate-done", note_id);
            }
            Err(e) => {
                // 失败：把错误信息写进 note.content，前端和 DB 都看得到
                let error_msg = format!("AI 讲解失败：{e}");
                if let Ok(db) = Db::open(&db_path) {
                    let _ = db.update_note(
                        note_id,
                        &error_msg,
                        placeholder_anchor.as_deref(),
                        placeholder_occurrence,
                        None,
                        placeholder_page_idx, // 流式追加阶段保留原 page_idx
                    );
                }
                let _ = app_clone.emit(
                    "ai-annotate-error",
                    AiErrorPayload {
                        note_id,
                        message: e,
                        final_content: error_msg,
                    },
                );
            }
        }
    });

    Ok(NoteDto::from(placeholder))
}

/// QA 场景的输入：用户在选中文字上提了一个问题。
/// 流程与 `start_ai_annotate` 平行：建占位笔记 → 立刻返回 → 后台任务拉流。
/// 占位笔记 `prompt` 字段写上用户问题，用于笔记卡片渲染"❓ 提问"引用块。
#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AiAskPayload {
    pub resource_id: i64,
    pub selected_text: String,
    pub context_before: String,
    pub context_after: String,
    pub section_title: String,
    /// 用户在浮层输入框里提交的问题。trim 后为空时拒绝请求。
    pub question: String,
    pub base_url: String,
    /// 与 AI 讲解共用高性能模型；启用 extended thinking（claude-* 前缀启发式）
    pub performance_model: String,
    pub api_key: String,
    /// PDF 笔记用的 0-based 页码。markdown/DOCX 不传（AI 问答会写入 None）。
    #[serde(default)]
    pub page_idx: Option<i64>,
}

#[tauri::command]
pub async fn start_ai_qa(payload: AiAskPayload, app: AppHandle) -> Result<NoteDto, String> {
    // 1. 校验
    let selected = payload.selected_text.trim();
    if selected.is_empty() {
        return Err("选中文本为空".to_string());
    }
    let question = payload.question.trim();
    if question.is_empty() {
        return Err("问题不能为空".to_string());
    }
    let api_key = payload.api_key.trim();
    if api_key.is_empty() {
        return Err("未配置 API Key，请在设置页填写".to_string());
    }
    let model = payload.performance_model.trim();
    if model.is_empty() {
        return Err("未配置高性能模型，请在设置页填写".to_string());
    }

    // 2. 创建占位笔记（content="", source="ai", prompt=用户问题）。
    let state: tauri::State<AppState> = app.state();
    let db_path = state.db_path.clone();
    let db = Db::open(&db_path).map_err(|e| e.to_string())?;
    let anchor_text = if selected.is_empty() {
        None
    } else {
        Some(selected.to_string())
    };
    let placeholder = db
        .insert_note(
            payload.resource_id,
            "",
            anchor_text.as_deref(),
            0,
            "ai",
            Some(question), // <-- QA 场景：prompt = 用户问题
            payload.page_idx,
        )
        .map_err(|e| e.to_string())?;
    let note_id = placeholder.id;
    let placeholder_anchor = placeholder.anchor_text.clone();
    let placeholder_occurrence = placeholder.anchor_occurrence;
    let placeholder_page_idx = placeholder.page_idx;

    // 3. 后台任务：与 AI 讲解共用流式逻辑，但事件名 + 错误前缀不同（QA 专属），
    // prompt 也由 build_qa_prompt 拼装（用户问题替换"讲解"角色）。
    let app_clone = app.clone();
    tokio::spawn(async move {
        let result = stream_ai_qa(
            app_clone.clone(),
            note_id,
            placeholder_anchor.clone(),
            placeholder_occurrence,
            payload,
            db_path.clone(),
        )
        .await;

        match result {
            Ok(()) => {
                let _ = app_clone.emit("ai-qa-done", note_id);
            }
            Err(e) => {
                // 失败：把错误信息写进 note.content，前端和 DB 都看得到
                let error_msg = format!("AI 问答失败：{e}");
                if let Ok(db) = Db::open(&db_path) {
                    let _ = db.update_note(
                        note_id,
                        &error_msg,
                        placeholder_anchor.as_deref(),
                        placeholder_occurrence,
                        None,
                        placeholder_page_idx, // 流式追加阶段保留原 page_idx
                    );
                }
                let _ = app_clone.emit(
                    "ai-qa-error",
                    AiErrorPayload {
                        note_id,
                        message: e,
                        final_content: error_msg,
                    },
                );
            }
        }
    });

    Ok(NoteDto::from(placeholder))
}

/// QA 场景的 prompt 模板：把章节标题 + 前后窗 + 选中片段作为"用户在读什么"摆出来，
/// 把"用户问题"摆在最显眼位置让模型立即知道回答什么。
///
/// 与 AI 讲解的差异：
/// - QA 强调"直接回答问题"，不允许复述选区/客套
/// - QA 输出更短（3-5 句），避免长篇大论
///
/// 章节/前文/后文/选区为空时分别用占位符（无章节标题）/（无前文）等，
/// 避免模型把空字符串误读为有效输入。
pub(crate) fn build_qa_prompt(
    section: &str,
    before: &str,
    after: &str,
    selected: &str,
    question: &str,
) -> String {
    format!(
        "你是知识问答助手。用户针对以下文字提出了问题，请直接、简洁地回答问题本身：\n\
         - 只回答用户问的事情，不要复述选区、不要客套、不要评价问题本身。\n\
         - 必要时用一句话给出相关背景或细节。\n\
         \n\
         输出 3-5 句中文纯文本段落。不要使用任何 markdown 标记（标题、列表、粗体、代码块、引用），笔记面板只渲染纯文本。\n\
         \n\
         ---\n\
         \n\
         章节：{section}\n\
         \n\
         前文：\n\
         {before}\n\
         \n\
         【用户选中的片段】\n\
         {selected}\n\
         \n\
         后文：\n\
         {after}\n\
         \n\
         【用户问题】\n\
         {question}",
        section = if section.is_empty() { "（无章节标题）" } else { section },
        before = if before.is_empty() { "（无前文）" } else { before },
        after = if after.is_empty() { "（无后文）" } else { after },
        selected = selected,
        question = question,
    )
}

/// 判断模型是否接受 Anthropic 原生的 `thinking` 字段。
/// 启发式：模型名前缀是 `claude-*` 才认为支持。
/// - Anthropic 自家模型：`claude-sonnet-4-5`、`claude-haiku-4-5-...` 等
/// - 火山方舟 Claude 兼容接入：`claude-...` 同样走 Anthropic 协议，支持 thinking
/// - 火山方舟非 Claude 模型（DeepSeek、Doubao）：`thinking` 字段会被服务端以 HTTP 400 拒绝
///
/// 这是保守启发式 —— 真实能力以服务端为准。如果用户配了带 `claude-` 前缀但服务端不
/// 支持的新模型，会因为请求里带了 thinking 而失败；此时把模型名改成不带该前缀即可。
pub fn model_supports_thinking(model: &str) -> bool {
    model.trim().to_ascii_lowercase().starts_with("claude")
}

/// 后台流式任务：调用 Anthropic stream API，解析 SSE，逐 chunk emit 事件并写 DB。
async fn stream_ai_annotation(
    app: AppHandle,
    note_id: i64,
    anchor_text: Option<String>,
    anchor_occurrence: i64,
    payload: AiAnnotatePayload,
    db_path: std::path::PathBuf,
) -> Result<(), String> {
    // 上下文策略同上一版：章节标题 + 前后 ~600 字窗口，绝不发整篇
    let section = payload.section_title.trim();
    let before = payload.context_before.trim();
    let after = payload.context_after.trim();
    let selected = payload.selected_text.trim();

    let prompt = format!(
        "你是知识讲解助手。用简体中文直接讲解用户选中的内容所涉及的知识本身：\n\
         - 解释它是什么、核心概念和关键含义（必要时附英文原名、公式或符号）。\n\
         - 必要时补充相关背景知识或在实际场景中的应用。\n\
         - 不要复述原文，不要评论文章结构，不要客套，不要使用敬语。\n\
         \n\
         输出 80-180 字的纯文本段落。不要使用任何 markdown 标记（标题、列表、粗体、代码块、引用），笔记面板只渲染纯文本。\n\
         \n\
         ---\n\
         \n\
         章节：{section}\n\
         \n\
         前文：\n\
         {before}\n\
         \n\
         【用户选中的片段】\n\
         {selected}\n\
         \n\
         后文：\n\
         {after}",
        section = if section.is_empty() { "（无章节标题）" } else { section },
        before = if before.is_empty() { "（无前文）" } else { before },
        after = if after.is_empty() { "（无后文）" } else { after },
        selected = selected,
    );

    // 显式 stream:true —— 不带的话响应是完整 JSON，没有 SSE 事件
    //
    // thinking 字段仅对 Claude 系模型生效（Anthropic 原生 + Volcano/方舟 等
    // 走 Anthropic 兼容协议的 Claude 模型）。其它模型（DeepSeek、Doubao 等）
    // 不识别 `thinking`，会直接 HTTP 400。
    // 用模型名前缀做启发式判断：claude-* 才带 thinking。
    let model = payload.performance_model.trim();
    let supports_thinking = model_supports_thinking(model);
    let mut req_body = serde_json::json!({
        "model": model,
        "max_tokens": 4096,
        "messages": [{ "role": "user", "content": prompt }],
        "stream": true,
    });
    if supports_thinking {
        req_body["thinking"] = serde_json::json!({
            "type": "enabled",
            "budget_tokens": 2048,
        });
    }

    let url = format!(
        "{}/v1/messages",
        payload.base_url.trim_end_matches('/')
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("http client: {e}"))?;

    let mut res = client
        .post(&url)
        .header("x-api-key", payload.api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&req_body)
        .send()
        .await
        .map_err(|e| format!("请求失败：{e}（请检查网络与 baseUrl）"))?;

    let status = res.status();
    if !status.is_success() {
        let body = res.text().await.unwrap_or_default();
        return Err(format!("HTTP {}：{}", status.as_u16(), body));
    }

    // 整段流式期间复用同一个 DB 连接（rusqlite::Connection 是 Send 但 !Sync，
    // 不能跨线程共享；这里都在一个 tokio 任务里，安全）
    let db = Db::open(&db_path).map_err(|e| format!("open db: {e}"))?;

    let mut parser = SseParser::new();
    let mut accumulated = String::new();

    // chunk() 是 reqwest 内置的，不依赖 futures-util
    while let Some(chunk) = res
        .chunk()
        .await
        .map_err(|e| format!("读取流失败：{e}"))?
    {
        let s = String::from_utf8_lossy(&chunk);
        for text in parser.feed(&s) {
            accumulated.push_str(&text);

            // 事件回推前端（前端 appendToNote → 笔记面板实时显示）
            let _ = app.emit(
                "ai-annotate-chunk",
                AiChunkPayload {
                    note_id,
                    text: text.clone(),
                },
            );

            // 同步写 DB —— 即使前端关掉面板，笔记也是完整的
            // source / page_idx 在流式追加阶段保持不变（仍是 'ai' / 原页码），
            // 用户编辑后才降级；传 None 让 update_note 走"保留"路径。
            let _ = db.update_note(
                note_id,
                &accumulated,
                anchor_text.as_deref(),
                anchor_occurrence,
                None,
                None,
            );
        }
    }

    if accumulated.is_empty() {
        return Err("模型返回了空内容".to_string());
    }
    Ok(())
}

/// QA 流式任务：与 `stream_ai_annotation` 几乎一致，
/// 唯一差异是 prompt 由 `build_qa_prompt` 拼装（包含用户问题）、
/// 事件名走 `ai-qa-chunk`（与 AI 讲解的 `ai-annotate-chunk` 物理隔离）。
///
/// 共享 HTTP/SSE/DB 写入逻辑不抽到公共函数：未来 prompt 路由或事件协议演化时
/// 各自调整，避免一处改动波及两个场景。如果以后 QA 也需要切到轻量模型或
/// 关闭 thinking，再考虑抽公共 stream helper。
async fn stream_ai_qa(
    app: AppHandle,
    note_id: i64,
    anchor_text: Option<String>,
    anchor_occurrence: i64,
    payload: AiAskPayload,
    db_path: std::path::PathBuf,
) -> Result<(), String> {
    let section = payload.section_title.trim();
    let before = payload.context_before.trim();
    let after = payload.context_after.trim();
    let selected = payload.selected_text.trim();
    let question = payload.question.trim();

    let prompt = build_qa_prompt(section, before, after, selected, question);

    let model = payload.performance_model.trim();
    let supports_thinking = model_supports_thinking(model);
    let mut req_body = serde_json::json!({
        "model": model,
        "max_tokens": 2048, // QA 输出 3-5 句，2048 token 充裕
        "messages": [{ "role": "user", "content": prompt }],
        "stream": true,
    });
    if supports_thinking {
        req_body["thinking"] = serde_json::json!({
            "type": "enabled",
            // QA 不需要长思考 —— 1024 token 足够模型理清问题并给出答案；
            // 比 AI 讲解（2048）省一半，避免不必要延迟
            "budget_tokens": 1024,
        });
    }

    let url = format!(
        "{}/v1/messages",
        payload.base_url.trim_end_matches('/')
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("http client: {e}"))?;

    let mut res = client
        .post(&url)
        .header("x-api-key", payload.api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&req_body)
        .send()
        .await
        .map_err(|e| format!("请求失败：{e}（请检查网络与 baseUrl）"))?;

    let status = res.status();
    if !status.is_success() {
        let body = res.text().await.unwrap_or_default();
        return Err(format!("HTTP {}：{}", status.as_u16(), body));
    }

    let db = Db::open(&db_path).map_err(|e| format!("open db: {e}"))?;

    let mut parser = SseParser::new();
    let mut accumulated = String::new();

    while let Some(chunk) = res
        .chunk()
        .await
        .map_err(|e| format!("读取流失败：{e}"))?
    {
        let s = String::from_utf8_lossy(&chunk);
        for text in parser.feed(&s) {
            accumulated.push_str(&text);

            let _ = app.emit(
                "ai-qa-chunk", // 与 AI 讲解的 "ai-annotate-chunk" 物理隔离
                AiChunkPayload {
                    note_id,
                    text: text.clone(),
                },
            );

            // 同步写 DB —— 即使前端关掉面板，笔记也是完整的
            // source / page_idx 在流式追加阶段保持不变（仍是 'ai' / 原页码），
            // 用户编辑后才降级；传 None 让 update_note 走"保留"路径。
            // prompt 在 insert 时已写入，update 不再触碰（保留用户原问题）。
            let _ = db.update_note(
                note_id,
                &accumulated,
                anchor_text.as_deref(),
                anchor_occurrence,
                None,
                None,
            );
        }
    }

    if accumulated.is_empty() {
        return Err("模型返回了空内容".to_string());
    }
    Ok(())
}

/// 简易 SSE 解析器：buffer 按 \n\n 切事件，data: 行 JSON 化，取 content_block_delta.delta.text。
struct SseParser {
    buffer: String,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// 把一段字节喂进来，返回当前能切出的所有 text_delta。剩余未完整的事件留在 buffer。
    fn feed(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);
        let mut texts = Vec::new();
        while let Some(idx) = self.buffer.find("\n\n") {
            let raw = self.buffer[..idx].to_string();
            self.buffer = self.buffer[idx + 2..].to_string();
            if let Some(text) = parse_sse_text_delta(&raw) {
                texts.push(text);
            }
        }
        texts
    }
}

fn parse_sse_text_delta(raw: &str) -> Option<String> {
    let mut data_lines: Vec<&str> = Vec::new();
    for line in raw.lines() {
        if let Some(d) = line.strip_prefix("data: ") {
            data_lines.push(d);
        }
    }
    if data_lines.is_empty() {
        return None;
    }
    let data = data_lines.join("\n");
    if data.trim() == "[DONE]" {
        return None;
    }
    let parsed: serde_json::Value = serde_json::from_str(&data).ok()?;
    if parsed.get("type")?.as_str()? != "content_block_delta" {
        return None;
    }
    Some(parsed.get("delta")?.get("text")?.as_str()?.to_string())
}

// === Phase 5: LLM 翻译 ===

/// 进程内翻译缓存。用户反复高亮同一术语（很常见）时直接命中、秒回，
/// 既省 token 也省首字延迟。键为 trim 后的原文，区分大小写以保留术语差异
/// （如 "Apple" 公司 vs "apple" 水果）。
static TRANSLATION_CACHE: Lazy<Mutex<HashMap<String, String>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranslatePayload {
    pub text: String,
    pub base_url: String,
    /// 轻量模型 ID。翻译这类简单任务走轻量模型，请求中**不携带 thinking 字段**，
    /// 高性能模型（Sonnet/Opus）的 extended thinking 会被跳过。
    pub lightweight_model: String,
    pub api_key: String,
}

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<AnthropicMessage<'a>>,
    // 显式不启用 thinking —— 字段缺失在 Anthropic API 里就等同于关闭。
    // 未来"讲解"类命令若要启用，应单独定义一个带 `thinking` 的请求结构，
    // 并使用 performance_model，不要在翻译里加。
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[tauri::command]
pub async fn translate_text(payload: TranslatePayload) -> Result<String, String> {
    let text = payload.text.trim().to_string();
    if text.is_empty() {
        return Err("选中文本为空".to_string());
    }
    let api_key = payload.api_key.trim();
    if api_key.is_empty() {
        return Err("未配置 API Key，请在设置页填写".to_string());
    }
    let model = payload.lightweight_model.trim();
    if model.is_empty() {
        return Err("未配置轻量模型，请在设置页填写".to_string());
    }

    // 1. 缓存命中：直接返回，跳过整个 HTTP 往返
    if let Some(cached) = TRANSLATION_CACHE
        .lock()
        .map_err(|e| format!("cache lock: {e}"))?
        .get(&text)
        .cloned()
    {
        return Ok(cached);
    }

    // 2. 调 LLM（轻量模型 + 不带 thinking）
    // - 简短 prompt：少送 token、低首字延迟
    // - 明确"全部译成中文"：避免单词被当作术语保留不译
    // - max_tokens 512：单词翻译几个 token 就够，2048 是浪费
    let prompt = format!(
        "Translate the following text to Simplified Chinese (简体中文). \
         Rules:\n\
         - Translate ALL natural language, including technical terms (give the standard Chinese name, \
         e.g. \"transformer\" → \"Transformer\", not \"变压器\").\n\
         - Keep code, URLs, file paths, identifiers untouched.\n\
         - Output ONLY the translation, no quotes, no explanation.\n\n\
         Text: {}",
        text
    );

    let req = MessagesRequest {
        model,
        max_tokens: 512,
        messages: vec![AnthropicMessage {
            role: "user",
            content: &prompt,
        }],
    };

    let base = payload.base_url.trim_end_matches('/');
    let url = format!("{}/v1/messages", base);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("http client: {e}"))?;

    let res = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&req)
        .send()
        .await
        .map_err(|e| format!("请求失败：{e}（请检查网络与 baseUrl）"))?;

    let status = res.status();
    if !status.is_success() {
        let body = res.text().await.unwrap_or_default();
        return Err(format!("HTTP {}：{}", status.as_u16(), body));
    }

    let parsed: MessagesResponse = res
        .json()
        .await
        .map_err(|e| format!("解析响应失败：{e}"))?;

    let result = parsed
        .content
        .into_iter()
        .find_map(|b| b.text)
        .ok_or_else(|| "响应中没有文本内容".to_string())?
        .trim()
        .to_string();

    if result.is_empty() {
        return Err("模型返回了空内容".to_string());
    }

    // 3. 写缓存，供下次同一原文直接命中
    TRANSLATION_CACHE
        .lock()
        .map_err(|e| format!("cache lock: {e}"))?
        .insert(text, result.clone());

    Ok(result)
}

/// 在 setup 阶段调用，创建 AppState 并注册到 Tauri。
pub fn build_state(app: &AppHandle) -> AppState {
    let db_path = resolve_db_path(app);
    let cwd = std::env::current_dir().unwrap_or_default();
    // 开发时从 cwd 向上找；安装后 cwd 不可靠（通常是 System32），
    // 回落到 app_data_dir/knowledge（旧版本曾用 $RESOURCE/knowledge，
    // 知识库不再打进安装包后该分支随之删除 —— 见 platform::pick_knowledge_root）。
    let mut app_data = app
        .path()
        .app_data_dir()
        .expect("app_data_dir resolvable");
    app_data.push("knowledge");
    let knowledge_root = crate::platform::pick_knowledge_root(
        crate::platform::find_knowledge_root(&cwd),
        app_data,
    );
    // 与 build_state 的 brief 对齐：保证扫描器 / materialize_seed 启动时目录一定存在
    std::fs::create_dir_all(&knowledge_root).ok();
    // setup() 钩子在主线程上、Tauri 的 tokio runtime 启动之前同步执行 —— 这是构造
    // HttpFetcher 的唯一安全时机（构造后共享给所有需要下载的命令）。失败仅记录日志，
    // 让用户仍能在纯本地模式下使用 app（无 sync_base_url 时本来也用不上）。
    let http_fetcher = match HttpFetcher::new() {
        Ok(f) => Some(Arc::new(f)),
        Err(e) => {
            eprintln!("警告：初始化 HTTP fetcher 失败，同步下载功能将不可用：{e:#}");
            None
        }
    };
    AppState {
        db_path,
        knowledge_root,
        http_fetcher,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_qa_prompt, model_supports_thinking, AiChunkPayload, AiErrorPayload, AppState,
    };
    use crate::db::Db;
    use crate::sync::{Manifest, ManifestFile};

    #[test]
    fn model_supports_thinking_only_for_claude_prefix() {
        // Anthropic 原生 Claude —— 支持 thinking
        assert!(model_supports_thinking("claude-sonnet-4-5"));
        assert!(model_supports_thinking("claude-haiku-4-5-20251001"));
        // 火山方舟走 Anthropic 兼容协议，模型名仍是 claude-* —— 也支持
        assert!(model_supports_thinking("claude-3-5-sonnet-20241022"));
        // 大小写不敏感
        assert!(model_supports_thinking("Claude-Sonnet-4-5"));

        // 火山方舟非 Claude 模型 —— 不识别 thinking 字段，会 HTTP 400
        assert!(!model_supports_thinking("DeepSeek-V4.1-Flash"));
        assert!(!model_supports_thinking("doubao-seed-2.0-mini"));
        // 空 / 空白 / 完全不相关的命名
        assert!(!model_supports_thinking(""));
        assert!(!model_supports_thinking("   "));
        assert!(!model_supports_thinking("gpt-4o"));
    }

    /// 事件载荷必须输出 camelCase JSON，否则前端 `e.payload.noteId` 之类
    /// 全部是 undefined —— listener 会拿到 undefined 的 noteId 然后被
    /// `if (noteId !== activeNoteId) return` 早期 return，导致用户看到
    /// "占位卡片永久空白 + 控制台无任何报错"。这是历史 bug 的回归守卫。
    #[test]
    fn ai_chunk_payload_serializes_camel_case() {
        let p = AiChunkPayload {
            note_id: 7,
            text: "你好".into(),
        };
        let json = serde_json::to_string(&p).expect("serialize");
        assert!(json.contains("\"noteId\":7"), "missing noteId: {json}");
        assert!(json.contains("\"text\":\"你好\""), "missing text: {json}");
        assert!(!json.contains("note_id"), "snake_case leaked: {json}");
    }

    #[test]
    fn ai_error_payload_serializes_camel_case() {
        let p = AiErrorPayload {
            note_id: 7,
            message: "HTTP 401".into(),
            final_content: "AI 讲解失败：HTTP 401".into(),
        };
        let json = serde_json::to_string(&p).expect("serialize");
        assert!(json.contains("\"noteId\":7"), "missing noteId: {json}");
        assert!(json.contains("\"message\":\"HTTP 401\""), "missing message: {json}");
        assert!(
            json.contains("\"finalContent\":\"AI 讲解失败：HTTP 401\""),
            "missing finalContent: {json}"
        );
        assert!(!json.contains("note_id"), "snake_case leaked: {json}");
        assert!(!json.contains("final_content"), "snake_case leaked: {json}");
    }

    /// QA prompt 必须把"用户问题"放在最显眼的末尾位置，让模型立刻知道回答什么；
    /// 也必须把章节标题 / 前文 / 后文 / 选中片段放进 prompt —— 与 AI 讲解共用同一套上下文策略，
    /// 这样模型能基于文章语境回答而不是空谈。
    #[test]
    fn build_qa_prompt_contains_required_sections() {
        let prompt = build_qa_prompt(
            "反向传播",
            "前文...梯度下降法...",
            "后文...权重更新...",
            "链式法则",
            "为什么这里用链式法则？",
        );

        assert!(prompt.contains("章节：反向传播"), "章节缺失：{prompt}");
        assert!(prompt.contains("前文：\n前文...梯度下降法..."), "前文缺失：{prompt}");
        assert!(
            prompt.contains("【用户选中的片段】\n链式法则"),
            "选中片段缺失：{prompt}"
        );
        assert!(prompt.contains("后文：\n后文...权重更新..."), "后文缺失：{prompt}");
        assert!(
            prompt.contains("【用户问题】\n为什么这里用链式法则？"),
            "用户问题缺失：{prompt}"
        );
    }

    /// 空章节标题走"（无章节标题）"占位，避免模型把空字符串误读为有效输入；
    /// 同 AI 讲解的占位约定。
    #[test]
    fn build_qa_prompt_handles_empty_section() {
        let prompt = build_qa_prompt("", "前文", "后文", "选区", "问题");
        assert!(
            prompt.contains("章节：（无章节标题）"),
            "占位符缺失：{prompt}"
        );
    }

    /// QA 场景不需要 thinking —— 用户已经明确问了什么，模型直接回答即可；
    /// 不在 prompt 层控制，由调用方在请求体里省略 thinking 字段（与 AI 讲解场景的
    /// supports_thinking 决策逻辑一致）。
    #[test]
    fn build_qa_prompt_strips_own_line_length() {
        // 单一职责：build_qa_prompt 只负责拼字符串，不应该 trim 后改变语义。
        // trim 的责任在调用方（与 stream_ai_annotation 的处理路径一致）。
        let prompt = build_qa_prompt("  ", "  ", "  ", "  ", "  ");
        // 不抛错，输出非空即可
        assert!(!prompt.is_empty());
    }

    // ---- Task 4: sync helpers ----

    fn tmp_db() -> (tempfile::TempDir, Db) {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open(&tmp.path().join("t.db")).unwrap();
        db.migrate().unwrap();
        (tmp, db)
    }

    #[test]
    fn pending_rows_skip_present_files() {
        let (_t, db) = tmp_db();
        db.upsert_manifest(&[
            ManifestFile { path: "a/x.md".into(), size: 10, sha256: None },
            ManifestFile { path: "a/y.md".into(), size: 20, sha256: None },
        ])
        .unwrap();
        db.mark_resource_present("a", "x.md").unwrap();
        let rows = super::pending_rows(&db).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rel_path, "y.md");
        assert_eq!(rows[0].size, 20);
    }

    /// `remote_hash IS NOT NULL` 是 pending_rows 的硬过滤：纯本地行（开发者手动扔
    /// 进 knowledge/ 的文件，没有 manifest 备份）绝对不能出现在下载队列里。
    /// 没有这条断言的话，过滤写漏只剩 `present = 0` 也会通过 —— 因为
    /// `upsert_manifest` 总是同时填 remote_hash，掩盖了过滤的真假。
    ///
    /// 关键：纯本地行必须 `present = 0`，否则 `present = 0` 这个 clause 也会排除它，
    /// 测试就变成"在断言两条互不依赖的过滤"的假阳性 —— 删掉 `remote_hash` 子句
    /// 仍然 GREEN。
    #[test]
    fn pending_rows_excludes_pure_local_rows() {
        let (_t, db) = tmp_db();
        db.upsert_manifest(&[
            ManifestFile { path: "a/remote.md".into(), size: 10, sha256: None },
        ])
        .unwrap();
        // 纯本地行：直接 upsert_resource，不走 manifest，remote_hash = NULL。
        // upsert_resource 强制 present=1（它表达"已扫描到本地文件"），
        // 这里显式把 present 翻成 0 —— 表达"还没拉到本地"的纯本地占位，
        // 让 `remote_hash IS NOT NULL` 成为唯一的排除依据。
        db.upsert_resource(crate::db::ResourceInput {
            category_path: "a".into(),
            rel_path: "local.md".into(),
            r#type: "markdown".into(),
            title: "local".into(),
            size_bytes: 5,
            mtime: 0,
            page_count: None,
            word_count: None,
        })
        .unwrap();
        db.conn
            .execute(
                "UPDATE resources SET present = 0 WHERE category_path = 'a' AND rel_path = 'local.md'",
                [],
            )
            .unwrap();
        // 健全性自检：现在 local 行真的是 present=0 + remote_hash=NULL，
        // 把 `remote_hash IS NOT NULL` 那条过滤删掉就会让 local 行通过。
        let (present, remote_hash): (i64, Option<String>) = db
            .conn
            .query_row(
                "SELECT present, remote_hash FROM resources WHERE rel_path='local.md'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(present, 0, "测试夹具必须 present=0，否则无法单独验证 remote_hash 过滤");
        assert!(
            remote_hash.is_none(),
            "测试夹具必须 remote_hash=NULL（纯本地行身份）"
        );

        let rows = super::pending_rows(&db).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rel_path, "remote.md");
    }

    #[test]
    fn apply_manifest_marks_downloaded_files_present() {
        let (tmp, db) = tmp_db();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(root.join("a/x.md"), b"12345").unwrap();
        db.upsert_manifest(&[ManifestFile { path: "a/x.md".into(), size: 5, sha256: None }]).unwrap();
        super::apply_manifest(&db, &root, &Manifest {
            version: "v".into(),
            files: vec![ManifestFile { path: "a/x.md".into(), size: 5, sha256: None }],
            indexes: vec![],
        }).unwrap();
        let present: i64 = db.conn.query_row("SELECT present FROM resources", [], |r| r.get(0)).unwrap();
        assert_eq!(present, 1);
    }

    /// Review Focus #1：无网络 / 未配置同步地址时点开未缓存文章，
    /// 错误信息必须是能照着做的人话，而不是 "resource not found"。
    #[test]
    fn ensure_cached_without_base_url_gives_actionable_chinese_error() {
        let (tmp, db) = tmp_db();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(root.join("a")).unwrap();
        db.upsert_manifest(&[ManifestFile { path: "a/new.md".into(), size: 5, sha256: None }]).unwrap();
        let id: i64 = db.conn.query_row("SELECT id FROM resources WHERE rel_path='new.md'", [], |r| r.get(0)).unwrap();
        let state = AppState {
            db_path: tmp.path().join("t.db"),
            knowledge_root: root,
            http_fetcher: None,
        };
        let err = super::ensure_cached(&state, &db, id, "a", "new.md").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("尚未缓存"), "实际信息：{msg}");
        assert!(msg.contains("同步地址"), "应告诉用户去哪里配置：{msg}");
    }

    /// Review Focus #1 的姊妹分支：文章已被 manifest 清掉（典型场景 —— 同步运行把
    /// 行 drop 掉，但前端的列表是上一次 scan 的快照，用户点了已不存在的条目）。
    /// 错误必须是中文 + 可操作，**绝不能**透出 rusqlite 的 "Query returned no rows"。
    #[test]
    fn lookup_resource_path_kind_missing_id_gives_actionable_chinese_error() {
        let (_t, db) = tmp_db();
        let err = super::lookup_resource_path_kind(&db, 9999).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("已不在知识库中"),
            "实际信息：{msg}"
        );
        assert!(
            msg.contains("返回列表刷新"),
            "应告诉用户去哪里操作：{msg}"
        );
        assert!(
            !msg.contains("Query returned no rows"),
            "绝不能把 rusqlite 内部错误原文透出来：{msg}"
        );
    }

    /// download_resource 的「找不到」分支：消息措辞要带"无法下载"以匹配上下文，
    /// 但同样不能透出 rusqlite 原文。
    #[test]
    fn lookup_resource_for_download_missing_id_gives_actionable_chinese_error() {
        let (_t, db) = tmp_db();
        let err = super::lookup_resource_for_download(&db, 9999).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("无法下载"), "实际信息：{msg}");
        assert!(
            !msg.contains("Query returned no rows"),
            "绝不能把 rusqlite 内部错误原文透出来：{msg}"
        );
    }
}