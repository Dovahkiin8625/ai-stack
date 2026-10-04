//! Tauri 命令入口
use crate::db::{Db, NoteRow, ResourceRow};
use crate::reader::{self, ResourceContent};
use crate::scanner::{self, ScanConfig, ScanSummary};
use anyhow::Context;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};

/// 跨命令共享的不可变路径配置。
/// 在 setup 阶段初始化一次；每个命令按需 `clone`。
pub struct AppState {
    pub db_path: PathBuf,
    pub knowledge_root: PathBuf,
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
    pub source: String,
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
            source: n.source,
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
pub fn read_resource(id: i64, app: AppHandle) -> Result<ResourceContent, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let (category_path, rel_path, kind) = db
        .conn
        .query_row(
            "SELECT category_path, rel_path, type FROM resources WHERE id = ?1",
            [id],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)),
        )
        .map_err(|e| format!("resource {id} not found: {e}"))?;
    let abs = reader::resolve_absolute(&state.knowledge_root, &category_path, &rel_path);
    reader::read(&abs, &kind).map_err(|e| format!("read {}: {e:#}", abs.display()))
}

/// 返回资源的原始字节。
///
/// 三类 reader 都用它拿原文件自己渲染：
/// - PDF：前端 pdf.js `getDocument({data})`，自带 text layer 支持选中/复制
/// - DOCX：前端 mammoth.js 把 docx → semantic HTML（保留 bold/italic/headings/lists/tables/images）
/// - PPTX：前端 pptxviewjs 把 pptx → canvas slide（按 slide 翻页，完整保留版式）
///
/// markdown reader 不走这里（后端已经返回 html）。
#[tauri::command]
pub fn read_resource_bytes(id: i64, app: AppHandle) -> Result<Vec<u8>, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let (category_path, rel_path, kind) = db
        .conn
        .query_row(
            "SELECT category_path, rel_path, type FROM resources WHERE id = ?1",
            [id],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)),
        )
        .map_err(|e| format!("resource {id} not found: {e}"))?;
    if !matches!(kind.as_str(), "pdf" | "docx" | "pptx") {
        return Err(format!(
            "read_resource_bytes 仅支持 PDF/DOCX/PPTX，resource {id} 是 {kind}"
        ));
    }
    let abs = reader::resolve_absolute(&state.knowledge_root, &category_path, &rel_path);
    std::fs::read(&abs).map_err(|e| format!("read {}: {e}", abs.display()))
}

/// 读取子分类目录下的 `_index.md` 并解析为 `[{ relPath, title, description }]`。
///
/// 给中间区目录列表的"每行附带描述"用。
/// 不入库（编辑 `_index.md` 不必触发重扫），每次调用现读现解析，路径 1KB 内的文件
/// 解析开销可以忽略；前端按 categoryPath 在 store 里缓存。
#[tauri::command]
pub fn read_subcategory_index(
    category_path: String,
    app: AppHandle,
) -> Result<crate::readers::index_md::ParsedIndex, String> {
    let state: tauri::State<AppState> = app.state();
    // _index.md 在该子分类目录下，文件名固定。
    let abs = state.knowledge_root.join(&category_path).join("_index.md");
    crate::readers::index_md_parse_file(&abs)
        .map_err(|e| format!("parse _index.md for {category_path}: {e:#}"))
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
        )
        .map_err(|e| e.to_string())?;
    let note_id = placeholder.id;
    let placeholder_anchor = placeholder.anchor_text.clone();
    let placeholder_occurrence = placeholder.anchor_occurrence;

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
            // source 在流式追加阶段保持不变（仍是 'ai'），用户编辑后才降级
            let _ = db.update_note(
                note_id,
                &accumulated,
                anchor_text.as_deref(),
                anchor_occurrence,
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
    let knowledge_root = find_knowledge_root(&cwd)
        .unwrap_or_else(|| cwd.join("resources").join("knowledge"));
    AppState {
        db_path,
        knowledge_root,
    }
}

/// 从 `start` 向上查找 `resources/knowledge/` 目录。
/// 解决 `cargo run`（cwd 在 src-tauri/）与 `tauri build`（cwd 在项目根）
/// 工作目录不一致的问题。
fn find_knowledge_root(start: &std::path::Path) -> Option<PathBuf> {
    let mut cur: Option<&std::path::Path> = Some(start);
    while let Some(p) = cur {
        let candidate = p.join("resources").join("knowledge");
        if candidate.is_dir() {
            return Some(candidate);
        }
        cur = p.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{find_knowledge_root, model_supports_thinking, AiChunkPayload, AiErrorPayload};

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
}