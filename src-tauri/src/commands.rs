//! Tauri 命令入口
use crate::db::{Db, ResourceRow};
use crate::reader::{self, ResourceContent};
use crate::scanner::{self, ScanConfig, ScanSummary};
use anyhow::Context;
use serde::Serialize;
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager, State};

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
    pub hash: String,
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
            hash: r.hash,
            indexed_at: r.indexed_at,
            page_count: r.page_count,
            word_count: r.word_count,
        }
    }
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

/// 在 setup 阶段调用，创建 AppState 并注册到 Tauri。
pub fn build_state(app: &AppHandle) -> AppState {
    let db_path = resolve_db_path(app);
    let cwd = std::env::current_dir().unwrap_or_default();
    let knowledge_root = cwd.join("resources").join("knowledge");
    AppState { db_path, knowledge_root }
}