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
    let knowledge_root = find_knowledge_root(&cwd)
        .unwrap_or_else(|| cwd.join("resources").join("knowledge"));
    AppState { db_path, knowledge_root }
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
    use super::find_knowledge_root;
    use std::path::PathBuf;

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