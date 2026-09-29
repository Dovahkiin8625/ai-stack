//! Tauri 命令入口
use crate::db::Db;
use crate::scanner::{ScanConfig, ScanSummary};
use std::path::PathBuf;

#[tauri::command]
pub async fn scan_library(force: bool) -> Result<ScanSummary, String> {
    let cfg = ScanConfig {
        knowledge_root: PathBuf::from("resources/knowledge"),
    };
    let _ = force;
    let db_path = PathBuf::from("data/library.db");
    let mut db = Db::open(&db_path).map_err(|e| e.to_string())?;
    db.migrate().map_err(|e| e.to_string())?;
    crate::scanner::scan(&db, &cfg).map_err(|e| e.to_string())
}