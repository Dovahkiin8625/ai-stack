//! Tauri 命令入口
use crate::scanner::{ScanConfig, ScanSummary};
use std::path::PathBuf;

#[tauri::command]
pub async fn scan_library(force: bool) -> Result<ScanSummary, String> {
    let cfg = ScanConfig {
        knowledge_root: PathBuf::from("resources/knowledge"),
    };
    let _ = force;
    crate::scanner::scan(&cfg).map_err(|e| e.to_string())
}