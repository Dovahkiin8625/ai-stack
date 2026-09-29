//! 扫描 resources/knowledge/ 并 upsert 到 DB
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub knowledge_root: PathBuf,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct ScanSummary {
    pub categories_count: usize,
    pub resources_count: usize,
    pub errors_count: usize,
    pub duration_ms: u128,
}

pub fn scan(_cfg: &ScanConfig) -> Result<ScanSummary, anyhow::Error> {
    // 任务 3 实现
    Ok(ScanSummary::default())
}

#[allow(dead_code)]
pub fn validate_root(root: &Path) -> anyhow::Result<()> {
    if !root.is_dir() {
        anyhow::bail!("knowledge root is not a directory: {}", root.display());
    }
    Ok(())
}