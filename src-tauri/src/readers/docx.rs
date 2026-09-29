use anyhow::Result;
use std::path::Path;
pub fn extract(_path: &Path) -> Result<(serde_json::Value, usize)> {
    anyhow::bail!("not implemented")
}