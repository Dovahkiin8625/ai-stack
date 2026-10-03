use anyhow::{Context, Result};
use std::path::Path;

/// 只统计页数（不栅格化），用 pdf-extract 文本分页。
///
/// 真正的渲染交给前端 pdf.js（拿到 `read_resource_bytes` 返的原始字节，自己渲染 + 文字层）。
/// 这里只数页数，给 UI 立即显示「第 N / 总页数」用 —— 拿到数字前 pdf.js 也能继续加载。
pub fn count_pages(path: &Path) -> Result<usize> {
    let text = pdf_extract::extract_text(path)
        .with_context(|| format!("pdf-extract {}", path.display()))?;
    let count = text.split('\x0c').filter(|s| !s.trim().is_empty()).count();
    if count == 0 {
        anyhow::bail!("pdf read failed: no pages extractable");
    }
    Ok(count)
}
