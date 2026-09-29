//! 文档读取入口，按扩展名分发
use anyhow::Result;
use std::path::Path;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum ResourceContent {
    Markdown { html: String, word_count: usize },
    Pdf { pages: Vec<PdfPageDataUrl>, page_count: usize },
    Docx { blocks: serde_json::Value, word_count: usize },
    Pptx { slides: serde_json::Value, slide_count: usize },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PdfPageDataUrl {
    pub index: usize,
    pub data_url: String,
}

pub fn read(_path: &Path, _kind: &str) -> Result<ResourceContent> {
    anyhow::bail!("reader not implemented (Task 4-7)")
}