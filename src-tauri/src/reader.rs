//! 文档读取入口，按扩展名分发
use crate::readers;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum ResourceContent {
    Markdown { html: String, word_count: usize },
    Pdf { pages: Vec<PdfPageDataUrl>, page_count: usize },
    Docx { blocks: serde_json::Value, word_count: usize },
    Pptx { slides: serde_json::Value, slide_count: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PdfPageDataUrl {
    pub index: usize,
    pub data_url: String,
}

pub fn read(absolute: &Path, kind: &str) -> Result<ResourceContent> {
    match kind {
        "markdown" => {
            let (html, wc) = readers::markdown_extract(absolute)?;
            Ok(ResourceContent::Markdown { html, word_count: wc })
        }
        "pdf" => {
            let (pages, count) = readers::pdf_extract(absolute)?;
            let pages = pages
                .into_iter()
                .map(|(index, data_url)| PdfPageDataUrl { index, data_url })
                .collect();
            Ok(ResourceContent::Pdf { pages, page_count: count })
        }
        "docx" => {
            let (blocks, wc) = readers::docx_extract(absolute)?;
            Ok(ResourceContent::Docx { blocks, word_count: wc })
        }
        "pptx" => {
            let (slides, count) = readers::pptx_extract(absolute)?;
            Ok(ResourceContent::Pptx { slides, slide_count: count })
        }
        other => anyhow::bail!("unsupported resource kind: {other}"),
    }
}

pub fn resolve_absolute(knowledge_root: &Path, category: &str, rel: &str) -> PathBuf {
    knowledge_root.join(category).join(rel)
}

#[allow(dead_code)]
pub fn touch_only_for_module_use() {
    let _ = resolve_absolute;
}