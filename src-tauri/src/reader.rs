//! 文档读取入口，按扩展名分发
//!
//! 设计：markdown 读 html（后端负责把 md → html）；PDF / DOCX / PPTX 都走
//! "后端只统计 number（页数 / slide 数）→ 前端拿原始字节用 JS 库渲染"
//! 的同一种模式 —— 让前端 JS 库各自处理自家格式的高保真渲染。
//! DOCX 用 mammoth.js，PPTX 用 pptxviewjs，PDF 用 pdfjs-dist。
use crate::readers;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum ResourceContent {
    Markdown { html: String, word_count: usize },
    /// PDF：只返回页数（用 pdf-extract 文本分页数）；前端 pdf.js 拿到
    /// `read_resource_bytes` 返回的原始字节后自行渲染并自带 text layer。
    Pdf { page_count: usize },
    /// DOCX：后端只统计 word 数（前端 mammoth 转换完后无法可靠数 cross-doc）；
    /// 实际渲染由前端 mammoth.js 完成。
    Docx { word_count: usize },
    /// PPTX：后端只统计 slide 数；实际渲染由前端 pptxviewjs 完成。
    Pptx { slide_count: usize },
}

pub fn read(absolute: &Path, kind: &str) -> Result<ResourceContent> {
    match kind {
        "markdown" => {
            let (html, wc) = readers::markdown_extract(absolute)?;
            Ok(ResourceContent::Markdown { html, word_count: wc })
        }
        "pdf" => {
            let count = readers::pdf_count_pages(absolute)?;
            Ok(ResourceContent::Pdf { page_count: count })
        }
        "docx" => {
            let wc = readers::docx_word_count(absolute)?;
            Ok(ResourceContent::Docx { word_count: wc })
        }
        "pptx" => {
            let count = readers::pptx_slide_count(absolute)?;
            Ok(ResourceContent::Pptx { slide_count: count })
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