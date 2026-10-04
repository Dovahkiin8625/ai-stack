//! Verify that all IPC-facing DTOs serialize to camelCase JSON so the
//! TypeScript side (which reads `parentPath`, `wordCount`, etc.) decodes
//! correctly.  The original `serde(rename_all = "camelCase")` was missing
//! on every type, so the tree rendered empty in production.  This test is
//! the regression guard.

use ai_stack_lib::commands::{CategoryDto, ResourceDto};
use ai_stack_lib::reader::ResourceContent;
use ai_stack_lib::scanner::ScanSummary;

#[test]
fn category_dto_serializes_camel_case() {
    let dto = CategoryDto {
        path: "01-foundations/01-math".into(),
        parent_path: Some("01-foundations".into()),
        title: "Mathematics".into(),
        sort_order: 1,
    };
    let json = serde_json::to_string(&dto).expect("serialize");
    assert!(json.contains("\"parentPath\""), "missing parentPath: {json}");
    assert!(json.contains("\"sortOrder\""), "missing sortOrder: {json}");
    assert!(!json.contains("parent_path"), "snake_case leaked: {json}");
    assert!(!json.contains("sort_order"), "snake_case leaked: {json}");
}

#[test]
fn resource_dto_serializes_camel_case() {
    let dto = ResourceDto {
        id: 7,
        category_path: "01-foundations/01-math".into(),
        rel_path: "notes.md".into(),
        r#type: "markdown".into(),
        title: "Notes".into(),
        size_bytes: 1024,
        indexed_at: "2026-09-29T00:00:00Z".into(),
        page_count: None,
        word_count: Some(120),
    };
    let json = serde_json::to_string(&dto).expect("serialize");
    for camel in ["categoryPath", "relPath", "sizeBytes", "indexedAt", "wordCount"] {
        assert!(json.contains(camel), "missing {camel}: {json}");
    }
    assert!(!json.contains("category_path"), "snake_case leaked: {json}");
    assert!(!json.contains("word_count"), "snake_case leaked: {json}");
    assert!(!json.contains("\"hash\""), "stale hash field leaked: {json}");
}

#[test]
fn scan_summary_serializes_camel_case() {
    let s = ScanSummary {
        categories_count: 3,
        resources_count: 12,
        errors_count: 0,
        duration_ms: 42,
    };
    let json = serde_json::to_string(&s).expect("serialize");
    for camel in [
        "categoriesCount",
        "resourcesCount",
        "errorsCount",
        "durationMs",
    ] {
        assert!(json.contains(camel), "missing {camel}: {json}");
    }
    assert!(!json.contains("categories_count"), "snake_case leaked: {json}");
}

#[test]
fn resource_content_markdown_variant_serializes_camel_case() {
    let content = ResourceContent::Markdown {
        html: "<p>hi</p>".into(),
        word_count: 1,
    };
    let json = serde_json::to_string(&content).expect("serialize");
    // Discriminator is lowercased from the PascalCase variant name.
    assert!(json.contains("\"type\":\"markdown\""), "missing type=markdown: {json}");
    assert!(json.contains("\"wordCount\":1"), "missing wordCount: {json}");
    assert!(!json.contains("word_count"), "snake_case leaked: {json}");
}

#[test]
fn resource_content_pdf_variant_serializes_camel_case() {
    // PDF variant 现在只携带页数。
    // 实际渲染交给前端 pdf.js（拿到 read_resource_bytes 返回的原始字节自己渲染 + text layer）。
    // 这里只校验 IPC 包的 shape。
    let content = ResourceContent::Pdf { page_count: 12 };
    let json = serde_json::to_string(&content).expect("serialize");
    assert!(json.contains("\"type\":\"pdf\""), "missing type=pdf: {json}");
    assert!(json.contains("\"pageCount\":12"), "missing pageCount: {json}");
    assert!(!json.contains("page_count"), "snake_case leaked: {json}");
    assert!(!json.contains("\"pages\""), "stale pages field leaked: {json}");
}

#[test]
fn resource_content_docx_variant_serializes_camel_case() {
    // DOCX 后端只数 word 数；实际渲染由前端 mammoth.js 完成。
    let content = ResourceContent::Docx { word_count: 42 };
    let json = serde_json::to_string(&content).expect("serialize");
    assert!(json.contains("\"type\":\"docx\""), "missing type=docx: {json}");
    assert!(json.contains("\"wordCount\":42"), "missing wordCount: {json}");
    assert!(!json.contains("word_count"), "snake_case leaked: {json}");
    // 老字段（blocks）必须不再出现 —— 前端 mammoth 替代它
    assert!(
        !json.contains("blocks"),
        "stale blocks field leaked: {json}"
    );
}

#[test]
fn resource_content_pptx_variant_serializes_camel_case() {
    // PPTX 后端只数 slide 数；实际渲染由前端 pptxviewjs 完成。
    let content = ResourceContent::Pptx { slide_count: 7 };
    let json = serde_json::to_string(&content).expect("serialize");
    assert!(json.contains("\"type\":\"pptx\""), "missing type=pptx: {json}");
    assert!(json.contains("\"slideCount\":7"), "missing slideCount: {json}");
    assert!(!json.contains("slide_count"), "snake_case leaked: {json}");
    // 老字段（slides）必须不再出现 —— 前端 pptxviewjs 替代它
    assert!(
        !json.contains("\"slides\""),
        "stale slides field leaked: {json}"
    );
}