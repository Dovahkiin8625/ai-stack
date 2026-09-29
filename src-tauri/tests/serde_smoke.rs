//! Verify that all IPC-facing DTOs serialize to camelCase JSON so the
//! TypeScript side (which reads `parentPath`, `wordCount`, etc.) decodes
//! correctly.  The original `serde(rename_all = "camelCase")` was missing
//! on every type, so the tree rendered empty in production.  This test is
//! the regression guard.

use ai_stack_lib::commands::{CategoryDto, ResourceDto};
use ai_stack_lib::reader::{PdfPageDataUrl, ResourceContent};
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
        hash: "abc".into(),
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
    for camel in ["categoriesCount", "resourcesCount", "errorsCount", "durationMs"] {
        assert!(json.contains(camel), "missing {camel}: {json}");
    }
    assert!(!json.contains("categories_count"), "snake_case leaked: {json}");
}

#[test]
fn pdf_page_data_url_serializes_camel_case() {
    let p = PdfPageDataUrl {
        index: 0,
        data_url: "data:image/png;base64,XXX".into(),
    };
    let json = serde_json::to_string(&p).expect("serialize");
    assert!(json.contains("\"dataUrl\""), "missing dataUrl: {json}");
    assert!(!json.contains("data_url"), "snake_case leaked: {json}");
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
    let content = ResourceContent::Pdf {
        pages: vec![PdfPageDataUrl {
            index: 0,
            data_url: "data:image/png;base64,XXX".into(),
        }],
        page_count: 1,
    };
    let json = serde_json::to_string(&content).expect("serialize");
    assert!(json.contains("\"type\":\"pdf\""), "missing type=pdf: {json}");
    assert!(json.contains("\"pageCount\":1"), "missing pageCount: {json}");
    assert!(json.contains("\"dataUrl\""), "missing dataUrl: {json}");
}

#[test]
fn resource_content_docx_variant_serializes_camel_case() {
    let content = ResourceContent::Docx {
        blocks: serde_json::json!([]),
        word_count: 0,
    };
    let json = serde_json::to_string(&content).expect("serialize");
    assert!(json.contains("\"type\":\"docx\""), "missing type=docx: {json}");
    assert!(json.contains("\"wordCount\":0"), "missing wordCount: {json}");
}

#[test]
fn resource_content_pptx_variant_serializes_camel_case() {
    let content = ResourceContent::Pptx {
        slides: serde_json::json!({"slides": []}),
        slide_count: 0,
    };
    let json = serde_json::to_string(&content).expect("serialize");
    assert!(json.contains("\"type\":\"pptx\""), "missing type=pptx: {json}");
    assert!(json.contains("\"slideCount\":0"), "missing slideCount: {json}");
}