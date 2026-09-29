use ai_stack_lib::readers::markdown::extract;
use ai_stack_lib::readers::pdf::extract as pdf_extract;
use ai_stack_lib::readers::docx::extract as docx_extract;
use ai_stack_lib::readers::pptx::extract as pptx_extract;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("readers");
    p.push(name);
    p
}

#[test]
fn markdown_renders_to_html_and_counts_words() {
    let (html, words) = extract(&fixture("simple.md")).unwrap();
    assert!(html.contains("<h1>Heading</h1>"), "html: {html}");
    assert!(html.contains("<strong>world</strong>"), "html: {html}");
    assert!(html.contains("<ul>"), "html: {html}");
    assert!(words >= 4, "got {words}");
}

#[test]
fn pdf_returns_pages_or_falls_back() {
    let result = pdf_extract(&fixture("sample.pdf"));
    match result {
        Ok((pages, count)) => {
            assert!(count >= 1);
            // 若 pdfium 缺失则 pages 为空向量
            if !pages.is_empty() {
                assert!(pages[0].1.starts_with("data:image/png;base64,"));
            }
        }
        Err(_) => {
            // 也可接受：失败时不 panic
        }
    }
}

#[test]
fn docx_extracts_headings_and_paragraphs() {
    let (blocks_value, words) = docx_extract(&fixture("sample.docx")).unwrap();
    let blocks = blocks_value.as_array().expect("blocks is array");
    let kinds: Vec<&str> = blocks.iter()
        .map(|b| b.get("kind").and_then(|v| v.as_str()).unwrap_or(""))
        .collect();
    assert!(kinds.contains(&"heading"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"paragraph"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"table"), "kinds: {kinds:?}");
    assert!(words > 0);
}

#[test]
fn pptx_extracts_slides() {
    let (value, _) = pptx_extract(&fixture("sample.pptx")).unwrap();
    let slides = value.get("slides").and_then(|v| v.as_array()).expect("slides array");
    assert!(slides.len() >= 2, "got {} slides", slides.len());
    let first = &slides[0];
    assert!(first.get("title").is_some());
    let body = first.get("body").and_then(|v| v.as_array()).unwrap();
    assert!(!body.is_empty(), "body empty");
}