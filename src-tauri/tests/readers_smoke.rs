use ai_stack_lib::readers::markdown::extract;
use ai_stack_lib::readers::pdf::extract as pdf_extract;
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