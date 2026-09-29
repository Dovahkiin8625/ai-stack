use ai_stack_lib::readers::markdown::extract;
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