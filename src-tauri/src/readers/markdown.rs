use anyhow::{Context, Result};
use comrak::{markdown_to_html, Options};
use std::path::Path;

pub fn extract(path: &Path) -> Result<(String, usize)> {
    let md = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut opts = Options::default();
    opts.extension.shortcodes = true;
    let html = markdown_to_html(&md, &opts);
    let words = count_words(&md);
    Ok((html, words))
}

fn count_words(s: &str) -> usize {
    s.split_whitespace().count()
}