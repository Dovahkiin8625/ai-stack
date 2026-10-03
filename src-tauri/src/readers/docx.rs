use anyhow::{Context, Result};
use std::path::Path;

/// 数 docx 里的 "word 数"。
///
/// docx 是 zip 包，文字都存在 `word/document.xml`。粗略做法：把所有 text
/// 节点串起来按空白 split —— 跟之前 markdown / PDF reader 数 word 数同策略，
/// DB 索引和 UI 显示只要"大致多少字"，不需要真正按 Word 的"word delimiter"
/// 算法去数。
///
/// 这里不再返回任何 blocks —— DOCX 的格式化渲染交给前端 mammoth.js，
/// 见 `src/components/library/reader/DocxReader.tsx`。
pub fn word_count(path: &Path) -> Result<usize> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read {}", path.display()))?;
    let docx = docx_rs::read_docx(&bytes)
        .with_context(|| format!("parse docx {}", path.display()))?;
    let mut all_text = String::new();
    for child in docx.document.children.iter() {
        if let DocumentChildOwned::Paragraph(p) = child {
            for r in &p.children {
                if let ParagraphChildOwned::Run(run) = r {
                    for tc in &run.children {
                        if let RunChildOwned::Text(t) = tc {
                            all_text.push_str(&t.text);
                            all_text.push(' ');
                        }
                    }
                }
            }
        }
    }
    Ok(all_text.split_whitespace().count())
}

// === Re-exports so `word_count` only needs to import what it touches. ===
use docx_rs::DocumentChild as DocumentChildOwned;
use docx_rs::ParagraphChild as ParagraphChildOwned;
use docx_rs::RunChild as RunChildOwned;

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests");
        p.push("fixtures");
        p.push("readers");
        p.push(name);
        p
    }

    #[test]
    fn docx_word_count_is_positive() {
        // 不精确断言具体数字（docx-rs 抽取后中文按 token 计未必分得清），
        // 只验证能成功 parse 且 word 数 > 0。
        let n = word_count(&fixture("sample.docx")).unwrap();
        assert!(n > 0, "got {n}");
    }
}