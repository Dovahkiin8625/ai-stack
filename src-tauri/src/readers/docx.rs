use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::path::Path;

pub fn extract(path: &Path) -> Result<(Value, usize)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read {}", path.display()))?;
    let docx = docx_rs::read_docx(&bytes)
        .with_context(|| format!("parse docx {}", path.display()))?;
    let mut blocks: Vec<Value> = Vec::new();
    let mut all_text = String::new();

    for child in docx.document.children.iter() {
        match child {
            docx_rs::DocumentChild::Paragraph(p) => {
                let text = collect_text(&p.children);
                all_text.push_str(&text);
                all_text.push(' ');
                if let Some(style) = p.property.style.as_ref().map(|s| s.val.to_string()) {
                    let level = match style.as_str() {
                        "Heading1" => Some(1),
                        "Heading2" => Some(2),
                        "Heading3" => Some(3),
                        _ => None,
                    };
                    if let Some(lvl) = level {
                        blocks.push(json!({"kind":"heading","level":lvl,"text":text}));
                        continue;
                    }
                }
                blocks.push(json!({"kind":"paragraph","text":text}));
            }
            docx_rs::DocumentChild::Table(t) => {
                let mut rows: Vec<Vec<String>> = Vec::new();
                for row_child in &t.rows {
                    let docx_rs::TableChild::TableRow(row) = row_child else { continue };
                    let mut cells: Vec<String> = Vec::new();
                    for cell_child in &row.cells {
                        let docx_rs::TableRowChild::TableCell(cell) = cell_child else { continue };
                        let mut cell_text = String::new();
                        for c in &cell.children {
                            if let docx_rs::TableCellContent::Paragraph(pp) = c {
                                cell_text.push_str(&collect_text(&pp.children));
                                cell_text.push('\n');
                            }
                        }
                        cells.push(cell_text.trim().to_string());
                    }
                    rows.push(cells);
                }
                blocks.push(json!({"kind":"table","rows":rows}));
            }
            _ => {}
        }
    }

    let words = all_text.split_whitespace().count();
    Ok((Value::Array(blocks), words))
}

fn collect_text(runs: &[docx_rs::ParagraphChild]) -> String {
    let mut out = String::new();
    for r in runs {
        if let docx_rs::ParagraphChild::Run(run) = r {
            for tc in &run.children {
                if let docx_rs::RunChild::Text(t) = tc {
                    out.push_str(&t.text);
                }
            }
        }
    }
    out
}