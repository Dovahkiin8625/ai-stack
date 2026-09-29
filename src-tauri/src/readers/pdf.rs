use anyhow::{Context, Result};
use base64::Engine;
use pdfium_render::prelude::*;
use std::path::Path;

const RASTER_DPI: f32 = 144.0;

pub fn extract(path: &Path) -> Result<(Vec<(usize, String)>, usize)> {
    let pages = match rasterize_with_pdfium(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[pdf] pdfium render failed ({e:#}), falling back to text");
            Vec::new()
        }
    };
    let text_count = text_page_count(path).unwrap_or_else(|e| {
        eprintln!("[pdf] text extract failed ({e:#})");
        0
    });
    let count = pages.len().max(text_count);
    if pages.is_empty() && count == 0 {
        anyhow::bail!("pdf read failed: no pages rasterizable and no text extractable");
    }
    Ok((pages, count))
}

fn rasterize_with_pdfium(path: &Path) -> Result<Vec<(usize, String)>> {
    let bindings = match Pdfium::bind_to_system_library() {
        Ok(b) => b,
        Err(e) => anyhow::bail!("pdfium init: {e}"),
    };
    let pdfium = Pdfium::new(bindings);
    let doc = pdfium
        .load_pdf_from_file(path, None)
        .with_context(|| format!("load pdf {}", path.display()))?;
    let mut out = Vec::new();
    for (i, page) in doc.pages().iter().enumerate() {
        let bitmap = page
            .render(RASTER_DPI as i32, RASTER_DPI as i32, None)
            .with_context(|| format!("render page {i}"))?;
        let img = bitmap.as_image();
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        img.write_to(&mut buf, image::ImageFormat::Png)
            .with_context(|| format!("encode page {i} png"))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
        out.push((i, format!("data:image/png;base64,{b64}")));
    }
    Ok(out)
}

fn text_page_count(path: &Path) -> Result<usize> {
    let text = pdf_extract::extract_text(path)
        .with_context(|| format!("pdf-extract {}", path.display()))?;
    Ok(text.split('\x0c').filter(|s| !s.trim().is_empty()).count())
}