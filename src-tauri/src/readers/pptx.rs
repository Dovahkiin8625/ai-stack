use anyhow::{Context, Result};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde_json::{json, Value};
use std::io::Read;
use std::path::Path;

pub fn extract(path: &Path) -> Result<(Value, usize)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("unzip {}", path.display()))?;

    // 1) 读取 presentation.xml.rels 取 slideId -> slide 文件名
    let mut rels_xml = String::new();
    zip.by_name("ppt/_rels/presentation.xml.rels")
        .context("read presentation.xml.rels")?
        .read_to_string(&mut rels_xml)?;
    let rels = parse_rels(&rels_xml);

    // 2) 读取 presentation.xml 取 slide 顺序
    let mut pres_xml = String::new();
    zip.by_name("ppt/presentation.xml")
        .context("read presentation.xml")?
        .read_to_string(&mut pres_xml)?;
    let slide_order = parse_slide_order(&pres_xml);

    // 3) 遍历每张 slide：先取 text，再尝试 notes
    let mut slides: Vec<Value> = Vec::new();
    for (i, rid) in slide_order.iter().enumerate() {
        let Some(target) = rels.get(rid) else { continue };
        let slide_path = format!("ppt/{}", target);
        let mut slide_xml = String::new();
        if let Ok(mut f) = zip.by_name(&slide_path) {
            f.read_to_string(&mut slide_xml).ok();
        }
        let (title, body) = extract_text_runs(&slide_xml);

        let notes_path = notes_target_for(&target, &rels);
        let mut notes_text: Option<String> = None;
        if let Some(np) = notes_path {
            if let Ok(mut f) = zip.by_name(&np) {
                let mut s = String::new();
                if f.read_to_string(&mut s).is_ok() {
                    let (n_title, n_body) = extract_text_runs(&s);
                    let mut joined = String::new();
                    if let Some(t) = n_title { joined.push_str(&t); joined.push('\n'); }
                    joined.push_str(&n_body.join("\n"));
                    notes_text = Some(joined.trim().to_string());
                }
            }
        }

        slides.push(json!({
            "index": i,
            "title": title,
            "body": body,
            "notes": notes_text,
        }));
    }

    let slide_count = slides.len();
    Ok((json!({"slides": slides, "slide_count": slide_count}), slide_count))
}

fn parse_rels(xml: &str) -> std::collections::HashMap<String, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut map = std::collections::HashMap::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                if e.local_name().as_ref() == b"Relationship" {
                    let mut id = None;
                    let mut target = None;
                    for attr in e.attributes().flatten() {
                        if attr.key.local_name().as_ref() == b"Id" { id = Some(attr.unescape_value().unwrap_or_default().to_string()); }
                        if attr.key.local_name().as_ref() == b"Target" { target = Some(attr.unescape_value().unwrap_or_default().to_string()); }
                    }
                    if let (Some(i), Some(t)) = (id, target) {
                        map.insert(i, t);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    map
}

fn parse_slide_order(xml: &str) -> Vec<String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut order = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                if e.local_name().as_ref() == b"sldId" {
                    for attr in e.attributes().flatten() {
                        if attr.key.local_name().as_ref() == b"id" {
                            order.push(attr.unescape_value().unwrap_or_default().to_string());
                        }
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    order
}

fn extract_text_runs(xml: &str) -> (Option<String>, Vec<String>) {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut title: Option<String> = None;
    let mut current: Option<String> = None;
    let mut paragraphs: Vec<String> = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name = e.local_name();
                if name.as_ref() == b"sp" || name.as_ref() == b"txBody" {
                    current = Some(String::new());
                }
            }
            Ok(Event::Empty(e)) => {
                if e.local_name().as_ref() == b"t" {
                    if let Some(cur) = current.as_mut() {
                        for attr in e.attributes().flatten() {
                            if attr.key.local_name().as_ref() == b"text" {
                                cur.push_str(&attr.unescape_value().unwrap_or_default());
                            }
                        }
                    }
                }
            }
            Ok(Event::End(e)) => {
                let name = e.local_name();
                if name.as_ref() == b"p" {
                    if let Some(mut p) = current.take() {
                        if paragraphs.is_empty() && title.is_none() && !p.is_empty() {
                            title = Some(p.clone());
                        }
                        if !p.is_empty() {
                            paragraphs.push(p);
                        }
                    }
                }
            }
            Ok(Event::Text(t)) => {
                if let Some(cur) = current.as_mut() {
                    cur.push_str(&t.unescape().unwrap_or_default());
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    let body: Vec<String> = paragraphs.into_iter().skip(if title.is_some() { 1 } else { 0 }).collect();
    (title, body)
}

fn notes_target_for(slide_target: &str, rels: &std::collections::HashMap<String, String>) -> Option<String> {
    // rels 中 rId -> "slides/slide1.xml"；尝试找 notesSlide 的 rId 不现实（需要解析每个 slide 的 rels）
    // 简化：直接假设 notes 路径为 "notesSlides/notesSlide{i}.xml"（PPTX 默认顺序）
    // 真实路径由 slideN.xml.rels 决定；为简化采用以下尝试顺序
    let _ = (slide_target, rels); // placeholder
    None
}