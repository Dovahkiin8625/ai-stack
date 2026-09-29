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

        // notes target lives in the slide's own .rels file: ppt/slides/_rels/slideN.xml.rels
        let mut notes_text: Option<String> = None;
        if let Some(slide_rels_path) = slide_rels_path_for(&slide_path) {
            // Scope the first borrow so it's released before the next by_name.
            let rels_xml = read_zip_string(&mut zip, &slide_rels_path);
            if let Some(rels_xml) = rels_xml {
                if let Some(notes_target) = find_notes_target(&rels_xml) {
                    let notes_path = resolve_notes_path(&slide_path, &notes_target);
                    let notes_xml = read_zip_string(&mut zip, &notes_path);
                    if let Some(s) = notes_xml {
                        let (n_title, n_body) = extract_text_runs(&s);
                        let mut joined = String::new();
                        if let Some(t) = n_title {
                            joined.push_str(&t);
                            joined.push('\n');
                        }
                        joined.push_str(&n_body.join("\n"));
                        let trimmed = joined.trim().to_string();
                        if !trimmed.is_empty() {
                            notes_text = Some(trimmed);
                        }
                    }
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

/// Compute the path of a slide's rels file from its slide path.
/// E.g. "ppt/slides/slide1.xml" -> "ppt/slides/_rels/slide1.xml.rels".
fn slide_rels_path_for(slide_path: &str) -> Option<String> {
    let (dir, file) = slide_path.rsplit_once('/')?;
    Some(format!("{}/_rels/{}.rels", dir, file))
}

/// Open a zip entry and read it fully into a String.  Returns None if the
/// entry is missing or can't be decoded.  Designed to be called inside a
/// short scope so the underlying `ZipFile` borrow is released before
/// another `by_name` is taken.
fn read_zip_string<R: std::io::Read + std::io::Seek>(
    zip: &mut zip::ZipArchive<R>,
    name: &str,
) -> Option<String> {
    let mut f = zip.by_name(name).ok()?;
    let mut s = String::new();
    f.read_to_string(&mut s).ok()?;
    Some(s)
}

/// Scan a slideN.xml.rels document for the `Target` of the `Relationship`
/// whose `Type` ends with `/notesSlide`. Returns the raw target string
/// (typically "../notesSlides/notesSlideN.xml") or None if absent.
fn find_notes_target(rels_xml: &str) -> Option<String> {
    let mut reader = Reader::from_str(rels_xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                if e.local_name().as_ref() == b"Relationship" {
                    let mut is_notes = false;
                    let mut target: Option<String> = None;
                    for attr in e.attributes().flatten() {
                        let key = attr.key.local_name();
                        let val = attr.unescape_value().unwrap_or_default().to_string();
                        if key.as_ref() == b"Type" && val.ends_with("/notesSlide") {
                            is_notes = true;
                        } else if key.as_ref() == b"Target" {
                            target = Some(val);
                        }
                    }
                    if is_notes {
                        return target;
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    None
}

/// Resolve a notesTarget (relative to the slide's location, e.g.
/// "../notesSlides/notesSlide1.xml") into a zip archive path rooted at
/// "ppt/...".  Slide is at "ppt/slides/slideN.xml"; we strip "../" and
/// join with the slide's directory.
fn resolve_notes_path(slide_path: &str, notes_target: &str) -> String {
    // notesTarget typically is "../notesSlides/notesSlideN.xml"
    let slide_dir = slide_path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
    // Strip leading "../" segments and walk them up.
    let mut parts: Vec<&str> = slide_dir.split('/').collect();
    let mut rel = notes_target;
    while let Some(rest) = rel.strip_prefix("../") {
        rel = rest;
        if !parts.is_empty() {
            parts.pop();
        }
    }
    if rel.starts_with('/') {
        rel = &rel[1..];
    }
    let mut path = parts.join("/");
    if !path.is_empty() {
        path.push('/');
    }
    path.push_str(rel);
    path
}