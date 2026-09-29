//! 扫描 resources/knowledge/ 并 upsert 到 DB
use crate::db::{Db, ResourceInput};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub knowledge_root: PathBuf,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanSummary {
    pub categories_count: usize,
    pub resources_count: usize,
    pub errors_count: usize,
    pub duration_ms: u128,
}

pub fn classify_type(ext: &str) -> Option<&'static str> {
    match ext.to_ascii_lowercase().as_str() {
        "md" | "markdown" => Some("markdown"),
        "pdf" => Some("pdf"),
        "docx" => Some("docx"),
        "pptx" => Some("pptx"),
        _ => None,
    }
}

pub fn scan(db: &Db, cfg: &ScanConfig) -> Result<ScanSummary> {
    let started = std::time::Instant::now();
    validate_root(&cfg.knowledge_root)?;

    // 1) 遍历收集 (category_path -> resources) + (categories 集合)
    let mut cat_paths: std::collections::BTreeSet<String> = Default::default();
    let mut cat_titles: std::collections::HashMap<String, (Option<String>, i64)> = Default::default();
    let mut per_cat_resources: std::collections::BTreeMap<String, Vec<ResourceInput>> = Default::default();
    let mut errors: usize = 0;
    let mut res_count: usize = 0;

    for entry in WalkDir::new(&cfg.knowledge_root).follow_links(false) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[scanner] walk error: {e}");
                continue;
            }
        };
        let path = entry.path();
        let rel = match path.strip_prefix(&cfg.knowledge_root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if entry.file_type().is_dir() {
            let cat_path = rel.to_string_lossy().replace('\\', "/");
            if cat_path.is_empty() {
                continue;
            }
            cat_paths.insert(cat_path.clone());
            cat_titles.entry(cat_path.clone()).or_insert((None, parse_sort_order(&cat_path)));
            continue;
        }
        let file_name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n,
            None => continue,
        };
        // _index.md 在分类目录下：解析分类 title，不入库为资源
        if file_name == "_index.md" {
            if let Some(parent) = rel.parent() {
                let cat_path = parent.to_string_lossy().replace('\\', "/");
                if let Ok(content) = std::fs::read_to_string(path) {
                    if let Some(title) = first_h1(&content) {
                        cat_titles
                            .entry(cat_path)
                            .and_modify(|t| t.0 = Some(title.clone()))
                            .or_insert((Some(title), 0));
                    }
                }
            }
            continue;
        }
        // 资源
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let Some(r#type) = classify_type(ext) else { continue };
        // docx/pptx 是 zip 容器，校验 magic bytes；不合法视为损坏
        if matches!(r#type, "docx" | "pptx") {
            match std::fs::read(path) {
                Ok(bytes) if bytes.len() >= 4 && &bytes[..4] == b"PK\x03\x04" => {}
                Ok(_) | Err(_) => {
                    eprintln!("[scanner] corrupt {}: {}", r#type, path.display());
                    errors += 1;
                    continue;
                }
            }
        }
        let parent_cat = rel
            .parent()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        if parent_cat.is_empty() {
            // 不允许资源直接放根目录
            continue;
        }
        cat_paths.insert(parent_cat.clone());
        cat_titles.entry(parent_cat.clone()).or_insert((None, parse_sort_order(&parent_cat)));
        let rel_path = file_name.to_string();
        let title = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&rel_path)
            .to_string();
        let size = entry.metadata().map(|m| m.len() as i64).unwrap_or(0);
        let hash = match sha256_file(path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[scanner] hash failed for {}: {e:#}", path.display());
                errors += 1;
                continue;
            }
        };
        per_cat_resources
            .entry(parent_cat)
            .or_default()
            .push(ResourceInput {
                category_path: String::new(), // patch after we know parent
                rel_path,
                r#type: r#type.to_string(),
                title,
                size_bytes: size,
                hash,
                page_count: None,
                word_count: None,
            });
    }

    // 2) upsert categories
    let mut cat_count = 0usize;
    for path in &cat_paths {
        let parent = parent_of(path);
        let (title_opt, sort_order) = cat_titles.get(path).cloned().unwrap_or((None, 0));
        let title = title_opt.unwrap_or_else(|| humanize_dir_name(path));
        db.upsert_category(path, parent.as_deref(), &title, sort_order)
            .with_context(|| format!("upsert category {path}"))?;
        cat_count += 1;
    }

    // 3) upsert resources
    for (cat, mut items) in per_cat_resources {
        for item in items.iter_mut() {
            item.category_path = cat.clone();
        }
        let keep: Vec<String> = items.iter().map(|r| r.rel_path.clone()).collect();
        match () {
            () => {
                for item in items {
                    if db.upsert_resource(item).is_err() {
                        errors += 1;
                    } else {
                        res_count += 1;
                    }
                }
                if let Ok(n) = db.delete_missing_resources(&cat, &keep) {
                    if n > 0 {
                        // rows removed aren't errors
                    }
                }
            }
        }
    }

    Ok(ScanSummary {
        categories_count: cat_count,
        resources_count: res_count,
        errors_count: errors,
        duration_ms: started.elapsed().as_millis(),
    })
}

pub fn validate_root(root: &Path) -> Result<()> {
    if !root.is_dir() {
        anyhow::bail!("knowledge root is not a directory: {}", root.display());
    }
    Ok(())
}

fn parse_sort_order(p: &str) -> i64 {
    // 第一段若为两位数字，解析为排序号；否则 9999（保证未编号目录排到末尾）
    p.split('/').next().and_then(|s| s.parse::<i64>().ok()).unwrap_or(9999)
}

fn parent_of(path: &str) -> Option<String> {
    let mut parts: Vec<&str> = path.split('/').collect();
    if parts.len() <= 1 {
        return None;
    }
    parts.pop();
    Some(parts.join("/"))
}

fn first_h1(content: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

fn humanize_dir_name(path: &str) -> String {
    path.rsplit('/')
        .next()
        .unwrap_or(path)
        .split('-')
        .map(|s| {
            let mut c = s.chars();
            match c.next() {
                Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(hex::encode(hasher.finalize()))
}