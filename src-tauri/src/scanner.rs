//! 扫描 `resources/knowledge/` 并 upsert 到 DB
//!
//! 启动期只做目录扫描 + 元数据采集（size, mtime）。**不读文件内容** ——
//! PDF/DOCX/PPTX/Markdown 都在用户真正打开时才解析（见 reader.rs）；
//! `_index.md` 也是 expand 分类时才读（见 `read_subcategory_index`）。
//!
//! 启动期 IO 预算：每次 walk 一次目录树，每个文件一次 `metadata()` syscall，
//! 没有额外 read/hash。
use crate::db::{Db, ResourceInput};
use anyhow::{Context, Result};
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

/// "透传"目录名：物理分组容器，不作为独立 category 出现在 sidebar，
/// 里面的文件归属上层分类（`rel_path` 保留该目录前缀以便 reader 拼路径）。
///
/// 当前只有 `三方资料/` —— 用来收 pdf/docx/pptx 等第三方资料。
/// 后续要再加（比如 `讲义/`）在这里追加即可。
const PASSTHROUGH_DIR_NAMES: &[&str] = &["三方资料"];

fn is_passthrough_dir(name: &str) -> bool {
    PASSTHROUGH_DIR_NAMES.contains(&name)
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

/// 计算一个资源路径的有效 category：
/// - 普通情况：父目录即为 category
/// - 若父目录是透传目录（如 `三方资料/`），则向上跳一层，把上层目录当作 category
///
/// 返回 `(category_path, rel_path)`，rel_path 包含透传目录前缀（如 `三方资料/foo.pdf`）。
fn effective_category_and_rel(rel: &Path) -> Option<(String, String)> {
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    let file_name = rel.file_name()?.to_str()?;
    let parent = rel.parent()?;
    let parent_str = parent.to_string_lossy().replace('\\', "/");
    let parent_name = parent.file_name().and_then(|s| s.to_str()).unwrap_or("");

    if is_passthrough_dir(parent_name) {
        // 跳到上一层目录作为 category
        let grandparent = parent.parent()?;
        let gp_str = grandparent.to_string_lossy().replace('\\', "/");
        let prefix = format!("{gp_str}/");
        let rel_path = rel_str.strip_prefix(&prefix).unwrap_or(&rel_str).to_string();
        Some((gp_str, rel_path))
    } else {
        let prefix = format!("{parent_str}/");
        let rel_path = rel_str.strip_prefix(&prefix).unwrap_or(&rel_str).to_string();
        // 如果 rel_path 就是 filename 且不在子目录里，留 file_name 以保证向后兼容
        let rel_path = if rel_path == file_name && !parent_str.is_empty() {
            file_name.to_string()
        } else {
            rel_path
        };
        Some((parent_str, rel_path))
    }
}

pub fn scan(db: &Db, cfg: &ScanConfig) -> Result<ScanSummary> {
    let started = std::time::Instant::now();
    validate_root(&cfg.knowledge_root)?;

    // 预读 DB 已有资源 (size, mtime) —— 跳过未变文件，避免每个启动都把
    // 整张 resources 表 upsert 一次。
    let existing = db.existing_resources_meta().unwrap_or_default();

    // 1) 遍历收集 (category_path -> resources) + (categories 集合)
    let mut cat_paths: std::collections::BTreeSet<String> = Default::default();
    let mut cat_titles: std::collections::HashMap<String, (Option<String>, i64)> = Default::default();
    let mut per_cat_resources: std::collections::BTreeMap<String, Vec<ResourceInput>> = Default::default();
    // 跳过的文件不写入 per_cat_resources，但要记录在 keep_set 里，防止
    // delete_missing_resources 把它们误删。
    let mut keep_set: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> = Default::default();
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
            // 透传目录（如 三方资料/）不进 cat_paths —— 它是物理分组，不是逻辑分类
            let dir_name = entry.file_name().to_string_lossy();
            if is_passthrough_dir(&dir_name) {
                continue;
            }
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
        // _index.md 不入库、不读内容 —— sidebar 标题先 fallback 到目录名，
        // 用户 expand 分类时由 read_subcategory_index 读真 H1。
        // 透传目录里的 _index.md 也不属于任何 category，跳过。
        if file_name == "_index.md" {
            continue;
        }
        // 资源
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let Some(r#type) = classify_type(ext) else { continue };
        // 一次 syscall 拿到 size + mtime —— 不要分两次 metadata() 调用。
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let size = meta.len() as i64;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        // 计算有效 category：透传目录下的文件归属上层分类
        let Some((parent_cat, rel_path)) = effective_category_and_rel(rel) else {
            eprintln!("[scanner] skip resource without category: {}", path.display());
            continue;
        };
        if parent_cat.is_empty() {
            // 不允许资源直接放根目录
            continue;
        }
        cat_paths.insert(parent_cat.clone());
        cat_titles.entry(parent_cat.clone()).or_insert((None, parse_sort_order(&parent_cat)));
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&rel_path)
            .to_string();

        // 跳过 (size, mtime) 未变的文件 —— 不读不哈希。
        let lookup_key = (parent_cat.clone(), rel_path.clone());
        if let Some((e_size, e_mtime)) = existing.get(&lookup_key) {
            if *e_size == size && *e_mtime == mtime {
                keep_set
                    .entry(parent_cat.clone())
                    .or_default()
                    .insert(rel_path.clone());
                res_count += 1; // 跳过 = 已存在的资源，仍计入总数
                continue;
            }
        }

        // 需要重新索引：title 用 stem（不读文件内容找 H1 —— 那是打开文档时才做的）
        keep_set
            .entry(parent_cat.clone())
            .or_default()
            .insert(rel_path.clone());
        per_cat_resources
            .entry(parent_cat)
            .or_default()
            .push(ResourceInput {
                category_path: String::new(), // patch after we know parent
                rel_path,
                r#type: r#type.to_string(),
                title: stem,
                size_bytes: size,
                mtime,
                page_count: None,
                word_count: None,
            });
    }

    // 2) upsert categories
    let mut cat_count = 0usize;
    for path in &cat_paths {
        let parent = parent_of(path);
        // H1 已经不在扫描时取了 —— 标题先走 humanize_dir_name fallback。
        // 用户 expand 分类时由 read_subcategory_index 返真 H1，UI 自行用之。
        let (title_opt, sort_order) = cat_titles.get(path).cloned().unwrap_or((None, 0));
        let title = title_opt.unwrap_or_else(|| humanize_dir_name(path));
        db.upsert_category(path, parent.as_deref(), &title, sort_order)
            .with_context(|| format!("upsert category {path}"))?;
        cat_count += 1;
    }

    // 3) upsert resources
    //
    // 遍历范围 = per_cat_resources ∪ keep_set 的所有 category：
    // - per_cat_resources 含本次需要 upsert 的资源（新增 / size|mtime 变动）
    // - keep_set 含被跳过的资源（但分类下若所有文件都跳过 / 全删了，
    //   per_cat_resources 为空，需要靠 keep_set 触发 delete_missing_resources）
    let mut all_cats: std::collections::BTreeSet<String> = per_cat_resources.keys().cloned().collect();
    all_cats.extend(keep_set.keys().cloned());
    for cat in all_cats {
        let mut items = per_cat_resources.remove(&cat).unwrap_or_default();
        for item in items.iter_mut() {
            item.category_path = cat.clone();
        }
        // keep 列表 = 本次扫到的所有资源（含 skipped），
        // 让 delete_missing_resources 不会误删被跳过的文件。
        let mut keep: Vec<String> = items.iter().map(|r| r.rel_path.clone()).collect();
        if let Some(skipped) = keep_set.get(&cat) {
            for s in skipped {
                if !keep.contains(s) {
                    keep.push(s.clone());
                }
            }
        }
        for item in items {
            if db.upsert_resource(item).is_err() {
                errors += 1;
            } else {
                res_count += 1;
            }
        }
        if let Ok(_n) = db.delete_missing_resources(&cat, &keep) {
            // rows removed aren't errors
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
