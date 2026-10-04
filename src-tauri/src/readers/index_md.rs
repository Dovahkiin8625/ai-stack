//! `_index.md` 解析：把人类可读的 markdown 目录文件解析成结构化的 `ParsedIndex`，
//! 供 ArticleIndexView 动态渲染页面使用。
//!
//! 解析规则：
//!
//! ```text
//! # Title                        → ParsedIndex.title
//!
//! > blockquote                   ↘
//! Some intro paragraph.          → ParsedIndex.preamble（到第一个 ## 为止）
//!
//! ## Section heading             → ParsedSection.heading
//! ### Subgroup                   → ParsedGroup.subheading
//! - [Title](./relative-path) — Description  → ParsedGroup.entries
//! - [Title](./x.pdf)             → ParsedGroup.entries（无 description）
//! - Non-link bullet / paragraph  → raw_markdown（按当前位置累加）
//! ```
//!
//! 行首允许缩进；分隔符 `—` (em-dash, U+2014) 或 `-` (ASCII)；rel_path 自动去掉 `./`。
//! H1 之后才进入 preamble 收集；遇到第一个 `##` 进入 section 模式，preamble 停止增长。
//!
//! 设计原则：**严格遵循 `_index.md` 结构**——H1 / preamble / ## section / ### group
//! / bullet entries / raw markdown 都按出现顺序进入数据结构，渲染层据此动态布局，
//! 不再做"忽略 H1"、"忽略 ###"、"混在一起渲染"的隐式拍平。
//!
//! Parser 不假设任何 section 名字——`## 文章目录` / `## 三方资料` / 任何自定义标题
//! 都按原文进入数据结构，由渲染层决定怎么显示。
use anyhow::{Context as _, Result};
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct IndexEntry {
    pub rel_path: String,
    pub title: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ParsedGroup {
    /// ### 子标题；None 表示没有 ### 子标题，entries 直接挂在 ## 下
    /// （包含"## section 下面又有一个 entry"的情形）。
    pub subheading: Option<String>,
    pub entries: Vec<IndexEntry>,
    /// 该 ### 下的非条目 markdown（段落 / 非链接 bullet）。
    pub raw_markdown: String,
}

#[derive(Debug, Clone, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ParsedSection {
    /// ## 标题，原文照录（parser 不假设任何特定名字）。
    pub heading: String,
    /// 该 section 下的 ### 子分组（含 entries）。
    pub groups: Vec<ParsedGroup>,
    /// 该 section 顶层（非任何 ### 下面）的非条目 markdown。
    pub raw_markdown: String,
}

#[derive(Debug, Clone, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ParsedIndex {
    /// # H1 标题（页面标题）。None 表示文件没 H1。
    pub title: Option<String>,
    /// H1 与第一个 ## 之间的 markdown（含 blockquote、段落）。
    pub preamble: Option<String>,
    /// 全部 ## sections（按文件顺序）。
    pub sections: Vec<ParsedSection>,
}

/// 读 `path` 指向的 `_index.md`，返回结构化数据。文件不存在 → 空结构（不报错）。
pub fn parse_file(path: &Path) -> Result<ParsedIndex> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    Ok(parse(&content))
}

/// 从字符串内容解析。便于测试。
pub fn parse(content: &str) -> ParsedIndex {
    let mut state = ParseState {
        in_preamble: true,
        ..Default::default()
    };
    for line in content.lines() {
        state.feed_line(line);
    }
    state.finish()
}

/// 单行解析辅助（用于测试 entry 解析；不影响顶层 parse() 的结构）。
fn parse_entry_body(body: &str) -> Option<IndexEntry> {
    let body = body.strip_prefix('[')?;
    let close_bracket = body.find(']')?;
    let title = body[..close_bracket].to_string();
    let after_title = &body[close_bracket + 1..];
    let body2 = after_title.strip_prefix('(')?;
    let close_paren = body2.find(')')?;
    let raw_rel = &body2[..close_paren];
    if raw_rel.is_empty() {
        return None;
    }
    let rel_path = raw_rel.strip_prefix("./").unwrap_or(raw_rel).to_string();
    let after_link = &body2[close_paren + 1..];
    let after_link = after_link.trim_start();
    let description = strip_separator(after_link)
        .map(|s| s.trim_start().trim_end())
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    Some(IndexEntry {
        rel_path,
        title,
        description,
    })
}

fn strip_separator(s: &str) -> Option<&str> {
    if let Some(rest) = s.strip_prefix('\u{2014}') {
        Some(rest)
    } else if let Some(rest) = s.strip_prefix('-') {
        Some(rest)
    } else {
        None
    }
}

#[derive(Default)]
struct ParseState {
    title: Option<String>,
    preamble: String,
    sections: Vec<ParsedSection>,
    /// 是否还在 preamble 阶段（H1 之后、第一个 ## 之前）
    in_preamble: bool,
}

impl ParseState {
    fn feed_line(&mut self, raw_line: &str) {
        // 处理掉换行后回填 —— raw_markdown 用 \n 拼接，便于 trim
        let trimmed_start = raw_line.trim_start();
        let line = raw_line;

        // H1: 仅在 preamble 阶段（第 1 个非空内容行 / 还没遇到 ##）才算 title。
        if self.in_preamble && self.title.is_none() && is_h1(trimmed_start) {
            self.title = Some(trimmed_start[2..].trim().to_string());
            return;
        }
        // 一旦 title 设置过，再见到 # 开头的行就当 raw markdown（罕见）
        // —— 这里不特殊处理，作为普通行继续累积到 preamble/raw

        // H2: 新 section
        if is_h2(trimmed_start) {
            let heading = trimmed_start[3..].trim().to_string();
            self.sections.push(ParsedSection {
                heading,
                groups: Vec::new(),
                raw_markdown: String::new(),
            });
            self.in_preamble = false;
            return;
        }

        // H3+: 新 group（仅在有 section 时）
        if is_h3(trimmed_start) {
            if let Some(sec) = self.sections.last_mut() {
                let subheading = trimmed_start[4..].trim().to_string();
                sec.groups.push(ParsedGroup {
                    subheading: Some(subheading),
                    entries: Vec::new(),
                    raw_markdown: String::new(),
                });
            }
            // ### 在 ## 之前：当作 raw markdown（继续落到 preamble/raw）
            return;
        }

        // Bullet 行
        if let Some(body) = bullet_body(trimmed_start) {
            if let Some(entry) = parse_entry_body(body) {
                // 链接型条目：进入当前 section 的某个 group
                if let Some(sec) = self.sections.last_mut() {
                    if sec.groups.is_empty() {
                        // 还没有 ### 子分组 → 创建一个默认 group
                        sec.groups.push(ParsedGroup::default());
                    }
                    sec.groups.last_mut().unwrap().entries.push(entry);
                    self.in_preamble = false;
                }
                // ## 之前的链接 bullet：忽略（不常见；可能是写错了）
                return;
            }
            // 非链接 bullet：作为 raw markdown 累积（保留 `- ` 前缀）
            self.append_raw(format!("- {body}"));
            return;
        }

        // 普通行：进入 preamble 或当前 section/group 的 raw
        self.append_raw(line.to_string());
    }

    fn append_raw(&mut self, line: String) {
        if self.in_preamble {
            self.preamble.push_str(&line);
            self.preamble.push('\n');
        } else if let Some(sec) = self.sections.last_mut() {
            if let Some(group) = sec.groups.last_mut() {
                group.raw_markdown.push_str(&line);
                group.raw_markdown.push('\n');
            } else {
                sec.raw_markdown.push_str(&line);
                sec.raw_markdown.push('\n');
            }
        }
        // 既不在 preamble 也不在任何 section 里 —— 行直接丢弃
        // （不应发生，因为 ## 是从 preamble 切换到 section 的唯一触发）
    }

    fn finish(mut self) -> ParsedIndex {
        let preamble = trim_trailing_blank_lines(&self.preamble);
        // 修剪每个 section/group 的 raw_markdown 尾部空白
        for sec in &mut self.sections {
            sec.raw_markdown = trim_trailing_blank_lines(&sec.raw_markdown);
            for group in &mut sec.groups {
                group.raw_markdown = trim_trailing_blank_lines(&group.raw_markdown);
            }
        }
        ParsedIndex {
            title: self.title,
            preamble: if preamble.is_empty() { None } else { Some(preamble) },
            sections: self.sections,
        }
    }
}

fn trim_trailing_blank_lines(s: &str) -> String {
    let mut end = s.len();
    while end > 0 {
        let prev = s[..end].char_indices().last().map(|(_, c)| c).unwrap_or(' ');
        if prev == '\n' || prev == ' ' || prev == '\r' {
            end = end - prev.len_utf8();
        } else {
            break;
        }
    }
    s[..end].to_string()
}

fn is_h1(s: &str) -> bool {
    s.starts_with("# ") && !s.starts_with("## ")
}

fn is_h2(s: &str) -> bool {
    s.starts_with("## ") && !s.starts_with("### ")
}

fn is_h3(s: &str) -> bool {
    s.starts_with("### ") && !s.starts_with("#### ")
}

fn bullet_body(s: &str) -> Option<&str> {
    if let Some(rest) = s.strip_prefix("- ") {
        Some(rest)
    } else if let Some(rest) = s.strip_prefix("* ") {
        Some(rest)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 模拟原 index_md parser 的 entry 解析行为（保证旧测试兼容 + 新结构）。
    #[test]
    fn entry_basic() {
        let e = parse_entry_body("[Title](./path.md) — Description text").unwrap();
        assert_eq!(e.title, "Title");
        assert_eq!(e.rel_path, "path.md");
        assert_eq!(e.description.as_deref(), Some("Description text"));
    }

    #[test]
    fn entry_ascii_dash() {
        let e = parse_entry_body("[Title](./path.md) - Description").unwrap();
        assert_eq!(e.description.as_deref(), Some("Description"));
    }

    #[test]
    fn entry_no_description() {
        let e = parse_entry_body("[Title](./path.md)").unwrap();
        assert_eq!(e.description, None);
    }

    #[test]
    fn entry_strips_dot_slash() {
        let e = parse_entry_body("[X](x.md) — d").unwrap();
        assert_eq!(e.rel_path, "x.md");
    }

    /// === 结构化解析 ===

    #[test]
    fn section_without_h3_creates_default_group() {
        let content = "\
## 文章目录

- [A](./a.md) — A
- [B](./b.md) — B
";
        let p = parse(content);
        assert_eq!(p.sections.len(), 1);
        let sec = &p.sections[0];
        assert_eq!(sec.groups.len(), 1);
        assert!(sec.groups[0].subheading.is_none());
        assert_eq!(sec.groups[0].entries.len(), 2);
    }

    #[test]
    fn no_h1_no_preamble_only_sections() {
        let content = "## S1\n\n- [A](./a.md) — d\n";
        let p = parse(content);
        assert!(p.title.is_none());
        assert!(p.preamble.is_none());
        assert_eq!(p.sections.len(), 1);
    }

    #[test]
    fn h1_directly_followed_by_h2_no_preamble() {
        let content = "# Title\n\n## S1\n\n- [A](./a.md)\n";
        let p = parse(content);
        assert_eq!(p.title.as_deref(), Some("Title"));
        assert!(p.preamble.is_none());
        assert_eq!(p.sections.len(), 1);
    }

    #[test]
    fn ignores_bullet_before_any_section() {
        let content = "\
# Title

- [orphan](./orphan.md) — should be ignored

## S1

- [A](./a.md)
";
        let p = parse(content);
        assert_eq!(p.title.as_deref(), Some("Title"));
        assert_eq!(p.sections.len(), 1);
        // S1 只有 1 个 entry（orphan 被忽略）
        assert_eq!(p.sections[0].groups[0].entries.len(), 1);
        assert_eq!(p.sections[0].groups[0].entries[0].rel_path, "a.md");
    }

    #[test]
    fn empty_file_yields_empty_structure() {
        let p = parse("");
        assert!(p.title.is_none());
        assert!(p.preamble.is_none());
        assert!(p.sections.is_empty());
    }

    #[test]
    fn preamble_strips_trailing_blank_lines() {
        let content = "# T\n\n\n\n## S\n";
        let p = parse(content);
        assert!(p.preamble.is_none(), "preamble should be empty after stripping: {p:?}");
    }

    #[test]
    fn mixed_section_has_both_entries_and_prose() {
        let content = "\
## 章节

Intro paragraph.

- [A](./a.md) — d

End paragraph.
";
        let p = parse(content);
        let sec = &p.sections[0];
        assert_eq!(sec.groups.len(), 1);
        assert_eq!(sec.groups[0].entries.len(), 1);
        // 段落"Intro paragraph"出现在 ## 之后、entry 之前 → sec.raw_markdown
        assert!(sec.raw_markdown.contains("Intro paragraph"), "sec.raw: {sec:?}");
        // "End paragraph"出现在 entry 之后、且当前 group 还"活动" → group.raw_markdown
        assert!(sec.groups[0].raw_markdown.contains("End paragraph"), "group.raw: {sec:?}");
        // entry 行不进 raw_markdown
        assert!(!sec.raw_markdown.contains("[A]"), "raw should not include entry: {sec:?}");
        assert!(!sec.groups[0].raw_markdown.contains("[A]"), "group raw should not include entry: {sec:?}");
    }

    #[test]
    fn h3_without_h2_is_ignored() {
        let content = "\
# Title

### orphan H3

- [A](./a.md)
";
        let p = parse(content);
        assert_eq!(p.title.as_deref(), Some("Title"));
        assert!(p.sections.is_empty());
        // ### 之前的 [A] 被忽略（不在任何 section 里）
    }
}
