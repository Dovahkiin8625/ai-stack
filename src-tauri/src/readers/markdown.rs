use anyhow::{Context, Result};
use comrak::{markdown_to_html, Options};
use std::path::Path;

/// PUA 占位符：把 `$...$` 内的 `_` / `*` / `\` 暂时换成这三个字符。
///
/// comrak 不识别 `$...$` 为数学，会按 CommonMark 严格做两件事：
///
/// 1. **emphasis 规则**：`$\mathbb{E}_\tau$` 里的第一个 `_` 同时左/右
///    符合（preceded by `}`、followed by `\`，都是 punctuation），会被
///    当成 `<em>` 开标签，整段公式被切成多段 text node，KaTeX auto-render
///    找不到完整 `$...$` 配对 → 静默不渲染。
/// 2. **escape 规则**：`\<punct>` 里的 `\` 会被吃掉，`\!` 变成 `!`、
///    `\;` 变成 `;`，KaTeX 看到的是字面字符而不是 thin-space 命令。
///
/// 解决思路：解析前把 math 内的 `_` / `*` / `\` 换成 PUA 占位符
/// （comrak 视为不识别字符，原样输出）；解析后再换回原字符，让 KaTeX
/// 看到的是干净的 LaTeX 源码。PUA 是 Unicode 私有区，comrak 不会对它
/// 做任何加工，也不会变成 HTML 实体。
const UNDERSCORE_PH: char = '\u{E000}';
const ASTERISK_PH: char = '\u{E001}';
const BACKSLASH_PH: char = '\u{E002}';

/// 读 markdown 文件 → (html, word_count, source)。
///
/// 三元组里 source 是原始 markdown 文本：编辑模式需要把原文回传给前端做
/// textarea 双向绑定，只回 html 不够 —— 用户编辑后切回预览用的是 comrak
/// 重新渲染，所以 round-trip 不会丢失信息。
pub fn extract(path: &Path) -> Result<(String, usize, String)> {
    let md = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut opts = Options::default();
    opts.extension.shortcodes = true;
    // GFM pipe table 必须显式开启：comrak 默认 extension.table = false，
    // 不开的话 `| a | b |` 这类语法会退化成段落文本而不是 <table>。
    opts.extension.table = true;
    // 保护 `$...$` / `$$...$$` 内的 `_` / `*`，避免被 comrak 当 emphasis
    let protected = protect_math_emphasis(&md);
    let html = markdown_to_html(&protected, &opts);
    let html = restore_math_emphasis(&html);
    let words = count_words(&md);
    Ok((html, words, md))
}

/// 在 `$...$` / `$$...$$` 区域内把 `_` / `*` 换成 PUA 占位符。
/// 跳过反引号代码块、行内代码、转义符 `\`（包括 `\$`、`\_`、`\*`），
/// 这些区域里的 `_` / `*` 不是 markdown emphasis 也不会被处理。
fn protect_math_emphasis(md: &str) -> String {
    let chars: Vec<char> = md.chars().collect();
    let len = chars.len();
    let mut out = String::with_capacity(md.len());
    let mut i = 0;

    while i < len {
        // 转义序列 `\$`/`\_`/`\*`/`\\`：原样复制两个字符。
        if chars[i] == '\\' && i + 1 < len {
            out.push(chars[i]);
            out.push(chars[i + 1]);
            i += 2;
            continue;
        }

        // 反引号围栏 / 代码块：里面的字符都按字面意思处理。
        if chars[i] == '`' {
            let mut j = i;
            while j < len && chars[j] == '`' {
                j += 1;
            }
            let bt = j - i;
            // 找等长的反引号作为收尾（用第一组匹配的，避免跨过 backtick 边界）
            while j < len {
                if chars[j] == '`' {
                    let mut k = j;
                    while k < len && chars[k] == '`' {
                        k += 1;
                    }
                    if k - j == bt {
                        j = k;
                        break;
                    }
                    j = k.max(j + 1);
                } else {
                    j += 1;
                }
            }
            out.push_str(&chars[i..j.min(len)].iter().collect::<String>());
            i = j;
            continue;
        }

        // `$$...$$` 显示块。必须先判 `$$`，否则单 `$` 配对会把首 `$` 误判成 `$...$` 的开。
        if chars[i] == '$' && i + 1 < len && chars[i + 1] == '$' {
            let content_start = i + 2;
            let mut j = content_start;
            let mut found = false;
            while j < len {
                if chars[j] == '\\' {
                    j += 2;
                    continue;
                }
                if chars[j] == '$' && j + 1 < len && chars[j + 1] == '$' {
                    found = true;
                    break;
                }
                j += 1;
            }
            if found {
                out.push_str("$$");
                // 把 `_` / `*` / `\` 全部换成 PUA，避免 comrak 的 emphasis / escape 规则
                // 污染 LaTeX 源码（`\!` 被剥成 `!`，`\tau` 被吃成 `tau` 都不会发生）。
                let mut k = content_start;
                while k < j {
                    let c = match chars[k] {
                        '_' => UNDERSCORE_PH,
                        '*' => ASTERISK_PH,
                        '\\' => BACKSLASH_PH,
                        _ => chars[k],
                    };
                    out.push(c);
                    k += 1;
                }
                out.push_str("$$");
                i = j + 2;
                continue;
            }
        }

        // `$...$` 行内公式。
        if chars[i] == '$' {
            let content_start = i + 1;
            let mut j = content_start;
            let mut found = false;
            while j < len {
                if chars[j] == '\\' {
                    j += 2;
                    continue;
                }
                if chars[j] == '$' {
                    // 别把 `$$` 的第二个 `$` 当行内 math 收尾。
                    if j + 1 < len && chars[j + 1] == '$' {
                        j += 2;
                        continue;
                    }
                    found = true;
                    break;
                }
                j += 1;
            }
            if found && j > content_start {
                out.push('$');
                let mut k = content_start;
                while k < j {
                    let c = match chars[k] {
                        '_' => UNDERSCORE_PH,
                        '*' => ASTERISK_PH,
                        '\\' => BACKSLASH_PH,
                        _ => chars[k],
                    };
                    out.push(c);
                    k += 1;
                }
                out.push('$');
                i = j + 1;
                continue;
            }
        }

        out.push(chars[i]);
        i += 1;
    }

    out
}

/// 把 HTML 里的 PUA 占位符还原成 `_` / `*` / `\`。
/// 占位符是私有区字符，comrak 不会对它做任何加工，也不会编码成 HTML 实体；
/// 整篇 HTML 里它们只可能来自 protect_math_emphasis，全局 replace 即可。
fn restore_math_emphasis(html: &str) -> String {
    html.replace(BACKSLASH_PH, "\\")
        .replace(UNDERSCORE_PH, "_")
        .replace(ASTERISK_PH, "*")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protect(s: &str) -> String {
        protect_math_emphasis(s)
    }

    #[test]
    fn underscore_inside_inline_math_is_protected() {
        let out = protect("$\\mathbb{E}_\\tau$");
        // `\` 和 `_` 都被保护，math 内不应残留字面字符
        assert!(!out.contains('_'), "_ 应被替换成 PUA，实际: {out}");
        assert!(!out.contains('\\'), r"\ 应被替换成 PUA，实际: {out}");
        // 仍保留 `$` 边界
        assert!(out.starts_with('$') && out.ends_with('$'));
    }

    #[test]
    fn asterisk_inside_inline_math_is_protected() {
        let out = protect(r"$a \cdot b$");
        assert!(!out.contains('*'), "* 应被替换成 PUA，实际: {out}");
        assert!(out.contains("$a "));
    }

    #[test]
    fn display_math_is_protected() {
        let out = protect("$$\\sum_x P(x) \\log P(x)$$");
        assert!(!out.contains('_'), "$$...$$ 内的 _ 应被保护，实际: {out}");
    }

    #[test]
    fn backslash_in_math_is_protected() {
        // comrak 默认对 `\<punct>` 做 CommonMark escape：`\!` → `!`。
        // protect 必须把 `\` 换成 PUA，否则 KaTeX 看到的不是 thin-space
        // 命令而是字面字符。下面这一行里：`\!` 的 `\` 换成 BACKSLASH_PH，
        // `_` 换成 UNDERSCORE_PH，整段 math 都不应残留字面 `\` / `_`。
        let out = protect(r"$\mathbb{E}\!\left(a_b\right)$");
        assert!(!out.contains('\\'), r"math 内的 `\` 应被换成 PUA，实际: {out}");
        assert!(!out.contains('_'), r"math 内的 `_` 应被换成 PUA，实际: {out}");
        assert!(out.contains(BACKSLASH_PH));
        assert!(out.contains(UNDERSCORE_PH));
    }

    #[test]
    fn backtick_outside_math_unchanged() {
        // 行内代码里的 `\!` / `_` 不属于 math，不应被换 PUA
        let out = protect("`code\\_with_underscore` $a_b$");
        // 反引号段原样保留
        assert!(
            out.starts_with("`code\\_with_underscore`"),
            "行内代码应原样保留，实际: {out}"
        );
        // 行内公式段里的占位符
        assert!(out.contains(UNDERSCORE_PH));
    }

    #[test]
    fn backtick_code_span_is_not_math() {
        // 行内代码里的 `_` 不应被替换成 PUA。
        let out = protect("`code_with_underscore` $a_b$");
        // 反引号段落原样保留，后面 `$a_b$` 才换。
        assert!(out.starts_with("`code_with_underscore`"));
        assert!(out.contains(UNDERSCORE_PH), "公式里的 _ 才该被换");
    }

    #[test]
    fn dollar_inside_code_block_is_not_math() {
        let out = protect("```\n$not_math$\n```");
        // 围栏代码块原样保留，里面的 `_` 不换。
        assert!(out.contains("$not_math$"), "围栏内的内容不应被改: {out}");
    }

    #[test]
    fn restore_reverts_placeholders() {
        let html = format!("<p>x {UNDERSCORE_PH} y {ASTERISK_PH}</p>");
        let back = restore_math_emphasis(&html);
        assert_eq!(back, "<p>x _ y *</p>");
    }

    #[test]
    fn display_math_does_not_eat_single_dollar_opener() {
        // `$$ 文本 \mathbb{E}_X $$`：$$ 对应单 `$` 配对，避免乱配。
        let out = protect(r"$$\mathbb{E}_X $$");
        assert!(
            out.starts_with("$$"),
            "应正确判 $$...$$ 为显示块，实际: {out}"
        );
        // 必须正好出现一次结尾 `$$`（开头 + 结尾）
        assert_eq!(out.matches("$$").count(), 2);
    }
}

fn count_words(s: &str) -> usize {
    s.split_whitespace().count()
}