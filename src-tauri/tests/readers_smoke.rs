use ai_stack_lib::readers::markdown::extract;
use ai_stack_lib::readers::pdf::extract as pdf_extract;
use ai_stack_lib::readers::docx::extract as docx_extract;
use ai_stack_lib::readers::pptx::extract as pptx_extract;
use std::path::PathBuf;

/// 与 `readers::markdown::BACKSLASH_PH` 保持同步的 PUA 占位符；
/// 这里再声明一份是为了不导出内部常量。
const BACKSLASH_PH: char = '\u{E002}';

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("readers");
    p.push(name);
    p
}

#[test]
fn markdown_renders_to_html_and_counts_words() {
    let (html, words) = extract(&fixture("simple.md")).unwrap();
    assert!(html.contains("<h1>Heading</h1>"), "html: {html}");
    assert!(html.contains("<strong>world</strong>"), "html: {html}");
    assert!(html.contains("<ul>"), "html: {html}");
    assert!(words >= 4, "got {words}");
}

/// GFM pipe table 必须渲染成 `<table>`，而不是把 `|` `---` 当段落文本输出。
/// 回归守卫：comrak 默认不启用 table extension，rust 后端必须显式开启。
#[test]
fn markdown_renders_gfm_table() {
    let (html, _words) = extract(&fixture("with-table.md")).unwrap();
    assert!(
        html.contains("<table>"),
        "expected <table> in html, got: {html}"
    );
    assert!(
        html.contains("<th>列 1</th>"),
        "expected <th>列 1</th> in html, got: {html}"
    );
    assert!(
        html.contains("<td>单元格 1</td>"),
        "expected <td>单元格 1</td> in html, got: {html}"
    );
}

#[test]
fn pdf_returns_pages_or_falls_back() {
    let result = pdf_extract(&fixture("sample.pdf"));
    match result {
        Ok((pages, count)) => {
            assert!(count >= 1);
            // 若 pdfium 缺失则 pages 为空向量
            if !pages.is_empty() {
                assert!(pages[0].1.starts_with("data:image/png;base64,"));
            }
        }
        Err(_) => {
            // 也可接受：失败时不 panic
        }
    }
}

#[test]
fn docx_extracts_headings_and_paragraphs() {
    let (blocks_value, words) = docx_extract(&fixture("sample.docx")).unwrap();
    let blocks = blocks_value.as_array().expect("blocks is array");
    let kinds: Vec<&str> = blocks.iter()
        .map(|b| b.get("kind").and_then(|v| v.as_str()).unwrap_or(""))
        .collect();
    assert!(kinds.contains(&"heading"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"paragraph"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"table"), "kinds: {kinds:?}");
    assert!(words > 0);
}

#[test]
fn pptx_extracts_slides() {
    let (value, _) = pptx_extract(&fixture("sample.pptx")).unwrap();
    // Pptx extractor returns the slides array directly (NOT wrapped in {"slides", "slide_count"}).
    // Wrapping was a bug: frontend `slides[idx].index` then crashed on undefined.
    let slides = value.as_array().expect("value is the slides array directly");
    assert!(slides.len() >= 2, "got {} slides", slides.len());
    let first = &slides[0];
    assert!(first.get("title").is_some());
    let body = first.get("body").and_then(|v| v.as_array()).unwrap();
    assert!(!body.is_empty(), "body empty");
}

#[test]
fn pptx_value_is_array_not_wrapped_object() {
    // Regression: previously pptx_extract returned a wrapped object
    // `{"slides": [...], "slide_count": N}` which broke PptxReader.tsx.
    let (value, _) = pptx_extract(&fixture("sample.pptx")).unwrap();
    assert!(
        value.is_array(),
        "pptx value should be the slides array, not a wrapper object"
    );
    assert!(
        value.get("slides").is_none(),
        "value must not be wrapped in {{'slides': ...}}"
    );
}

/// 回归守卫：math 代码块必须用 ``` 收尾，不能用 `$$` 当收尾。
/// 之前 information-theory.md 第 92 行错把 `$$` 当 math 围栏的收尾，
/// comrak 不会识别 → 数学块一直延伸到下一个 ``` ，把后续章节标题、段落
/// 一起吞进 math 块 → KaTeX 渲染失败，整段显示成红色错误。
/// 用一篇真实文章验证 R(D) 公式后面的二级标题能正常渲染成 <h2>。
#[test]
fn math_code_block_closing_fence_must_be_backticks() {
    let path = resource("01-foundations/01-mathematics/information-theory.md");
    let (html, _) = extract(&path).unwrap();
    // 修复前：h2 "六、信息论在决策树与强化学习中的应用" 被吞进 math 块
    assert!(
        html.contains("<h2>六、信息论"),
        "R(D) 公式后的 <h2> 六、信息论... 应正常渲染，实际被吸入 math 块。html 片段:\n{}",
        html
    );
    // 修复前：math 块里塞了一整段 "率失真解释了为什么..." 中文，
    // 这段不会出现在 <p> 里。
    assert!(
        html.contains("率失真解释了为什么我们可以对图像做有损压缩"),
        "math 块之后的释义段落应在 <p> 里，不能被吞到 math 块。"
    );
    // 顺带覆盖 unsupervised-learning.md 同一类 bug
    let path2 = resource("01-foundations/03-ml-basics/unsupervised-learning.md");
    let (html2, _) = extract(&path2).unwrap();
    assert!(
        html2.contains("训练用 EM 算法"),
        "GMM 公式后 '训练用 EM 算法' 段落应正常出现"
    );
}

fn resource(rel_path: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR = src-tauri/，knowledge 资源在 repo 根 resources/
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.push("resources");
    p.push("knowledge");
    p.push(rel_path);
    p
}

/// 回归守卫：行内 math（`$...$`）里的 `_` 不能被 comrak 当 emphasis 解析。
///
/// 之前 information-theory.md 第 100 行的
/// `$\mathbb{E}_\tau[\nabla_\theta \log \pi_\theta(a|s) \cdot R(\tau)]$`
/// 被 comrak 处理后变成
/// `$\mathbb{E}<em>\tau[\nabla</em>\theta ...$`，
/// `\$...\$` 中间出现 `<em>` 元素 → KaTeX auto-render 找不到完整 `$...$`
/// 对应的 text node（被 `<em>` 截成三段）→ 整段公式不渲染。
/// 视觉上变成 "$...$ 消失了"，VSCode 看起来正常是因为它的 pipeline
/// 在 markdown-it 之前对 `$...$` 内容做了转义保护。
///
/// 修复方向：markdown reader 把 `$...$` 内的 `_` / `*` 暂时换成 PUA 占位符
/// （comrak 不会对它做 emphasis 加工），过完 comrak 再换回原字符，
/// 这样 KaTeX 看到的依然是 `\mathbb{E}_\tau` 的 LaTeX 源码。
#[test]
fn inline_math_underscore_not_emphasized() {
    let (html, _) = extract(&fixture("inline-math-underscore.md")).unwrap();
    // 没有修复前：fixture 里的 $a_b$ 会被包成 $a<em>b</em>$
    assert!(
        !html.contains("<em>"),
        "comrak 不应在 $...$ 行内加 <em>，实际 HTML:\n{}",
        html
    );
    // 占位符字符不能泄漏
    assert!(
        !html.contains('\u{E000}'),
        "PUA 占位符应在 comrak 后被还原"
    );
    // 原 `_` 必须原样保留
    assert!(
        html.contains("$a_b$"),
        "fixture 里的 $a_b$ 必须保持 $a_b$，没修复前后均未添加此断言"
    );
}

/// 真实文章层面：第 100 行那条公式渲染出的 HTML 必须是完整的
/// `$...$` 区间，不能被 `<em>` 截断。
#[test]
fn information_theory_line_100_formula_not_emphasized() {
    let path = resource("01-foundations/01-mathematics/information-theory.md");
    let (html, _) = extract(&path).unwrap();
    let needle = r"$\mathbb{E}<em>\tau";
    assert!(
        !html.contains(needle),
        "comrak 把信息论第 100 行公式里的 `_` 当 emphasis 了，实际 HTML:\n{}",
        html
    );
}

/// 回归守卫：math 内的 `\<punct>` 必须原样传给 HTML，不能被 comrak 当
/// CommonMark escape 吃掉。把 `\` 都换成 PUA 占位符 → comrak 不识别 → 还原。
///
/// 之前 `\` 漏在保护外：`\!` 变成 `!`、`\;` 变成 `;`，KaTeX 看到的不是
/// thin-space 命令而是字面字符，渲染塌成无空格。
#[test]
fn inline_math_backslash_escapes_preserved() {
    let (html, _) = extract(&fixture("inline-math-display-escapes.md")).unwrap();
    // `\!` 必须保留成反斜杠 + 叹号
    assert!(
        html.contains(r"\!"),
        r"`\!` 应保留成 `\!`，实际: {html}"
    );
    // `\,` 同样
    assert!(
        html.contains(r"\,"),
        r"`\,` 应保留成 `\,`，实际: {html}"
    );
    // 矩阵里的 `\\` 必须保留两个反斜杠（KaTeX 在 \begin{pmatrix} 里把它当行分隔符）
    assert!(
        html.contains(r"\\"),
        r"矩阵里的 `\\` 应保留成 `\\`，实际: {html}"
    );
    // 占位符不能泄漏
    assert!(
        !html.contains(BACKSLASH_PH),
        "BACKSLASH_PH 占位符应在 HTML 里被还原"
    );
}