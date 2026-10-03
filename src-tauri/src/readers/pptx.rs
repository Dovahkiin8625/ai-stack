use anyhow::{Context, Result};
use std::io::Read;
use std::path::Path;

/// 数 pptx 里的 slide 数。
///
/// pptx 是 zip 包，slide 顺序由 `ppt/presentation.xml` 里的 `<p:sldIdLst>`
/// 列出（每个 `<p:sldId>` 一个 slide），与 `ppt/_rels/presentation.xml.rels`
/// 里的 Relationship target 对应到具体的 slide XML 文件。
///
/// 这里只统计 sldId 数量 —— 实际渲染交给前端 pptxviewjs，
/// 见 `src/components/library/reader/PptxReader.tsx`。
pub fn slide_count(path: &Path) -> Result<usize> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("unzip {}", path.display()))?;
    let mut pres_xml = String::new();
    zip.by_name("ppt/presentation.xml")
        .context("read ppt/presentation.xml")?
        .read_to_string(&mut pres_xml)?;
    Ok(count_sld_id(&pres_xml))
}

fn count_sld_id(xml: &str) -> usize {
    // 简单字符串计数：`<p:sldId ` 出现次数（带属性的开场标签）。
    // 不严格区分 namespace prefix 是为了兼容带其它 prefix 的 pptx。
    // 用 `matches` + `<p:sldId ` / `<sldId ` 两条 needle 都数，
    // 避免某个文件用 ns0: 前缀时漏算。
    xml.matches("<p:sldId ").count() + xml.matches("<sldId ").count()
}

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
    fn pptx_slide_count_is_at_least_two() {
        // fixture 里 sample.pptx 至少 2 张 slide（与之前 pptx_extract 测试一致）
        let n = slide_count(&fixture("sample.pptx")).unwrap();
        assert!(n >= 2, "got {n}");
    }
}