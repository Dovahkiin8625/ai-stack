//! 真实 _index.md 解析 smoke：跑 index_md::parse_file()，
//! 验证真实 `_index.md` 文件能正确解析（## sections、### subgroups、entries）。
//! 测试 fixture 是真文件，作者精简 `_index.md` 后需同步更新本测试。
use ai_stack_lib::readers::index_md_parse_file;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    // src-tauri/target/debug/deps/ → 跳到项目根
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // src-tauri -> ai-stack
    p
}

#[test]
fn parses_third_party_section_in_real_index_md() {
    let root = workspace_root();
    let samples = [
        (
            "02-deep-learning/transformers/_index.md",
            vec![
                ("三方资料/Attention_Is_All_You_Need.pdf", "Attention Is All You Need"),
                ("三方资料/sd1.pptx", "Stable Diffusion 讲解稿"),
                ("三方资料/transformer.docx", "Transformer 综述讲义"),
            ],
        ),
        (
            "08-ai-agents/planning-reasoning/_index.md",
            vec![
                ("三方资料/cot.pdf", "Chain-of-Thought 论文"),
                ("三方资料/tree-of-thoughts.pdf", "Tree of Thoughts 论文"),
            ],
        ),
        (
            "04-computer-vision/image-classification/_index.md",
            vec![("三方资料/vit.pdf", "ViT 论文")],
        ),
    ];

    for (rel, expected) in samples {
        let abs = root.join("resources").join("knowledge").join(rel);
        let parsed = index_md_parse_file(&abs)
            .unwrap_or_else(|e| panic!("parse {rel} failed: {e}"));
        // 找 ## 三方资料 section，遍历它的所有 group.entries
        let third_party = parsed
            .sections
            .iter()
            .find(|s| s.heading == "三方资料")
            .unwrap_or_else(|| panic!("{rel}: 缺 ## 三方资料 section, got sections: {:?}", parsed.sections));
        let entries: Vec<_> = third_party
            .groups
            .iter()
            .flat_map(|g| g.entries.iter())
            .collect();
        for (want_rel, want_title) in expected {
            let hit = entries
                .iter()
                .find(|e| e.rel_path == want_rel)
                .unwrap_or_else(|| panic!("{rel}: 缺 {want_rel}, got: {entries:?}"));
            assert_eq!(
                hit.title, want_title,
                "{rel}: title mismatch for {want_rel}"
            );
            assert!(
                hit.description.is_some(),
                "{rel}: {want_rel} 应有 description"
            );
        }
    }
}

/// 验证 _index.md 的结构化解析能正确处理 H1 + preamble + ## 文章目录（### 子分组）+ ## 三方资料。
/// 注意：parser 不写死任何 section 名——本测试针对当前 `cnn/_index.md` 的实际内容。
/// 若作者精简/扩充 `_index.md`，应同步更新本测试（这是 fixture 测试的固有权衡）。
#[test]
fn cnn_index_md_has_full_structure() {
    let root = workspace_root();
    let abs = root
        .join("resources")
        .join("knowledge")
        .join("02-deep-learning/cnn/_index.md");
    let p = index_md_parse_file(&abs).expect("parse cnn/_index.md");

    // H1 + preamble（含 blockquote + 段落）
    assert_eq!(p.title.as_deref(), Some("卷积神经网络（CNN）"));
    let preamble = p.preamble.as_deref().unwrap_or("");
    assert!(preamble.contains("> 分类"), "preamble: {preamble:?}");
    assert!(preamble.contains("> 路径"), "preamble: {preamble:?}");
    assert!(preamble.contains("本目录覆盖"), "preamble: {preamble:?}");

    // 文章目录：2 个 ### 子分组 + 3 个 entries
    let articles_sec = p
        .sections
        .iter()
        .find(|s| s.heading == "文章目录")
        .expect("missing ## 文章目录");
    assert_eq!(articles_sec.groups.len(), 2);
    assert_eq!(articles_sec.groups[0].subheading.as_deref(), Some("基础与经典"));
    assert_eq!(articles_sec.groups[1].subheading.as_deref(), Some("现代趋势与选型"));
    let total_entries: usize = articles_sec.groups.iter().map(|g| g.entries.len()).sum();
    assert_eq!(total_entries, 3);

    // 三方资料：无 ### 子分组 → 单个 default group
    let tp_sec = p
        .sections
        .iter()
        .find(|s| s.heading == "三方资料")
        .expect("missing ## 三方资料");
    assert_eq!(tp_sec.groups.len(), 1);
    assert!(tp_sec.groups[0].subheading.is_none());
    assert_eq!(tp_sec.groups[0].entries.len(), 2);
}

/// 没有 ### 子分组的 ## section 也应正确归入 default group
/// （如 frameworks/_index.md 只有一个 ## 文章目录，内含两个 entry 但没 ###）。
#[test]
fn section_without_h3_uses_default_group() {
    let root = workspace_root();
    let abs = root
        .join("resources")
        .join("knowledge")
        .join("02-deep-learning/frameworks/_index.md");
    let p = index_md_parse_file(&abs).expect("parse frameworks/_index.md");

    assert_eq!(p.title.as_deref(), Some("深度学习框架"));
    let articles = p
        .sections
        .iter()
        .find(|s| s.heading == "文章目录")
        .expect("missing 文章目录");
    // 单个 default group（无 ###）
    assert_eq!(articles.groups.len(), 1);
    assert!(articles.groups[0].subheading.is_none());
    assert_eq!(articles.groups[0].entries.len(), 2);

    // ## 三方资料 也在
    assert!(p.sections.iter().any(|s| s.heading == "三方资料"));
}

