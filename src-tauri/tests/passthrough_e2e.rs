//! 端到端 smoke：在临时目录里建一个和真实结构相似的目录，
//! 然后跑 scan()，验证 三方资料/ 下的文件归属正确。

use ai_stack_lib::db::Db;
use ai_stack_lib::scanner::{scan, ScanConfig};

#[test]
fn passthrough_e2e_against_realistic_layout() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("kt");

    // 02-deep-learning/cnn/
    let cnn = root.join("02-deep-learning").join("cnn");
    let cnn_third = cnn.join("三方资料");
    std::fs::create_dir_all(&cnn_third).unwrap();
    std::fs::write(cnn.join("_index.md"), "# CNN\n").unwrap();
    std::fs::write(cnn.join("cnn-fundamentals.md"), "# CNN Fundamentals\n\ntext\n").unwrap();
    std::fs::write(cnn_third.join("efficientnet.pdf"), b"%PDF-1.4\n").unwrap();
    std::fs::write(cnn_third.join("resnet.pdf"), b"%PDF-1.4\n").unwrap();

    // 02-deep-learning/transformers/（含 docx/pptx）
    let tx = root.join("02-deep-learning").join("transformers");
    let tx_third = tx.join("三方资料");
    std::fs::create_dir_all(&tx_third).unwrap();
    std::fs::write(tx.join("_index.md"), "# Transformer\n").unwrap();
    std::fs::write(tx_third.join("Attention_Is_All_You_Need.pdf"), b"%PDF-1.4\n").unwrap();
    std::fs::write(tx_third.join("sd1.pptx"), b"PK\x03\x04rest").unwrap();

    let db_path = tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    let summary = scan(&db, &ScanConfig { knowledge_root: root.clone() }).unwrap();
    assert_eq!(summary.errors_count, 0, "errors: {summary:?}");

    // 不应该有 三方资料/ 这个 category
    let cats = db.list_categories().unwrap();
    let cat_paths: Vec<String> = cats.iter().map(|c| c.path.clone()).collect();
    assert!(!cat_paths.iter().any(|p| p.contains("三方资料")), "passthrough leaked into categories: {cat_paths:?}");
    assert!(cat_paths.iter().any(|p| p == "02-deep-learning/cnn"), "missing cnn category");
    assert!(cat_paths.iter().any(|p| p == "02-deep-learning/transformers"), "missing transformers category");

    // cnn 下应有 md + 三方资料/ 下的两个 pdf
    let cnn_res = db.list_resources("02-deep-learning/cnn").unwrap();
    let cnn_rels: Vec<String> = cnn_res.iter().map(|r| r.rel_path.clone()).collect();
    assert!(cnn_rels.contains(&"cnn-fundamentals.md".to_string()), "cnn md missing: {cnn_rels:?}");
    assert!(cnn_rels.contains(&"三方资料/efficientnet.pdf".to_string()), "cnn efficientnet rel_path wrong: {cnn_rels:?}");
    assert!(cnn_rels.contains(&"三方资料/resnet.pdf".to_string()), "cnn resnet rel_path wrong: {cnn_rels:?}");

    // transformers 下应有 Attention_Is_All_You_Need.pdf 和 sd1.pptx，rel_path 含 三方资料/
    let tx_res = db.list_resources("02-deep-learning/transformers").unwrap();
    let tx_rels: Vec<String> = tx_res.iter().map(|r| r.rel_path.clone()).collect();
    assert!(tx_rels.contains(&"三方资料/Attention_Is_All_You_Need.pdf".to_string()), "tx pdf rel_path wrong: {tx_rels:?}");
    assert!(tx_rels.contains(&"三方资料/sd1.pptx".to_string()), "tx pptx rel_path wrong: {tx_rels:?}");

    // 验证文件类型正确归类
    let eff = cnn_res.iter().find(|r| r.rel_path == "三方资料/efficientnet.pdf").unwrap();
    assert_eq!(eff.r#type, "pdf");
    let sd = tx_res.iter().find(|r| r.rel_path == "三方资料/sd1.pptx").unwrap();
    assert_eq!(sd.r#type, "pptx");
}
