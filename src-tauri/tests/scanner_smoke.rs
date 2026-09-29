use ai_stack_lib::db::Db;
use ai_stack_lib::scanner::{scan, ScanConfig, classify_type};
use std::path::PathBuf;

fn fixtures_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("knowledge-tree");
    p
}

#[test]
fn classify_type_basic() {
    assert_eq!(classify_type("md"), Some("markdown"));
    assert_eq!(classify_type("PDF"), Some("pdf"));
    assert_eq!(classify_type("docx"), Some("docx"));
    assert_eq!(classify_type("pptx"), Some("pptx"));
    assert_eq!(classify_type("txt"), None);
}

#[test]
fn scan_creates_categories_and_resources_from_fixture() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();

    let cfg = ScanConfig {
        knowledge_root: fixtures_root(),
    };
    let summary = scan(&db, &cfg).unwrap();
    assert!(summary.errors_count == 0, "errors: {summary:?}");
    assert!(summary.categories_count >= 3, "got {summary:?}");
    assert!(summary.resources_count >= 2, "got {summary:?}");

    let cats = db.list_categories().unwrap();
    let titles: Vec<String> = cats.iter().map(|c| c.title.clone()).collect();
    assert!(titles.contains(&"数学基础".to_string()), "titles: {titles:?}");
    assert!(titles.contains(&"深度学习".to_string()));
    assert!(titles.contains(&"Transformer".to_string()));

    let res = db.list_resources("01-foundations/01-mathematics").unwrap();
    let rels: Vec<String> = res.iter().map(|r| r.rel_path.clone()).collect();
    assert!(rels.contains(&"linear-algebra-notes.md".to_string()));
    assert!(rels.contains(&"calculus-notes.md".to_string()));
    // _index.md 不应入 resources
    assert!(!rels.iter().any(|p| p.ends_with("_index.md")));
}

#[test]
fn scan_idempotent_no_changes() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    let cfg = ScanConfig {
        knowledge_root: fixtures_root(),
    };
    let s1 = scan(&db, &cfg).unwrap();
    let s2 = scan(&db, &cfg).unwrap();
    assert_eq!(s1.resources_count, s2.resources_count);
}

#[test]
fn scan_removes_resource_when_file_deleted() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    let cfg = ScanConfig {
        knowledge_root: fixtures_root(),
    };
    scan(&db, &cfg).unwrap();

    // 删除一个资源文件，再扫描
    let target = fixtures_root().join("01-foundations").join("01-mathematics").join("calculus-notes.md");
    std::fs::remove_file(&target).unwrap();

    let summary = scan(&db, &cfg).unwrap();
    assert!(summary.resources_count < 4, "got {summary:?}");
    let res = db.list_resources("01-foundations/01-mathematics").unwrap();
    assert!(res.iter().all(|r| r.rel_path != "calculus-notes.md"));
}

#[test]
fn scan_continues_after_corrupt_file() {
    // 在 readers/ 下复制一个 corrupt.docx 进 fixtures 子目录，扫描不应 panic
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("kt");
    let cat_dir = root.join("01-cat").join("01-sub");
    std::fs::create_dir_all(&cat_dir).unwrap();
    std::fs::write(cat_dir.join("_index.md"), "# Sub\n").unwrap();
    std::fs::write(cat_dir.join("good.md"), "# Good\n\ntext\n").unwrap();
    std::fs::write(cat_dir.join("bad.docx"), b"not a real docx").unwrap();

    let db_path = tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    let summary = scan(&db, &ScanConfig { knowledge_root: root }).unwrap();
    let res = db.list_resources("01-cat/01-sub").unwrap();
    assert!(res.iter().any(|r| r.rel_path == "good.md"), "good.md missing");
    assert!(!res.iter().any(|r| r.rel_path == "bad.docx"), "corrupt file should not be indexed");
    assert!(summary.errors_count >= 1, "corrupt file should bump errors_count");
}