use ai_stack_lib::db::{Db, ResourceInput};

#[test]
fn migrate_is_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test.db");
    {
        let mut db = Db::open(&path).unwrap();
        db.migrate().unwrap();
    }
    // 第二次打开 + migrate 不应出错
    let mut db = Db::open(&path).unwrap();
    db.migrate().unwrap();
}

#[test]
fn upsert_resource_returns_stable_id_across_updates() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test.db");
    let mut db = Db::open(&path).unwrap();
    db.migrate().unwrap();

    let input = ResourceInput {
        category_path: "01-foundations/01-mathematics".into(),
        rel_path: "linear-algebra/notes.md".into(),
        r#type: "markdown".into(),
        title: "Linear Algebra Notes".into(),
        size_bytes: 1024,
        hash: "abc123".into(),
        page_count: None,
        word_count: None,
    };
    let id1 = db.upsert_resource(input.clone()).unwrap();
    let id2 = db.upsert_resource(input.clone()).unwrap();
    assert_eq!(id1, id2, "upsert on identical key should return same id");

    // 修改 hash 后 id 不变
    let mut changed = input;
    changed.hash = "def456".into();
    let id3 = db.upsert_resource(changed).unwrap();
    assert_eq!(id1, id3);
}

#[test]
fn list_resources_filters_by_category() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test.db");
    let mut db = Db::open(&path).unwrap();
    db.migrate().unwrap();

    for (cat, title) in [
        ("01-foundations", "F"),
        ("02-deep-learning", "D"),
    ] {
        db.upsert_category(cat.into(), None, title.into(), 1).unwrap();
    }
    db.upsert_resource(ResourceInput {
        category_path: "01-foundations".into(),
        rel_path: "a.md".into(),
        r#type: "markdown".into(),
        title: "A".into(),
        size_bytes: 1, hash: "h1".into(),
        page_count: None, word_count: None,
    }).unwrap();
    db.upsert_resource(ResourceInput {
        category_path: "02-deep-learning".into(),
        rel_path: "b.md".into(),
        r#type: "markdown".into(),
        title: "B".into(),
        size_bytes: 1, hash: "h2".into(),
        page_count: None, word_count: None,
    }).unwrap();

    let rows = db.list_resources("01-foundations").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].title, "A");
}

#[test]
fn delete_missing_removes_rows_not_in_set() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test.db");
    let mut db = Db::open(&path).unwrap();
    db.migrate().unwrap();

    for rel in ["a.md", "b.md", "c.md"] {
        db.upsert_resource(ResourceInput {
            category_path: "x".into(),
            rel_path: rel.into(),
            r#type: "markdown".into(),
            title: rel.into(),
            size_bytes: 1, hash: format!("h-{rel}").into(),
            page_count: None, word_count: None,
        }).unwrap();
    }
    let removed = db
        .delete_missing_resources("x", &["a.md".to_string(), "c.md".to_string()])
        .unwrap();
    assert_eq!(removed, 1);
    let remaining: Vec<String> = db.list_resources("x").unwrap()
        .into_iter().map(|r| r.rel_path).collect();
    assert_eq!(remaining, vec!["a.md".to_string(), "c.md".to_string()]);
}