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

    db.upsert_category("01-foundations".into(), None, "Foundations".into(), 1).unwrap();
    db.upsert_category(
        "01-foundations/01-mathematics".into(),
        Some("01-foundations".into()),
        "Mathematics".into(),
        1,
    ).unwrap();

    let input = ResourceInput {
        category_path: "01-foundations/01-mathematics".into(),
        rel_path: "linear-algebra/notes.md".into(),
        r#type: "markdown".into(),
        title: "Linear Algebra Notes".into(),
        size_bytes: 1024,
        mtime: 0,
        page_count: None,
        word_count: None,
    };
    let id1 = db.upsert_resource(input.clone()).unwrap();
    let id2 = db.upsert_resource(input.clone()).unwrap();
    assert_eq!(id1, id2, "upsert on identical key should return same id");

    // 修改 size_bytes 后 id 不变（hash 字段已删除，用 size 代替做差异验证）
    let mut changed = input;
    changed.size_bytes = 2048;
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
        size_bytes: 1,
        mtime: 0,
        page_count: None, word_count: None,
    }).unwrap();
    db.upsert_resource(ResourceInput {
        category_path: "02-deep-learning".into(),
        rel_path: "b.md".into(),
        r#type: "markdown".into(),
        title: "B".into(),
        size_bytes: 1,
        mtime: 0,
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

    db.upsert_category("x".into(), None, "X".into(), 1).unwrap();

    for rel in ["a.md", "b.md", "c.md"] {
        db.upsert_resource(ResourceInput {
            category_path: "x".into(),
            rel_path: rel.into(),
            r#type: "markdown".into(),
            title: rel.into(),
            size_bytes: 1,
            mtime: 0,
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

/// 准备一个能装笔记的资源分类，避免每个 case 重复 setup
fn fixture_with_resource(db: &mut Db) -> i64 {
    db.upsert_category("x".into(), None, "X".into(), 1).unwrap();
    db.upsert_resource(ResourceInput {
        category_path: "x".into(),
        rel_path: "a.md".into(),
        r#type: "markdown".into(),
        title: "A".into(),
        size_bytes: 1,
        mtime: 0,
        page_count: None,
        word_count: None,
    }).unwrap()
}

#[test]
fn update_note_without_source_keeps_existing_source() {
    // 用户编辑时（前端不传 source）：保留原 source 不变
    let tmp = tempfile::tempdir().unwrap();
    let mut db = Db::open(&tmp.path().join("t.db")).unwrap();
    db.migrate().unwrap();
    let rid = fixture_with_resource(&mut db);
    let ai = db.insert_note(rid, "AI 内容", None, 0, "ai", None).unwrap();
    assert_eq!(ai.source, "ai");

    // 普通编辑，source 传 None
    let after = db.update_note(ai.id, "编辑后", None, 0, None).unwrap();
    assert_eq!(after.source, "ai", "未传 source 时必须保留原值");
    assert_eq!(after.content, "编辑后");
}

#[test]
fn update_note_with_source_can_demote_ai_to_user() {
    // 用户编辑 AI 笔记后降级为 user
    let tmp = tempfile::tempdir().unwrap();
    let mut db = Db::open(&tmp.path().join("t.db")).unwrap();
    db.migrate().unwrap();
    let rid = fixture_with_resource(&mut db);
    let ai = db.insert_note(rid, "AI 内容", None, 0, "ai", None).unwrap();

    let after = db.update_note(ai.id, "用户改写", None, 0, Some("user")).unwrap();
    assert_eq!(after.source, "user");
    assert_eq!(after.content, "用户改写");
}

#[test]
fn list_notes_reflects_source_after_update() {
    // 降级后 list_notes 拿到的也是 user
    let tmp = tempfile::tempdir().unwrap();
    let mut db = Db::open(&tmp.path().join("t.db")).unwrap();
    db.migrate().unwrap();
    let rid = fixture_with_resource(&mut db);
    let ai = db.insert_note(rid, "AI 内容", None, 0, "ai", None).unwrap();
    db.update_note(ai.id, "改写", None, 0, Some("user")).unwrap();
    let listed = db.list_notes(rid).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].source, "user");
    assert_eq!(listed[0].content, "改写");
}

#[test]
fn migrate_adds_prompt_column_to_notes() {
    // Phase 5 增量：notes 表新增 prompt 列（用户问 AI 的问题）。
    // 老库（无 prompt 列）必须被自动 ALTER，老笔记的 prompt 全为 NULL。
    let tmp = tempfile::tempdir().unwrap();
    let mut db = Db::open(&tmp.path().join("t.db")).unwrap();
    db.migrate().unwrap();
    let rid = fixture_with_resource(&mut db);

    let has: i64 = db
        .conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('notes') WHERE name='prompt'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(has, 1, "notes 表必须有 prompt 列");

    // 老笔记（未传 prompt）插入后 prompt 为 NULL，向后兼容
    let n = db.insert_note(rid, "AI 讲解内容", None, 0, "ai", None).unwrap();
    assert_eq!(n.prompt, None);
}

#[test]
fn insert_note_with_prompt_stores_prompt() {
    // 新建 AI 问答笔记时附带 prompt（用户问题），必须被持久化并回读
    let tmp = tempfile::tempdir().unwrap();
    let mut db = Db::open(&tmp.path().join("t.db")).unwrap();
    db.migrate().unwrap();
    let rid = fixture_with_resource(&mut db);

    let n = db
        .insert_note(
            rid,
            "AI 回答内容",
            Some("选区文字"),
            0,
            "ai",
            Some("为什么用 sigmoid?"),
        )
        .unwrap();
    assert_eq!(n.prompt.as_deref(), Some("为什么用 sigmoid?"));
    assert_eq!(n.content, "AI 回答内容");
    assert_eq!(n.anchor_text.as_deref(), Some("选区文字"));

    // list_notes 也能读回 prompt
    let listed = db.list_notes(rid).unwrap();
    assert_eq!(listed[0].prompt.as_deref(), Some("为什么用 sigmoid?"));
}

#[test]
fn update_note_preserves_prompt_by_default() {
    // 用户编辑笔记时不应清掉 prompt（prompt 是元数据不是内容）
    let tmp = tempfile::tempdir().unwrap();
    let mut db = Db::open(&tmp.path().join("t.db")).unwrap();
    db.migrate().unwrap();
    let rid = fixture_with_resource(&mut db);

    let n = db
        .insert_note(
            rid,
            "初始回答",
            None,
            0,
            "ai",
            Some("用户原问题"),
        )
        .unwrap();
    assert_eq!(n.prompt.as_deref(), Some("用户原问题"));

    // 普通编辑（不传 source）→ prompt 必须保留
    let edited = db.update_note(n.id, "用户改写后", None, 0, None).unwrap();
    assert_eq!(edited.prompt.as_deref(), Some("用户原问题"));
    assert_eq!(edited.content, "用户改写后");
}