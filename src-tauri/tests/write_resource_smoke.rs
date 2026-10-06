//! `write_resource` 命令的端到端 smoke：写文件 + 重新提取 + 更新 DB 行。
//!
//! 测试的核心约束：
//! - 文件内容必须真的改了（最直观的回归点：写文件成功但忘了更新 DB，或反之）
//! - DB 行的 size_bytes / word_count / indexed_at 必须同步刷新，
//!   否则下次扫库会拿旧 size 当差异判断依据，永远不再重 extract
//! - 返回的 html 必须是新 markdown 重新 comrak 出来的，不是旧缓存

use ai_stack_lib::commands::write_resource_markdown;
use ai_stack_lib::db::{Db, ResourceInput};

/// 准备一个能写 markdown 的资源分类 + 一篇 tmp 文件，返回 (resource_id, abs_path)。
///
/// tmp dir 用 `mem::forget` 留着 —— 测试期间路径要有效，drop 后面 read 会失败。
fn fixture(db: &mut Db) -> (i64, std::path::PathBuf) {
    db.upsert_category("x".into(), None, "X".into(), 1).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("note.md");
    std::fs::write(&path, "# Old\n\nOriginal body.\n").unwrap();
    let id = db
        .upsert_resource(ResourceInput {
            category_path: "x".into(),
            rel_path: "note.md".into(),
            r#type: "markdown".into(),
            title: "Note".into(),
            size_bytes: std::fs::metadata(&path).unwrap().len() as i64,
            mtime: 0,
            page_count: None,
            word_count: None,
        })
        .unwrap();
    std::mem::forget(tmp);
    (id, path)
}

#[test]
fn write_resource_markdown_writes_file_and_updates_db_row() {
    let db_tmp = tempfile::tempdir().unwrap();
    let mut db = Db::open(&db_tmp.path().join("t.db")).unwrap();
    db.migrate().unwrap();
    let (id, abs_path) = fixture(&mut db);

    let new_md = "# New Heading\n\n- one\n- two\n- three\n";
    let (new_html, new_words) = write_resource_markdown(&db, id, &abs_path, new_md).unwrap();

    // 文件被新内容覆盖（不能只更新 DB 行就当写完 —— 这是写命令最基本的契约）
    let on_disk = std::fs::read_to_string(&abs_path).unwrap();
    assert_eq!(on_disk, new_md, "文件必须被新 markdown 覆盖");

    // 返回的 html 是新 md 重新 comrak 出来的（不能返旧 html）
    assert!(new_html.contains("New Heading"), "html 应反映新内容: {new_html}");
    assert!(new_words >= 4, "word 数应反映新内容, got {new_words}");

    // DB 行：size_bytes / word_count / indexed_at 都要更新
    let row = db
        .conn
        .query_row(
            "SELECT size_bytes, word_count, indexed_at FROM resources WHERE id = ?1",
            [id],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<i64>>(1)?, r.get::<_, String>(2)?)),
        )
        .unwrap();
    let (size, words, _indexed_at) = row;
    assert_eq!(
        size,
        new_md.len() as i64,
        "size_bytes 必须等于新文件字节数"
    );
    assert_eq!(
        words,
        Some(new_words as i64),
        "word_count 必须等于新 word 数"
    );
}

/// 编辑器写的 markdown 可以是空串 —— 用户清空文件 → 落盘成 0 字节文件。
/// 不应被服务端拒绝（之前某些写命令默认拒绝空字符串）。
#[test]
fn write_resource_markdown_accepts_empty_content() {
    let db_tmp = tempfile::tempdir().unwrap();
    let mut db = Db::open(&db_tmp.path().join("t.db")).unwrap();
    db.migrate().unwrap();
    let (id, abs_path) = fixture(&mut db);

    let (html, words) = write_resource_markdown(&db, id, &abs_path, "").unwrap();

    assert_eq!(std::fs::read_to_string(&abs_path).unwrap(), "");
    assert_eq!(html, "", "空 md 的 html 应为空");
    assert_eq!(words, 0);
    let size: i64 = db
        .conn
        .query_row(
            "SELECT size_bytes FROM resources WHERE id = ?1",
            [id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(size, 0, "空文件 size_bytes 应为 0");
}