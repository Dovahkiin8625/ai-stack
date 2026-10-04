use ai_stack_lib::db::Db;
use ai_stack_lib::scanner::{scan, ScanConfig, classify_type};
use std::collections::HashMap;
use std::path::PathBuf;

fn fixtures_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("knowledge-tree");
    p
}

fn isolated_fixture() -> PathBuf {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixtures_root();
    let dst: PathBuf = tmp.path().to_path_buf();
    for entry in walkdir::WalkDir::new(&src) {
        let entry = entry.unwrap();
        let rel = entry.path().strip_prefix(&src).unwrap();
        let target = dst.join(rel);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&target).unwrap();
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
    // Keep tmp alive for the lifetime of the test by leaking it.
    std::mem::forget(tmp);
    dst
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
    let db_tmp = tempfile::tempdir().unwrap();
    let db_path = db_tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();

    let cfg = ScanConfig {
        knowledge_root: isolated_fixture(),
    };
    let summary = scan(&db, &cfg).unwrap();
    assert!(summary.errors_count == 0, "errors: {summary:?}");
    assert!(summary.categories_count >= 3, "got {summary:?}");
    assert!(summary.resources_count >= 2, "got {summary:?}");

    let cats = db.list_categories().unwrap();
    let titles: Vec<String> = cats.iter().map(|c| c.title.clone()).collect();
    // scan 不再读 _index.md 取 H1 —— 标题先走 humanize_dir_name fallback（保留
    // 数字前缀，"- " 切分并首字母大写）。用户 expand 分类时由
    // read_subcategory_index 拿到真 H1。
    assert!(titles.contains(&"01 Mathematics".to_string()), "titles: {titles:?}");
    assert!(titles.contains(&"02 Deep Learning".to_string()));
    assert!(titles.contains(&"Transformers".to_string()));

    let res = db.list_resources("01-foundations/01-mathematics").unwrap();
    let rels: Vec<String> = res.iter().map(|r| r.rel_path.clone()).collect();
    assert!(rels.contains(&"linear-algebra-notes.md".to_string()));
    assert!(rels.contains(&"calculus-notes.md".to_string()));
    // _index.md 不应入 resources
    assert!(!rels.iter().any(|p| p.ends_with("_index.md")));
}

#[test]
fn scan_idempotent_no_changes() {
    let db_tmp = tempfile::tempdir().unwrap();
    let db_path = db_tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    let cfg = ScanConfig {
        knowledge_root: isolated_fixture(),
    };
    let s1 = scan(&db, &cfg).unwrap();
    let s2 = scan(&db, &cfg).unwrap();
    assert_eq!(s1.resources_count, s2.resources_count);
}

#[test]
fn scan_removes_resource_when_file_deleted() {
    let db_tmp = tempfile::tempdir().unwrap();
    let db_path = db_tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    let root = isolated_fixture();
    let cfg = ScanConfig {
        knowledge_root: root.clone(),
    };
    scan(&db, &cfg).unwrap();

    // 删除一个资源文件，再扫描
    let target = root.join("01-foundations").join("01-mathematics").join("calculus-notes.md");
    std::fs::remove_file(&target).unwrap();

    let summary = scan(&db, &cfg).unwrap();
    assert!(summary.resources_count < 4, "got {summary:?}");
    let res = db.list_resources("01-foundations/01-mathematics").unwrap();
    assert!(res.iter().all(|r| r.rel_path != "calculus-notes.md"));
}

#[test]
fn scan_continues_after_corrupt_file() {
    // scan 不读文件内容 —— corrupt 文件无法在 scan 时发现，
    // 会照常入库。corruption 在用户打开（reader 阶段）时才报错。
    // 这里只验证 scan 不会 panic、两个文件都进 DB。
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
    // 旧实现会在 scan 时做 BadMagic 校验，把 corrupt 文件挡在外面。
    // 新实现不读文件，所以 corrupt 文件也会入库 —— 这是有意的（按需读取原则）。
    assert!(
        res.iter().any(|r| r.rel_path == "bad.docx"),
        "scan 不读文件，corrupt 文件也会正常入库"
    );
    assert_eq!(summary.errors_count, 0, "scan 不再读文件，不会因 corrupt 报错");
}

/// `三方资料/` 是物理分组目录，不应在 sidebar 出现为子分类；
/// 它的文件归属上层分类，`rel_path` 包含 `三方资料/` 前缀。
#[test]
fn scan_third_party_passthrough_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("kt");
    let cat_dir = root.join("01-cat").join("01-sub");
    let third_party = cat_dir.join("三方资料");
    std::fs::create_dir_all(&third_party).unwrap();
    std::fs::write(cat_dir.join("_index.md"), "# Sub\n").unwrap();
    std::fs::write(cat_dir.join("article.md"), "# Article\n\ntext\n").unwrap();
    std::fs::write(third_party.join("paper.pdf"), b"%PDF-1.4\n").unwrap();
    // _index.md 放在 三方资料/ 内 —— 不应被当作 category title 来读
    std::fs::write(third_party.join("_index.md"), "# Bad Title\n").unwrap();

    let db_path = tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    scan(&db, &ScanConfig { knowledge_root: root }).unwrap();

    // 三方资料/ 不应该是独立 category
    let cats = db.list_categories().unwrap();
    let paths: Vec<String> = cats.iter().map(|c| c.path.clone()).collect();
    assert!(
        !paths.iter().any(|p| p.contains("三方资料")),
        "三方资料/ should not appear as a category, got: {paths:?}"
    );

    // pdf 应归属 "01-cat/01-sub"，rel_path 带 三方资料/ 前缀
    let res = db.list_resources("01-cat/01-sub").unwrap();
    let rels: Vec<String> = res.iter().map(|r| r.rel_path.clone()).collect();
    assert!(
        rels.iter().any(|p| p == "三方资料/paper.pdf"),
        "paper.pdf rel_path: {rels:?}"
    );
    assert!(
        rels.contains(&"article.md".to_string()),
        "article.md missing: {rels:?}"
    );
    // 三方资料/_index.md 不应入库
    assert!(
        !rels.iter().any(|p| p.ends_with("_index.md")),
        "_index.md should not be a resource: {rels:?}"
    );
}

/// mtime 列必须存在 —— scanner 跳过未变文件依赖它做 (size, mtime) 比对。
#[test]
fn resources_table_has_mtime_column() {
    let db_tmp = tempfile::tempdir().unwrap();
    let db_path = db_tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    // 通过现有接口触发一次最小扫描，保证表里至少一行
    let cfg = ScanConfig {
        knowledge_root: isolated_fixture(),
    };
    scan(&db, &cfg).unwrap();
    // pragma_table_info 应能查到 mtime 列
    let cols: Vec<String> = db
        .conn
        .prepare("SELECT name FROM pragma_table_info('resources')")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    assert!(
        cols.iter().any(|c| c == "mtime"),
        "resources table missing mtime column; got: {cols:?}"
    );
}

/// 扫过一次后，existing_resources_meta() 应返回完整的 (size, mtime) 映射。
/// scanner 据此判断哪些文件可跳过 —— 没有这个 API 整个优化失效。
#[test]
fn existing_resources_meta_returns_full_map_after_scan() {
    let db_tmp = tempfile::tempdir().unwrap();
    let db_path = db_tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    let cfg = ScanConfig {
        knowledge_root: isolated_fixture(),
    };
    let summary = scan(&db, &cfg).unwrap();
    let meta: HashMap<(String, String), (i64, i64)> =
        db.existing_resources_meta().unwrap();
    assert_eq!(
        meta.len(),
        summary.resources_count,
        "meta should cover every scanned resource"
    );
    // 至少一条记录的 mtime > 0（unix epoch seconds）
    let any_positive = meta.values().any(|(_, mtime)| *mtime > 0);
    assert!(any_positive, "at least one mtime should be positive");
}

/// 第二次扫描（文件未变）：所有 (size, mtime) 都未变 → 所有资源被跳过，
/// DB 里的 size / mtime 不变。
/// 这条断言 catch 掉"忘了 (size,mtime) 比对，每次都重写 DB 行"的回归。
#[test]
fn second_scan_preserves_meta_when_files_unchanged() {
    let db_tmp = tempfile::tempdir().unwrap();
    let db_path = db_tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    let cfg = ScanConfig {
        knowledge_root: isolated_fixture(),
    };
    scan(&db, &cfg).unwrap();

    let first: HashMap<(String, String), (i64, i64)> =
        db.existing_resources_meta().unwrap();
    let s2 = scan(&db, &cfg).unwrap();
    assert_eq!(
        s2.resources_count,
        first.len() as usize,
        "second scan should keep same resource count: got {s2:?}"
    );

    let second: HashMap<(String, String), (i64, i64)> =
        db.existing_resources_meta().unwrap();
    assert_eq!(first.len(), second.len());
    for (key, (size1, mtime1)) in &first {
        let (size2, mtime2) = second
            .get(key)
            .unwrap_or_else(|| panic!("missing resource after re-scan: {key:?}"));
        assert_eq!(size1, size2, "size changed for unchanged file {key:?}");
        assert_eq!(mtime1, mtime2, "mtime changed for unchanged file {key:?}");
    }
}

/// 修改文件内容后扫描：该文件的 (size, mtime) 变了 → DB 行被刷新，
/// 其它未变文件跳过（资源数不变，只是这一个的 size/mtime 更新）。
#[test]
fn second_scan_detects_modified_file() {
    let db_tmp = tempfile::tempdir().unwrap();
    let db_path = db_tmp.path().join("test.db");
    let mut db = Db::open(&db_path).unwrap();
    db.migrate().unwrap();
    let cfg = ScanConfig {
        knowledge_root: isolated_fixture(),
    };
    scan(&db, &cfg).unwrap();
    let first: HashMap<(String, String), (i64, i64)> =
        db.existing_resources_meta().unwrap();

    // 改一个 md 文件内容（append 一行），mtime 自动更新，size 也会变
    let target = cfg
        .knowledge_root
        .join("01-foundations")
        .join("01-mathematics")
        .join("linear-algebra-notes.md");
    let mut content = std::fs::read_to_string(&target).unwrap();
    content.push_str("\n\n## appended\n");
    std::fs::write(&target, &content).unwrap();

    // 把 mtime 推后 3 秒以避免 Windows NTFS 2 秒精度被同秒写吞掉
    let new_mtime = std::time::SystemTime::now() + std::time::Duration::from_secs(3);
    filetime_set(&target, new_mtime);

    let s2 = scan(&db, &cfg).unwrap();
    assert_eq!(
        s2.resources_count,
        first.len() as usize,
        "resource count should stay the same: got {s2:?}"
    );

    let second: HashMap<(String, String), (i64, i64)> =
        db.existing_resources_meta().unwrap();
    let key: (String, String) = (
        "01-foundations/01-mathematics".to_string(),
        "linear-algebra-notes.md".to_string(),
    );
    let (old_size, _) = first.get(&key).expect("original record");
    let (new_size, new_mtime) = second.get(&key).expect("record after rescan");
    assert_ne!(
        old_size, new_size,
        "size should change when file content changes"
    );
    assert!(*new_mtime > 0, "mtime should be set");
}

#[cfg(unix)]
fn filetime_set(path: &std::path::Path, t: std::time::SystemTime) {
    let ft = filetime::FileTime::from_system_time(t);
    filetime::set_file_mtime(path, ft).unwrap();
}

#[cfg(not(unix))]
fn filetime_set(path: &std::path::Path, t: std::time::SystemTime) {
    // Windows: 直接 std::fs::write 已经更新了 mtime。这里再 Sleep 一下保证跨平台语义。
    // （测试只针对修改是否被检测到，mtime 精度靠 +3s 兜底。）
    let _ = (path, t);
}