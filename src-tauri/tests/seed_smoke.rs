//! platform 模块对外 API 的烟雾测试 —— 只覆盖 inline 测试**没有**覆盖的边界：
//! - `copy_seed_entries` 空入参走零写入/零跳过路径
//! - `migrate_legacy_bundled` 在标记已写 / src 缺失时是真正的 no-op（零写入）
//! - `migrate_legacy_bundled` 标记守门：用户编辑后第二次跑不覆盖
//!
//! 桌面 `materialize_seed` **不**在此文件测试 —— 它的函数体只有
//! `Ok(SeedStats::default())`，没有可观察行为可断言；现有 cargo build / test
//! 走桌面 cfg 分支通过本身已证明它能编译并返回 Ok。
use ai_stack_lib::platform::{copy_seed_entries, migrate_legacy_bundled, SeedEntry};

#[test]
fn copy_seed_entries_empty_list_writes_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("knowledge");
    // 即便 root 不存在，空入参也不应触发任何 fs 写：
    // 实现里有 create_dir_all(root)，所以 root 会被建出来 —— 测的是 0 写入。
    let read = |_p: &str| -> anyhow::Result<Vec<u8>> {
        panic!("空入参不该调用 read");
    };
    let stats = copy_seed_entries(&[], &read, &root).expect("ok");
    assert_eq!((stats.written, stats.skipped), (0, 0));
    assert!(root.is_dir());
}

#[test]
fn copy_seed_entries_skips_when_size_matches_with_zero_bytes() {
    // 边界：size=0 的 entry。空文件 size==0 也算"已就位"。
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("knowledge");
    std::fs::create_dir_all(root.join("empty-cat")).unwrap();
    std::fs::write(root.join("empty-cat/empty.md"), b"").unwrap();
    let read = |_p: &str| -> anyhow::Result<Vec<u8>> {
        panic!("大小已匹配的 entry 不该调 read");
    };
    let entries = vec![SeedEntry {
        path: "empty-cat/empty.md".into(),
        size: 0,
    }];
    let stats = copy_seed_entries(&entries, &read, &root).expect("ok");
    assert_eq!((stats.written, stats.skipped), (0, 1), "空文件 size=0 也算命中");
}

#[test]
fn migrate_legacy_bundled_noop_when_marker_present() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("old");
    let dest = dir.path().join("new");
    let marker = dir.path().join(".legacy-migrated");
    std::fs::create_dir_all(src.join("a")).unwrap();
    std::fs::write(src.join("a/x.md"), b"x").unwrap();
    // 标记已存在（场景：之前已迁过）。即使 src 与 dest 此刻没有任何内容，函数必须
    // 完全短路，不读 src、不碰 dest、不改 marker。
    std::fs::write(&marker, b"").unwrap();

    let n = migrate_legacy_bundled(&src, &dest, &marker).unwrap();
    assert_eq!(n, 0, "marker 存在 → 立即 0 退出");
    assert!(!dest.exists(), "marker 已写时不应再去 walk / 创建 dest");
}

#[test]
fn migrate_legacy_bundled_short_circuits_preserve_user_edits() {
    // 升级用户的最关键路径：迁移跑过一次 → 用户编辑 dest → 再次启动。
    // 新版契约是"标记闸门 + 单次内可恢复"，所以第二次跑必须 0、dest 原样。
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("old");
    let dest = dir.path().join("new");
    let marker = dir.path().join(".legacy-migrated");
    std::fs::create_dir_all(src.join("a")).unwrap();
    std::fs::write(src.join("a/x.md"), b"original-content").unwrap();

    // 第一次：迁移完成 + 写标记
    let first = migrate_legacy_bundled(&src, &dest, &marker).unwrap();
    assert_eq!(first, 1);
    assert!(marker.exists());

    // 用户编辑（长度变了）
    std::fs::write(dest.join("a/x.md"), b"USER EDIT, longer now").unwrap();

    // 模拟第二次启动
    let second = migrate_legacy_bundled(&src, &dest, &marker).unwrap();
    assert_eq!(second, 0, "标记存在 → 整轮短路，绝不去碰 dest");
    assert_eq!(
        std::fs::read(dest.join("a/x.md")).unwrap(),
        b"USER EDIT, longer now",
        "用户编辑必须原样保留 —— 这是数据丢失回归守卫"
    );
}
