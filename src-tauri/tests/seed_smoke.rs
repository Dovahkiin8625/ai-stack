//! platform 模块对外 API 的烟雾测试 —— 只覆盖 inline 测试**没有**覆盖的边界：
//! - 桌面端 `materialize_seed` 不 panic 且返回零（验 cfg 分支健在）
//! - `copy_seed_entries` 空入参走零写入/零跳过路径
//! - `migrate_legacy_bundled` 在 dest 全部就位时是真正的 no-op（零写入）
use ai_stack_lib::platform::{
    copy_seed_entries, materialize_seed, migrate_legacy_bundled, SeedEntry,
};
use std::path::Path;

/// 桌面 `materialize_seed` 不引用 `_app` / `_root` —— 永远 `Ok(default)`。
///
/// 难点：`tauri::AppHandle` 在测试环境里造不出来（要起 Tauri runtime），
/// 但桌面分支**从未**读取这俩参数；只校验"调用能跑过且返回 zero stats"。
/// 这里用 NonNull::dangling 占一个类型对齐的合法"指针"，cast 成引用。
/// 它**永不**被解引用 —— unsafe 块只用于满足类型系统，行为是 sound 的。
fn fake_app_handle() -> &'static tauri::AppHandle {
    let dangling = std::ptr::NonNull::<tauri::AppHandle>::dangling().as_ptr();
    // SAFETY: `materialize_seed` 桌面分支签名是 `&AppHandle` 但函数体从未解引用；
    // 该 fake handle 只用于让调用通过类型检查，运行时不会触发任何读操作。
    unsafe { &*dangling }
}

#[test]
fn materialize_seed_no_op_on_desktop() {
    let app = fake_app_handle();
    let root = Path::new("ignored-on-desktop");
    let stats = materialize_seed(app, root).expect("desktop 分支必须返回 Ok");
    assert_eq!(stats.written, 0);
    assert_eq!(stats.skipped, 0);
}

#[test]
fn copy_seed_entries_empty_list_writes_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("knowledge");
    // 即使 root 路径是空的（不存在），空入参也不应触发任何 fs 写：
    // 实现里有 create_dir_all(root)，所以 root 会被建出来 —— 测的是 0 写入。
    let read = |_p: &str| -> anyhow::Result<Vec<u8>> {
        panic!("空入参不该调用 read");
    };
    let stats = copy_seed_entries(&[], &read, &root).expect("ok");
    assert_eq!((stats.written, stats.skipped), (0, 0));
    assert!(root.is_dir());
}

#[test]
fn migrate_legacy_bundled_noop_when_already_present() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("old");
    let dest = dir.path().join("new");
    // src 与 dest 内容一致：先正常迁一次，再迁第二次应零写入。
    std::fs::create_dir_all(src.join("a")).unwrap();
    std::fs::write(src.join("a/x.md"), b"x").unwrap();
    let first = migrate_legacy_bundled(&src, &dest).unwrap();
    assert_eq!(first, 1, "首次迁移应复制 1 个文件");
    let second = migrate_legacy_bundled(&src, &dest).unwrap();
    assert_eq!(second, 0, "全部就位后再跑应零写入 —— 这才是真正的 no-op");
    assert_eq!(std::fs::read(dest.join("a/x.md")).unwrap(), b"x");
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
