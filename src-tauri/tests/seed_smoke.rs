//! platform 模块对外 API 的烟雾测试：覆盖 parse_seed_manifest / copy_seed_entries
//! 在集成测试层面（与 inline 测试同等断言，但走 `pub` 接口）。
use ai_stack_lib::platform::{copy_seed_entries, parse_seed_manifest, SeedEntry, SeedStats};

#[test]
fn parse_seed_manifest_smoke_returns_entries() {
    let json = r#"{"files":[{"path":"a/b.md","size":7}]}"#;
    let e = parse_seed_manifest(json).expect("valid manifest");
    assert_eq!(e.len(), 1);
    assert_eq!(e[0].path, "a/b.md");
    assert_eq!(e[0].size, 7);
}

#[test]
fn copy_seed_entries_smoke_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("knowledge");
    let entries = vec![SeedEntry { path: "cat/seed.md".into(), size: 11 }];
    let read = |p: &str| -> anyhow::Result<Vec<u8>> {
        if p == "cat/seed.md" {
            Ok(b"seed-body!".to_vec())
        } else {
            anyhow::bail!("unexpected asset: {p}")
        }
    };
    let stats: SeedStats = copy_seed_entries(&entries, &read, &root).expect("copy");
    assert_eq!((stats.written, stats.skipped), (1, 0));
    assert_eq!(std::fs::read(root.join("cat/seed.md")).expect("read back"), b"seed-body!");
}