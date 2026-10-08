# Android 支持与知识库远程分发 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 AI Stack 能在 Android 上运行并打出 APK，同时把 218MB 知识库从安装包改为"远程 manifest + 按需下载到本地缓存"，桌面端共用同一套逻辑。

**Architecture:** `knowledge/` 目录始终是真实文件系统目录（桌面与 Android 都是 `app_data_dir/knowledge`，开发模式例外用 cwd）。远端 `manifest.json` 的条目 upsert 进现有 `resources` 表并带 `present` 标记，因此资源列表在下载前就完整；`read_resource` 发现文件缺失时自动下载再读。Android 端 APK 内只带一份 <1MB 的种子库，首次启动释放到 `knowledge/`。

**Tech Stack:** Tauri 2.1 / Rust 2021（rusqlite 0.32、reqwest 0.12、serde、tokio）/ React 19 + TypeScript + Zustand + TailwindCSS / vitest

**Spec:** `docs/superpowers/specs/2026-10-08-android-support-design.md`

---

## Global Constraints

- Rust edition 2021，`rust-version = "1.78"`。不升级工具链版本。
- **不新增任何 npm 依赖。** 不新增任何 Rust crate —— `reqwest`（rustls）已在 `Cargo.toml`，`sha256` 由 Node 脚本计算，Rust 侧首版只用 `size` 校验。
- Tauri 命令参数一律 camelCase（前端 `invoke` 侧），Rust 侧用 `#[serde(rename_all = "camelCase")]`。
- 事件名沿用项目现有的下划线风格：`sync_progress`、`seed_progress`（不要用 `://`）。
- DB schema 变更一律走 `migrate()` 里 `pragma_table_info` 探测 + `ALTER TABLE` 的既有写法；新表走 `CREATE TABLE IF NOT EXISTS`。禁止 `DROP`/重建表。
- `resources` 表被 manifest 管理的行（`remote_hash IS NOT NULL`）**不得**被 scanner 的 `delete_missing_resources` 删除；只有纯本地行（`remote_hash IS NULL`）才允许删。
- `knowledge/` 目录内不得留下任何临时文件或临时目录 —— 下载中的 `.tmp` 一律放在 `knowledge/` 的同级 `.tmp/` 目录，否则 scanner 会把临时目录扫成一个新的分类。
- Rust 测试命令：`cd src-tauri && cargo test`。前端测试命令：`npm test`。类型检查：`npm run typecheck`。
- 现有测试全部保持通过（`cargo test`、`npm test`、`npm run typecheck`），每个 task 结束前都要跑一次。

## Review Focus

以下五类输入是 spec 隐含、但任何任务的单元测试都不会自然覆盖的。它们的测试已分别钉在对应 task 里。

1. **无网络时点开未缓存的文章** → 应得到明确的中文失败提示（"下载失败，请检查网络"），而不是 "resource not found" 或白屏。→ Task 4
2. **恶意/损坏的 manifest 路径**（绝对路径、`..`、反斜杠） → 必须拒绝并跳过该条，绝不能写到 `knowledge_root` 之外。→ Task 1
3. **下载中途断网** → 不得留下半个文件；下次点开自动重下；`.tmp` 不被 scanner 扫成分类。→ Task 3、Task 2
4. **用户在本地编辑保存过 markdown（`write_resource`）后再同步** → 本地编辑内容不被 manifest 覆盖（`present=1` 的行不重下）。→ Task 2
5. **手机窄屏打开文章后笔记面板** → 面板必须能关闭，不能永久遮住正文。→ Task 11

---

## 文件结构

| 文件 | 职责 | 任务 |
|---|---|---|
| `src-tauri/src/sync.rs` **新建** | manifest 类型、解析、路径规范化、Fetcher trait、ensure_local / download_all / sync_status 纯逻辑 | 1, 3 |
| `src-tauri/src/platform.rs` **新建** | knowledge_root 解析、种子库释放（Android）、旧 bundle 目录迁移（桌面） | 5 |
| `src-tauri/src/db.rs` | migration v2、manifest upsert、present 标记、index_files 表、sync_status 查询 | 2 |
| `src-tauri/src/scanner.rs` | 把 `humanize_dir_name` / `parse_sort_order` / `parent_of` / `classify_type` 改为 `pub` 供 sync 复用 | 2 |
| `src-tauri/src/commands.rs` | AppState 改造、`read_resource` 自动下载、新命令、lib.rs handler 注册 | 4 |
| `src-tauri/src/lib.rs` | window-state 仅 desktop 注册 | 13 |
| `src-tauri/Cargo.toml` | window-state 移到 desktop-only target 段 | 13 |
| `src-tauri/tauri.conf.json` | 删除 `bundle.resources` | 6 |
| `src-tauri/capabilities/default.json` | 补 `$RESOURCE` fs scope | 13 |
| `scripts/gen-remote-manifest.mjs` **新建** | 扫描全量知识库，输出 manifest.json + sha256 + 可托管目录 | 6 |
| `scripts/serve-dist.mjs` **新建** | 本地静态服务器，验证远端链路 | 6 |
| `src/lib/sync.ts` **新建** | 同步相关 invoke 封装 + 进度事件订阅 | 7 |
| `src/stores/ui.ts` **新建** | 移动端抽屉开关（zustand） | 10 |
| `src/stores/library.ts` | syncStatus / syncPhase / downloads 状态与动作 | 7 |
| `src/components/sync/SyncStatusBar.tsx` **新建** | 顶部同步状态条 + 下载全部 / 暂停 | 9 |
| `src/components/settings/SyncForm.tsx` **新建** | syncBaseUrl 配置表单 | 8 |

---

## Phase A — 后端同步核心

### Task 1: manifest 类型、解析与路径规范化

**Files:**
- Create: `src-tauri/src/sync.rs`
- Modify: `src-tauri/src/lib.rs`（加 `pub mod sync;`）

**Interfaces:**
- Produces:
  - `pub struct Manifest { pub version: String, pub files: Vec<ManifestFile>, pub indexes: Vec<ManifestFile> }`
  - `pub struct ManifestFile { pub path: String, pub size: u64, pub sha256: Option<String> }`
  - `pub fn parse_manifest(json: &str) -> Result<Manifest>`
  - `pub fn normalize_rel_path(raw: &str) -> Result<String>`
  - `pub fn split_category(path: &str) -> Option<(String, String)>` —— `("a/b/c.md") → ("a/b", "c.md")`

- [ ] **Step 1: 写失败测试**

在 `src-tauri/src/sync.rs` 末尾先写测试模块：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_rel_path_converts_backslash() {
        assert_eq!(normalize_rel_path("a\\b\\c.md").unwrap(), "a/b/c.md");
    }

    #[test]
    fn normalize_rel_path_strips_leading_slash_dot() {
        assert_eq!(normalize_rel_path("./a/b.md").unwrap(), "a/b.md");
    }

    #[test]
    fn normalize_rel_path_rejects_parent_traversal() {
        assert!(normalize_rel_path("../../etc/passwd").is_err());
        assert!(normalize_rel_path("a/../../b.md").is_err());
    }

    #[test]
    fn normalize_rel_path_rejects_absolute_and_empty() {
        assert!(normalize_rel_path("/etc/passwd").is_err());
        assert!(normalize_rel_path("C:/win.md").is_err());
        assert!(normalize_rel_path("").is_err());
    }

    #[test]
    fn split_category_splits_at_last_slash() {
        assert_eq!(
            split_category("01-基础/线性回归.md"),
            Some(("01-基础".into(), "线性回归.md".into()))
        );
        // 根目录下的文件没有分类 —— scanner 同样拒绝根目录资源
        assert_eq!(split_category("root.md"), None);
    }

    #[test]
    fn parse_manifest_reads_files_and_indexes() {
        let json = r#"{
            "version": "2026-10-08",
            "files": [{"path": "01-基础/a.md", "size": 12, "sha256": "abc"}],
            "indexes": [{"path": "01-基础/_index.md", "size": 30}]
        }"#;
        let m = parse_manifest(json).unwrap();
        assert_eq!(m.files.len(), 1);
        assert_eq!(m.files[0].size, 12);
        assert_eq!(m.files[0].sha256.as_deref(), Some("abc"));
        assert_eq!(m.indexes.len(), 1);
        // sha256 缺省时为 None，不报错（首版只用 size 校验）
        assert_eq!(m.indexes[0].sha256, None);
    }

    #[test]
    fn parse_manifest_skips_entries_with_bad_paths_instead_of_failing() {
        let json = r#"{
            "version": "v",
            "files": [
                {"path": "01-基础/good.md", "size": 3},
                {"path": "../../evil.md", "size": 3},
                {"path": "01-基础/no-slash.md", "size": 3}
            ]
        }"#;
        let m = parse_manifest(json).unwrap();
        // 只有一条既路径合法、又有分类的条目被保留
        assert_eq!(m.files.len(), 1);
        assert_eq!(m.files[0].path, "01-基础/good.md");
    }

    #[test]
    fn parse_manifest_rejects_malformed_json() {
        assert!(parse_manifest("{ not json").is_err());
    }
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd src-tauri && cargo test sync:: 2>&1 | tail -20`
Expected: 编译失败，`cannot find function normalize_rel_path`

- [ ] **Step 3: 写最小实现**

`src-tauri/src/sync.rs` 开头：

```rust
//! 知识库远程同步：manifest 解析、路径规范化、下载与落盘。
//!
//! 设计要点（见 docs/superpowers/specs/2026-10-08-android-support-design.md）：
//! - manifest 里的 `path` 相对 `knowledge/`，必须先过 `normalize_rel_path`
//!   再拼绝对路径，杜绝 `..` 目录穿越。
//! - 单个条目的路径/分类不合法时**跳过**而不是让整份 manifest 失败 —— 一条坏数据
//!   不该让 380 篇文档全部同步不上。
use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub files: Vec<ManifestFile>,
    #[serde(default)]
    pub indexes: Vec<ManifestFile>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ManifestFile {
    pub path: String,
    pub size: u64,
    #[serde(default)]
    pub sha256: Option<String>,
}

pub fn parse_manifest(json: &str) -> Result<Manifest> {
    let raw: Manifest = serde_json::from_str(json)?;
    Ok(Manifest {
        version: raw.version,
        files: sanitize(raw.files),
        indexes: sanitize(raw.indexes),
    })
}

/// 过滤掉路径不合法 / 没有分类的条目，并把反斜杠归一成 `/`。
fn sanitize(mut entries: Vec<ManifestFile>) -> Vec<ManifestFile> {
    entries.retain_mut(|e| match normalize_rel_path(&e.path) {
        Ok(normalized) if split_category(&normalized).is_some() => {
            e.path = normalized;
            true
        }
        _ => false,
    });
    entries
}

/// 归一化相对路径：反斜杠 → `/`，去掉前导 `./` 与 `/`。
/// 拒绝绝对路径、Windows 盘符与任何 `..` 段。
pub fn normalize_rel_path(raw: &str) -> Result<String> {
    let mut s = raw.trim().replace('\\', "/");
    while let Some(rest) = s.strip_prefix("./") {
        s = rest.to_string();
    }
    let s = s.trim_start_matches('/').to_string();
    if s.is_empty() {
        anyhow::bail!("manifest path is empty");
    }
    if s.contains(':') {
        anyhow::bail!("manifest path must not contain ':': {raw}");
    }
    if s.split('/').any(|seg| seg == ".." || seg == ".") {
        anyhow::bail!("manifest path must not traverse: {raw}");
    }
    Ok(s)
}

/// 拆成 (category_path, rel_path)。根目录下的文件返回 None。
pub fn split_category(path: &str) -> Option<(String, String)> {
    let idx = path.rfind('/')?;
    let category = &path[..idx];
    let rel = &path[idx + 1..];
    if category.is_empty() || rel.is_empty() {
        return None;
    }
    Some((category.to_string(), rel.to_string()))
}
```

在 `src-tauri/src/lib.rs` 顶部加 `pub mod sync;`（放在 `pub mod db;` 一行之前）。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd src-tauri && cargo test sync:: 2>&1 | tail -20`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/sync.rs src-tauri/src/lib.rs
git commit -m "feat(sync): manifest 类型、解析与路径规范化"
```

---

### Task 2: DB migration v2 + manifest upsert

**Files:**
- Modify: `src-tauri/src/db.rs`
- Modify: `src-tauri/src/scanner.rs`（派生函数改 `pub`）
- Test: `src-tauri/src/db.rs`（inline `#[cfg(test)] mod tests`）

**Interfaces:**
- Consumes: `crate::sync::{Manifest, ManifestFile, split_category}`
- Produces:
  - `ResourceRow` 新增 `pub present: bool`、`pub remote_hash: Option<String>`
  - `pub struct SyncStatus { pub total: i64, pub present: i64, pub total_bytes: i64, pub cached_bytes: i64 }`
  - `pub fn upsert_manifest(&self, files: &[ManifestFile]) -> Result<()>`
  - `pub fn drop_resources_absent_from_manifest(&self, keep: &[(String, String)]) -> Result<usize>`
  - `pub fn mark_resource_present(&self, category_path: &str, rel_path: &str) -> Result<()>`
  - `pub fn upsert_index_files(&self, files: &[ManifestFile]) -> Result<()>`
  - `pub fn index_file(&self, path: &str) -> Result<Option<(u64, bool)>>`
  - `pub fn mark_index_present(&self, path: &str) -> Result<()>`
  - `pub fn sync_status(&self) -> Result<SyncStatus>`
  - `pub fn get_config(&self, key: &str) -> Result<Option<String>>` / `pub fn set_config(&self, key: &str, value: &str) -> Result<()>`
  - `pub fn ensure_category_chain(&self, category_path: &str) -> Result<()>`
  - scanner: `pub fn humanize_dir_name`, `pub fn parse_sort_order`, `pub fn parent_of`

- [ ] **Step 1: 写失败测试**

在 `src-tauri/src/db.rs` 末尾追加：

```rust
#[cfg(test)]
mod manifest_tests {
    use super::*;
    use crate::sync::ManifestFile;

    fn opened() -> (tempfile::TempDir, Db) {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open(&tmp.path().join("t.db")).unwrap();
        db.migrate().unwrap();
        (tmp, db)
    }

    fn file(path: &str, size: u64) -> ManifestFile {
        ManifestFile { path: path.into(), size, sha256: None }
    }

    #[test]
    fn migrate_adds_present_and_remote_hash_columns() {
        let (_t, db) = opened();
        let cols: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('resources') WHERE name IN ('present','remote_hash')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cols, 2);
        // 新表也在
        db.conn
            .query_row("SELECT COUNT(*) FROM index_files", [], |r| r.get::<_, i64>(0))
            .unwrap();
    }

    #[test]
    fn upsert_manifest_creates_placeholder_rows_not_present() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("01-基础/回归.md", 500)]).unwrap();
        let (present, size, hash): (i64, i64, Option<String>) = db
            .conn
            .query_row(
                "SELECT present, size_bytes, remote_hash FROM resources WHERE rel_path = '回归.md'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(present, 0);
        assert_eq!(size, 500);
        assert!(hash.is_none());
        // 分类被自动补齐（FK 要求）
        let n: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM categories WHERE path='01-基础'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn upsert_manifest_is_idempotent() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("01-基础/回归.md", 500)]).unwrap();
        db.upsert_manifest(&[file("01-基础/回归.md", 500)]).unwrap();
        let n: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM resources", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn scan_local_file_flips_present_to_one() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("01-基础/回归.md", 500)]).unwrap();
        db.upsert_resource(ResourceInput {
            category_path: "01-基础".into(),
            rel_path: "回归.md".into(),
            r#type: "markdown".into(),
            title: "回归".into(),
            size_bytes: 500,
            mtime: 0,
            page_count: None,
            word_count: None,
        })
        .unwrap();
        let present: i64 = db
            .conn
            .query_row("SELECT present FROM resources WHERE rel_path='回归.md'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(present, 1);
        // remote_hash 保留 —— 决定这条不再被 scan 的 delete_missing_resources 误删
        let still_managed: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM resources WHERE remote_hash IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(still_managed, 1);
    }

    #[test]
    fn delete_missing_resources_keeps_manifest_managed_rows() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("01-基础/远端.md", 10), file("01-基础/本地.md", 10)]).unwrap();
        // 本地.md 被 scanner 扫到（present=1），远端.md 只有 manifest 占位（present=0）
        db.upsert_resource(ResourceInput {
            category_path: "01-基础".into(),
            rel_path: "本地.md".into(),
            r#type: "markdown".into(),
            title: "本地".into(),
            size_bytes: 10,
            mtime: 0,
            page_count: None,
            word_count: None,
        })
        .unwrap();
        // scanner 只 keep 了 本地.md；远端.md 虽没扫到也必须留下
        db.delete_missing_resources("01-基础", &["本地.md".to_string()]).unwrap();
        let n: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM resources WHERE rel_path='远端.md'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn drop_resources_absent_from_manifest_removes_stale() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("01-基础/旧.md", 10), file("01-基础/新.md", 10)]).unwrap();
        db.drop_resources_absent_from_manifest(&[("01-基础".into(), "新.md".into())])
            .unwrap();
        let n: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM resources WHERE rel_path='旧.md'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn index_files_roundtrip() {
        let (_t, db) = opened();
        db.upsert_index_files(&[file("01-基础/_index.md", 42)]).unwrap();
        assert_eq!(db.index_file("01-基础/_index.md").unwrap(), Some((42, false)));
        db.mark_index_present("01-基础/_index.md").unwrap();
        assert_eq!(db.index_file("01-基础/_index.md").unwrap(), Some((42, true)));
        assert_eq!(db.index_file("不存在/_index.md").unwrap(), None);
    }

    #[test]
    fn sync_status_counts_present_and_bytes() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("a/x.md", 100), file("a/y.md", 200)]).unwrap();
        db.mark_resource_present("a", "x.md").unwrap();
        let st = db.sync_status().unwrap();
        assert_eq!(st.total, 2);
        assert_eq!(st.present, 1);
        assert_eq!(st.total_bytes, 300);
        assert_eq!(st.cached_bytes, 100);
    }

    #[test]
    fn config_roundtrip() {
        let (_t, db) = opened();
        assert_eq!(db.get_config("sync_base_url").unwrap(), None);
        db.set_config("sync_base_url", "https://example.com/kb").unwrap();
        assert_eq!(
            db.get_config("sync_base_url").unwrap().as_deref(),
            Some("https://example.com/kb")
        );
    }
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd src-tauri && cargo test manifest_tests 2>&1 | tail -20`
Expected: 编译失败，`no function named upsert_manifest`

- [ ] **Step 3: 实施 DB 改动**

`db.rs` 改动要点，逐条落地：

**(a) `ResourceRow` 加两个字段**：

```rust
pub struct ResourceRow {
    pub id: i64,
    pub category_path: String,
    pub rel_path: String,
    pub r#type: String,
    pub title: String,
    pub size_bytes: i64,
    pub indexed_at: String,
    pub page_count: Option<i64>,
    pub word_count: Option<i64>,
    /// 文件是否已在本地 knowledge/ 目录里。manifest 有但没下过的条目为 false。
    pub present: bool,
    /// 非 NULL = 该行由 manifest 管理，scanner 的 delete_missing_resources 不得删它。
    pub remote_hash: Option<String>,
}
```

**(b) `migrate()` 末尾（`INSERT OR IGNORE schema_version (1, ...)` 之前）追加**，沿用既有 pragma 探测写法：

```rust
        let has_remote_hash: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('resources') WHERE name='remote_hash'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if has_remote_hash == 0 {
            tx.execute("ALTER TABLE resources ADD COLUMN remote_hash TEXT", [])?;
        }
        let has_present: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('resources') WHERE name='present'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if has_present == 0 {
            // 默认 1：老库里的每一行都是已经躺在磁盘上的本地文件。
            tx.execute("ALTER TABLE resources ADD COLUMN present INTEGER NOT NULL DEFAULT 1", [])?;
        }
        // `_index.md` 不进 resources（scanner 刻意跳过），单独一张小表记录远端大小，
        // 供 read_subcategory_index 在文件缺失时按需下载。
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS index_files (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                present INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS app_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
             );",
        )?;
```

并把版本记录改为同时写 1 和 2：

```rust
        tx.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (2, ?1)",
            params![now],
        )?;
```

（保留原有的 version=1 插入语句。）

**(c) `upsert_resource` 写入 `present = 1`**：INSERT 列表加 `present`，值 `1`；`ON CONFLICT DO UPDATE SET` 末尾加 `present = 1,`。**不要动 `remote_hash`** —— 同步进来的行即使本地有文件也仍然受 manifest 管理。

**(d) `delete_missing_resources` 改为只删非 manifest 行**：把 `DELETE FROM resources WHERE category_path = ?1 AND rel_path = ?2` 改成

```sql
DELETE FROM resources WHERE category_path = ?1 AND rel_path = ?2 AND remote_hash IS NULL
```

**(e) `list_resources` 读出新列**：

```rust
    pub fn list_resources(&self, category_path: &str) -> Result<Vec<ResourceRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, category_path, rel_path, type, title,
                    size_bytes, indexed_at, page_count, word_count, present, remote_hash
             FROM resources WHERE category_path = ?1
             ORDER BY rel_path",
        )?;
        // ... row.get(9)? => present != 0, row.get(10)? => remote_hash
    }
```

**(f) 新增方法**（全部挂在 `impl Db` 内）：

```rust
    /// manifest 条目 upsert 为占位行（present=0）。分类链按顶层到叶子顺序补齐，
    /// 满足 resources.category_path → categories.path 的外键。
    pub fn upsert_manifest(&self, files: &[crate::sync::ManifestFile]) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        for f in files {
            let Some((category_path, rel_path)) = crate::sync::split_category(&f.path) else {
                continue;
            };
            self.ensure_category_chain(&category_path)?;
            let title = title_from_file_name(&rel_path);
            let Some(r#type) = crate::scanner::classify_type(
                rel_path.rsplit('.').next().unwrap_or(""),
            ) else {
                continue; // 未知扩展名不进资源列表
            };
            self.conn.execute(
                "INSERT INTO resources (category_path, rel_path, type, title, size_bytes,
                                        mtime, indexed_at, page_count, word_count,
                                        remote_hash, present)
                 VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6, NULL, NULL, NULL, 0)
                 ON CONFLICT(category_path, rel_path) DO UPDATE SET
                    size_bytes = excluded.size_bytes,
                    remote_hash = excluded.remote_hash,
                    indexed_at = excluded.indexed_at",
                params![category_path, rel_path, r#type, title, f.size as i64, now, f.sha256],
            )?;
        }
        Ok(())
    }

    pub fn ensure_category_chain(&self, category_path: &str) -> Result<()> {
        let segs: Vec<&str> = category_path.split('/').collect();
        for i in 1..segs.len() {
            let path = segs[..i].join("/");
            let parent = if i == 1 { None } else { Some(segs[..i - 1].join("/")) };
            let title = crate::scanner::humanize_dir_name(&path);
            let order = crate::scanner::parse_sort_order(&path) as i64;
            self.conn.execute(
                "INSERT INTO categories (path, parent_path, title, sort_order)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(path) DO UPDATE SET
                    parent_path = excluded.parent_path",
                params![path, parent, title, order],
            )?;
        }
        Ok(())
    }

    pub fn mark_resource_present(&self, category_path: &str, rel_path: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE resources SET present = 1 WHERE category_path = ?1 AND rel_path = ?2",
            params![category_path, rel_path],
        )?;
        Ok(())
    }

    /// manifest 里已经没有的远端条目 → 删除。（纯本地行 remote_hash IS NULL 不动。）
    pub fn drop_resources_absent_from_manifest(&self, keep: &[(String, String)]) -> Result<usize> {
        let mut stmt = self.conn.prepare("SELECT category_path, rel_path FROM resources WHERE remote_hash IS NOT NULL")?;
        let all: Vec<(String, String)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut removed = 0;
        for (cat, rel) in all {
            if keep.contains(&(cat.clone(), rel.clone())) {
                continue;
            }
            removed += self.conn.execute(
                "DELETE FROM resources WHERE category_path = ?1 AND rel_path = ?2",
                params![cat, rel],
            )?;
        }
        Ok(removed)
    }

    pub fn upsert_index_files(&self, files: &[crate::sync::ManifestFile]) -> Result<()> {
        for f in files {
            self.conn.execute(
                "INSERT INTO index_files (path, size, present) VALUES (?1, ?2, 0)
                 ON CONFLICT(path) DO UPDATE SET size = excluded.size",
                params![f.path, f.size as i64],
            )?;
        }
        Ok(())
    }

    pub fn index_file(&self, path: &str) -> Result<Option<(u64, bool)>> {
        let mut stmt = self.conn.prepare("SELECT size, present FROM index_files WHERE path = ?1")?;
        let mut rows = stmt.query([path])?;
        match rows.next()? {
            Some(r) => Ok(Some((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? != 0))),
            None => Ok(None),
        }
    }

    pub fn mark_index_present(&self, path: &str) -> Result<()> {
        self.conn.execute("UPDATE index_files SET present = 1 WHERE path = ?1", [path])?;
        Ok(())
    }

    pub fn sync_status(&self) -> Result<SyncStatus> {
        let mut stmt = self.conn.prepare(
            "SELECT COUNT(*),
                    COALESCE(SUM(present), 0),
                    COALESCE(SUM(size_bytes), 0),
                    COALESCE(SUM(CASE WHEN present = 1 THEN size_bytes ELSE 0 END), 0)
             FROM resources WHERE remote_hash IS NOT NULL",
        )?;
        let mut rows = stmt.query([])?;
        let row = rows.next()?.expect("aggregate always yields one row");
        Ok(SyncStatus {
            total: row.get(0)?,
            present: row.get(1)?,
            total_bytes: row.get(2)?,
            cached_bytes: row.get(3)?,
        })
    }

    pub fn get_config(&self, key: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT value FROM app_config WHERE key = ?1")?;
        let mut rows = stmt.query([key])?;
        match rows.next()? {
            Some(r) => Ok(Some(r.get(0)?)),
            None => Ok(None),
        }
    }

    pub fn set_config(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO app_config (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }
```

另加文件顶部的结构体与辅助函数：

```rust
#[derive(Debug, Clone, Default)]
pub struct SyncStatus {
    pub total: i64,
    pub present: i64,
    pub total_bytes: i64,
    pub cached_bytes: i64,
}

/// manifest 条目的标题 = 文件名去扩展名（与 scanner 的 stem 规则一致）。
fn title_from_file_name(rel_path: &str) -> String {
    rel_path
        .rsplit('/')
        .next()
        .unwrap_or(rel_path)
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(rel_path)
        .to_string()
}
```

- [ ] **Step 4: scanner 派生函数改 pub**

`src-tauri/src/scanner.rs`：把 `fn humanize_dir_name`、`fn parse_sort_order`、`fn parent_of` 三处的 `fn` 改为 `pub fn`（`classify_type` 已经是 pub）。不改任何逻辑。

- [ ] **Step 5: 运行测试确认通过**

Run: `cd src-tauri && cargo test 2>&1 | tail -30`
Expected: 全部 passed（含原有测试）

若 `upsert_manifest` 的 `ON CONFLICT` 触发 SQLite "ON CONFLICT clause does not match" —— 检查 `resources` 是否已有 `UNIQUE(category_path, rel_path)`（有，见现有 schema）。

- [ ] **Step 6: Commit**

```bash
git add src-tauri/src/db.rs src-tauri/src/scanner.rs
git commit -m "feat(db): migration v2 — manifest 占位行、present 标记、index_files 与 app_config"
```

---

### Task 3: 下载器（Fetcher trait + ensure_local + download_all）

**Files:**
- Modify: `src-tauri/src/sync.rs`
- Test: `src-tauri/src/sync.rs`（inline tests）

**Interfaces:**
- Produces:
  - `pub trait Fetcher: Send + Sync { fn get(&self, url: &str) -> Result<Vec<u8>>; }`
  - `pub struct HttpFetcher { client: reqwest::Client }` + `impl HttpFetcher { pub fn new() -> Result<Self> }`
  - `pub fn tmp_dir_for(root: &Path) -> PathBuf` —— `root` 的同级 `.tmp` 目录
  - `pub struct PendingFile { pub category_path: String, pub rel_path: String, pub size: u64 }`
  - `pub fn ensure_local(root: &Path, entry: &PendingFile, fetcher: &dyn Fetcher) -> Result<()>` —— 命中且 size 相符则直接返回；否则下载到 `.tmp` 再原子改名
  - `pub fn download_all<F>(root: &Path, pending: Vec<PendingFile>, fetcher: &Arc<F>, on_progress: &(dyn Fn(usize, usize) + Sync)) -> Result<()>`
  - `pub fn remote_url(base_url: &str, path: &str) -> String`
  - `pub fn pending_from_db(status_rows: &[(String, String, i64)]) -> Vec<PendingFile>`

- [ ] **Step 1: 写失败测试**

在 `src-tauri/src/sync.rs` 的 `mod tests` 内追加：

```rust
    // ---- 下载器 ----

    struct FakeFetcher {
        map: std::collections::HashMap<String, Vec<u8>>,
        calls: std::sync::Mutex<Vec<String>>,
    }

    impl FakeFetcher {
        fn new(pairs: &[(&str, &[u8])]) -> Self {
            Self {
                map: pairs.iter().map(|(k, v)| (k.to_string(), v.to_vec())).collect(),
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl Fetcher for FakeFetcher {
        fn get(&self, url: &str) -> Result<Vec<u8>> {
            self.calls.lock().unwrap().push(url.to_string());
            self.map
                .get(url)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("404 {url}"))
        }
    }

    #[test]
    fn remote_url_joins_base_and_path() {
        assert_eq!(
            remote_url("https://x.com/kb/", "a/b.md"),
            "https://x.com/kb/a/b.md"
        );
        assert_eq!(remote_url("https://x.com/kb", "a/b.md"), "https://x.com/kb/a/b.md");
    }

    #[test]
    fn ensure_local_skips_download_when_file_already_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(root.join("a/x.md"), b"hello").unwrap();
        let f = FakeFetcher::new(&[]);
        ensure_local_at(&root, &PendingFile { category_path: "a".into(), rel_path: "x.md".into(), size: 5 }, "https://x/kb", &f).unwrap();
        assert!(f.calls.lock().unwrap().is_empty(), "本地已有且大小相符，不应发请求");
    }

    #[test]
    fn ensure_local_redownloads_when_size_differs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(root.join("a/x.md"), b"stale").unwrap();
        let f = FakeFetcher::new(&[("https://x/kb/a/x.md", b"fresh-content")]);
        ensure_local_at(&root, &PendingFile { category_path: "a".into(), rel_path: "x.md".into(), size: 13 }, "https://x/kb", &f).unwrap();
        assert_eq!(std::fs::read(root.join("a/x.md")).unwrap(), b"fresh-content");
    }

    #[test]
    fn ensure_local_downloads_missing_file_via_tmp_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(&root).unwrap();
        let f = FakeFetcher::new(&[("https://x/kb/a/y.md", b"body")]);
        ensure_local_at(&root, &PendingFile { category_path: "a".into(), rel_path: "y.md".into(), size: 4 }, "https://x/kb", &f).unwrap();
        assert_eq!(std::fs::read(root.join("a/y.md")).unwrap(), b"body");
        // 临时文件不能留在 knowledge/ 里（否则 scanner 会把它扫成新分类）
        let leftovers: Vec<_> = walkdir::WalkDir::new(&root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && e.file_name() != "y.md")
            .collect();
        assert!(leftovers.is_empty(), "knowledge/ 里残留了临时文件");
    }

    #[test]
    fn ensure_local_leaves_no_partial_file_when_download_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(&root).unwrap();
        let f = FakeFetcher::new(&[]); // 一律 404
        let err = ensure_local_at(&root, &PendingFile { category_path: "a".into(), rel_path: "z.md".into(), size: 9 }, "https://x/kb", &f);
        assert!(err.is_err());
        assert!(!root.join("a/z.md").exists(), "失败后不能留下半个文件");
    }

    #[test]
    fn ensure_local_rejects_size_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(&root).unwrap();
        let f = FakeFetcher::new(&[("https://x/kb/a/s.md", b"12345")]); // manifest 说 100 字节
        let err = ensure_local_at(&root, &PendingFile { category_path: "a".into(), rel_path: "s.md".into(), size: 100 }, "https://x/kb", &f);
        assert!(err.is_err());
        assert!(!root.join("a/s.md").exists(), "校验不过不能落盘");
    }

    #[test]
    fn download_all_processes_every_pending_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(&root).unwrap();
        let f = std::sync::Arc::new(FakeFetcher::new(&[
            ("https://x/kb/a/1.md", b"1"),
            ("https://x/kb/a/2.md", b"2"),
            ("https://x/kb/b/3.md", b"3"),
        ]));
        let pending = vec![
            PendingFile { category_path: "a".into(), rel_path: "1.md".into(), size: 1 },
            PendingFile { category_path: "a".into(), rel_path: "2.md".into(), size: 1 },
            PendingFile { category_path: "b".into(), rel_path: "3.md".into(), size: 1 },
        ];
        let seen = std::sync::Mutex::new(Vec::new());
        download_all(&root, "https://x/kb", pending, f, &|d, t| {
            seen.lock().unwrap().push((d, t));
        }).unwrap();
        assert_eq!(f.calls.lock().unwrap().len(), 3);
        assert_eq!(std::fs::read(root.join("b/3.md")).unwrap(), b"3");
        assert_eq!(seen.lock().unwrap().last().copied(), Some((3, 3)));
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd src-tauri && cargo test sync:: 2>&1 | tail -20`
Expected: 编译失败，`trait Fetcher not found`

- [ ] **Step 3: 写实现**

追加到 `src-tauri/src/sync.rs`：

```rust
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 下载来源抽象。生产用 HttpFetcher；单测用 FakeFetcher，不碰网络。
pub trait Fetcher: Send + Sync {
    fn get(&self, url: &str) -> Result<Vec<u8>>;
}

pub struct HttpFetcher {
    client: reqwest::Client,
}

impl HttpFetcher {
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?;
        Ok(Self { client })
    }
}

impl Fetcher for HttpFetcher {
    fn get(&self, url: &str) -> Result<Vec<u8>> {
        let bytes = self
            .client
            .get(url)
            .send()?
            .error_for_status()?
            .bytes()?;
        Ok(bytes.to_vec())
    }
}

pub fn remote_url(base_url: &str, path: &str) -> String {
    format!("{}/{}", base_url.trim_end_matches('/'), path)
}

/// 下载中的临时文件目录 —— 必须是 knowledge/ 的**同级**，绝不能落在 knowledge/ 里面：
/// scanner 会 walk 整个 knowledge/，多出来的目录会被当成一个新分类。
pub fn tmp_dir_for(root: &Path) -> PathBuf {
    root.parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".tmp")
}

#[derive(Debug, Clone)]
pub struct PendingFile {
    pub category_path: String,
    pub rel_path: String,
    pub size: u64,
}

fn local_path(root: &Path, entry: &PendingFile) -> PathBuf {
    // path 已在 upsert 阶段过 normalize_rel_path，这里再拼一次以防万一
    root.join(&entry.category_path).join(&entry.rel_path)
}

fn temp_name(entry: &PendingFile) -> String {
    format!("{}.{}.part", entry.category_path.replace('/', "__"), entry.rel_path.replace('/', "__"))
}

/// 确保本地文件就位。`base_url` 为空表示纯本地模式，不允许触发任何下载。
/// 本地已有且大小与 manifest 一致 → 直接返回，完全不联网。
pub fn ensure_local_at(root: &Path, entry: &PendingFile, base_url: &str, fetcher: &dyn Fetcher) -> Result<()> {
    let dest = local_path(root, entry);
    if let Ok(meta) = std::fs::metadata(&dest) {
        if meta.len() == entry.size {
            return Ok(());
        }
    }
    if base_url.trim().is_empty() {
        anyhow::bail!(
            "{} 尚未缓存，且未配置知识库同步地址",
            entry.rel_path
        );
    }
    let tmp_dir = tmp_dir_for(root);
    std::fs::create_dir_all(&tmp_dir)?;
    let tmp = tmp_dir.join(temp_name(entry));
    let url = remote_url(base_url, &format!("{}/{}", entry.category_path, entry.rel_path));
    let body = fetcher.get(&url).map_err(|e| anyhow::anyhow!("下载 {url} 失败：{e}"))?;
    if body.len() as u64 != entry.size {
        let _ = std::fs::remove_file(&tmp);
        anyhow::bail!(
            "下载校验失败 {}：收到 {} 字节，服务器声明 {} 字节",
            entry.rel_path,
            body.len(),
            entry.size
        );
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&tmp, &body)?;
    std::fs::rename(&tmp, &dest)?;
    Ok(())
}
```

并把上面测试里的 `ensure_local(&root, entry, &f)` 全部改成 `ensure_local_at(&root, entry, "https://x/kb", &f)`。

`download_all`（3 并发，`std::thread::scope` 实现，不新增依赖）：

```rust
/// 并发（最多 3 路）下载全部待缓存文件。单个文件失败不中断整批，
/// 最后一个参数把 (已完成, 总数) 报给调用方用于进度条。
pub fn download_all<F: Fetcher + 'static>(
    root: &Path,
    base_url: &str,
    pending: Vec<PendingFile>,
    fetcher: Arc<F>,
    on_progress: &(dyn Fn(usize, usize) + Sync),
) -> Result<()> {
    let total = pending.len();
    if total == 0 {
        on_progress(0, 0);
        return Ok(());
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let root_ref = root.to_path_buf();
    let base = base_url.to_string();
    let progress: &(dyn Fn(usize, usize) + Sync) = on_progress;
    let queue = std::sync::Mutex::new(pending);
    std::thread::scope(|scope| {
        for _ in 0..3 {
            let next = &next;
            let done = &done;
            let queue = &queue;
            let root_ref = &root_ref;
            let base = &base;
            let fetcher = fetcher.clone();
            scope.spawn(move || loop {
                let item = {
                    let mut q = queue.lock().unwrap();
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    q.get(i).cloned()
                };
                let Some(item) = item else { break };
                // 单个文件失败不拖垮整批 —— 用户点「下载全部」时一两个 404 不该让整次失败
                let _ = ensure_local_at(&root_ref, &item, &base, fetcher.as_ref());
                let d = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                progress(d, total);
            });
        }
    });
    Ok(())
}
```

把上面 `download_all` 的测试调用改为：

```rust
        download_all(&root, "https://x/kb", pending, f, &|d, t| {
```

并把 `mod tests` 顶部加上 `use std::sync::Arc;`。
- [ ] **Step 4: 运行测试确认通过**

Run: `cd src-tauri && cargo test 2>&1 | tail -30`
Expected: 全部 passed

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/sync.rs
git commit -m "feat(sync): Fetcher trait、ensure_local 落盘与 3 并发 download_all"
```

---

### Task 4: commands 接入 + 读文件自动下载

**Files:**
- Modify: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/lib.rs`（注册新命令）
- Test: `src-tauri/src/commands.rs`（inline tests）

**Interfaces:**
- Consumes: `sync::{ensure_local_at, download_all, HttpFetcher, PendingFile}`、`db::{upsert_manifest, sync_status, ...}`
- Produces 新 Tauri 命令（前端 invoke 名）：
  - `set_sync_base_url(url: String) -> ()` —— 同时写 `app_config.sync_base_url`
  - `sync_manifest() -> SyncManifestDto { files: i64, indexes: i64, skipped: bool }`
  - `sync_status() -> SyncStatusDto { total, present, totalBytes, cachedBytes, configured }`
  - `download_resource(id: i64) -> ()`
  - `download_all() -> ()` —— 后台跑，进度走 `sync_progress` 事件
  - `ResourceDto` 新增 `present: bool`

- [ ] **Step 1: 写失败测试**

在 `commands.rs` 现有 `mod tests` 内追加（这些是不需要 AppHandle 的纯逻辑部分）：

```rust
    use crate::sync::{ManifestFile, PendingFile};

    fn tmp_db() -> (tempfile::TempDir, Db) {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open(&tmp.path().join("t.db")).unwrap();
        db.migrate().unwrap();
        (tmp, db)
    }

    #[test]
    fn pending_rows_skip_present_files() {
        let (_t, db) = tmp_db();
        db.upsert_manifest(&[
            ManifestFile { path: "a/x.md".into(), size: 10, sha256: None },
            ManifestFile { path: "a/y.md".into(), size: 20, sha256: None },
        ])
        .unwrap();
        db.mark_resource_present("a", "x.md").unwrap();
        let rows = super::pending_rows(&db).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rel_path, "y.md");
        assert_eq!(rows[0].size, 20);
    }

    #[test]
    fn apply_manifest_marks_downloaded_files_present() {
        let (tmp, db) = tmp_db();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(root.join("a/x.md"), b"12345").unwrap();
        db.upsert_manifest(&[ManifestFile { path: "a/x.md".into(), size: 5, sha256: None }]).unwrap();
        super::apply_manifest(&db, &root, &crate::sync::Manifest {
            version: "v".into(),
            files: vec![ManifestFile { path: "a/x.md".into(), size: 5, sha256: None }],
            indexes: vec![],
        }).unwrap();
        let present: i64 = db.conn.query_row("SELECT present FROM resources", [], |r| r.get(0)).unwrap();
        assert_eq!(present, 1);
    }

    /// Review Focus #1：无网络 / 未配置同步地址时点开未缓存文章，
    /// 错误信息必须是能照着做的人话，而不是 "resource not found"。
    #[test]
    fn ensure_cached_without_base_url_gives_actionable_chinese_error() {
        let (tmp, db) = tmp_db();
        let root = tmp.path().join("knowledge");
        std::fs::create_dir_all(root.join("a")).unwrap();
        db.upsert_manifest(&[ManifestFile { path: "a/new.md".into(), size: 5, sha256: None }]).unwrap();
        let id: i64 = db.conn.query_row("SELECT id FROM resources WHERE rel_path='new.md'", [], |r| r.get(0)).unwrap();
        let state = AppState { db_path: tmp.path().join("t.db"), knowledge_root: root };
        let err = super::ensure_cached(&state, &db, id, "a", "new.md").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("尚未缓存"), "实际信息：{msg}");
        assert!(msg.contains("同步地址"), "应告诉用户去哪里配置：{msg}");
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd src-tauri && cargo test commands:: 2>&1 | tail -20`
Expected: 编译失败，`cannot find function pending_rows`

- [ ] **Step 3: 实现纯逻辑 helper**（放在 `commands.rs`，`#[cfg(test)]` 之外）

```rust
/// DB 里 present=0 的远端条目 —— 待下载队列。
pub fn pending_rows(db: &Db) -> Result<Vec<crate::sync::PendingFile>> {
    let mut stmt = db.conn.prepare(
        "SELECT category_path, rel_path, size_bytes FROM resources
         WHERE remote_hash IS NOT NULL AND present = 0",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok(crate::sync::PendingFile {
                category_path: r.get(0)?,
                rel_path: r.get(1)?,
                size: r.get::<_, i64>(2)?.max(0) as u64,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// manifest 落库后，把本地已经存在的文件标成 present=1（用户可能在同步前就下载过）。
pub fn apply_manifest(db: &Db, knowledge_root: &Path, manifest: &crate::sync::Manifest) -> Result<()> {
    db.upsert_manifest(&manifest.files)?;
    db.upsert_index_files(&manifest.indexes)?;
    let keep: Vec<(String, String)> = manifest
        .files
        .iter()
        .filter_map(|f| crate::sync::split_category(&f.path))
        .collect();
    db.drop_resources_absent_from_manifest(&keep)?;
    for f in &manifest.files {
        let Some((cat, rel)) = crate::sync::split_category(&f.path) else { continue };
        let abs = reader::resolve_absolute(knowledge_root, &cat, &rel);
        if std::fs::metadata(&abs).map(|m| m.len() == f.size).unwrap_or(false) {
            db.mark_resource_present(&cat, &rel)?;
        }
    }
    for f in &manifest.indexes {
        if std::fs::metadata(knowledge_root.join(&f.path)).map(|m| m.len() == f.size).unwrap_or(false) {
            db.mark_index_present(&f.path)?;
        }
    }
    Ok(())
}
```

- [ ] **Step 4: 接入命令**

**(a) `ResourceDto` 加字段**：在结构体里加 `pub present: bool`，并在 `From<ResourceRow>` 里填 `r.present`。

**(b) `read_resource` 自动下载**：在 `let abs = reader::resolve_absolute(...)` 之前插入：

```rust
    ensure_cached(&state, &db, id, &category_path, &rel_path)
        .map_err(|e| format!("{e:#}"))?;
```

并新增：

```rust
/// 文件本地缺失时自动从远端补齐再读。命中缓存或纯本地模式（无 base_url）时静默返回。
/// 下载失败必须给出可读的中文原因 —— 用户在无网络下点开未缓存文章时，
/// 不能看到 "resource not found" 这种天书。
pub fn ensure_cached(
    state: &AppState,
    db: &Db,
    id: i64,
    category_path: &str,
    rel_path: &str,
) -> Result<()> {
    let abs = reader::resolve_absolute(&state.knowledge_root, category_path, rel_path);
    let size: i64 = db.conn.query_row(
        "SELECT size_bytes FROM resources WHERE id = ?1",
        [id],
        |r| r.get(0),
    )?;
    if std::fs::metadata(&abs).map(|m| m.len() as i64 == size).unwrap_or(false) {
        return Ok(());
    }
    let base_url = db.get_config("sync_base_url")?.unwrap_or_default();
    if base_url.trim().is_empty() {
        anyhow::bail!(
            "「{rel_path}」尚未缓存到本机，且未配置知识库同步地址（设置 → 知识库同步）"
        );
    }
    let fetcher = crate::sync::HttpFetcher::new()?;
    crate::sync::ensure_local_at(
        &state.knowledge_root,
        &crate::sync::PendingFile {
            category_path: category_path.to_string(),
            rel_path: rel_path.to_string(),
            size: size.max(0) as u64,
        },
        &base_url,
        &fetcher,
    )?;
    db.mark_resource_present(category_path, rel_path)?;
    Ok(())
}
```

`read_resource_bytes` 同样在读之前调用 `ensure_cached`。

**(c) `read_subcategory_index` 自动下载 `_index.md`**：在 `index_md_parse_file(&abs)` 之前：

```rust
    let index_path = format!("{category_path}/_index.md");
    if !abs.is_file() {
        if let Some((size, present)) = db.index_file(&index_path)? {
            if !present {
                let base_url = db.get_config("sync_base_url")?.unwrap_or_default();
                if !base_url.trim().is_empty() {
                    let (cat, rel) = index_path.split_once('/').map(|(a, b)| (a.to_string(), b.to_string()))
                        .ok_or_else(|| anyhow::anyhow!("bad index path"))?;
                    let _ = crate::sync::ensure_local_at(
                        &state.knowledge_root,
                        &crate::sync::PendingFile { category_path: cat, rel_path: rel, size },
                        &base_url,
                        &crate::sync::HttpFetcher::new()?,
                    );
                    if abs.is_file() {
                        db.mark_index_present(&index_path)?;
                    }
                }
            }
        }
    }
```

（`read_subcategory_index` 当前没有开 DB，需要加 `let db = Db::open(&state.db_path)...`。）

**(d) 新增 5 个命令**：

```rust
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncStatusDto {
    pub total: i64,
    pub present: i64,
    pub total_bytes: i64,
    pub cached_bytes: i64,
    /// 是否配置了远端地址 —— false 时前端不显示状态条
    pub configured: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncManifestDto {
    pub files: i64,
    pub indexes: i64,
    pub skipped: bool,
}

#[tauri::command]
pub fn set_sync_base_url(url: String, app: AppHandle) -> Result<(), String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    db.set_config("sync_base_url", url.trim())
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn sync_status(app: AppHandle) -> Result<SyncStatusDto, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let st = db.sync_status().map_err(|e| e.to_string())?;
    let configured = db
        .get_config("sync_base_url")
        .map_err(|e| e.to_string())?
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    Ok(SyncStatusDto {
        total: st.total,
        present: st.present,
        total_bytes: st.total_bytes,
        cached_bytes: st.cached_bytes,
        configured,
    })
}

#[tauri::command]
pub async fn sync_manifest(app: AppHandle) -> Result<SyncManifestDto, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let base_url = db
        .get_config("sync_base_url")
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    if base_url.trim().is_empty() {
        // 纯本地模式（开发）不报错，静默跳过 —— 前端据此不显示状态条
        return Ok(SyncManifestDto { files: 0, indexes: 0, skipped: true });
    }
    let url = crate::sync::remote_url(&base_url, "manifest.json");
    let body = crate::sync::HttpFetcher::new()
        .map_err(|e| e.to_string())?
        .get(&url)
        .map_err(|e| format!("拉取 {url} 失败：{e}"))?;
    let text = String::from_utf8(body).map_err(|e| e.to_string())?;
    let manifest = crate::sync::parse_manifest(&text).map_err(|e| format!("manifest 解析失败：{e}"))?;
    let root = state.knowledge_root.clone();
    tauri::async_runtime::spawn_blocking(move || -> anyhow::Result<()> {
        let db = Db::open(&state.db_path)?;
        db.migrate()?;
        apply_manifest(&db, &root, &manifest)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    Ok(SyncManifestDto {
        files: manifest.files.len() as i64,
        indexes: manifest.indexes.len() as i64,
        skipped: false,
    })
}

#[tauri::command]
pub fn download_resource(id: i64, app: AppHandle) -> Result<(), String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let (cat, rel, size) = db.conn.query_row(
        "SELECT category_path, rel_path, size_bytes FROM resources WHERE id = ?1",
        [id],
        |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?)),
    ).map_err(|e| format!("resource {id} not found: {e}"))?;
    let base_url = db.get_config("sync_base_url").map_err(|e| e.to_string())?.unwrap_or_default();
    crate::sync::ensure_local_at(
        &state.knowledge_root,
        &crate::sync::PendingFile { category_path: cat, rel_path: rel, size: size.max(0) as u64 },
        &base_url,
        &crate::sync::HttpFetcher::new().map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{e:#}"))?;
    db.mark_resource_present_unchecked(id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn download_all(app: AppHandle) -> Result<(), String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let base_url = db.get_config("sync_base_url").map_err(|e| e.to_string())?.unwrap_or_default();
    let pending = pending_rows(&db).map_err(|e| e.to_string())?;
    if pending.is_empty() {
        return Ok(());
    }
    let root = state.knowledge_root.clone();
    let db_path = state.db_path.clone();
    let emit_app = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> anyhow::Result<()> {
        let total = pending.len();
        let fetcher = std::sync::Arc::new(crate::sync::HttpFetcher::new()?);
        crate::sync::download_all(&root, &base_url, pending, fetcher, &move |done, total| {
            let _ = emit_app.emit(
                "sync_progress",
                serde_json::json!({"done": done, "total": total}),
            );
        })?;
        // 下载完统一刷新 present 标记
        let db = Db::open(&db_path)?;
        mark_all_present(&db)
    });
    Ok(())
}

fn mark_all_present(db: &Db) -> anyhow::Result<()> {
    for p in pending_rows(db)? {
        db.mark_resource_present(&p.category_path, &p.rel_path)?;
    }
    Ok(())
}
```

- [ ] **Step 5: 注册命令**

`src-tauri/src/lib.rs` 的 `generate_handler!` 里追加：

```rust
            commands::set_sync_base_url,
            commands::sync_manifest,
            commands::sync_status,
            commands::download_resource,
            commands::download_all,
```

并在 `commands.rs` 顶部确认 `use std::path::Path;`、`use tauri::Manager;` 已有（`Emitter` 也要有，`sync_progress` 要用）。

**注意**：`db.rs` 里需要补一个按 id 标记的便捷方法 `mark_resource_present_unchecked(id)`：

```rust
    pub fn mark_resource_present_unchecked(&self, id: i64) -> Result<()> {
        self.conn.execute("UPDATE resources SET present = 1 WHERE id = ?1", [id])?;
        Ok(())
    }
```

- [ ] **Step 6: 运行测试确认通过**

Run: `cd src-tauri && cargo test 2>&1 | tail -30`
Expected: 全部 passed

- [ ] **Step 7: Commit**

```bash
git add src-tauri/src/commands.rs src-tauri/src/lib.rs src-tauri/src/db.rs
git commit -m "feat(commands): 同步命令族 + read_resource 缺文件自动下载"
```

---

### Task 5: knowledge_root 解析 + 种子库释放

**Files:**
- Create: `src-tauri/src/platform.rs`
- Modify: `src-tauri/src/commands.rs`（`build_state` 改为调用 `platform::resolve_knowledge_root`，删除 `pick_knowledge_root` 及其 3 个测试）
- Modify: `src-tauri/src/lib.rs`（`pub mod platform;`）
- Test: `src-tauri/src/platform.rs`（inline tests）+ `src-tauri/tests/seed_smoke.rs`

**Interfaces:**
- Produces:
  - `pub fn find_knowledge_root(start: &Path) -> Option<PathBuf>`（从 commands.rs 迁到这里）
  - `pub fn pick_knowledge_root(dev_root: Option<PathBuf>, app_data: PathBuf) -> PathBuf`
  - `pub struct SeedManifest { pub files: Vec<SeedEntry> }` / `pub struct SeedEntry { pub path: String, pub size: u64 }`
  - `pub fn parse_seed_manifest(json: &str) -> Result<Vec<SeedEntry>>`
  - `pub fn copy_seed_entries(entries: &[SeedEntry], read: &dyn Fn(&str) -> Result<Vec<u8>>, root: &Path) -> Result<SeedStats>` —— 读内容的动作由调用方注入（Android 上是 `app.fs()`，单测里是内存 map）
  - `pub struct SeedStats { pub written: usize, pub skipped: usize }`
  - `pub fn migrate_legacy_bundled(src: &Path, dest: &Path) -> Result<usize>` —— 桌面端老版本迁移

- [ ] **Step 1: 写失败测试**

`src-tauri/src/platform.rs` 内：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_seed_manifest_reads_paths_and_sizes() {
        let json = r#"{"files":[{"path":"01-基础/_index.md","size":12},{"path":"01-基础/a.md","size":34}]}"#;
        let e = parse_seed_manifest(json).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].path, "01-基础/_index.md");
        assert_eq!(e[1].size, 34);
    }

    #[test]
    fn parse_seed_manifest_rejects_traversal() {
        assert!(parse_seed_manifest(r#"{"files":[{"path":"../evil.md","size":1}]}"#).is_err());
    }

    #[test]
    fn copy_seed_entries_writes_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("knowledge");
        let body = |p: &str| -> Result<Vec<u8>> {
            if p == "01-基础/a.md" { Ok(b"hello".to_vec()) } else { anyhow::bail!("no asset {p}") }
        };
        let entries = vec![SeedEntry { path: "01-基础/a.md".into(), size: 5 }];
        let s1 = copy_seed_entries(&entries, &body, &root).unwrap();
        assert_eq!((s1.written, s1.skipped), (1, 0));
        let s2 = copy_seed_entries(&entries, &body, &root).unwrap();
        assert_eq!((s2.written, s2.skipped), (0, 1), "第二次应全部跳过");
        assert_eq!(std::fs::read(root.join("01-基础/a.md")).unwrap(), b"hello");
    }

    #[test]
    fn copy_seed_entries_propagates_read_error() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("knowledge");
        let body = |_p: &str| -> Result<Vec<u8>> { anyhow::bail!("asset 读取失败") };
        let err = copy_seed_entries(&[SeedEntry { path: "a/b.md".into(), size: 1 }], &body, &root);
        assert!(err.is_err());
    }

    #[test]
    fn pick_knowledge_root_prefers_dev_dir() {
        assert_eq!(
            pick_knowledge_root(Some(PathBuf::from("/dev/resources/knowledge")), PathBuf::from("/data/knowledge")),
            PathBuf::from("/dev/resources/knowledge")
        );
    }

    #[test]
    fn pick_knowledge_root_falls_back_to_app_data() {
        assert_eq!(
            pick_knowledge_root(None, PathBuf::from("/data/knowledge")),
            PathBuf::from("/data/knowledge")
        );
    }

    #[test]
    fn migrate_legacy_bundled_copies_tree_once() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("old");
        let dest = dir.path().join("new");
        std::fs::create_dir_all(src.join("a")).unwrap();
        std::fs::write(src.join("a/x.md"), b"x").unwrap();
        let n = migrate_legacy_bundled(&src, &dest).unwrap();
        assert_eq!(n, 1);
        assert_eq!(std::fs::read(dest.join("a/x.md")).unwrap(), b"x");
    }
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd src-tauri && cargo test platform:: 2>&1 | tail -20`
Expected: 文件不存在

- [ ] **Step 3: 实现**

`src-tauri/src/platform.rs`：

```rust
//! knowledge_root 的平台解析 + Android 种子库释放。
//!
//! 规则（见 spec §5）：
//! - 开发模式（cwd 向上能找到 `resources/knowledge/`）→ 用它，改文件立刻生效
//! - 其余（已安装桌面端 / Android）→ `app_data_dir/knowledge`
//!
//! `$RESOURCE` 回落分支已随"知识库不再打进安装包"一起删除。
use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub fn find_knowledge_root(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        let candidate = d.join("resources").join("knowledge");
        if candidate.is_dir() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

pub fn pick_knowledge_root(dev_root: Option<PathBuf>, app_data: PathBuf) -> PathBuf {
    dev_root.unwrap_or(app_data)
}

#[derive(Debug, Clone, Deserialize)]
pub struct SeedEntry {
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Clone, Default)]
pub struct SeedStats {
    pub written: usize,
    pub skipped: usize,
}

pub fn parse_seed_manifest(json: &str) -> Result<Vec<SeedEntry>> {
    #[derive(Deserialize)]
    struct Raw {
        #[serde(default)]
        files: Vec<SeedEntry>,
    }
    let raw: Raw = serde_json::from_str(json)?;
    for e in &raw.files {
        // 与 sync::normalize_rel_path 同一套拒绝规则
        crate::sync::normalize_rel_path(&e.path)?;
    }
    Ok(raw.files)
}

/// 把种子库从（Android 上不可直接读的）asset URI 释放到真实目录。
/// `read` 由调用方注入：Android 传 `app.fs()`，测试传内存 map。
/// 已存在且大小相符 → 跳过（幂等）。
pub fn copy_seed_entries(
    entries: &[SeedEntry],
    read: &dyn Fn(&str) -> Result<Vec<u8>>,
    root: &Path,
) -> Result<SeedStats> {
    let mut stats = SeedStats::default();
    std::fs::create_dir_all(root)?;
    for e in entries {
        let dest = root.join(&e.path);
        if std::fs::metadata(&dest).map(|m| m.len() == e.size).unwrap_or(false) {
            stats.skipped += 1;
            continue;
        }
        let body = read(&e.path)?;
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&dest, body)
            .map_err(|err| anyhow::anyhow!("write seed {}: {err}", e.path))?;
        stats.written += 1;
    }
    Ok(stats)
}

/// 桌面端老版本升级迁移：知识库原先随安装包打在 $RESOURCE/knowledge。
/// 目标目录为空时才搬（只搬一次），搬完用户本地编辑过的文件不会被覆盖。
pub fn migrate_legacy_bundled(src: &Path, dest: &Path) -> Result<usize> {
    if !src.is_dir() {
        return Ok(0);
    }
    std::fs::create_dir_all(dest)?;
    if std::fs::read_dir(dest)?.next().is_some() {
        return Ok(0);
    }
    let mut copied = 0usize;
    for entry in WalkDir::new(src) {
        let entry = match entry { Ok(e) => e, Err(_) => continue };
        let rel = entry.path().strip_prefix(src)?;
        if rel.as_os_str().is_empty() {
            continue;
        }
        let target = dest.join(rel);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
            copied += 1;
        }
    }
    Ok(copied)
}
```

`commands.rs` 的 `build_state` 改为：

```rust
pub fn build_state(app: &AppHandle) -> AppState {
    let db_path = resolve_db_path(app);
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut app_data = app
        .path()
        .app_data_dir()
        .expect("app_data_dir resolvable");
    app_data.push("knowledge");
    let knowledge_root = platform::pick_knowledge_root(
        platform::find_knowledge_root(&cwd),
        app_data,
    );
    std::fs::create_dir_all(&knowledge_root).ok();
    AppState { db_path, knowledge_root }
}
```

删除 `commands.rs` 中的 `pick_knowledge_root` 函数与它对应的 3 个测试（`pick_knowledge_root_*`）。`find_knowledge_root` 及其 2 个测试一并从 commands.rs 迁到 platform.rs。

`lib.rs` 加 `pub mod platform;`。

- [ ] **Step 4: Android 侧接线（`#[cfg(target_os = "android")]`）**

在 `platform.rs` 追加：

```rust
/// Android 启动时把 APK assets 里的种子库释放到 knowledge/。
/// 桌面端是 no-op —— 桌面不打包任何资源。
#[cfg(target_os = "android")]
pub fn materialize_seed(app: &tauri::AppHandle, root: &Path) -> Result<SeedStats> {
    use tauri_plugin_fs::FsExt;
    let manifest_path = app
        .path()
        .resolve("seed-manifest.json", tauri::path::BaseDirectory::Resource)?;
    let json = app
        .fs()
        .read_to_string(&manifest_path)
        .map_err(|e| anyhow::anyhow!("read seed-manifest: {e}"))?;
    let entries = parse_seed_manifest(&json)?;
    let total = entries.len();
    let mut done = 0usize;
    let emit_app = app.clone();
    let read = move |p: &str| -> Result<Vec<u8>> {
        let asset = format!("knowledge/{p}");
        let full = app.path().resolve(&asset, tauri::path::BaseDirectory::Resource)?;
        let bytes = app.fs().read(&full).map_err(|e| anyhow::anyhow!("read asset {asset}: {e}"))?;
        done += 1;
        let _ = emit_app.emit(
            "seed_progress",
            serde_json::json!({"done": done, "total": total}),
        );
        Ok(bytes)
    };
    copy_seed_entries(&entries, &read, root)
}

#[cfg(not(target_os = "android"))]
pub fn materialize_seed(_app: &tauri::AppHandle, _root: &Path) -> Result<SeedStats> {
    Ok(SeedStats::default())
}
```

在 `lib.rs` 的 `setup` 里调用（`build_state` 之后）：

```rust
        .setup(|app| {
            let state = commands::build_state(app.handle());
            let root = state.knowledge_root.clone();
            // 种子库只在 Android 有意义；失败不阻断启动（远端同步仍可补齐）
            if let Err(e) = platform::materialize_seed(app.handle(), &root) {
                eprintln!("[seed] materialize failed: {e:#}");
            }
            app.manage(state);
            Ok(())
        })
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd src-tauri && cargo test 2>&1 | tail -30`
Expected: 全部 passed（桌面编译即可，Android 分支不参与测试）

- [ ] **Step 6: Commit**

```bash
git add src-tauri/src/platform.rs src-tauri/src/commands.rs src-tauri/src/lib.rs
git commit -m "feat(platform): knowledge_root 平台解析、种子库释放与旧 bundle 迁移"
```

---

### Task 6: manifest 生成脚本 + 移除 bundle.resources

**Files:**
- Create: `scripts/gen-remote-manifest.mjs`
- Create: `scripts/serve-dist.mjs`
- Modify: `src-tauri/tauri.conf.json`
- Modify: `package.json`（加 `gen:manifest`、`serve:dist`）
- Test: `scripts/gen-remote-manifest.test.mjs`（node:test，随 `npm test` 之外的独立命令跑）

**Interfaces:**
- `gen-remote-manifest.mjs` 导出 `buildManifest(root, { seedFiles })`，供 node:test 直接断言
- CLI：`node scripts/gen-remote-manifest.mjs --out resources/dist --seed`（`--seed` 只收种子子集）

- [ ] **Step 1: 写失败测试**

`scripts/gen-remote-manifest.test.mjs`：

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildManifest } from './gen-remote-manifest.mjs';

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'kb-'));
  mkdirSync(join(root, '01-基础', '回归'), { recursive: true });
  mkdirSync(join(root, '01-基础', '三方资料'), { recursive: true });
  writeFileSync(join(root, '01-基础', '_index.md'), '# 基础\n');
  writeFileSync(join(root, '01-基础', '回归', '线性.md'), 'linear\n');
  writeFileSync(join(root, '01-基础', '三方资料', 'book.pdf'), 'PDF');
  writeFileSync(join(root, '01-基础', 'notes.txt'), 'ignored');
  return root;
}

test('manifest 只收 md/pdf/docx/pptx，_index.md 进 indexes', () => {
  const { files, indexes } = buildManifest(fixture());
  assert.deepEqual(files.map((f) => f.path).sort(), ['01-基础/三方资料/book.pdf', '01-基础/回归/线性.md']);
  assert.deepEqual(indexes.map((f) => f.path), ['01-基础/_index.md']);
});

test('每条都带 size 与 sha256', () => {
  const { files } = buildManifest(fixture());
  const md = files.find((f) => f.path.endsWith('.md'));
  assert.equal(md.size, Buffer.byteLength('linear\n'));
  assert.match(md.sha256, /^[0-9a-f]{64}$/);
});

test('seed 子集只保留显式列出的路径', () => {
  const { files } = buildManifest(fixture(), { seedFiles: ['01-基础/_index.md'] });
  assert.deepEqual(files, []);
});
```

- [ ] **Step 2: 运行测试确认失败**

Run: `node --test scripts/gen-remote-manifest.test.mjs`
Expected: `Cannot find module .../gen-remote-manifest.mjs`

- [ ] **Step 3: 实现脚本**

`scripts/gen-remote-manifest.mjs`：

```js
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync, writeFileSync, mkdirSync, cpSync } from 'node:fs';
import { join, relative, sep, dirname } from 'node:path';

const DOC_EXT = new Set(['.md', '.pdf', '.docx', '.pptx']);

/** 递归列出 root 下所有文件（相对路径，统一用 / 分隔）。 */
function walk(root, dir = root, out = []) {
  for (const name of readdirSync(dir)) {
    const abs = join(dir, name);
    if (statSync(abs).isDirectory()) walk(root, abs, out);
    else out.push(relative(root, abs).split(sep).join('/'));
  }
  return out;
}

/**
 * 生成 manifest。_index.md 归入 indexes（scanner 不把它当资源），
 * 未知扩展名直接忽略。seedFiles 非空时只保留这些路径 —— 用于打包 APK 种子库。
 */
export function buildManifest(root, { seedFiles = null } = {}) {
  const allow = seedFiles ? new Set(seedFiles) : null;
  const files = [];
  const indexes = [];
  for (const rel of walk(root)) {
    if (allow && !allow.has(rel)) continue;
    const buf = readFileSync(join(root, rel));
    const entry = {
      path: rel,
      size: buf.length,
      sha256: createHash('sha256').update(buf).digest('hex'),
    };
    if (rel.endsWith('/_index.md') || rel === '_index.md') indexes.push(entry);
    else if (DOC_EXT.has(rel.slice(rel.lastIndexOf('.')).toLowerCase())) files.push(entry);
  }
  files.sort((a, b) => a.path.localeCompare(b.path));
  indexes.sort((a, b) => a.path.localeCompare(b.path));
  return { version: new Date().toISOString(), files, indexes };
}

/** 种子库：只挑每个二级分类下的 _index.md —— 目录骨架可见、正文按需下载。 */
export function seedSelection(root) {
  return walk(root).filter((rel) => rel.endsWith('/_index.md') || rel === '_index.md');
}

function parseArgs(argv) {
  const out = { out: 'resources/dist', seed: false };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--out') out.out = argv[++i];
    else if (argv[i] === '--seed') out.seed = true;
  }
  return out;
}

const invokedDirectly = process.argv[1] && process.argv[1].endsWith('gen-remote-manifest.mjs');
if (invokedDirectly) {
  const args = parseArgs(process.argv.slice(2));
  const root = 'resources/knowledge';
  const manifest = args.seed
    ? buildManifest(root, { seedFiles: seedSelection(root) })
    : buildManifest(root);

  if (args.seed) {
    // 种子库交给 Tauri 打包：文件 + seed-manifest.json
    const seedDir = join('src-tauri', 'seed');
    mkdirSync(seedDir, { recursive: true });
    writeFileSync(join(seedDir, 'seed-manifest.json'), JSON.stringify(manifest, null, 2));
    for (const e of manifest.files) {
      const dest = join(seedDir, 'knowledge', e.path);
      mkdirSync(dirname(dest), { recursive: true });
      cpSync(join(root, e.path), dest);
    }
    console.log(`[seed] ${manifest.files.length} files -> ${seedDir}/knowledge`);
  } else {
    mkdirSync(args.out, { recursive: true });
    writeFileSync(join(args.out, 'manifest.json'), JSON.stringify(manifest, null, 2));
    cpSync(root, join(args.out, 'knowledge'), { recursive: true });
    console.log(
      `[manifest] ${manifest.files.length} files, ${manifest.indexes.length} indexes -> ${args.out}`,
    );
    console.log('把这个目录整个传到任意静态托管即可。');
  }
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `node --test scripts/gen-remote-manifest.test.mjs`
Expected: 3 passed

- [ ] **Step 5: 本地静态服务器**

`scripts/serve-dist.mjs`：

```js
import { createServer } from 'node:http';
import { createReadStream, statSync, existsSync } from 'node:fs';
import { join, normalize, extname } from 'node:path';

const ROOT = process.argv[2] ?? 'resources/dist';
const PORT = Number(process.argv[3] ?? 8787);
const TYPES = {
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.pdf': 'application/pdf',
  '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
};

createServer((req, res) => {
  // 目录穿越防护：规范化后必须仍在 ROOT 内
  const rel = normalize(decodeURIComponent(req.url.split('?')[0])).replace(/^([/\\])+/, '');
  const abs = join(ROOT, rel);
  if (!abs.startsWith(normalize(ROOT)) || !existsSync(abs) || !statSync(abs).isFile()) {
    res.writeHead(404).end('not found');
    return;
  }
  res.writeHead(200, {
    'content-type': TYPES[extname(abs)] ?? 'application/octet-stream',
    'content-length': statSync(abs).size,
    'access-control-allow-origin': '*',
  });
  createReadStream(abs).pipe(res);
}).listen(PORT, () => console.log(`serving ${ROOT} at http://127.0.0.1:${PORT}/`));
```

- [ ] **Step 6: 生成种子库并改 `tauri.conf.json`**

先跑一次生成，**必须先有目录再改配置** —— Tauri 在 `bundle.resources` 指向不存在的路径时会直接构建失败：

```bash
npm run gen:seed
ls src-tauri/seed/knowledge
```

把 `tauri.conf.json` 的 `bundle` 段改成：

```json
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.ico"
    ],
    "resources": {
      "seed/seed-manifest.json": "seed-manifest.json",
      "seed/knowledge/": "knowledge/"
    }
  }
```

知识库本体不再进包；`seed/` 内容只有各分类的 `_index.md`（几十 KB）。

`.gitignore` 追加（**只**忽略生成物 `resources/dist/`，`src-tauri/seed/` 必须纳入版本管理，否则干净 clone 上构建会因缺目录而失败）：

```
resources/dist/
```

- [ ] **Step 7: package.json 脚本**

```json
    "gen:manifest": "node scripts/gen-remote-manifest.mjs",
    "gen:seed": "node scripts/gen-remote-manifest.mjs --seed",
    "serve:dist": "node scripts/serve-dist.mjs",
```

- [ ] **Step 8: 端到端验证脚本能跑**

```bash
node scripts/gen-remote-manifest.mjs --out resources/dist
node scripts/serve-dist.mjs resources/dist 8787 &
curl -s http://127.0.0.1:8787/manifest.json | head -c 120
```

Expected: 输出一段 JSON（含 `"version"`）

- [ ] **Step 9: Commit**

```bash
git add scripts/gen-remote-manifest.mjs scripts/gen-remote-manifest.test.mjs scripts/serve-dist.mjs src-tauri/tauri.conf.json src-tauri/seed package.json .gitignore
git commit -m "feat(dist): manifest 生成与本地静态托管；知识库改为远程分发，仅打包种子库"
```

---

## Phase B — 前端同步

### Task 7: sync.ts + 类型 + store

**Files:**
- Create: `src/lib/sync.ts`
- Modify: `src/types/index.ts`（`Resource` 加 `present`，新增 `SyncStatus` / `SyncProgress`）
- Modify: `src/stores/library.ts`
- Test: `src/lib/sync.test.ts`

**Interfaces:**
- Produces: `sync.ts` 的 `syncManifest()` / `syncStatus()` / `setSyncBaseUrl(url)` / `downloadResource(id)` / `downloadAll()`，以及 `library` store 的 `syncStatus` / `syncPhase` / `downloadProgress` / `syncNow()` / `downloadAll()` / `stopAll()`

- [ ] **Step 1: 写失败测试**

`src/lib/sync.test.ts`：

```ts
import { describe, it, expect, vi } from 'vitest';

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(async (cmd: string) => {
    if (cmd === 'sync_status')
      return { total: 380, present: 37, totalBytes: 1000, cachedBytes: 100, configured: true };
    if (cmd === 'sync_manifest') return { files: 380, indexes: 12, skipped: false };
    if (cmd === 'set_sync_base_url') return null;
    if (cmd === 'download_resource') return null;
    if (cmd === 'download_all') return null;
    return null;
  }),
}));

import { syncStatus, syncManifest, setSyncBaseUrl } from './sync';

describe('sync', () => {
  it('syncStatus returns counts', async () => {
    const s = await syncStatus();
    expect(s.total).toBe(380);
    expect(s.present).toBe(37);
    expect(s.configured).toBe(true);
  });
  it('syncManifest returns file counts', async () => {
    const m = await syncManifest();
    expect(m.files).toBe(380);
    expect(m.skipped).toBe(false);
  });
  it('setSyncBaseUrl trims trailing slash', async () => {
    const { invoke } = await import('@tauri-apps/api/core');
    await setSyncBaseUrl('https://x.com/kb/');
    expect(invoke).toHaveBeenCalledWith('set_sync_base_url', { url: 'https://x.com/kb/' });
  });
});
```

- [ ] **Step 2: 运行确认失败**

Run: `npm test -- src/lib/sync.test.ts`
Expected: Cannot find module './sync'

- [ ] **Step 3: 实现 `src/lib/sync.ts`**

```ts
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type { SyncProgress, SyncStatus } from '../types';

export type { SyncProgress, SyncStatus };

export async function syncStatus(): Promise<SyncStatus> {
  return invoke<SyncStatus>('sync_status');
}

export async function syncManifest(): Promise<{ files: number; indexes: number; skipped: boolean }> {
  return invoke('sync_manifest');
}

export async function setSyncBaseUrl(url: string): Promise<void> {
  await invoke('set_sync_base_url', { url });
}

export async function downloadResource(id: number): Promise<void> {
  await invoke('download_resource', { id });
}

export async function downloadAll(): Promise<void> {
  await invoke('download_all');
}

/** 后端 download_all 是 fire-and-forget；进度靠事件回传。 */
export async function onSyncProgress(cb: (p: SyncProgress) => void): Promise<UnlistenFn> {
  return listen<SyncProgress>('sync_progress', (e) => cb(e.payload));
}

export async function onSeedProgress(cb: (p: SyncProgress) => void): Promise<UnlistenFn> {
  return listen<SyncProgress>('seed_progress', (e) => cb(e.payload));
}
```

- [ ] **Step 4: 类型**

`src/types/index.ts`：

```ts
export interface Resource {
  // ...现有字段
  /** 文件是否已缓存到本地 knowledge/ 目录。false = 点开会自动下载。 */
  present: boolean;
}

export interface SyncStatus {
  total: number;
  present: number;
  totalBytes: number;
  cachedBytes: number;
  /** 是否配置了远端地址；false 时 UI 不显示同步状态条 */
  configured: boolean;
}

export interface SyncProgress {
  done: number;
  total: number;
}
```

- [ ] **Step 5: store 扩展**

`src/stores/library.ts` 增加：

```ts
  syncStatus: SyncStatus | null;
  syncPhase: 'idle' | 'manifest' | 'downloading' | 'error';
  downloadProgress: { done: number; total: number } | null;
  syncNow: () => Promise<void>;
  refreshSyncStatus: () => Promise<void>;
  downloadAll: () => Promise<void>;
```

实现（action 部分）。注意 `library-api.ts` 里没有同步函数，同步走独立的 `sync.ts` —— 在文件顶部加 `import * as sync from '../lib/sync';`，然后：

```ts
  refreshSyncStatus: async () => {
    try {
      set({ syncStatus: await sync.syncStatus() });
    } catch {
      // 状态条是增强信息，拉不到就静默隐藏
    }
  },

  syncNow: async () => {
    set({ syncPhase: 'manifest' });
    try {
      const r = await sync.syncManifest();
      // 拉到新条目后重新扫描，把本地已存在的文件标成 present=1
      await get().scan(false);
      await get().refreshSyncStatus();
      set({ syncPhase: 'idle' });
    } catch (e) {
      set({ syncPhase: 'error', error: String(e) });
    }
  },

  downloadAll: async () => {
    set({ syncPhase: 'downloading', downloadProgress: { done: 0, total: 0 } });
    try {
      await sync.downloadAll();
    } catch (e) {
      set({ syncPhase: 'error', error: String(e) });
    }
  },
```

`reset` 里补上三个新字段的初值（`syncStatus: null`、`syncPhase: 'idle'`、`downloadProgress: null`）。

- [ ] **Step 6: 运行测试**

Run: `npm test && npm run typecheck`
Expected: 全部通过

- [ ] **Step 7: Commit**

```bash
git add src/lib/sync.ts src/lib/sync.test.ts src/types/index.ts src/stores/library.ts .gitignore
git commit -m "feat(sync): 前端同步 API 与 store 状态"
```

---

### Task 8: 设置页的知识库同步配置

**Files:**
- Create: `src/components/settings/SyncForm.tsx`
- Modify: `src/lib/tauri.ts`
- Modify: `src/routes/Settings.tsx`
- Test: `src/lib/tauri.test.ts`

- [ ] **Step 1: 写失败测试**

`src/lib/tauri.test.ts`：

```ts
import { describe, it, expect, vi, beforeEach } from 'vitest';

const state = new Map<string, unknown>();
vi.mock('@tauri-apps/plugin-store', () => ({
  load: vi.fn(async () => ({
    get: async (k: string) => state.get(k),
    set: async (k: string, v: unknown) => { state.set(k, v); },
    save: async () => {},
  })),
}));

import { getSettings, saveSettings, DEFAULTS } from './tauri';

describe('syncBaseUrl', () => {
  beforeEach(() => state.clear());

  it('defaults to empty string', async () => {
    expect((await getSettings()).syncBaseUrl).toBe('');
  });

  it('round-trips', async () => {
    await saveSettings({ syncBaseUrl: 'https://x.com/kb' });
    expect((await getSettings()).syncBaseUrl).toBe('https://x.com/kb');
  });

  it('DEFAULTS contains syncBaseUrl', () => {
    expect(DEFAULTS.syncBaseUrl).toBe('');
  });
});
```

- [ ] **Step 2: 运行确认失败**

Run: `npm test -- src/lib/tauri.test.ts`
Expected: TS/断言失败（属性不存在）

- [ ] **Step 3: `src/lib/tauri.ts` 改造**

加常量 `const KEY_SYNC_BASE_URL = 'syncBaseUrl';`，`AppSettings` 加 `syncBaseUrl: string`，`DEFAULTS` 加 `syncBaseUrl: ''`，`getSettings` 里读 `?? DEFAULTS.syncBaseUrl`，`saveSettings` 里写入。

- [ ] **Step 4: `SyncForm.tsx`**

```tsx
import { useEffect, useState } from 'react';
import { Save } from 'lucide-react';
import { getSettings, saveSettings } from '../../lib/tauri';
import { setSyncBaseUrl } from '../../lib/sync';
import { useLibraryStore } from '../../stores/library';

export default function SyncForm() {
  const [url, setUrl] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'saved' | 'error'>('idle');
  const syncNow = useLibraryStore((s) => s.syncNow);

  useEffect(() => {
    getSettings()
      .then((s) => { setUrl(s.syncBaseUrl); setStatus('idle'); })
      .catch(() => setStatus('error'));
  }, []);

  async function onSave(e: React.FormEvent) {
    e.preventDefault();
    try {
      await saveSettings({ syncBaseUrl: url });
      // 后端也要拿到它 —— read_resource 的自动下载读的是 app_config
      await setSyncBaseUrl(url);
      await syncNow();
      setStatus('saved');
      setTimeout(() => setStatus('idle'), 1500);
    } catch {
      setStatus('error');
    }
  }

  return (
    <form onSubmit={onSave} className="space-y-3">
      <div>
        <label className="mb-1 block text-sm font-medium">知识库同步地址</label>
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://your-host/kb"
          className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
        />
        <p className="mt-1 text-xs text-text-muted">
          指向存放 <code>manifest.json</code> 与 <code>knowledge/</code> 的目录。
          留空则只用本机已缓存的内容（开发模式下仍读取仓库里的 resources/knowledge）。
        </p>
      </div>
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={status === 'loading'}
          className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
        >
          <Save size={16} />
          保存并同步
        </button>
        {status === 'saved' && <span className="text-sm text-green-600">已保存并同步</span>}
        {status === 'error' && <span className="text-sm text-red-600">同步失败</span>}
      </div>
    </form>
  );
}
```

- [ ] **Step 5: 挂到 Settings**

在 `src/routes/Settings.tsx` 顶部描述下方插入：

```tsx
      <section className="mt-6">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-text-muted">
          知识库同步
        </h3>
        <SyncForm />
      </section>
```

并加 `import SyncForm from '../components/settings/SyncForm';`

- [ ] **Step 6: 运行测试**

Run: `npm test && npm run typecheck`
Expected: 通过

- [ ] **Step 7: Commit**

```bash
git add src/components/settings/SyncForm.tsx src/routes/Settings.tsx src/lib/tauri.ts src/lib/tauri.test.ts
git commit -m "feat(settings): 知识库同步地址配置"
```

---

### Task 9: 同步状态条 + 未缓存角标

**Files:**
- Create: `src/components/sync/SyncStatusBar.tsx`
- Modify: `src/components/layout/Topbar.tsx`
- Modify: `src/components/library/ArticleIndexView.tsx`
- Modify: `src/components/layout/Sidebar.tsx`
- Modify: `src/App.tsx`（启动时拉一次 manifest）
- Test: `src/components/sync/SyncStatusBar.test.tsx`

- [ ] **Step 1: 写失败测试**

`src/components/sync/SyncStatusBar.test.tsx`：

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import SyncStatusBar from './SyncStatusBar';
import { useLibraryStore } from '../../stores/library';

describe('SyncStatusBar', () => {
  it('未配置远端时不渲染', () => {
    useLibraryStore.setState({ syncStatus: { total: 0, present: 0, totalBytes: 0, cachedBytes: 0, configured: false } });
    const { container } = render(<SyncStatusBar />);
    expect(container.firstChild).toBeNull();
  });

  it('显示已缓存数量并提供下载全部', () => {
    useLibraryStore.setState({
      syncStatus: { total: 380, present: 37, totalBytes: 1000, cachedBytes: 100, configured: true },
      syncPhase: 'idle',
      downloadProgress: null,
    });
    render(<SyncStatusBar />);
    expect(screen.getByText(/37 \/ 380/)).toBeTruthy();
    expect(screen.getByRole('button', { name: '下载全部' })).toBeTruthy();
  });

  it('全部缓存后不渲染', () => {
    useLibraryStore.setState({
      syncStatus: { total: 380, present: 380, totalBytes: 1000, cachedBytes: 1000, configured: true },
    });
    const { container } = render(<SyncStatusBar />);
    expect(container.firstChild).toBeNull();
  });

  it('下载中显示进度', () => {
    useLibraryStore.setState({
      syncStatus: { total: 380, present: 37, totalBytes: 1000, cachedBytes: 100, configured: true },
      syncPhase: 'downloading',
      downloadProgress: { done: 12, total: 343 },
    });
    render(<SyncStatusBar />);
    expect(screen.getByText(/12 \/ 343/)).toBeTruthy();
  });
});
```

- [ ] **Step 2: 运行确认失败**

Run: `npm test -- src/components/sync/SyncStatusBar.test.tsx`
Expected: 模块不存在

- [ ] **Step 3: 实现组件**

```tsx
import { useEffect } from 'react';
import { CloudDownload, Loader2 } from 'lucide-react';
import { useLibraryStore } from '../../stores/library';
import { onSyncProgress } from '../../lib/sync';

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export default function SyncStatusBar() {
  const status = useLibraryStore((s) => s.syncStatus);
  const phase = useLibraryStore((s) => s.syncPhase);
  const progress = useLibraryStore((s) => s.downloadProgress);
  const downloadAll = useLibraryStore((s) => s.downloadAll);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    void onSyncProgress((p) => {
      useLibraryStore.setState({ downloadProgress: { done: p.done, total: p.total } });
      if (p.total > 0 && p.done >= p.total) {
        useLibraryStore.setState({ syncPhase: 'idle', downloadProgress: null });
        void useLibraryStore.getState().refreshSyncStatus();
      }
    }).then((u) => { unlisten = u; });
    return () => unlisten?.();
  }, []);

  if (!status || !status.configured) return null;
  if (status.total === 0 || status.present >= status.total) return null;

  const downloading = phase === 'downloading';
  return (
    <div
      className="flex items-center gap-3 border-b border-border bg-surface-2 px-4 py-1.5 text-xs text-text-muted"
      role="status"
      aria-live="polite"
    >
      <CloudDownload size={13} className="shrink-0" />
      <span className="shrink-0">
        {downloading && progress
          ? `正在下载 ${progress.done} / ${progress.total}`
          : `已缓存 ${status.present} / ${status.total}`}
        {' · '}
        {fmtBytes(status.cachedBytes)} / {fmtBytes(status.totalBytes)}
      </span>
      {!downloading && (
        <button
          type="button"
          onClick={() => void downloadAll()}
          className="ml-auto shrink-0 rounded px-2 py-1 text-accent hover:bg-accent/10"
        >
          下载全部
        </button>
      )}
      {downloading && <Loader2 size={13} className="ml-auto shrink-0 animate-spin" />}
    </div>
  );
}
```

（`SyncStatusBar.test.tsx` 里 `onSyncProgress` 依赖 `@tauri-apps/api/event`，需在测试文件顶部 `vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn(async () => () => {}) }))`。）

- [ ] **Step 4: 接入 Topbar**

`Topbar.tsx` 里在 `<header>` 的下一个位置（Layout 中）渲染。在 `Layout.tsx` 里：

```tsx
      <div className="relative flex min-w-0 flex-1 flex-col">
        <Topbar />
        <SyncStatusBar />
        <main className="flex-1 overflow-auto">
```

- [ ] **Step 5: 未缓存角标**

`ArticleIndexView.tsx` 里文章条目右侧，`formatMeta` 旁边追加：

```tsx
              {!r.present && (
                <CloudDownload size={12} className="shrink-0 text-text-muted" aria-label="未缓存" />
              )}
```

并 `import { CloudDownload } from 'lucide-react'`（合并进已有的 lucide import）。

`Sidebar.tsx` 的文章 NavLink 同理，在 `<span className="truncate">{r.title}</span>` 后加同一个图标。

- [ ] **Step 6: 启动时同步一次**

`App.tsx` 的启动序列改为：

```tsx
    void (async () => {
      await loadCachedCategories();
      await scan(false);
      // 远端清单只在配置了同步地址时才有内容；内部会自动跳过未配置的情况
      await useLibraryStore.getState().syncNow();
      await useLibraryStore.getState().refreshSyncStatus();
    })();
```

- [ ] **Step 7: 运行测试**

Run: `npm test && npm run typecheck`
Expected: 通过（含新增 4 个 SyncStatusBar 用例）

- [ ] **Step 8: Commit**

```bash
git add src/components/sync src/components/layout/Topbar.tsx src/components/layout/Layout.tsx src/components/library/ArticleIndexView.tsx src/components/layout/Sidebar.tsx src/App.tsx
git commit -m "feat(sync): 同步状态条与未缓存角标"
```

---

## Phase C — 响应式移动端 UI

### Task 10: 侧栏抽屉 + 顶栏菜单/返回按钮

**Files:**
- Create: `src/stores/ui.ts`
- Modify: `src/components/layout/Layout.tsx`
- Modify: `src/components/layout/Sidebar.tsx`
- Modify: `src/components/layout/Topbar.tsx`
- Modify: `src/components/library/NotesPanel.tsx`

- [ ] **Step 1: 写失败测试**

`src/stores/ui.test.ts`：

```ts
import { describe, it, expect, beforeEach } from 'vitest';
import { useUiStore } from './ui';

describe('ui store', () => {
  beforeEach(() => useUiStore.setState({ navOpen: false }));
  it('opens and closes the nav drawer', () => {
    useUiStore.getState().openNav();
    expect(useUiStore.getState().navOpen).toBe(true);
    useUiStore.getState().closeNav();
    expect(useUiStore.getState().navOpen).toBe(false);
  });
  it('toggles', () => {
    useUiStore.getState().toggleNav();
    expect(useUiStore.getState().navOpen).toBe(true);
    useUiStore.getState().toggleNav();
    expect(useUiStore.getState().navOpen).toBe(false);
  });
});
```

- [ ] **Step 2: 运行确认失败**

Run: `npm test -- src/stores/ui.test.ts`
Expected: 模块不存在

- [ ] **Step 3: 实现 `src/stores/ui.ts`**

```ts
import { create } from 'zustand';

interface UiState {
  /** 窄屏下侧栏抽屉是否展开 */
  navOpen: boolean;
  openNav: () => void;
  closeNav: () => void;
  toggleNav: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  navOpen: false,
  openNav: () => set({ navOpen: true }),
  closeNav: () => set({ navOpen: false }),
  toggleNav: () => set((s) => ({ navOpen: !s.navOpen })),
}));
```

- [ ] **Step 4: Layout 去掉最小宽度**

```tsx
    <div className="flex h-full min-w-0 bg-bg text-text">
      <Sidebar />
      <div className="relative flex min-w-0 flex-1 flex-col">
```

- [ ] **Step 5: Sidebar 变抽屉**

把 `<aside ...>` 的 className 换成：

```tsx
      className={`flex h-full w-72 shrink-0 flex-col border-r border-border bg-surface
        transition-transform duration-200 ease-out motion-reduce:transition-none
        max-md:fixed max-md:inset-y-0 max-md:left-0 max-md:z-50
        ${navOpen ? 'max-md:translate-x-0' : 'max-md:-translate-x-full'}`}
```

组件顶部取 `const navOpen = useUiStore((s) => s.navOpen);`，并加 `useEffect` 在 pathname 变化时关闭抽屉：

```tsx
  const location = useLocation();
  useEffect(() => { useUiStore.getState().closeNav(); }, [location.pathname]);
```

在 `<aside>` 之后（同级）渲染遮罩：

```tsx
      {navOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/40 md:hidden"
          aria-hidden="true"
          onClick={closeNav}
        />
      )}
```

（`closeNav` 从 store 取。）

- [ ] **Step 6: Topbar 加汉堡与返回**

```tsx
import { useNavigate } from 'react-router-dom';
import { ChevronLeft, Menu } from 'lucide-react';
import { useUiStore } from '../../stores/ui';

  const navigate = useNavigate();
  const toggleNav = useUiStore((s) => s.toggleNav);
  // 窄屏且打开了某篇文章时，返回按钮回到该子分类的列表
  const isMobile = useIsMobile();
  const articleRoute = /^\/library\/[^/]+\/[^/]+\/[^/]+/.test(pathname);
  const listRoute = articleRoute
    ? `/${pathname.split('/').slice(0, 4).join('/')}`
    : null;
```

在 `<header>` 内左侧 `<div className="flex min-w-0 items-center gap-3">` 的最前面插入：

```tsx
        <button
          type="button"
          onClick={toggleNav}
          aria-label="打开分类导航"
          className="-ml-2 rounded-md p-2 text-text-muted hover:bg-surface-2 md:hidden"
        >
          <Menu size={18} />
        </button>
        {isMobile && listRoute && (
          <button
            type="button"
            onClick={() => navigate(listRoute)}
            aria-label="返回列表"
            className="-ml-2 rounded-md p-2 text-text-muted hover:bg-surface-2 md:hidden"
          >
            <ChevronLeft size={18} />
          </button>
        )}
```

新增 hook `src/lib/useIsMobile.ts`：

```ts
import { useEffect, useState } from 'react';

/** 匹配 Tailwind 的 md 断点（768px）。SSR / 测试环境下默认 false = 桌面布局。 */
export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 767px)');
    const onChange = () => setIsMobile(mq.matches);
    onChange();
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return isMobile;
}
```

- [ ] **Step 7: NotesPanel 移动端底部抽屉**

`NotesPanel.tsx` 第 104-108 行的 `<aside>` className 替换为（先把类名提成导出的常量，好让 Step 8 的测试能钉住它）：

```tsx
/** 笔记面板定位：桌面 = 右侧固定栏；窄屏 = 底部抽屉。
 *  关闭态必须是 transform 移出视口，不能只改透明度 —— 否则面板会永久遮住正文。 */
export const PANEL_CLASSES = {
  base: 'fixed z-30 flex flex-col border-border bg-surface',
  desktop:
    'md:right-0 md:top-14 md:bottom-0 md:w-[320px] md:border-l md:shadow-[-4px_0_12px_rgba(0,0,0,0.04)]',
  mobile: 'max-md:inset-x-0 max-md:bottom-0 max-md:h-[70vh] max-md:rounded-t-2xl max-md:border-t',
  motion: 'transition-transform duration-200 ease-out motion-reduce:transition-none',
  open: 'translate-x-0',
  closed: 'max-md:translate-y-full pointer-events-none md:translate-x-full',
};
```

```tsx
      className={`${PANEL_CLASSES.base} ${PANEL_CLASSES.desktop} ${PANEL_CLASSES.mobile} ${PANEL_CLASSES.motion} ${
        panelOpen ? PANEL_CLASSES.open : PANEL_CLASSES.closed
      }`}
```

`aria-hidden={!panelOpen}` 保持不变。

- [ ] **Step 8: 钉住「窄屏关闭态移出视口」**

`src/components/library/NotesPanel.test.tsx`：

```tsx
import { describe, it, expect } from 'vitest';
import { PANEL_CLASSES } from './NotesPanel';

describe('NotesPanel 移动端定位', () => {
  it('窄屏是底部抽屉', () => {
    expect(PANEL_CLASSES.mobile).toContain('max-md:bottom-0');
    expect(PANEL_CLASSES.mobile).toContain('max-md:h-[70vh]');
  });
  it('关闭态把面板移出视口，不只是变透明（Review Focus #5）', () => {
    expect(PANEL_CLASSES.closed).toContain('max-md:translate-y-full');
    expect(PANEL_CLASSES.closed).toContain('pointer-events-none');
  });
  it('桌面仍是右侧固定栏', () => {
    expect(PANEL_CLASSES.desktop).toContain('md:w-[320px]');
    expect(PANEL_CLASSES.desktop).toContain('md:right-0');
  });
});
```

若 `NotesPanel` 的 import 链在测试环境里报错（reader 相关依赖），改用 `vi.mock` 挡掉即可 —— 断言的是纯常量，不依赖渲染。

- [ ] **Step 9: 运行测试**

Run: `npm test && npm run typecheck`
Expected: 通过

- [ ] **Step 10: Commit**

```bash
git add src/stores/ui.ts src/stores/ui.test.ts src/lib/useIsMobile.ts src/components/layout src/components/library/NotesPanel.tsx src/components/library/NotesPanel.test.tsx
git commit -m "feat(mobile): 侧栏抽屉、顶栏菜单/返回与笔记面板底部化"
```

---

### Task 11: 触控尺寸与 hover→active

**Files:**
- Modify: `src/components/layout/Sidebar.tsx`
- Modify: `src/components/library/ArticleIndexView.tsx`
- Modify: `src/components/layout/Topbar.tsx`

- [ ] **Step 1: 找出所有可点击目标**

Run: `grep -rn "hover:bg-surface-2\|hover:bg-accent" src/components/ | wc -l`
记录基线数字

- [ ] **Step 2: 侧栏导航项触控高度**

`Sidebar.tsx` 里分类 NavLink 的 `px-2 py-1.5` 改为 `px-2 py-2 min-h-[44px] max-md:min-h-[48px]`；文章项的 `px-2 py-1 text-xs` 改为 `px-2 py-1.5 text-xs min-h-[40px] max-md:min-h-[48px]`。展开按钮 `h-6 w-6` 改为 `h-9 w-9 max-md:h-11 max-md:w-11`。

- [ ] **Step 3: hover → hover/active 并存**

`Sidebar.tsx` 与 `ArticleIndexView.tsx` 里的 hover 类统一改为 `hover:bg-surface-2 active:bg-surface-2/70`（保留 hover，触屏用 active）。

- [ ] **Step 4: 运行测试与构建**

Run: `npm test && npm run typecheck && npm run build`
Expected: 全部通过

- [ ] **Step 5: Commit**

```bash
git add src/components/layout/Sidebar.tsx src/components/library/ArticleIndexView.tsx src/components/layout/Topbar.tsx
git commit -m "feat(mobile): 触控目标尺寸与 active 反馈"
```

---

## Phase D — Android 构建与交付

### Task 12: 平台隔离（window-state / capabilities / gitignore）

**Files:**
- Modify: `src-tauri/Cargo.toml`
- Modify: `src-tauri/src/lib.rs`
- Modify: `src-tauri/capabilities/default.json`
- Modify: `.gitignore`

- [ ] **Step 1: Cargo.toml 把 window-state 移出主依赖**

从 `[dependencies]` 删掉 `tauri-plugin-window-state = "2.0"`，在文件末尾加：

```toml
# window-state 是桌面独占插件（不支持 Android/iOS）。
# 放在 target 段里，交叉编译到 Android 时 Cargo 根本不会解析它。
[target.'cfg(not(any(target_os = "android", target_os = "ios")))'.dependencies]
tauri-plugin-window-state = "2.0"
```

- [ ] **Step 2: lib.rs 条件注册**

```rust
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_sql::Builder::default().build());
    // window-state 桌面独占：Android/iOS 上单窗口模型没有窗口位置可保存
    #[cfg(desktop)]
    let builder = builder.plugin(tauri_plugin_window_state::Builder::default().build());
    builder
        .setup(|app| { /* ... */ })
        .invoke_handler(tauri::generate_handler![/* ... */])
        .run(tauri::generate_context!())
        .expect("error while running ai-stack application");
}
```

- [ ] **Step 3: capabilities 补资源读取 scope**

在 `permissions` 数组里把 `"fs:default"` 换成：

```json
    "fs:default",
    {
      "identifier": "fs:allow-read-file",
      "allow": [{ "path": "$RESOURCE/**" }]
    },
```

- [ ] **Step 4: gitignore 放行 gen/android**

`.gitignore` 里把 `src-tauri/gen/` 改成：

```
src-tauri/gen/schemas/
```

其余（`src-tauri/gen/android/`）纳入版本管理 —— `AndroidManifest.xml` 的权限改动不能每次 init 后重做。

- [ ] **Step 5: 桌面编译验证**

Run: `cd src-tauri && cargo check 2>&1 | tail -10`
Expected: 无错误

- [ ] **Step 6: Commit**

```bash
git add src-tauri/Cargo.toml src-tauri/src/lib.rs src-tauri/capabilities/default.json .gitignore
git commit -m "build: window-state 桌面隔离、fs 资源 scope、gen/android 纳入版本管理"
```

---

### Task 13: 安装 Android 工具链并生成 Android 工程

**Files:**
- Create: `src-tauri/gen/android/**`（由 `tauri android init` 生成）
- Modify: `src-tauri/gen/android/app/src/main/AndroidManifest.xml`（INTERNET 权限）

- [ ] **Step 1: 安装 JDK 17**

到 https://adoptium.net/ 下载 Temurin 17 (Windows x64 MSI) 安装。

设置用户环境变量（PowerShell，**新开终端**后生效）：
```powershell
[Environment]::SetEnvironmentVariable('JAVA_HOME', 'C:\Program Files\Eclipse Adoptium\jdk-17.0.13.11-hotspot', 'User')
[Environment]::SetEnvironmentVariable('ANDROID_HOME', "$env:LOCALAPPDATA\Android\Sdk", 'User')
[Environment]::SetEnvironmentVariable('ANDROID_SDK_ROOT', "$env:LOCALAPPDATA\Android\Sdk", 'User')
```

验证：`java -version` 应显示 17。

- [ ] **Step 2: 安装 Android SDK 命令行工具**

下载 commandlinetools-win.zip，解压到 `%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\`（注意 `latest` 这层目录必须存在）。

```powershell
sdkmanager --licenses
sdkmanager "platform-tools" "platforms;android-35" "build-tools;35.0.0"
```

- [ ] **Step 3: 安装 NDK**

Tauri 2 要求 NDK r27 系列。先装 r27，再按 `tauri android build` 的实际报错调整：

```powershell
sdkmanager "ndk;27.1.12297006"
[Environment]::SetEnvironmentVariable('NDK_HOME', "$env:LOCALAPPDATA\Android\Sdk\ndk\27.1.12297006", 'User')
```

若 `tauri android dev/build` 提示版本不符，按提示改装对应版本，并把 `NDK_HOME` 指向它。

- [ ] **Step 4: 装 Rust Android target**

```bash
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
```

- [ ] **Step 5: 生成种子库**

```bash
npm run gen:seed
ls src-tauri/seed/knowledge
```
Expected: 每个二级分类下若干 `_index.md`

- [ ] **Step 6: 生成 Android 工程**

```bash
npm run tauri android init
```
Expected: 生成 `src-tauri/gen/android/`

- [ ] **Step 7: 加 INTERNET 权限**

打开 `src-tauri/gen/android/app/src/main/AndroidManifest.xml`，在 `<manifest>` 下（`<application>` 之前）加入：

```xml
    <uses-permission android:name="android.permission.INTERNET" />
```

- [ ] **Step 8: 验证**

```bash
git status --short
```
Expected: `src-tauri/gen/android/` 下的新文件（Task 12 已放行 gitignore）

- [ ] **Step 9: Commit**

```bash
git add src-tauri/gen/android .gitignore
git commit -m "build(android): 生成 Android 工程并添加 INTERNET 权限"
```

---

### Task 14: 构建 debug APK 并装机验证

**Files:** 无代码改动（验证任务）

- [ ] **Step 1: 构建 arm64 debug APK**

```bash
npm run tauri android build -- --debug --apk --target aarch64
```

首次会编译 `rusqlite` 的 SQLite C 代码，**耗时 10-30 分钟**属正常。产物：

```
src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-universal-debug.apk
```

若报错指向 NDK 版本或缺失 target，按错误信息修好后重跑。

- [ ] **Step 2: 装机验证（真机）**

```bash
adb install -r src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-universal-debug.apk
adb logcat -s ai_stack | tail -50
```

依次验证：

1. 首启后 seed 已释放：`adb shell run-as com.aistack.app ls files/knowledge`
2. 打开 App，设置里填入 `http://<你的电脑IP>:8787/`（电脑跑 `npm run serve:dist`），点「保存并同步」→ 列表出现全部文章
3. 点一篇**未缓存**的文章 → 自动下载 → 正常打开
4. 关掉电脑上的静态服务器 → 点另一篇未缓存文章 → 显示中文失败提示，不是崩溃
5. 重新联网后打开已缓存文章 → 正常

- [ ] **Step 3: 模拟器验证（可选）**

```bash
npm run tauri android dev -- --target aarch64
```
模拟器需 x86_64 系统镜像时加 `--target x86_64`（需先 `rustup target add x86_64-linux-android`）。

- [ ] **Step 4: Release 签名包**

```bash
keytool -genkey -v -keystore ~/ai-stack-release.jks -keyalg RSA -keysize 2048 -validity 10000 -alias ai-stack
```
把口令写进 `src-tauri/tauri.android.conf.json`（**不要提交到 git**）：
```json
{ "bundle": { "android": { "keystorePath": "C:/Users/<你>/ai-stack-release.jks", "password": "…", "alias": "…", "keyPassword": "…" } } }
```

```bash
npm run tauri android build -- --apk --target aarch64
```

- [ ] **Step 5: 记录 APK 路径与体积**

把最终 APK 路径和体积写进 README（Task 15）。

---

### Task 15: 文档

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`

- [ ] **Step 1: README 更新构建章节**

把「构建」段替换为：

````markdown
## 构建

### 桌面端

```bash
npm run tauri build    # 产出安装包到 src-tauri/target/release/bundle/
```

安装包**不再包含知识库**。安装后首次启动需要知识库时，去设置页填「知识库同步地址」。

### 知识库发布

```bash
npm run gen:manifest   # 输出 resources/dist/{manifest.json,knowledge/}
npm run serve:dist     # 本地静态服务器 http://127.0.0.1:8787/ 验证用
```

把 `resources/dist/` 整个传到任意静态托管（Nginx / 对象存储 / GitHub Release）。
设置页填的地址就是这个目录的 URL 根。

### Android

前置：JDK 17、Android SDK（cmdline-tools / platform-tools / build-tools 35 / NDK r27）、
`rustup target add aarch64-linux-android armv7-linux-androideabi`。

```bash
npm run gen:seed                       # 生成 APK 内置的种子库（只有各分类 _index.md）
npm run tauri android init             # 生成 src-tauri/gen/android（已提交，勿删）
npm run tauri android build -- --debug --apk --target aarch64
```

产物：`src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-universal-debug.apk`

签名发布见 `docs/superpowers/plans/2026-10-08-android-support.md` Task 14。
````

- [ ] **Step 2: README 顶部定位更新**

把「Windows 平台上的 AI 学习工作台」改成「Windows + Android 平台上的 AI 学习工作台」，并在「状态」里补一句知识库改为远程分发的说明。

- [ ] **Step 3: architecture.md 补一节**

追加「## 知识库分发」小节，描述：`knowledge/` 恒为真实目录、manifest → `resources` 表占位行、`read_resource` 自动补齐、`present` 语义、`.tmp` 目录为何在 `knowledge/` 之外。内容与 spec §4/§7 保持一致。

- [ ] **Step 4: 提交**

```bash
git add README.md docs/architecture.md
git commit -m "docs: Android 构建与知识库远程分发说明"
```

---

## 收尾验收

全部任务完成后，逐条跑一遍：

```bash
npm run typecheck
npm test
node --test scripts/gen-remote-manifest.test.mjs
cd src-tauri && cargo test && cd ..
npm run build
npm run tauri build
```

并人工确认：

- [ ] 桌面端开发模式（不配 syncBaseUrl）仍能用 `resources/knowledge/` 工作
- [ ] 桌面端安装后（不配 syncBaseUrl）知识库为空且**不报错**
- [ ] 配好 syncBaseUrl 后桌面端能拉到并打开远端文章
- [ ] Android APK 装机后 §Task 14 的 5 条验证项全过
- [ ] 断网时已缓存文章仍可读，未缓存文章给出中文提示
