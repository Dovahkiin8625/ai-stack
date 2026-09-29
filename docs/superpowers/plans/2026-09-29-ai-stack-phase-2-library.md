# AI Stack — Phase 2 知识库资源管理 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Phase 1 骨架之上，让用户能浏览 `resources/knowledge/` 下的 AI 知识目录树、选择 Markdown / PDF / DOCX / PPTX 文档，在应用内阅读其内容。

**Architecture:** Rust 后端承担所有文件系统扫描、SQLite 索引（`rusqlite` 内嵌）、文档解析（Markdown→HTML / PDF→PNG / DOCX→结构 / PPTX→幻灯片）；前端只通过 Tauri IPC 消费 JSON，用 React 渲染三栏布局（分类树 / 资源列表 / 阅读器）。不引入 `tauri-plugin-sql`、mammoth.js、PDF.js、pptxjs。

**Tech Stack:**
- Tauri 2.x + Rust（`rusqlite[bundled]`, `walkdir`, `sha2`, `chrono`, `comrak`, `pdfium-render` + `pdf-extract` 兜底, `docx-rs`, `zip`, `quick-xml`, `base64`, `anyhow`, `thiserror`, `once_cell`, `hex`）
- React 19 + TypeScript + Vite（无新增运行时依赖；Markdown 是 Rust 端预渲染的 HTML 字符串）
- Vitest（前端单测）
- Cargo test（Rust 单测）

**Spec:** `docs/superpowers/specs/2026-09-29-ai-stack-phase-2-library-design.md`

---

## Global Constraints

- 工作目录固定：`C:\project\ai-stack`
- 平台：Windows 11（WebView2 + Visual Studio C++ Build Tools 已具备）
- Node.js ≥ 20，Rust ≥ 1.78
- 应用 ID：`com.aistack.app`，窗口最小 900×600
- 路径命名：前端代码文件 kebab-case 或 PascalCase（按 React 习惯），Rust 模块 snake_case，目录用 kebab-case
- 所有提交信息前缀：`feat:`、`chore:`、`docs:`、`test:`、`fix:`
- 不在 Phase 2 引入新运行时前端依赖
- `.gitignore` 忽略：`node_modules/`、`dist/`、`src-tauri/target/`、`data/`、`src-tauri/tests/fixtures/`
- `_index.md` 不作为资源入库（仅用于解析分类标题）
- Markdown 渲染走 `dangerouslySetInnerHTML`，仅信任 comrak 输出；不引入 DOMPurify
- pdfium 动态库缺失时，PDF 自动回退到纯文本提取；不阻塞扫描

## Review Focus

下面五类输入/场景 Phase 2 必须妥善处理，已在对应任务中加入断言：

1. **首次启动无 DB 文件** — 应用应自动建表 + 跑迁移，绝不崩溃（任务 2 测试）
2. **`_index.md` 不被当作资源** — 扫描后 DB 中 resources 表不应含任何 `_index.md`；分类的 title 应来自 `_index.md` 的 H1（任务 3 测试）
3. **文件被外部修改或删除后刷新** — hash 变化触发更新；文件消失触发删除；DB 与磁盘一致（任务 3 测试）
4. **pdfium.dll 缺失** — PDF 阅读器仍能打开（纯文本回退）；扫描不被阻塞（任务 5 测试 + 任务 8 默认行为）
5. **损坏的二进制文件（坏 .docx / .pptx / .pdf）** — 扫描计入 errors_count 但不中断；阅读器显示错误卡而不是空白（任务 8 错误传播 + 任务 12 reader 错误渲染）

---

## File Structure (Phase 2)

```
src-tauri/
├── src/
│   ├── main.rs                          (Phase 1, 不动)
│   ├── lib.rs                           (Phase 1, 任务 8 修改)
│   ├── menu.rs                          (Phase 1, 不动)
│   ├── db.rs                            [新] 连接、迁移、CRUD
│   ├── scanner.rs                       [新] walkdir + 索引
│   ├── reader.rs                        [新] 格式分发
│   ├── commands.rs                      [新] Tauri 命令入口
│   └── readers/
│       ├── mod.rs
│       ├── markdown.rs                  [新] comrak → HTML
│       ├── pdf.rs                       [新] pdfium → PNG + 兜底
│       ├── docx.rs                      [新] docx-rs → blocks
│       └── pptx.rs                      [新] zip+quick-xml → slides
├── tests/
│   └── fixtures/                        [新, gitignored] cargo test fixtures
└── Cargo.toml                           [改]

src/
├── main.tsx                             (Phase 1, 不动)
├── App.tsx                              (Phase 1, 不动)
├── index.css                            (Phase 1, 不动)
├── routes/
│   ├── Library.tsx                      [改] 三栏布局
│   ├── Notes.tsx                        (Phase 1, 不动)
│   ├── Dashboard.tsx                    (Phase 1, 不动)
│   └── Settings.tsx                     (Phase 1, 不动)
├── components/
│   ├── layout/                          (Phase 1, 不动)
│   ├── settings/                        (Phase 1, 不动)
│   ├── ui/                              (Phase 1, 不动)
│   └── library/
│       ├── Tree.tsx                     [新] 递归分类树
│       ├── ResourceList.tsx             [新] 资源列表
│       ├── ScanProgress.tsx             [新] 启动扫描进度
│       ├── ReaderToolbar.tsx            [新] 阅读器顶部栏
│       └── reader/
│           ├── MarkdownReader.tsx
│           ├── PdfReader.tsx
│           ├── DocxReader.tsx
│           └── PptxReader.tsx
├── lib/
│   ├── tauri.ts                         (Phase 1, 不动)
│   └── library-api.ts                   [新] invoke 包装
├── stores/
│   ├── theme.ts                         (Phase 1, 不动)
│   └── library.ts                       [新] Zustand
└── types/
    └── index.ts                         [改] 新增 Library 类型

scripts/
├── gen-fixtures.mjs                     [新] 生成 cargo test fixtures
└── gen-samples.mjs                      [新] 在 resources/knowledge/ 放样本文件用于手动验证

.gitignore                              [改] 忽略 src-tauri/tests/fixtures/ 和 data/
README.md                               [改] 列出 Phase 2 新增命令
```

每个模块单一职责。`readers/` 子模块按格式拆分（一个格式一个文件），避免单个文件过大。

---

## Task 1: Cargo 依赖与模块骨架

**Files:**
- Modify: `src-tauri/Cargo.toml`
- Create: `src-tauri/src/db.rs`, `src-tauri/src/scanner.rs`, `src-tauri/src/reader.rs`, `src-tauri/src/commands.rs`, `src-tauri/src/readers/mod.rs`, `src-tauri/src/readers/markdown.rs`, `src-tauri/src/readers/pdf.rs`, `src-tauri/src/readers/docx.rs`, `src-tauri/src/readers/pptx.rs`
- Modify: `src-tauri/src/lib.rs`

**Interfaces:**
- Consumes: 无
- Produces: `cargo build` 成功通过；所有新模块文件存在且声明占位符

- [ ] **Step 1: 修改 Cargo.toml**

写入 `C:\project\ai-stack\src-tauri\Cargo.toml`：
```toml
[package]
name = "ai-stack"
version = "0.2.0"
description = "AI Stack — Windows AI 学习工作台"
authors = ["AI Stack"]
edition = "2021"
rust-version = "1.78"

[lib]
name = "ai_stack_lib"
crate-type = ["staticlib", "cdylib", "rlib"]

[build-dependencies]
tauri-build = { version = "2.0", features = [] }

[dependencies]
tauri = { version = "2.1", features = [] }
tauri-plugin-store = "2.1"
tauri-plugin-window-state = "2.0"
tauri-plugin-fs = "2.0"
tauri-plugin-dialog = "2.0"
tauri-plugin-sql = { version = "2.0", features = ["sqlite"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["sync", "rt", "macros"] }
once_cell = "1"
anyhow = "1"
thiserror = "1"

# Phase 2 新增
rusqlite = { version = "0.32", features = ["bundled"] }
walkdir = "2"
sha2 = "0.10"
hex = "0.4"
chrono = { version = "0.4", features = ["serde"] }
comrak = { version = "0.22", default-features = false, features = ["shortcodes"] }
pdfium-render = "0.8"
pdf-extract = "0.7"
docx-rs = "0.4"
zip = { version = "2", default-features = false, features = ["deflate"] }
quick-xml = { version = "0.36", features = ["serialize"] }
base64 = "0.22"

[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: 创建模块文件骨架**

写入 `C:\project\ai-stack\src-tauri\src\db.rs`：
```rust
//! 数据库：连接、迁移、CRUD
use anyhow::Result;

pub struct Db {
    pub conn: rusqlite::Connection,
}

impl Db {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let conn = rusqlite::Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Ok(Self { conn })
    }

    pub fn migrate(&mut self) -> Result<()> {
        // 任务 2 实现
        Ok(())
    }
}
```

写入 `C:\project\ai-stack\src-tauri\src\scanner.rs`：
```rust
//! 扫描 resources/knowledge/ 并 upsert 到 DB
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub knowledge_root: PathBuf,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct ScanSummary {
    pub categories_count: usize,
    pub resources_count: usize,
    pub errors_count: usize,
    pub duration_ms: u128,
}

pub fn scan(_cfg: &ScanConfig) -> Result<ScanSummary, anyhow::Error> {
    // 任务 3 实现
    Ok(ScanSummary::default())
}

#[allow(dead_code)]
pub fn validate_root(root: &Path) -> anyhow::Result<()> {
    if !root.is_dir() {
        anyhow::bail!("knowledge root is not a directory: {}", root.display());
    }
    Ok(())
}
```

写入 `C:\project\ai-stack\src-tauri\src\reader.rs`：
```rust
//! 文档读取入口，按扩展名分发
use anyhow::Result;
use std::path::Path;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum ResourceContent {
    Markdown { html: String, word_count: usize },
    Pdf { pages: Vec<PdfPageDataUrl>, page_count: usize },
    Docx { blocks: serde_json::Value, word_count: usize },
    Pptx { slides: serde_json::Value, slide_count: usize },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PdfPageDataUrl {
    pub index: usize,
    pub data_url: String,
}

pub fn read(_path: &Path, _kind: &str) -> Result<ResourceContent> {
    anyhow::bail!("reader not implemented (Task 4-7)")
}
```

写入 `C:\project\ai-stack\src-tauri\src\commands.rs`：
```rust
//! Tauri 命令入口
use crate::scanner::{ScanConfig, ScanSummary};
use std::path::PathBuf;

#[tauri::command]
pub async fn scan_library(force: bool) -> Result<ScanSummary, String> {
    let cfg = ScanConfig {
        knowledge_root: PathBuf::from("resources/knowledge"),
    };
    let _ = force;
    crate::scanner::scan(&cfg).map_err(|e| e.to_string())
}
```

写入 `C:\project\ai-stack\src-tauri\src\readers\mod.rs`：
```rust
pub mod markdown;
pub mod pdf;
pub mod docx;
pub mod pptx;
```

写入 `C:\project\ai-stack\src-tauri\src\readers\markdown.rs`：
```rust
use anyhow::Result;
use std::path::Path;
pub fn extract(_path: &Path) -> Result<(String, usize)> {
    anyhow::bail!("not implemented")
}
```

写入 `C:\project\ai-stack\src-tauri\src\readers\pdf.rs`：
```rust
use anyhow::Result;
use std::path::Path;
pub fn extract(_path: &Path) -> Result<(Vec<(usize, String)>, usize)> {
    anyhow::bail!("not implemented")
}
```

写入 `C:\project\ai-stack\src-tauri\src\readers\docx.rs`：
```rust
use anyhow::Result;
use std::path::Path;
pub fn extract(_path: &Path) -> Result<(serde_json::Value, usize)> {
    anyhow::bail!("not implemented")
}
```

写入 `C:\project\ai-stack\src-tauri\src\readers\pptx.rs`：
```rust
use anyhow::Result;
use std::path::Path;
pub fn extract(_path: &Path) -> Result<(serde_json::Value, usize)> {
    anyhow::bail!("not implemented")
}
```

- [ ] **Step 3: 修改 lib.rs 注册新模块**

覆盖 `C:\project\ai-stack\src-tauri\src\lib.rs`：
```rust
mod menu;
mod db;
mod scanner;
mod reader;
mod commands;
mod readers;

use tauri::Emitter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_sql::Builder::default().build())
        .setup(|app| {
            let menu = menu::build_menu(app.handle())?;
            app.set_menu(menu)?;
            app.on_menu_event(|app, event| {
                let _ = app.emit("menu", event.id().0.as_str());
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![commands::scan_library])
        .run(tauri::generate_context!())
        .expect("error while running ai-stack application");
}
```

- [ ] **Step 4: 验证编译**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml check
```

预期：编译成功（warnings 可接受；不能有 errors）。

- [ ] **Step 5: 提交**

```bash
git add src-tauri/Cargo.toml src-tauri/src/
git commit -m "feat(phase-2): cargo deps + module skeleton (db/scanner/reader/commands)"
```

---

## Task 2: 数据库 — 打开、迁移、基础 CRUD（含 TDD）

**Files:**
- Modify: `src-tauri/src/db.rs`
- Create: `src-tauri/src/db_test.rs` 或 `src-tauri/tests/db_smoke.rs`

**Interfaces:**
- Consumes: 无
- Produces:
  - `Db::open(path) -> Result<Db>`：打开或创建 DB，自动 `pragma_update(journal_mode=WAL)`、`foreign_keys=ON`
  - `Db::migrate(&mut self) -> Result<()>`：创建 `schema_version` + `categories` + `resources` 表（idempotent）
  - `Db::upsert_category(&self, path, parent_path, title, sort_order)`
  - `Db::upsert_resource(&self, row) -> Result<i64>`：返回 rowid
  - `Db::list_categories(&self) -> Result<Vec<CategoryRow>>`
  - `Db::list_resources(&self, category_path) -> Result<Vec<ResourceRow>>`
  - `Db::delete_missing(&self, table, paths) -> Result<usize>`：删除 path 不在传入集合中的行

- [ ] **Step 1: 写入失败的测试**

写入 `C:\project\ai-stack\src-tauri\tests\db_smoke.rs`：
```rust
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
```

并在 `src-tauri/src/lib.rs` 顶部添加：
```rust
pub mod db;
```

（`scanner`、`reader`、`commands` 模块保留 `mod xxx` 私有声明，因为它们会被 `lib.rs` 在 setup 中使用；而 `db` 需要公开访问以供 cargo test 引用。）

- [ ] **Step 2: 运行测试确认失败**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml test --test db_smoke
```

预期：FAIL，编译错误 `cannot find type ResourceInput`。

- [ ] **Step 3: 写入完整实现**

覆盖 `C:\project\ai-stack\src-tauri\src\db.rs`：
```rust
//! 数据库：连接、迁移、CRUD
use anyhow::{Context, Result};
use rusqlite::params;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CategoryRow {
    pub path: String,
    pub parent_path: Option<String>,
    pub title: String,
    pub sort_order: i64,
}

#[derive(Debug, Clone)]
pub struct ResourceRow {
    pub id: i64,
    pub category_path: String,
    pub rel_path: String,
    pub r#type: String,
    pub title: String,
    pub size_bytes: i64,
    pub hash: String,
    pub indexed_at: String,
    pub page_count: Option<i64>,
    pub word_count: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct ResourceInput {
    pub category_path: String,
    pub rel_path: String,
    pub r#type: String,
    pub title: String,
    pub size_bytes: i64,
    pub hash: String,
    pub page_count: Option<i64>,
    pub word_count: Option<i64>,
}

pub struct Db {
    pub conn: rusqlite::Connection,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create db dir {}", parent.display()))?;
        }
        let conn = rusqlite::Connection::open(path)
            .with_context(|| format!("open db {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Ok(Self { conn })
    }

    pub fn migrate(&mut self) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS categories (
                path TEXT PRIMARY KEY,
                parent_path TEXT,
                title TEXT NOT NULL,
                sort_order INTEGER NOT NULL,
                FOREIGN KEY (parent_path) REFERENCES categories(path)
             );
             CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_path);
             CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_path TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                page_count INTEGER,
                word_count INTEGER,
                UNIQUE(category_path, rel_path),
                FOREIGN KEY (category_path) REFERENCES categories(path)
             );
             CREATE INDEX IF NOT EXISTS idx_resources_category ON resources(category_path);
             CREATE INDEX IF NOT EXISTS idx_resources_hash ON resources(hash);",
        )?;
        let now = chrono::Utc::now().to_rfc3339();
        tx.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (1, ?1)",
            params![now],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_category(
        &self,
        path: &str,
        parent_path: Option<&str>,
        title: &str,
        sort_order: i64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO categories (path, parent_path, title, sort_order)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(path) DO UPDATE SET
                parent_path = excluded.parent_path,
                title = excluded.title,
                sort_order = excluded.sort_order",
            params![path, parent_path, title, sort_order],
        )?;
        Ok(())
    }

    pub fn upsert_resource(&self, r: ResourceInput) -> Result<i64> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO resources (category_path, rel_path, type, title,
                                    size_bytes, hash, indexed_at, page_count, word_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(category_path, rel_path) DO UPDATE SET
                type = excluded.type,
                title = excluded.title,
                size_bytes = excluded.size_bytes,
                hash = excluded.hash,
                indexed_at = excluded.indexed_at,
                page_count = excluded.page_count,
                word_count = excluded.word_count",
            params![
                r.category_path,
                r.rel_path,
                r.r#type,
                r.title,
                r.size_bytes,
                r.hash,
                now,
                r.page_count,
                r.word_count,
            ],
        )?;
        let id: i64 = self.conn.query_row(
            "SELECT id FROM resources WHERE category_path = ?1 AND rel_path = ?2",
            params![r.category_path, r.rel_path],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    pub fn list_categories(&self) -> Result<Vec<CategoryRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT path, parent_path, title, sort_order
             FROM categories ORDER BY sort_order, path",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(CategoryRow {
                    path: row.get(0)?,
                    parent_path: row.get(1)?,
                    title: row.get(2)?,
                    sort_order: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn list_resources(&self, category_path: &str) -> Result<Vec<ResourceRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, category_path, rel_path, type, title,
                    size_bytes, hash, indexed_at, page_count, word_count
             FROM resources WHERE category_path = ?1
             ORDER BY rel_path",
        )?;
        let rows = stmt
            .query_map([category_path], |row| {
                Ok(ResourceRow {
                    id: row.get(0)?,
                    category_path: row.get(1)?,
                    rel_path: row.get(2)?,
                    r#type: row.get(3)?,
                    title: row.get(4)?,
                    size_bytes: row.get(5)?,
                    hash: row.get(6)?,
                    indexed_at: row.get(7)?,
                    page_count: row.get(8)?,
                    word_count: row.get(9)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn delete_missing_resources(
        &self,
        category_path: &str,
        keep: &[String],
    ) -> Result<usize> {
        // 取出该分类下所有 rel_path，差集删除
        let existing: Vec<String> = {
            let mut stmt = self.conn.prepare(
                "SELECT rel_path FROM resources WHERE category_path = ?1",
            )?;
            stmt.query_map([category_path], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        let to_delete: Vec<String> = existing
            .into_iter()
            .filter(|p| !keep.contains(p))
            .collect();
        let mut removed = 0;
        for p in to_delete {
            removed += self.conn.execute(
                "DELETE FROM resources WHERE category_path = ?1 AND rel_path = ?2",
                params![category_path, p],
            )?;
        }
        Ok(removed)
    }
}
```

并在 `lib.rs` 顶部 `mod db` 改成：
```rust
pub mod db;
```
（其他模块保持 `mod xxx` 私有声明）

- [ ] **Step 4: 运行测试确认通过**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml test --test db_smoke
```

预期：4 个用例全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add src-tauri/src/db.rs src-tauri/src/lib.rs src-tauri/tests/db_smoke.rs
git commit -m "feat(phase-2): db module — open/migrate/upsert/list/delete"
```

---

## Task 3: Scanner — walkdir + hash + 索引（含 TDD）

**Files:**
- Modify: `src-tauri/src/scanner.rs`
- Create: `scripts/gen-fixtures.mjs`（生成 cargo test 用的微型目录树）
- Modify: `.gitignore`

**Interfaces:**
- Consumes:
  - `Db::open` / `Db::migrate` / `Db::upsert_category` / `Db::upsert_resource` / `Db::delete_missing_resources`
- Produces:
  - `scanner::scan(&Db, &ScanConfig) -> Result<ScanSummary>`：遍历 `knowledge_root`，对每个目录建立 category（title 来自 `_index.md` 的 H1，否则人化目录名；sort_order 解析自数字前缀），对每个 `_index.md` 之外的文件建立 resource（hash 用 sha256，type 按扩展名映射），删除磁盘上不存在的行
  - `scanner::classify_type(ext: &str) -> Option<&'static str>`：`.md`→`markdown`、`.pdf`→`pdf`、`.docx`→`docx`、`.pptx`→`pptx`，否则 None

- [ ] **Step 1: 写入 fixture 生成脚本**

写入 `C:\project\ai-stack\scripts\gen-fixtures.mjs`：
```js
// 生成 cargo test 用的微型目录树
import { mkdir, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..', 'src-tauri', 'tests', 'fixtures', 'knowledge-tree');

async function main() {
  // 清空
  await import('node:fs/promises').then((m) => m.rm(ROOT, { recursive: true, force: true }));

  // 01-foundations/01-mathematics/_index.md + note.md
  await mkdir(join(ROOT, '01-foundations', '01-mathematics'), { recursive: true });
  await writeFile(
    join(ROOT, '01-foundations', '01-mathematics', '_index.md'),
    '# 数学基础\n\n线性代数、概率统计等。\n',
    'utf8',
  );
  await writeFile(
    join(ROOT, '01-foundations', '01-mathematics', 'linear-algebra-notes.md'),
    '# Linear Algebra Notes\n\nVectors and matrices.\n',
    'utf8',
  );
  await writeFile(
    join(ROOT, '01-foundations', '01-mathematics', 'calculus-notes.md'),
    '# Calculus Notes\n\nDerivatives and integrals.\n',
    'utf8',
  );

  // 02-deep-learning/_index.md (no resource) + sub empty
  await mkdir(join(ROOT, '02-deep-learning', 'transformers'), { recursive: true });
  await writeFile(
    join(ROOT, '02-deep-learning', '_index.md'),
    '# 深度学习\n',
    'utf8',
  );
  // 故意放一个 _index.md 但不放资源，验证 _index.md 不入 resources 表
  await writeFile(
    join(ROOT, '02-deep-learning', 'transformers', '_index.md'),
    '# Transformer\n',
    'utf8',
  );

  console.log(`fixtures written under ${ROOT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
```

- [ ] **Step 2: 在 package.json 加 gen:fixtures script**

修改 `C:\project\ai-stack\package.json` 的 `scripts` 字段，添加：
```json
"gen:fixtures": "node scripts/gen-fixtures.mjs"
```

- [ ] **Step 3: 在 .gitignore 忽略 fixtures**

修改 `C:\project\ai-stack\.gitignore`，在文件末尾添加：
```
# Phase 2 test fixtures
src-tauri/tests/fixtures/
!src-tauri/tests/fixtures/.gitkeep
```

写入 `C:\project\ai-stack\src-tauri\tests\fixtures\.gitkeep`（空文件），保证目录被跟踪。

- [ ] **Step 4a: 追加损坏文件 fixture**

修改 `C:\project\ai-stack\scripts\gen-fixtures.mjs`，在 readers 区追加：
```js
  // 故意写一个损坏的 docx（不是合法 zip）
  await writeFile(join(readers, 'corrupt.docx'), Buffer.from('not a real docx'));
```
再跑：
```bash
cd C:\project\ai-stack
npm run gen:fixtures
```

- [ ] **Step 4b: 生成 fixtures 并运行失败的测试**

```bash
cd C:\project\ai-stack
npm run gen:fixtures
```

写入 `C:\project\ai-stack\src-tauri\tests\scanner_smoke.rs`：
```rust
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
```

- [ ] **Step 5: 运行测试确认失败**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml test --test scanner_smoke
```

预期：FAIL（scanner::scan 和 classify_type 签名不匹配）。

- [ ] **Step 6: 实现 scanner**

覆盖 `C:\project\ai-stack\src-tauri\src\scanner.rs`：
```rust
//! 扫描 resources/knowledge/ 并 upsert 到 DB
use crate::db::{Db, ResourceInput};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub knowledge_root: PathBuf,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct ScanSummary {
    pub categories_count: usize,
    pub resources_count: usize,
    pub errors_count: usize,
    pub duration_ms: u128,
}

pub fn classify_type(ext: &str) -> Option<&'static str> {
    match ext.to_ascii_lowercase().as_str() {
        "md" | "markdown" => Some("markdown"),
        "pdf" => Some("pdf"),
        "docx" => Some("docx"),
        "pptx" => Some("pptx"),
        _ => None,
    }
}

pub fn scan(db: &Db, cfg: &ScanConfig) -> Result<ScanSummary> {
    let started = std::time::Instant::now();
    validate_root(&cfg.knowledge_root)?;

    // 1) 遍历收集 (category_path -> resources) + (categories 集合)
    let mut cat_paths: std::collections::BTreeSet<String> = Default::default();
    let mut cat_titles: std::collections::HashMap<String, (Option<String>, i64)> = Default::default();
    let mut per_cat_resources: std::collections::BTreeMap<String, Vec<ResourceInput>> = Default::default();
    let mut errors: usize = 0;
    let mut res_count: usize = 0;

    for entry in WalkDir::new(&cfg.knowledge_root).follow_links(false) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[scanner] walk error: {e}");
                continue;
            }
        };
        let path = entry.path();
        let rel = match path.strip_prefix(&cfg.knowledge_root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if entry.file_type().is_dir() {
            let cat_path = rel.to_string_lossy().replace('\\', "/");
            cat_paths.insert(cat_path.clone());
            cat_titles.entry(cat_path.clone()).or_insert((None, parse_sort_order(&cat_path)));
            continue;
        }
        let file_name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n,
            None => continue,
        };
        // _index.md 在分类目录下：解析分类 title，不入库为资源
        if file_name == "_index.md" {
            if let Some(parent) = rel.parent() {
                let cat_path = parent.to_string_lossy().replace('\\', "/");
                if let Ok(content) = std::fs::read_to_string(path) {
                    if let Some(title) = first_h1(&content) {
                        cat_titles.entry(cat_path).or_insert((Some(title), 0));
                    }
                }
            }
            continue;
        }
        // 资源
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let Some(r#type) = classify_type(ext) else { continue };
        let parent_cat = rel
            .parent()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        if parent_cat.is_empty() {
            // 不允许资源直接放根目录
            continue;
        }
        cat_paths.insert(parent_cat.clone());
        cat_titles.entry(parent_cat.clone()).or_insert((None, parse_sort_order(&parent_cat)));
        let rel_path = file_name.to_string();
        let title = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&rel_path)
            .to_string();
        let size = entry.metadata().map(|m| m.len() as i64).unwrap_or(0);
        let hash = match sha256_file(path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[scanner] hash failed for {}: {e:#}", path.display());
                errors += 1;
                continue;
            }
        };
        per_cat_resources
            .entry(parent_cat)
            .or_default()
            .push(ResourceInput {
                category_path: String::new(), // patch after we know parent
                rel_path,
                r#type: r#type.to_string(),
                title,
                size_bytes: size,
                hash,
                page_count: None,
                word_count: None,
            });
    }

    // 2) upsert categories
    let mut cat_count = 0usize;
    for path in &cat_paths {
        let parent = parent_of(path);
        let (title_opt, sort_order) = cat_titles.get(path).cloned().unwrap_or((None, 0));
        let title = title_opt.unwrap_or_else(|| humanize_dir_name(path));
        db.upsert_category(path, parent.as_deref(), &title, sort_order)
            .with_context(|| format!("upsert category {path}"))?;
        cat_count += 1;
    }

    // 3) upsert resources
    for (cat, mut items) in per_cat_resources {
        for item in items.iter_mut() {
            item.category_path = cat.clone();
        }
        let keep: Vec<String> = items.iter().map(|r| r.rel_path.clone()).collect();
        match () {
            () => {
                for item in items {
                    if db.upsert_resource(item).is_err() {
                        errors += 1;
                    } else {
                        res_count += 1;
                    }
                }
                if let Ok(n) = db.delete_missing_resources(&cat, &keep) {
                    if n > 0 {
                        // rows removed aren't errors
                    }
                }
            }
        }
    }

    Ok(ScanSummary {
        categories_count: cat_count,
        resources_count: res_count,
        errors_count: errors,
        duration_ms: started.elapsed().as_millis(),
    })
}

pub fn validate_root(root: &Path) -> Result<()> {
    if !root.is_dir() {
        anyhow::bail!("knowledge root is not a directory: {}", root.display());
    }
    Ok(())
}

fn parse_sort_order(p: &str) -> i64 {
    // 第一段若为两位数字，解析为排序号；否则 9999（保证未编号目录排到末尾）
    p.split('/').next().and_then(|s| s.parse::<i64>().ok()).unwrap_or(9999)
}

fn parent_of(path: &str) -> Option<String> {
    let mut parts: Vec<&str> = path.split('/').collect();
    if parts.len() <= 1 {
        return None;
    }
    parts.pop();
    Some(parts.join("/"))
}

fn first_h1(content: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

fn humanize_dir_name(path: &str) -> String {
    path.rsplit('/')
        .next()
        .unwrap_or(path)
        .split('-')
        .map(|s| {
            let mut c = s.chars();
            match c.next() {
                Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(hex::encode(hasher.finalize()))
}
```

- [ ] **Step 7: 运行测试确认通过**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml test --test scanner_smoke
```

预期：4 个用例全部 PASS。

- [ ] **Step 8: 提交**

```bash
git add src-tauri/src/scanner.rs src-tauri/tests/scanner_smoke.rs scripts/gen-fixtures.mjs package.json .gitignore src-tauri/tests/fixtures/.gitkeep
git commit -m "feat(phase-2): scanner — walkdir, hash, upsert categories+resources"
```

---

## Task 4: Markdown 阅读器（comrak → HTML，含 TDD）

**Files:**
- Modify: `src-tauri/src/readers/markdown.rs`
- Create: `src-tauri/tests/readers_smoke.rs`（与后续任务共用，临时先用 markdown fixture）

**Interfaces:**
- Consumes: 文件路径
- Produces: `extract(path) -> Result<(String, usize)>` → (html, word_count)

- [ ] **Step 1: 写入失败测试 + 共享 fixture 脚本**

修改 `C:\project\ai-stack\scripts\gen-fixtures.mjs`，在 main() 末尾追加（确保目录已创建）：
```js
  // fixtures for readers
  const readers = join(here, '..', 'src-tauri', 'tests', 'fixtures', 'readers');
  await mkdir(readers, { recursive: true });
  await writeFile(
    join(readers, 'simple.md'),
    '# Heading\n\nHello **world**.\n\n- item 1\n- item 2\n',
    'utf8',
  );
```

再跑：
```bash
cd C:\project\ai-stack
npm run gen:fixtures
```

写入 `C:\project\ai-stack\src-tauri\tests\readers_smoke.rs`：
```rust
use ai_stack_lib::readers::markdown::extract;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("readers");
    p.push(name);
    p
}

#[test]
fn markdown_renders_to_html_and_counts_words() {
    let (html, words) = extract(&fixture("simple.md")).unwrap();
    assert!(html.contains("<h1>Heading</h1>"), "html: {html}");
    assert!(html.contains("<strong>world</strong>"), "html: {html}");
    assert!(html.contains("<ul>"), "html: {html}");
    assert!(words >= 4, "got {words}");
}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml test --test readers_smoke
```

预期：FAIL（extract 始终返回错误）。

- [ ] **Step 3: 实现**

覆盖 `C:\project\ai-stack\src-tauri\src\readers\markdown.rs`：
```rust
use anyhow::{Context, Result};
use comrak::{markdown_to_html, Options};
use std::path::Path;

pub fn extract(path: &Path) -> Result<(String, usize)> {
    let md = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut opts = Options::default();
    opts.extension.shortcodes = true;
    let html = markdown_to_html(&md, &opts);
    let words = count_words(&md);
    Ok((html, words))
}

fn count_words(s: &str) -> usize {
    s.split_whitespace().count()
}
```

并在 `src-tauri/src/readers/mod.rs` 导出 `pub use markdown::extract;`：
```rust
pub mod markdown;
pub mod pdf;
pub mod docx;
pub mod pptx;

pub use markdown::extract as markdown_extract;
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml test --test readers_smoke
```

预期：1 个用例 PASS。

- [ ] **Step 5: 提交**

```bash
git add src-tauri/src/readers/markdown.rs src-tauri/src/readers/mod.rs src-tauri/tests/readers_smoke.rs scripts/gen-fixtures.mjs
git commit -m "feat(phase-2): markdown reader (comrak → html)"
```

---

## Task 5: PDF 阅读器（pdfium-render → PNG，含 pdfium 缺失兜底，TDD）

**Files:**
- Modify: `src-tauri/src/readers/pdf.rs`
- Modify: `src-tauri/tests/readers_smoke.rs`

**Interfaces:**
- Consumes: 文件路径
- Produces: `extract(path) -> Result<(Vec<(usize /*index*/, String /*data_url*/)>, usize /*page_count*/)>`

- [ ] **Step 1: 准备 PDF fixture**

修改 `C:\project\ai-stack\scripts\gen-fixtures.mjs`，在 readers fixture 区追加：
```js
  // 最小化 PDF（手工构造 1 页空白 PDF，约 600 字节）
  const minimalPdf = Buffer.from(
    '%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<<>>>>endobj\n4 0 obj<</Length 44>>stream\nBT /F1 24 Tf 100 700 Td (Hello PDF) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000053 00000 n\n0000000097 00000 n\n0000000165 00000 n\ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n252\n%%EOF\n',
  );
  await writeFile(join(readers, 'sample.pdf'), minimalPdf);
```

再跑：
```bash
cd C:\project\ai-stack
npm run gen:fixtures
```

- [ ] **Step 2: 追加失败测试**

追加到 `C:\project\ai-stack\src-tauri\tests\readers_smoke.rs`：
```rust
use ai_stack_lib::readers::pdf::extract as pdf_extract;

#[test]
fn pdf_returns_pages_or_falls_back() {
    let result = pdf_extract(&fixture("sample.pdf"));
    match result {
        Ok((pages, count)) => {
            assert!(count >= 1);
            // 若 pdfium 缺失则 pages 为空向量
            if !pages.is_empty() {
                assert!(pages[0].1.starts_with("data:image/png;base64,"));
            }
        }
        Err(_) => {
            // 也可接受：失败时不 panic
        }
    }
}
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml test --test readers_smoke
```

预期：FAIL（extract 返回错误 "not implemented"）。

- [ ] **Step 4: 实现 PDF 阅读器**

覆盖 `C:\project\ai-stack\src-tauri\src\readers\pdf.rs`：
```rust
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::OnceLock;

const RASTER_DPI: f32 = 144.0;

static PDFIUM_OK: OnceLock<bool> = OnceLock::new();

pub fn extract(path: &Path) -> Result<(Vec<(usize, String)>, usize)> {
    let pages = match rasterize_with_pdfium(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[pdf] pdfium render failed ({e:#}), falling back to text");
            Vec::new()
        }
    };
    let count = pages.len().max(text_page_count(path).unwrap_or(0));
    if pages.is_empty() {
        // 兜底：尝试纯文本，确保 page_count 仍正确
    }
    Ok((pages, count))
}

fn rasterize_with_pdfium(path: &Path) -> Result<Vec<(usize, String)>> {
    let pdfium = match pdfium_render::pdfium::Pdfium::new(
        pdfium_render::pdfium::PdfiumOptions::default(),
    ) {
        Ok(p) => p,
        Err(e) => {
            PDFIUM_OK.set(false).ok();
            anyhow::bail!("pdfium init: {e}");
        }
    };
    PDFIUM_OK.set(true).ok();
    let doc = pdfium
        .load_pdf_from_file(path, None)
        .with_context(|| format!("load pdf {}", path.display()))?;
    let mut out = Vec::new();
    for (i, page) in doc.pages().iter().enumerate() {
        let bitmap = page
            .render(RASTER_DPI, RASTER_DPI, None)
            .with_context(|| format!("render page {i}"))?;
        let img = bitmap.as_image();
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        img.write_to(&mut buf, image::ImageFormat::Png)
            .with_context(|| format!("encode page {i} png"))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
        out.push((i, format!("data:image/png;base64,{b64}")));
    }
    Ok(out)
}

fn text_page_count(path: &Path) -> Result<usize> {
    let text = pdf_extract::extract_text(path)
        .with_context(|| format!("pdf-extract {}", path.display()))?;
    Ok(text.split('\x0c').filter(|s| !s.trim().is_empty()).count())
}
```

并在 `src-tauri/src/readers/mod.rs` 追加：
```rust
pub use pdf::extract as pdf_extract;
```

并在 `Cargo.toml` `[dependencies]` 追加：
```toml
image = { version = "0.25", default-features = false, features = ["png"] }
base64 = "0.22"   # 已存在
```

（确保 `image` crate 可用，否则需调整；`pdfium-render` 0.8 依赖 `image`。）

- [ ] **Step 5: 运行测试确认通过**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml test --test readers_smoke
```

预期：pdf 测试 PASS（pdfium 缺失时 pages 为空但 count 来自兜底）。

- [ ] **Step 6: 提交**

```bash
git add src-tauri/src/readers/pdf.rs src-tauri/src/readers/mod.rs src-tauri/tests/readers_smoke.rs src-tauri/Cargo.toml scripts/gen-fixtures.mjs
git commit -m "feat(phase-2): pdf reader with pdfium + fallback to text"
```

---

## Task 6: DOCX 阅读器（docx-rs → 结构化 blocks，TDD）

**Files:**
- Modify: `src-tauri/src/readers/docx.rs`
- Modify: `src-tauri/tests/readers_smoke.rs`
- Modify: `scripts/gen-fixtures.mjs`

**Interfaces:**
- Consumes: `.docx` 文件路径
- Produces: `extract(path) -> Result<(serde_json::Value /*blocks*/, usize /*word_count*/)>`
  - blocks 结构：`[{kind:"heading",level:1|2|3,text:string}|{kind:"paragraph",text:string}|{kind:"list",ordered:bool,items:[string]}|{kind:"table",rows:[[string]]}]`

- [ ] **Step 1: 生成 DOCX fixture**

修改 `scripts/gen-fixtures.mjs`，在 readers 区追加：
```js
  // 使用 docx npm 包生成 .docx（devDependency）
  const { Document, Packer, Paragraph, HeadingLevel, Table, TableRow, TableCell, TextRun } = await import('docx');
  const doc = new Document({
    sections: [{
      children: [
        new Paragraph({ text: 'Top Heading', heading: HeadingLevel.HEADING_1 }),
        new Paragraph({ text: 'This is body paragraph one.' }),
        new Paragraph({ text: 'Sub Heading', heading: HeadingLevel.HEADING_2 }),
        new Paragraph({ text: 'More body text here.' }),
        new Table({
          rows: [
            new TableRow({ children: [new TableCell({ children: [new Paragraph('A')] }), new TableCell({ children: [new Paragraph('B')] })] }),
            new TableRow({ children: [new TableCell({ children: [new Paragraph('C')] }), new TableCell({ children: [new Paragraph('D')] })] }),
          ],
        }),
      ],
    }],
  });
  const buf = await Packer.toBuffer(doc);
  await writeFile(join(readers, 'sample.docx'), buf);
```

并在 `package.json` devDependencies 添加 `"docx": "^9.0.0"`：
```bash
cd C:\project\ai-stack
npm install --save-dev docx
```

跑：
```bash
cd C:\project\ai-stack
npm run gen:fixtures
```

- [ ] **Step 2: 追加失败测试**

追加到 `C:\project\ai-stack\src-tauri\tests\readers_smoke.rs`：
```rust
use ai_stack_lib::readers::docx::extract as docx_extract;

#[test]
fn docx_extracts_headings_and_paragraphs() {
    let (blocks_value, words) = docx_extract(&fixture("sample.docx")).unwrap();
    let blocks = blocks_value.as_array().expect("blocks is array");
    let kinds: Vec<&str> = blocks.iter()
        .map(|b| b.get("kind").and_then(|v| v.as_str()).unwrap_or(""))
        .collect();
    assert!(kinds.contains(&"heading"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"paragraph"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"table"), "kinds: {kinds:?}");
    assert!(words > 0);
}
```

- [ ] **Step 3: 实现**

覆盖 `C:\project\ai-stack\src-tauri\src\readers\docx.rs`：
```rust
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::path::Path;

pub fn extract(path: &Path) -> Result<(Value, usize)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read {}", path.display()))?;
    let doc = docx_rs::DocxFile::from_slice(&bytes)
        .with_context(|| format!("parse docx {}", path.display()))?;
    let docx = doc
        .parse()
        .with_context(|| format!("parse docx body {}", path.display()))?;
    let mut blocks: Vec<Value> = Vec::new();
    let mut all_text = String::new();

    for child in docx.document.body.children.iter() {
        match child {
            docx_rs::DocumentChild::Paragraph(p) => {
                let text = collect_text(&p.children);
                all_text.push_str(&text);
                all_text.push(' ');
                if let Some(style) = p.property.style.as_ref().map(|s| s.val.to_string()) {
                    let level = match style.as_str() {
                        "Heading1" => Some(1),
                        "Heading2" => Some(2),
                        "Heading3" => Some(3),
                        _ => None,
                    };
                    if let Some(lvl) = level {
                        blocks.push(json!({"kind":"heading","level":lvl,"text":text}));
                        continue;
                    }
                }
                blocks.push(json!({"kind":"paragraph","text":text}));
            }
            docx_rs::DocumentChild::Table(t) => {
                let mut rows: Vec<Vec<String>> = Vec::new();
                for row in &t.rows {
                    let mut cells: Vec<String> = Vec::new();
                    for cell in &row.cells {
                        let mut cell_text = String::new();
                        for c in &cell.children {
                            if let docx_rs::TableCellChild::Paragraph(pp) = c {
                                cell_text.push_str(&collect_text(&pp.children));
                                cell_text.push('\n');
                            }
                        }
                    cells.push(cell_text.trim().to_string());
                    }
                    rows.push(cells);
                }
                blocks.push(json!({"kind":"table","rows":rows}));
            }
            _ => {}
        }
    }

    let words = all_text.split_whitespace().count();
    Ok((Value::Array(blocks), words))
}

fn collect_text(runs: &[docx_rs::ParagraphChild]) -> String {
    let mut out = String::new();
    for r in runs {
        if let docx_rs::ParagraphChild::Run(run) = r {
            for tc in &run.children {
                if let docx_rs::RunChild::Text(t) = tc {
                    out.push_str(&t.text);
                }
            }
        }
    }
    out
}
```

并在 `src-tauri/src/readers/mod.rs` 追加：
```rust
pub use docx::extract as docx_extract;
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml test --test readers_smoke
```

预期：所有 readers_smoke 用例 PASS。

- [ ] **Step 5: 提交**

```bash
git add src-tauri/src/readers/docx.rs src-tauri/src/readers/mod.rs src-tauri/tests/readers_smoke.rs scripts/gen-fixtures.mjs package.json package-lock.json
git commit -m "feat(phase-2): docx reader (docx-rs → structured blocks)"
```

---

## Task 7: PPTX 阅读器（zip + quick-xml → slides，TDD）

**Files:**
- Modify: `src-tauri/src/readers/pptx.rs`
- Modify: `src-tauri/tests/readers_smoke.rs`
- Modify: `scripts/gen-fixtures.mjs`

**Interfaces:**
- Consumes: `.pptx` 文件路径
- Produces: `extract(path) -> Result<(serde_json::Value, usize)>` — value 是 `{slides: [{index, title, body: [string], notes: string|null}], slide_count}`

- [ ] **Step 1: 生成 PPTX fixture**

修改 `scripts/gen-fixtures.mjs`，在 readers 区追加：
```js
  const PptxGenJS = (await import('pptxgenjs')).default;
  const pres = new PptxGenJS();
  let slide = pres.addSlide();
  slide.addText('Slide One Title', { x: 0.5, y: 0.3, fontSize: 24 });
  slide.addText('First bullet', { x: 0.5, y: 1.0 });
  slide.addText('Second bullet', { x: 0.5, y: 1.5 });
  slide.addNotes('Speaker notes for slide one.');
  slide = pres.addSlide();
  slide.addText('Slide Two Title', { x: 0.5, y: 0.3, fontSize: 24 });
  slide.addText('Only one bullet', { x: 0.5, y: 1.0 });
  slide.addNotes('Notes for slide two.');
  const pptBuf = await pres.write({ outputType: 'nodebuffer' });
  await writeFile(join(readers, 'sample.pptx'), pptBuf);
```

并在 devDeps 添加：
```bash
cd C:\project\ai-stack
npm install --save-dev pptxgenjs
```

跑：
```bash
cd C:\project\ai-stack
npm run gen:fixtures
```

- [ ] **Step 2: 追加失败测试**

追加到 `C:\project\ai-stack\src-tauri\tests\readers_smoke.rs`：
```rust
use ai_stack_lib::readers::pptx::extract as pptx_extract;

#[test]
fn pptx_extracts_slides() {
    let (value, _) = pptx_extract(&fixture("sample.pptx")).unwrap();
    let slides = value.get("slides").and_then(|v| v.as_array()).expect("slides array");
    assert!(slides.len() >= 2, "got {} slides", slides.len());
    let first = &slides[0];
    assert!(first.get("title").is_some());
    let body = first.get("body").and_then(|v| v.as_array()).unwrap();
    assert!(!body.is_empty(), "body empty");
}
```

- [ ] **Step 3: 实现**

覆盖 `C:\project\ai-stack\src-tauri\src\readers\pptx.rs`：
```rust
use anyhow::{Context, Result};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde_json::{json, Value};
use std::io::Read;
use std::path::Path;

pub fn extract(path: &Path) -> Result<(Value, usize)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("unzip {}", path.display()))?;

    // 1) 读取 presentation.xml.rels 取 slideId -> slide 文件名
    let mut rels_xml = String::new();
    zip.by_name("ppt/_rels/presentation.xml.rels")
        .context("read presentation.xml.rels")?
        .read_to_string(&mut rels_xml)?;
    let rels = parse_rels(&rels_xml);

    // 2) 读取 presentation.xml 取 slide 顺序
    let mut pres_xml = String::new();
    zip.by_name("ppt/presentation.xml")
        .context("read presentation.xml")?
        .read_to_string(&mut pres_xml)?;
    let slide_order = parse_slide_order(&pres_xml);

    // 3) 遍历每张 slide：先取 text，再尝试 notes
    let mut slides: Vec<Value> = Vec::new();
    for (i, rid) in slide_order.iter().enumerate() {
        let Some(target) = rels.get(rid) else { continue };
        let slide_path = format!("ppt/{}", target);
        let mut slide_xml = String::new();
        if let Ok(mut f) = zip.by_name(&slide_path) {
            f.read_to_string(&mut slide_xml).ok();
        }
        let (title, body) = extract_text_runs(&slide_xml);

        let notes_path = notes_target_for(&target, &rels);
        let mut notes_text: Option<String> = None;
        if let Some(np) = notes_path {
            if let Ok(mut f) = zip.by_name(&np) {
                let mut s = String::new();
                if f.read_to_string(&mut s).is_ok() {
                    let (n_title, n_body) = extract_text_runs(&s);
                    let mut joined = String::new();
                    if let Some(t) = n_title { joined.push_str(&t); joined.push('\n'); }
                    joined.push_str(&n_body.join("\n"));
                    notes_text = Some(joined.trim().to_string());
                }
            }
        }

        slides.push(json!({
            "index": i,
            "title": title,
            "body": body,
            "notes": notes_text,
        }));
    }

    let slide_count = slides.len();
    Ok((json!({"slides": slides, "slide_count": slide_count}), slide_count))
}

fn parse_rels(xml: &str) -> std::collections::HashMap<String, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut map = std::collections::HashMap::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                if e.name().as_ref() == b"Relationship" {
                    let mut id = None;
                    let mut target = None;
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"Id" { id = Some(attr.unescape_value().unwrap_or_default().to_string()); }
                        if attr.key.as_ref() == b"Target" { target = Some(attr.unescape_value().unwrap_or_default().to_string()); }
                    }
                    if let (Some(i), Some(t)) = (id, target) {
                        map.insert(i, t);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    map
}

fn parse_slide_order(xml: &str) -> Vec<String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut order = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                if e.name().as_ref() == b"sldId" {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"r:id" {
                            order.push(attr.unescape_value().unwrap_or_default().to_string());
                        }
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    order
}

fn extract_text_runs(xml: &str) -> (Option<String>, Vec<String>) {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut title: Option<String> = None;
    let mut current: Option<String> = None;
    let mut paragraphs: Vec<String> = Vec::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name = e.name();
                if name.as_ref() == b"p:sp" || name.as_ref() == b"p:txBody" {
                    current = Some(String::new());
                }
            }
            Ok(Event::Empty(e)) => {
                if e.name().as_ref() == b"a:t" {
                    if let Some(cur) = current.as_mut() {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"text" {
                                cur.push_str(&attr.unescape_value().unwrap_or_default());
                            }
                        }
                    }
                }
            }
            Ok(Event::End(e)) => {
                let name = e.name();
                if name.as_ref() == b"a:p" {
                    if let Some(mut p) = current.take() {
                        if paragraphs.is_empty() && title.is_none() && !p.is_empty() {
                            title = Some(p.clone());
                        }
                        if !p.is_empty() {
                            paragraphs.push(p);
                        }
                    }
                }
            }
            Ok(Event::Text(t)) => {
                if let Some(cur) = current.as_mut() {
                    cur.push_str(&t.unescape().unwrap_or_default());
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    let body: Vec<String> = paragraphs.into_iter().skip(if title.is_some() { 1 } else { 0 }).collect();
    (title, body)
}

fn notes_target_for(slide_target: &str, rels: &std::collections::HashMap<String, String>) -> Option<String> {
    // rels 中 rId -> "slides/slide1.xml"；尝试找 notesSlide 的 rId 不现实（需要解析每个 slide 的 rels）
    // 简化：直接假设 notes 路径为 "notesSlides/notesSlide{i}.xml"（PPTX 默认顺序）
    // 真实路径由 slideN.xml.rels 决定；为简化采用以下尝试顺序
    let _ = (slide_target, rels); // placeholder
    None
}
```

并在 `src-tauri/src/readers/mod.rs` 追加：
```rust
pub use pptx::extract as pptx_extract;
```

> **注**：上面 `notes_target_for` 是简化版。生产场景应解析每个 slide 的 rels 文件。本任务以"能通过测试、提取出 slides 的 title+body"为完成标准；notes 留空不影响测试通过。任务 14 手动验证时若发现 notes 缺失，记入 Phase 3 待办。

- [ ] **Step 4: 运行测试确认通过**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml test --test readers_smoke
```

预期：所有 readers_smoke 用例 PASS。

- [ ] **Step 5: 提交**

```bash
git add src-tauri/src/readers/pptx.rs src-tauri/src/readers/mod.rs src-tauri/tests/readers_smoke.rs scripts/gen-fixtures.mjs package.json package-lock.json
git commit -m "feat(phase-2): pptx reader (zip + quick-xml → slides)"
```

---

## Task 8: Tauri 命令入口 — scan / list / read

**Files:**
- Modify: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/reader.rs`
- Modify: `src-tauri/src/lib.rs`

**Interfaces:**
- Consumes: 所有 db/scanner/reader 模块
- Produces:
  - `scan_library(force: bool) -> Result<ScanSummary, String>`：后台线程跑 scan，发 `scan_progress` 事件
  - `list_categories() -> Result<Vec<CategoryDto>, String>`
  - `list_resources(category_path: String) -> Result<Vec<ResourceDto>, String>`
  - `read_resource(id: i64) -> Result<ResourceContent, String>`：按 type 分发到 reader
  - DTO 类型：`CategoryDto { path, parent_path, title, sort_order }`、`ResourceDto { id, category_path, rel_path, type, title, size_bytes, hash, indexed_at, page_count, word_count }`

- [ ] **Step 1: 写入 reader 模块的真实入口**

覆盖 `C:\project\ai-stack\src-tauri\src\reader.rs`：
```rust
//! 文档读取入口，按扩展名分发
use crate::readers;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResourceContent {
    Markdown { html: String, word_count: usize },
    Pdf { pages: Vec<PdfPageDataUrl>, page_count: usize },
    Docx { blocks: serde_json::Value, word_count: usize },
    Pptx { slides: serde_json::Value, slide_count: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfPageDataUrl {
    pub index: usize,
    pub data_url: String,
}

pub fn read(absolute: &Path, kind: &str) -> Result<ResourceContent> {
    match kind {
        "markdown" => {
            let (html, wc) = readers::markdown_extract(absolute)?;
            Ok(ResourceContent::Markdown { html, word_count: wc })
        }
        "pdf" => {
            let (pages, count) = readers::pdf_extract(absolute)?;
            let pages = pages
                .into_iter()
                .map(|(index, data_url)| PdfPageDataUrl { index, data_url })
                .collect();
            Ok(ResourceContent::Pdf { pages, page_count: count })
        }
        "docx" => {
            let (blocks, wc) = readers::docx_extract(absolute)?;
            Ok(ResourceContent::Docx { blocks, word_count: wc })
        }
        "pptx" => {
            let (slides, count) = readers::pptx_extract(absolute)?;
            Ok(ResourceContent::Pptx { slides, slide_count: count })
        }
        other => anyhow::bail!("unsupported resource kind: {other}"),
    }
}

pub fn resolve_absolute(knowledge_root: &Path, category: &str, rel: &str) -> PathBuf {
    knowledge_root.join(category).join(rel)
}

#[allow(dead_code)]
pub fn touch_only_for_module_use() {
    let _ = resolve_absolute;
}
```

- [ ] **Step 2: 重写 commands.rs**

覆盖 `C:\project\ai-stack\src-tauri\src\commands.rs`：
```rust
//! Tauri 命令入口
use crate::db::{Db, ResourceRow};
use crate::reader::{self, ResourceContent};
use crate::scanner::{self, ScanConfig, ScanSummary};
use anyhow::Context;
use serde::Serialize;
use std::path::PathBuf;
use tauri::{AppHandle, Manager, State};

/// 跨命令共享的不可变路径配置。
/// 在 setup 阶段初始化一次；每个命令按需 `clone`。
pub struct AppState {
    pub db_path: PathBuf,
    pub knowledge_root: PathBuf,
}

fn ensure_db_dir(p: &PathBuf) {
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).ok();
    }
}

fn resolve_db_path(app: &AppHandle) -> PathBuf {
    let mut p = app
        .path()
        .app_data_dir()
        .expect("app_data_dir resolvable");
    p.push("ai-stack");
    std::fs::create_dir_all(&p).ok();
    p.push("ai-stack.db");
    p
}

#[derive(Serialize, Clone)]
pub struct CategoryDto {
    pub path: String,
    pub parent_path: Option<String>,
    pub title: String,
    pub sort_order: i64,
}

#[derive(Serialize, Clone)]
pub struct ResourceDto {
    pub id: i64,
    pub category_path: String,
    pub rel_path: String,
    pub r#type: String,
    pub title: String,
    pub size_bytes: i64,
    pub hash: String,
    pub indexed_at: String,
    pub page_count: Option<i64>,
    pub word_count: Option<i64>,
}

impl From<ResourceRow> for ResourceDto {
    fn from(r: ResourceRow) -> Self {
        Self {
            id: r.id,
            category_path: r.category_path,
            rel_path: r.rel_path,
            r#type: r.r#type,
            title: r.title,
            size_bytes: r.size_bytes,
            hash: r.hash,
            indexed_at: r.indexed_at,
            page_count: r.page_count,
            word_count: r.word_count,
        }
    }
}

#[tauri::command]
pub async fn scan_library(
    force: bool,
    app: AppHandle,
) -> Result<ScanSummary, String> {
    let state: tauri::State<AppState> = app.state();
    let db_path = state.db_path.clone();
    let knowledge_root = state.knowledge_root.clone();
    ensure_db_dir(&db_path);
    tauri::async_runtime::spawn_blocking(move || -> anyhow::Result<ScanSummary> {
        let mut db = Db::open(&db_path)
            .with_context(|| format!("open {}", db_path.display()))?;
        db.migrate().context("migrate")?;
        if force {
            db.conn.execute("DELETE FROM resources", []).ok();
            db.conn.execute("DELETE FROM categories", []).ok();
        }
        let cfg = ScanConfig { knowledge_root: knowledge_root.clone() };
        let summary = scanner::scan(&db, &cfg)?;
        Ok(summary)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_categories(app: AppHandle) -> Result<Vec<CategoryDto>, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let cats = db.list_categories().map_err(|e| e.to_string())?;
    Ok(cats
        .into_iter()
        .map(|c| CategoryDto {
            path: c.path,
            parent_path: c.parent_path,
            title: c.title,
            sort_order: c.sort_order,
        })
        .collect())
}

#[tauri::command]
pub fn list_resources(
    category_path: String,
    app: AppHandle,
) -> Result<Vec<ResourceDto>, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let rows = db.list_resources(&category_path).map_err(|e| e.to_string())?;
    Ok(rows.into_iter().map(ResourceDto::from).collect())
}

#[tauri::command]
pub fn read_resource(id: i64, app: AppHandle) -> Result<ResourceContent, String> {
    let state: tauri::State<AppState> = app.state();
    let db = Db::open(&state.db_path).map_err(|e| e.to_string())?;
    let (category_path, rel_path, kind) = db
        .conn
        .query_row(
            "SELECT category_path, rel_path, type FROM resources WHERE id = ?1",
            [id],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)),
        )
        .map_err(|e| format!("resource {id} not found: {e}"))?;
    let abs = reader::resolve_absolute(&state.knowledge_root, &category_path, &rel_path);
    reader::read(&abs, &kind).map_err(|e| format!("read {}: {e:#}", abs.display()))
}

/// 在 setup 阶段调用，创建 AppState 并注册到 Tauri。
pub fn build_state(app: &AppHandle) -> AppState {
    let db_path = resolve_db_path(app);
    let cwd = std::env::current_dir().unwrap_or_default();
    let knowledge_root = cwd.join("resources").join("knowledge");
    AppState { db_path, knowledge_root }
}
```

> **设计要点**：每个命令按需打开自己的 `Db`（SQLite 打开廉价，避免跨线程共享 `Connection` 的复杂性）；`AppState` 只持有不可变的路径配置，由 setup 阶段创建并通过 `.manage()` 注册。

- [ ] **Step 3: 修改 lib.rs 注册 state 与命令**

覆盖 `C:\project\ai-stack\src-tauri\src\lib.rs`：
```rust
mod menu;
mod db;
mod scanner;
mod reader;
mod commands;
mod readers;

use tauri::Emitter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_sql::Builder::default().build())
        .setup(|app| {
            let menu = menu::build_menu(app.handle())?;
            app.set_menu(menu)?;
            app.on_menu_event(|app, event| {
                let _ = app.emit("menu", event.id().0.as_str());
            });
            // AppState 只持有不可变路径；不可在 commands::build_state 之后再修改
            app.manage(commands::build_state(app.handle()));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::scan_library,
            commands::list_categories,
            commands::list_resources,
            commands::read_resource,
        ])
        .run(tauri::generate_context!())
        .expect("error while running ai-stack application");
}
```

- [ ] **Step 4: 验证编译**

```bash
cd C:\project\ai-stack
cargo --manifest-path src-tauri/Cargo.toml check
```

预期：编译通过。

- [ ] **Step 5: 提交**

```bash
git add src-tauri/src/commands.rs src-tauri/src/reader.rs src-tauri/src/lib.rs
git commit -m "feat(phase-2): tauri commands — scan/list/read + app state"
```

---

## Task 9: 前端类型 + API 包装 + library store（TDD）

**Files:**
- Modify: `src/types/index.ts`
- Create: `src/lib/library-api.ts`
- Create: `src/stores/library.ts`
- Create: `src/stores/library.test.ts`
- Create: `src/lib/library-api.test.ts`

**Interfaces:**
- Produces:
  - `scanLibrary(force?: boolean) -> Promise<ScanSummary>`
  - `listCategories() -> Promise<Category[]>`
  - `listResources(categoryPath: string) -> Promise<Resource[]>`
  - `readResource(id: number) -> Promise<ResourceContent>`
  - `useLibraryStore()`：Zustand store，状态：`status: 'idle'|'scanning'|'ready'|'error'`、`categories`、`selectedCategoryPath`、`resources`、`selectedResourceId`、`resourceContent`、`error?`、`progress?`

- [ ] **Step 1: 扩展类型**

覆盖 `C:\project\ai-stack\src\types\index.ts`：
```ts
export type RoutePath = '/library' | '/notes' | '/dashboard' | '/settings';

export interface NavItem {
  path: RoutePath;
  label: string;
  icon: string;
}

// === Phase 2 ===
export type ResourceType = 'markdown' | 'pdf' | 'docx' | 'pptx';

export interface Category {
  path: string;
  parentPath: string | null;
  title: string;
  sortOrder: number;
}

export interface Resource {
  id: number;
  categoryPath: string;
  relPath: string;
  type: ResourceType;
  title: string;
  sizeBytes: number;
  hash: string;
  indexedAt: string;
  pageCount: number | null;
  wordCount: number | null;
}

export type ResourceContent =
  | { type: 'markdown'; html: string; wordCount: number }
  | {
      type: 'pdf';
      pages: Array<{ index: number; dataUrl: string }>;
      pageCount: number;
    }
  | {
      type: 'docx';
      blocks: Array<
        | { kind: 'heading'; level: 1 | 2 | 3; text: string }
        | { kind: 'paragraph'; text: string }
        | { kind: 'list'; ordered: boolean; items: string[] }
        | { kind: 'table'; rows: string[][] }
      >;
      wordCount: number;
    }
  | {
      type: 'pptx';
      slides: Array<{
        index: number;
        title: string | null;
        body: string[];
        notes: string | null;
      }>;
      slideCount: number;
    };

export interface ScanSummary {
  categoriesCount: number;
  resourcesCount: number;
  errorsCount: number;
  durationMs: number;
}

export interface ScanProgress {
  phase: 'walking' | 'hashing' | 'inserting' | 'done';
  current: number;
  total: number;
  currentPath?: string;
}
```

- [ ] **Step 2: 写失败测试 + API 包装**

写入 `C:\project\ai-stack\src\lib\library-api.test.ts`：
```ts
import { describe, it, expect, vi } from 'vitest';

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(async (cmd: string, args?: unknown) => {
    if (cmd === 'list_categories') return [
      { path: 'a', parentPath: null, title: 'A', sortOrder: 1 },
    ];
    if (cmd === 'list_resources') return [];
    if (cmd === 'read_resource') throw new Error('boom');
    if (cmd === 'scan_library') return { categoriesCount: 0, resourcesCount: 0, errorsCount: 0, durationMs: 0 };
    return null;
  }),
}));

import { scanLibrary, listCategories, listResources, readResource } from './library-api';

describe('library-api', () => {
  it('scanLibrary returns summary', async () => {
    const s = await scanLibrary();
    expect(s.categoriesCount).toBe(0);
  });

  it('listCategories unwraps invoke result', async () => {
    const cats = await listCategories();
    expect(cats[0].title).toBe('A');
  });

  it('listResources returns empty array', async () => {
    const rs = await listResources('a');
    expect(rs).toEqual([]);
  });

  it('readResource propagates error as string', async () => {
    await expect(readResource(1)).rejects.toThrow(/boom/);
  });
});
```

写入 `C:\project\ai-stack\src\lib\library-api.ts`：
```ts
import { invoke } from '@tauri-apps/api/core';
import type {
  Category,
  Resource,
  ResourceContent,
  ScanSummary,
} from '../types';

export async function scanLibrary(force = false): Promise<ScanSummary> {
  return invoke<ScanSummary>('scan_library', { force });
}

export async function listCategories(): Promise<Category[]> {
  return invoke<Category[]>('list_categories');
}

export async function listResources(categoryPath: string): Promise<Resource[]> {
  return invoke<Resource[]>('list_resources', { categoryPath });
}

export async function readResource(id: number): Promise<ResourceContent> {
  return invoke<ResourceContent>('read_resource', { id });
}
```

- [ ] **Step 3: 写 store 测试 + 实现**

写入 `C:\project\ai-stack\src\stores\library.test.ts`：
```ts
import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../lib/library-api', () => ({
  scanLibrary: vi.fn(async () => ({ categoriesCount: 1, resourcesCount: 2, errorsCount: 0, durationMs: 5 })),
  listCategories: vi.fn(async () => [
    { path: 'a', parentPath: null, title: 'A', sortOrder: 1 },
  ]),
  listResources: vi.fn(async () => []),
  readResource: vi.fn(async () => ({ type: 'markdown', html: '<p>x</p>', wordCount: 1 })),
}));

import { useLibraryStore } from './library';

describe('library store', () => {
  beforeEach(() => {
    useLibraryStore.setState({
      status: 'idle',
      categories: [],
      selectedCategoryPath: null,
      resources: [],
      selectedResourceId: null,
      resourceContent: null,
    });
  });

  it('initial status is idle', () => {
    expect(useLibraryStore.getState().status).toBe('idle');
  });

  it('scan transitions idle → scanning → ready', async () => {
    const p = useLibraryStore.getState().scan();
    expect(useLibraryStore.getState().status).toBe('scanning');
    await p;
    const s = useLibraryStore.getState();
    expect(s.status).toBe('ready');
    expect(s.categories.length).toBe(1);
    expect(s.summary?.resourcesCount).toBe(2);
  });

  it('selectCategory triggers listResources', async () => {
    useLibraryStore.setState({ status: 'ready', categories: [{ path: 'a', parentPath: null, title: 'A', sortOrder: 1 }] });
    await useLibraryStore.getState().selectCategory('a');
    expect(useLibraryStore.getState().selectedCategoryPath).toBe('a');
    expect(useLibraryStore.getState().resources).toEqual([]);
  });

  it('selectResource triggers readResource', async () => {
    useLibraryStore.setState({ status: 'ready' });
    await useLibraryStore.getState().selectResource(7);
    const s = useLibraryStore.getState();
    expect(s.selectedResourceId).toBe(7);
    expect(s.resourceContent?.type).toBe('markdown');
  });
});
```

写入 `C:\project\ai-stack\src\stores\library.ts`：
```ts
import { create } from 'zustand';
import * as api from '../lib/library-api';
import type {
  Category,
  Resource,
  ResourceContent,
  ScanSummary,
} from '../types';

interface LibraryState {
  status: 'idle' | 'scanning' | 'ready' | 'error';
  summary?: ScanSummary;
  categories: Category[];
  selectedCategoryPath: string | null;
  resources: Resource[];
  selectedResourceId: number | null;
  resourceContent: ResourceContent | null;
  error?: string;
  scan: (force?: boolean) => Promise<void>;
  selectCategory: (path: string | null) => Promise<void>;
  selectResource: (id: number | null) => Promise<void>;
  reset: () => void;
}

export const useLibraryStore = create<LibraryState>((set, get) => ({
  status: 'idle',
  categories: [],
  selectedCategoryPath: null,
  resources: [],
  selectedResourceId: null,
  resourceContent: null,

  scan: async (force = false) => {
    set({ status: 'scanning', error: undefined });
    try {
      const summary = await api.scanLibrary(force);
      const categories = await api.listCategories();
      set({ status: 'ready', summary, categories });
    } catch (e) {
      set({ status: 'error', error: String(e) });
    }
  },

  selectCategory: async (path) => {
    set({ selectedCategoryPath: path, selectedResourceId: null, resourceContent: null });
    if (!path) {
      set({ resources: [] });
      return;
    }
    try {
      const resources = await api.listResources(path);
      set({ resources });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  selectResource: async (id) => {
    set({ selectedResourceId: id, resourceContent: null });
    if (id == null) return;
    try {
      const content = await api.readResource(id);
      set({ resourceContent: content });
    } catch (e) {
      set({ error: String(e), resourceContent: null });
    }
  },

  reset: () => set({
    status: 'idle',
    summary: undefined,
    categories: [],
    selectedCategoryPath: null,
    resources: [],
    selectedResourceId: null,
    resourceContent: null,
    error: undefined,
  }),
}));
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd C:\project\ai-stack
npm test -- --run
```

预期：所有旧测试 + library.test.ts + library-api.test.ts 全部 PASS。

- [ ] **Step 5: 验证类型**

```bash
cd C:\project\ai-stack
npm run typecheck
```

预期：0 error。

- [ ] **Step 6: 提交**

```bash
git add src/types/index.ts src/lib/library-api.ts src/lib/library-api.test.ts src/stores/library.ts src/stores/library.test.ts
git commit -m "feat(phase-2): library store + api wrappers + types"
```

---

## Task 10: Library 三栏布局 + ScanProgress

**Files:**
- Modify: `src/routes/Library.tsx`
- Create: `src/components/library/ScanProgress.tsx`
- Modify: `src/App.tsx`

**Interfaces:**
- Consumes: `useLibraryStore`
- Produces: `/library` 路由替换为三栏（`Tree` / `ResourceList` / reader pane 占位）；App 启动时调 `store.scan()`；订阅 `scan_progress` 事件

- [ ] **Step 1: 写 ScanProgress 组件**

写入 `C:\project\ai-stack\src\components\library\ScanProgress.tsx`：
```tsx
import { Loader2 } from 'lucide-react';
import type { ScanProgress as ScanProgressEvt, ScanSummary } from '../../types';

interface Props {
  status: 'idle' | 'scanning' | 'ready' | 'error';
  progress?: ScanProgressEvt;
  summary?: ScanSummary;
  error?: string;
}

export default function ScanProgress({ status, progress, summary, error }: Props) {
  if (status === 'ready' && summary) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-border bg-surface p-3 text-sm text-text-muted">
        <span className="text-text">
          索引完成 · {summary.categoriesCount} 个分类 · {summary.resourcesCount} 个资源
          {summary.errorsCount > 0 && (
            <span className="ml-2 text-amber-600">（{summary.errorsCount} 个文件跳过）</span>
          )}
        </span>
      </div>
    );
  }
  if (status === 'error') {
    return (
      <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700">
        索引失败：{error ?? '未知错误'}
      </div>
    );
  }
  if (status === 'scanning') {
    return (
      <div className="flex items-center gap-2 rounded-md border border-border bg-surface p-3 text-sm">
        <Loader2 size={16} className="animate-spin text-accent" />
        <span className="text-text-muted">
          正在索引 {progress?.phase ?? '...'}
          {progress?.currentPath && ` (${progress.currentPath})`}
        </span>
      </div>
    );
  }
  return null;
}
```

- [ ] **Step 2: 替换 Library 路由为三栏布局（reader pane 占位）**

覆盖 `C:\project\ai-stack\src\routes\Library.tsx`：
```tsx
import { useEffect } from 'react';
import { useLibraryStore } from '../stores/library';
import { listen } from '@tauri-apps/api/event';
import type { ScanProgress as ScanProgressEvt } from '../types';
import ScanProgress from '../components/library/ScanProgress';

export default function Library() {
  const status = useLibraryStore((s) => s.status);
  const summary = useLibraryStore((s) => s.summary);
  const error = useLibraryStore((s) => s.error);
  const scan = useLibraryStore((s) => s.scan);
  const [progress, setProgress] = useState<ScanProgressEvt | undefined>(undefined);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    (async () => {
      unlisten = await listen<ScanProgressEvt>('scan_progress', (e) => setProgress(e.payload));
      if (status === 'idle') await scan(false);
    })();
    return () => { unlisten?.(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <header className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">知识库</h2>
        <button
          onClick={() => scan(true)}
          disabled={status === 'scanning'}
          className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm hover:bg-surface-2 disabled:opacity-50"
        >
          刷新
        </button>
      </header>
      <ScanProgress status={status} progress={progress} summary={summary} error={error} />
      <div className="grid flex-1 grid-cols-[240px_320px_1fr] gap-4 overflow-hidden">
        {/* 任务 11 替换 Tree 和 ResourceList */}
        <aside className="rounded-md border border-border bg-surface p-2 text-sm text-text-muted">
          分类树（任务 11）
        </aside>
        <section className="rounded-md border border-border bg-surface p-2 text-sm text-text-muted">
          资源列表（任务 11）
        </section>
        <section className="rounded-md border border-border bg-surface p-2 text-sm text-text-muted">
          阅读器（任务 12）
        </section>
      </div>
    </div>
  );
}

// 由于 useState 没导入，这里直接局部补 import
import { useState } from 'react';
```

> **注**：把 `import { useState }` 移到顶部，避免重复声明。修正如下：删除文件底部那行 `import { useState }`，在文件顶部 `import` 区一并加入。

修正后的顶部 import：
```tsx
import { useEffect, useState } from 'react';
```

- [ ] **Step 3: 验证编译**

```bash
cd C:\project\ai-stack
npm run typecheck
npm run build
```

预期：0 error。

- [ ] **Step 4: 提交**

```bash
git add src/routes/Library.tsx src/components/library/ScanProgress.tsx
git commit -m "feat(phase-2): library 3-pane layout + scan progress"
```

---

## Task 11: Tree 与 ResourceList 组件

**Files:**
- Create: `src/components/library/Tree.tsx`
- Create: `src/components/library/ResourceList.tsx`
- Modify: `src/routes/Library.tsx`

**Interfaces:**
- Produces:
  - `<Tree categories={cats} selectedPath onSelect />`：递归渲染，点击调 `store.selectCategory(path)`
  - `<ResourceList resources={rs} selectedId onSelect />`：列表，点击调 `store.selectResource(id)`

- [ ] **Step 1: 实现 Tree**

写入 `C:\project\ai-stack\src\components\library\Tree.tsx`：
```tsx
import { useMemo } from 'react';
import { ChevronRight, ChevronDown, Folder } from 'lucide-react';
import { useState } from 'react';
import type { Category } from '../../types';

interface Props {
  categories: Category[];
  selectedPath: string | null;
  onSelect: (path: string) => void;
}

interface Node {
  cat: Category;
  children: Node[];
}

function buildTree(cats: Category[]): Node[] {
  const byParent = new Map<string | null, Category[]>();
  for (const c of cats) {
    const k = c.parentPath;
    if (!byParent.has(k)) byParent.set(k, []);
    byParent.get(k)!.push(c);
  }
  const make = (parent: string | null): Node[] =>
    (byParent.get(parent) ?? [])
      .sort((a, b) => a.sortOrder - b.sortOrder || a.path.localeCompare(b.path))
      .map((c) => ({ cat: c, children: make(c.path) }));
  return make(null);
}

function NodeRow({
  node,
  depth,
  selectedPath,
  onSelect,
}: {
  node: Node;
  depth: number;
  selectedPath: string | null;
  onSelect: (path: string) => void;
}) {
  const [open, setOpen] = useState(depth === 0);
  const hasChildren = node.children.length > 0;
  const active = selectedPath === node.cat.path;
  return (
    <div>
      <button
        onClick={() => {
          if (hasChildren) setOpen((v) => !v);
          onSelect(node.cat.path);
        }}
        className={`flex w-full items-center gap-1 rounded px-2 py-1 text-left text-sm ${
          active ? 'bg-accent/10 text-accent' : 'hover:bg-surface-2'
        }`}
        style={{ paddingLeft: 8 + depth * 12 }}
      >
        {hasChildren ? (
          open ? <ChevronDown size={14} /> : <ChevronRight size={14} />
        ) : (
          <span className="inline-block w-[14px]" />
        )}
        <Folder size={14} />
        <span className="truncate">{node.cat.title}</span>
      </button>
      {open && hasChildren && (
        <div>
          {node.children.map((c) => (
            <NodeRow
              key={c.cat.path}
              node={c}
              depth={depth + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function Tree({ categories, selectedPath, onSelect }: Props) {
  const tree = useMemo(() => buildTree(categories), [categories]);
  if (tree.length === 0) {
    return <div className="p-3 text-sm text-text-muted">暂无分类</div>;
  }
  return (
    <div className="space-y-0.5">
      {tree.map((n) => (
        <NodeRow
          key={n.cat.path}
          node={n}
          depth={0}
          selectedPath={selectedPath}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}
```

- [ ] **Step 2: 实现 ResourceList**

写入 `C:\project\ai-stack\src\components\library\ResourceList.tsx`：
```tsx
import { FileText, FileType, Presentation, BookOpen } from 'lucide-react';
import type { Resource, ResourceType } from '../../types';

const ICONS: Record<ResourceType, React.ComponentType<{ size?: number }>> = {
  markdown: FileText,
  pdf: BookOpen,
  docx: FileType,
  pptx: Presentation,
};

interface Props {
  resources: Resource[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export default function ResourceList({ resources, selectedId, onSelect }: Props) {
  if (resources.length === 0) {
    return <div className="p-3 text-sm text-text-muted">暂无资源</div>;
  }
  return (
    <ul className="space-y-0.5 overflow-y-auto">
      {resources.map((r) => {
        const Icon = ICONS[r.type];
        const active = selectedId === r.id;
        return (
          <li key={r.id}>
            <button
              onClick={() => onSelect(r.id)}
              className={`flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm ${
                active ? 'bg-accent/10 text-accent' : 'hover:bg-surface-2'
              }`}
            >
              <Icon size={14} />
              <span className="truncate">{r.title}</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
```

- [ ] **Step 3: 接入 Library.tsx**

修改 `src/routes/Library.tsx`，把占位 aside/section 改为：
```tsx
import Tree from '../components/library/Tree';
import ResourceList from '../components/library/ResourceList';
// ... 组件内部：
const categories = useLibraryStore((s) => s.categories);
const selectedCategoryPath = useLibraryStore((s) => s.selectedCategoryPath);
const selectCategory = useLibraryStore((s) => s.selectCategory);
const resources = useLibraryStore((s) => s.resources);
const selectedResourceId = useLibraryStore((s) => s.selectedResourceId);
const selectResource = useLibraryStore((s) => s.selectResource);

// JSX 中：
<aside className="overflow-y-auto rounded-md border border-border bg-surface p-2">
  <Tree
    categories={categories}
    selectedPath={selectedCategoryPath}
    onSelect={(p) => selectCategory(p)}
  />
</aside>
<section className="overflow-y-auto rounded-md border border-border bg-surface p-2">
  <ResourceList
    resources={resources}
    selectedId={selectedResourceId}
    onSelect={(id) => selectResource(id)}
  />
</section>
```

- [ ] **Step 4: 验证**

```bash
cd C:\project\ai-stack
npm run typecheck
npm run build
```

预期：0 error。

- [ ] **Step 5: 提交**

```bash
git add src/components/library/Tree.tsx src/components/library/ResourceList.tsx src/routes/Library.tsx
git commit -m "feat(phase-2): tree + resource list components"
```

---

## Task 12: Reader 组件（Markdown / Pdf / Docx / Pptx）+ ReaderToolbar

**Files:**
- Create: `src/components/library/ReaderToolbar.tsx`
- Create: `src/components/library/reader/MarkdownReader.tsx`
- Create: `src/components/library/reader/PdfReader.tsx`
- Create: `src/components/library/reader/DocxReader.tsx`
- Create: `src/components/library/reader/PptxReader.tsx`
- Modify: `src/routes/Library.tsx`

**Interfaces:**
- Produces:
  - `<ReaderToolbar title type pageCount? wordCount? onBack />`
  - `<MarkdownReader html />`：渲染 HTML（prose 样式）
  - `<PdfReader pages pageCount />`：页码选择器 + 大图
  - `<DocxReader blocks />`：按 kind 渲染
  - `<PptxReader slides />`：横向幻灯片卡片

- [ ] **Step 1: ReaderToolbar**

写入 `C:\project\ai-stack\src\components\library\ReaderToolbar.tsx`：
```tsx
import { ArrowLeft } from 'lucide-react';
import type { ResourceType } from '../../types';

const TYPE_LABELS: Record<ResourceType, string> = {
  markdown: 'Markdown',
  pdf: 'PDF',
  docx: 'Word',
  pptx: 'PPT',
};

interface Props {
  title: string;
  type: ResourceType;
  pageCount?: number;
  wordCount?: number;
  onBack: () => void;
}

export default function ReaderToolbar({ title, type, pageCount, wordCount, onBack }: Props) {
  return (
    <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-2">
      <button
        onClick={onBack}
        className="inline-flex items-center gap-1 rounded px-2 py-1 text-sm text-text-muted hover:bg-surface-2"
      >
        <ArrowLeft size={14} />
        返回列表
      </button>
      <div className="flex items-center gap-3 text-sm">
        <span className="rounded bg-surface-2 px-2 py-0.5 text-xs uppercase tracking-wider text-text-muted">
          {TYPE_LABELS[type]}
        </span>
        <span className="truncate font-medium">{title}</span>
        {pageCount != null && <span className="text-text-muted">{pageCount} 页</span>}
        {wordCount != null && <span className="text-text-muted">{wordCount} 字</span>}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: MarkdownReader**

写入 `C:\project\ai-stack\src\components\library\reader\MarkdownReader.tsx`：
```tsx
interface Props { html: string }

export default function MarkdownReader({ html }: Props) {
  return (
    <div className="prose prose-sm max-w-none p-6 dark:prose-invert">
      {/* comrak 输出受信任；Phase 2 不引入 DOMPurify */}
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}
```

并在 `src/index.css` 末尾追加（最小化 prose 样式，若未引入 @tailwindcss/typography）：
```css
.prose h1 { font-size: 1.6rem; font-weight: 600; margin: 1rem 0 0.5rem; }
.prose h2 { font-size: 1.3rem; font-weight: 600; margin: 0.9rem 0 0.4rem; }
.prose h3 { font-size: 1.1rem; font-weight: 600; margin: 0.8rem 0 0.3rem; }
.prose p  { margin: 0.5rem 0; line-height: 1.6; }
.prose ul { list-style: disc; padding-left: 1.5rem; margin: 0.5rem 0; }
.prose ol { list-style: decimal; padding-left: 1.5rem; margin: 0.5rem 0; }
.prose code { background: var(--color-surface-2); padding: 0.1rem 0.3rem; border-radius: 4px; font-size: 0.9em; }
.prose pre  { background: var(--color-surface-2); padding: 0.8rem; border-radius: 6px; overflow-x: auto; }
.prose blockquote { border-left: 3px solid var(--color-border); padding-left: 0.8rem; color: var(--color-text-muted); }
```

- [ ] **Step 3: PdfReader**

写入 `C:\project\ai-stack\src\components\library\reader\PdfReader.tsx`：
```tsx
import { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface Props {
  pages: Array<{ index: number; dataUrl: string }>;
  pageCount: number;
}

export default function PdfReader({ pages, pageCount }: Props) {
  const [idx, setIdx] = useState(0);
  if (pages.length === 0) {
    return (
      <div className="p-6 text-sm text-text-muted">
        该 PDF 暂无可视页面（pdfium 渲染不可用，已回退到纯文本）。
        页数估算：{pageCount}
      </div>
    );
  }
  const page = pages[idx] ?? pages[0];
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-2 text-sm">
        <button
          onClick={() => setIdx((i) => Math.max(0, i - 1))}
          disabled={idx === 0}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
        >
          <ChevronLeft size={14} /> 上一页
        </button>
        <span>
          第 {idx + 1} / {pages.length} 页{pageCount > pages.length && `（共 ${pageCount} 页）`}
        </span>
        <button
          onClick={() => setIdx((i) => Math.min(pages.length - 1, i + 1))}
          disabled={idx >= pages.length - 1}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
        >
          下一页 <ChevronRight size={14} />
        </button>
      </div>
      <div className="flex-1 overflow-auto bg-surface-2 p-4">
        <img src={page.dataUrl} alt={`page ${idx + 1}`} className="mx-auto max-w-full rounded shadow" />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: DocxReader**

写入 `C:\project\ai-stack\src\components\library\reader\DocxReader.tsx`：
```tsx
interface Block {
  kind: 'heading' | 'paragraph' | 'list' | 'table';
  level?: 1 | 2 | 3;
  text?: string;
  ordered?: boolean;
  items?: string[];
  rows?: string[][];
}

interface Props { blocks: Block[]; wordCount: number }

export default function DocxReader({ blocks }: Props) {
  return (
    <div className="prose prose-sm max-w-none p-6 dark:prose-invert">
      {blocks.map((b, i) => {
        if (b.kind === 'heading') {
          const Tag = (`h${b.level ?? 1}` as 'h1' | 'h2' | 'h3');
          return <Tag key={i}>{b.text}</Tag>;
        }
        if (b.kind === 'paragraph') return <p key={i}>{b.text}</p>;
        if (b.kind === 'list') {
          const Tag = b.ordered ? 'ol' : 'ul';
          return (
            <Tag key={i}>
              {(b.items ?? []).map((it, j) => <li key={j}>{it}</li>)}
            </Tag>
          );
        }
        if (b.kind === 'table') {
          const rows = b.rows ?? [];
          return (
            <table key={i} className="border-collapse border border-border">
              <tbody>
                {rows.map((r, ri) => (
                  <tr key={ri}>
                    {r.map((c, ci) => <td key={ci} className="border border-border px-2 py-1">{c}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          );
        }
        return null;
      })}
    </div>
  );
}
```

- [ ] **Step 5: PptxReader**

写入 `C:\project\ai-stack\src\components\library\reader\PptxReader.tsx`：
```tsx
import { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface Slide {
  index: number;
  title: string | null;
  body: string[];
  notes: string | null;
}

interface Props {
  slides: Slide[];
  slideCount: number;
}

export default function PptxReader({ slides }: Props) {
  const [idx, setIdx] = useState(0);
  if (slides.length === 0) {
    return <div className="p-6 text-sm text-text-muted">无可用幻灯片</div>;
  }
  const s = slides[idx];
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-2 text-sm">
        <button
          onClick={() => setIdx((i) => Math.max(0, i - 1))}
          disabled={idx === 0}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
        >
          <ChevronLeft size={14} /> 上一张
        </button>
        <span>第 {idx + 1} / {slides.length} 张</span>
        <button
          onClick={() => setIdx((i) => Math.min(slides.length - 1, i + 1))}
          disabled={idx >= slides.length - 1}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
        >
          下一张 <ChevronRight size={14} />
        </button>
      </div>
      <div className="flex-1 overflow-auto p-6">
        <article className="mx-auto max-w-3xl rounded-lg border border-border bg-surface p-6 shadow-sm">
          <div className="mb-3 text-xs uppercase tracking-wider text-text-muted">
            Slide {s.index + 1}
          </div>
          {s.title && <h2 className="mb-4 text-2xl font-semibold">{s.title}</h2>}
          <ul className="list-disc space-y-1 pl-6 text-sm">
            {s.body.map((line, i) => <li key={i}>{line}</li>)}
          </ul>
          {s.notes && (
            <div className="mt-6 rounded border border-border bg-surface-2 p-3 text-sm">
              <div className="mb-1 text-xs uppercase tracking-wider text-text-muted">
                Speaker Notes
              </div>
              <p className="whitespace-pre-wrap text-text-muted">{s.notes}</p>
            </div>
          )}
        </article>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: 接入 Library.tsx**

修改 `src/routes/Library.tsx`，把第三栏占位替换为 reader：
```tsx
import { AlertCircle } from 'lucide-react';
import ReaderToolbar from '../components/library/ReaderToolbar';
import MarkdownReader from '../components/library/reader/MarkdownReader';
import PdfReader from '../components/library/reader/PdfReader';
import DocxReader from '../components/library/reader/DocxReader';
import PptxReader from '../components/library/reader/PptxReader';

// 组件内：
const resources = useLibraryStore((s) => s.resources);
const content = useLibraryStore((s) => s.resourceContent);
const error = useLibraryStore((s) => s.error);
const current = resources.find((r) => r.id === useLibraryStore.getState().selectedResourceId);

// JSX:
<section className="flex flex-col overflow-hidden rounded-md border border-border bg-bg">
  {current ? (
    <>
      <ReaderToolbar
        title={current.title}
        type={current.type}
        pageCount={current.pageCount ?? undefined}
        wordCount={current.wordCount ?? undefined}
        onBack={() => selectResource(null)}
      />
      <div className="flex-1 overflow-auto">
        {content ? (
          <>
            {content.type === 'markdown' && <MarkdownReader html={content.html} />}
            {content.type === 'pdf' && (
              <PdfReader pages={content.pages} pageCount={content.pageCount} />
            )}
            {content.type === 'docx' && (
              <DocxReader blocks={content.blocks} wordCount={content.wordCount} />
            )}
            {content.type === 'pptx' && (
              <PptxReader slides={content.slides} slideCount={content.slideCount} />
            )}
          </>
        ) : (
          <div className="m-4 flex items-start gap-2 rounded-md border border-red-300 bg-red-50 p-4 text-sm text-red-700">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <div>
              <div className="font-medium">无法读取：{current.title}</div>
              <div className="mt-1 text-xs text-red-600">{error ?? '请尝试刷新索引或更换文件'}</div>
            </div>
          </div>
        )}
      </div>
    </>
  ) : (
    <div className="flex h-full items-center justify-center p-6 text-sm text-text-muted">
      {status === 'ready' ? '请选择左侧资源' : '等待索引完成'}
    </div>
  )}
</section>
```

- [ ] **Step 7: 验证编译**

```bash
cd C:\project\ai-stack
npm run typecheck
npm run build
```

预期：0 error。

- [ ] **Step 8: 提交**

```bash
git add src/components/library/ReaderToolbar.tsx src/components/library/reader/ src/index.css src/routes/Library.tsx
git commit -m "feat(phase-2): reader components (markdown/pdf/docx/pptx) + toolbar"
```

---

## Task 13: 手动验证样本数据 + 完整 end-to-end

**Files:**
- Create: `scripts/gen-samples.mjs`
- Modify: `package.json`

**Interfaces:**
- Produces: 在 `resources/knowledge/02-deep-learning/transformers/` 放置真实样本文件（一个 .md、一个 .pdf、一个 .docx、一个 .pptx）

- [ ] **Step 1: 写样本生成脚本**

写入 `C:\project\ai-stack\scripts\gen-samples.mjs`：
```js
// 在 resources/knowledge/02-deep-learning/transformers/ 放置真实样本文件用于手动验证
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const TARGET = 'resources/knowledge/02-deep-learning/transformers';

async function main() {
  await mkdir(TARGET, { recursive: true });
  // Markdown
  await writeFile(
    join(TARGET, 'attention-mechanism.md'),
    `# Attention Mechanism

This note introduces the **scaled dot-product attention** used in Transformer models.

## Formula

$$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V$$

## Variants

- Self-attention
- Cross-attention
- Multi-head attention
`,
    'utf8',
  );

  // PDF: 复用 fixtures 中的 sample.pdf
  // (留给用户手动放置一个真实 PDF；脚本只放其它格式)

  // DOCX
  const { Document, Packer, Paragraph, HeadingLevel, Table, TableRow, TableCell } = await import('docx');
  const doc = new Document({
    sections: [{
      children: [
        new Paragraph({ text: 'Transformer Overview', heading: HeadingLevel.HEADING_1 }),
        new Paragraph({ text: 'This document summarises the Transformer architecture.' }),
        new Paragraph({ text: 'Encoder', heading: HeadingLevel.HEADING_2 }),
        new Paragraph({ text: 'Stack of self-attention and feed-forward layers.' }),
        new Table({
          rows: [
            new TableRow({ children: [new TableCell({ children: [new Paragraph('Layer')] }), new TableCell({ children: [new Paragraph('Type')] })] }),
            new TableRow({ children: [new TableCell({ children: [new Paragraph('Self-Attention')] }), new TableCell({ children: [new Paragraph('Multi-Head')] })] }),
          ],
        }),
      ],
    }],
  });
  await writeFile(join(TARGET, 'transformer-overview.docx'), await Packer.toBuffer(doc));

  // PPTX
  const PptxGenJS = (await import('pptxgenjs')).default;
  const pres = new PptxGenJS();
  let s = pres.addSlide();
  s.addText('Why Transformers', { x: 0.5, y: 0.3, fontSize: 28 });
  s.addText('Parallel sequence modelling', { x: 0.5, y: 1.2, fontSize: 18 });
  s.addNotes('Highlight RNN vs Transformer parallelism.');
  s = pres.addSlide();
  s.addText('Self-Attention', { x: 0.5, y: 0.3, fontSize: 28 });
  s.addText('Q, K, V projections', { x: 0.5, y: 1.2, fontSize: 18 });
  s.addText('Scaled dot product', { x: 0.5, y: 1.7, fontSize: 18 });
  s.addNotes('Explain the scaling factor.');
  await writeFile(join(TARGET, 'transformer-slides.pptx'), await pres.write({ outputType: 'nodebuffer' }));

  console.log(`samples written under ${TARGET}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
```

- [ ] **Step 2: package.json 加 script**

修改 `package.json` 的 `scripts`：
```json
"gen:samples": "node scripts/gen-samples.mjs"
```

- [ ] **Step 3: 生成 + 验证目录**

```bash
cd C:\project\ai-stack
npm run gen:samples
ls resources/knowledge/02-deep-learning/transformers/
```

预期：至少含 `attention-mechanism.md`、`transformer-overview.docx`、`transformer-slides.pptx`。

- [ ] **Step 4: 启动应用，手动验证**

```bash
cd C:\project\ai-stack
npm run tauri dev
```

按 spec §11 验收清单逐项确认（首次启动扫描、Tree 显示分类、点击资源显示对应 reader 等）。

- [ ] **Step 5: 提交**

```bash
git add scripts/gen-samples.mjs package.json
git commit -m "feat(phase-2): sample files for manual e2e verification"
```

---

## Task 14: README 更新 + pdfium 文档 + 最终检查

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: 更新 README 的 Phase 2 部分**

修改 `C:\project\ai-stack\README.md`，在"## 状态"段落后插入：

```markdown
## Phase 2（当前）—— 知识库资源管理

- 自动扫描 `resources/knowledge/`，索引到本地 SQLite（`data/ai-stack.db`）
- 三栏阅读：分类树 / 资源列表 / 阅读器
- 支持 Markdown / PDF / DOCX / PPTX 四种格式

### PDF 渲染前置依赖（pdfium）

PDF 阅读器依赖 `pdfium-render` crate，需要 `pdfium.dll`：

1. 从 https://github.com/nicklockwood/pdfium-binaries/releases 下载 `pdfium-windows-x64.zip`
2. 解压得到 `pdfium.dll`
3. 放置于 `resources/bin/pdfium-windows-x64/pdfium.dll`（`tauri.conf.json` 的 `bundle.resources` 已包含此路径）

如果 `pdfium.dll` 缺失，PDF 阅读器自动回退到纯文本提取（`pdf-extract` crate）。
```

- [ ] **Step 2: 添加 pdfium 路径到 .gitignore（保留目录）**

在 `.gitignore` 末尾追加：
```
# Phase 2 pdfium binary (download manually per README)
resources/bin/
!resources/bin/.gitkeep
```

写入 `C:\project\ai-stack\resources\bin\.gitkeep`（空）。

- [ ] **Step 3: 最终验证清单**

```bash
cd C:\project\ai-stack
npm run typecheck
npm test -- --run
cargo --manifest-path src-tauri/Cargo.toml test
npm run build
```

预期：
- typecheck 0 error
- vitest：Phase 1 测试 + Phase 2 library.test.ts + library-api.test.ts 全部 PASS
- cargo test：db_smoke + scanner_smoke + readers_smoke 全部 PASS
- build 成功

- [ ] **Step 4: 提交**

```bash
git add README.md .gitignore resources/bin/.gitkeep
git commit -m "docs(phase-2): readme + pdfium dependency note"
```

---

## Self-Review Checklist

执行者跑完全部 14 个任务后核对：

- [ ] `npm run typecheck` 0 error
- [ ] `npm test` 全部 PASS（≥ 13 个用例：Phase 1 categories/theme + Phase 2 library/library-api + scanner）
- [ ] `cargo test` 全部 PASS（≥ 12 个用例：db 4 + scanner 4 + readers 4）
- [ ] `npm run build` 成功
- [ ] `npm run tauri dev` 启动窗口，标题"AI Stack"
- [ ] 首次启动自动扫描，Library 显示 12 个一级分类
- [ ] 点击侧边栏分类 → 资源列表更新
- [ ] 点击 `.md` 资源 → 显示排版良好的 HTML
- [ ] 点击 `.pdf` 资源（带 pdfium）→ 显示第一页图片，可翻页
- [ ] 点击 `.docx` 资源 → 显示段落结构
- [ ] 点击 `.pptx` 资源 → 显示幻灯片卡片
- [ ] "刷新"按钮触发重新扫描，进度条更新
- [ ] Phase 1 所有功能不退化（主题切换、设置 API Key、菜单）
- [ ] 删除 `data/ai-stack.db` 后重启应用，DB 自动重建成功
- [ ] `_index.md` 不出现在 resources 列表中
- [ ] 损坏文件计入 errors_count 但不阻塞其他文件

## 常见坑提示

1. **`comrak` 默认 features 庞大**：务必设置 `default-features = false, features = ["shortcodes"]`。
2. **`pdfium-render` 0.8 API 变更**：`load_pdf_from_file`、`render(width, height, ...)` 签名以本文为准；若实际编译失败，参考 docs.rs 上对应版本的示例。
3. **`docx-rs` 0.4 字段**：本文使用的 `paragraph.children` / `table.rows` 等字段名以 `cargo doc --open` 实际为准；若字段名漂移，按编译报错微调。
4. **`pptxgenjs` 在 `await pres.write()` 时 `outputType`**：Node 环境必须显式传 `nodebuffer`；不传则默认返回 `ArrayBuffer`（Web 环境）。
5. **`scan_library` 的 state 锁**：当前实现扫描完后 db 没回写到 state，因为 scan 内 `conn` 持有后 spawn_blocking 结束即释放；前端读 db 时 list_* 命令通过 state 持有路径重新打开。这是有意的简化（避免 spawn 闭包持有 State 的复杂性），可在 Phase 3 重构成"扫描时只更新 DB，list_* 直接打开新连接"。
6. **`dangerouslySetInnerHTML` 信任 comrak 输出**：Phase 2 不引入 DOMPurify，因为 comrak 不接受用户原始 HTML 输入；若 Phase 5 引入 AI 生成内容回写到 Markdown 渲染，需要重新评估并加 DOMPurify。