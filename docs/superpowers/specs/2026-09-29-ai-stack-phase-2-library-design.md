# AI Stack — Phase 2 知识库资源管理 设计文档

| 项 | 值 |
|---|---|
| 日期 | 2026-09-29 |
| 路径 | `C:\project\ai-stack` |
| 阶段 | Phase 2（实施） |
| 状态 | 待评审 |
| 前置 | `2026-09-29-ai-stack-design.md`、Phase 1 已完成 |

## 1. 目标与成功标准

**目标：** 在 Phase 1 骨架之上，让用户能浏览 `resources/knowledge/` 下的 AI 知识目录树、选择文档（Markdown / PDF / Word / PPTX），在应用内阅读其内容。

**成功标准：**

- 应用启动后自动扫描 `resources/knowledge/`，将分类和资源写入 `data/ai-stack.db`，期间前端显示进度。
- `/library` 页面提供三栏布局：分类树、资源列表、阅读器。分类与资源均来自 SQLite，而非前端硬编码。
- 点击任一资源，2 秒内（本地文件，5MB 以内）显示内容。Markdown 渲染为排版良好的 HTML；PDF 按页栅格化为图片；DOCX / PPTX 提取结构化内容（段落、标题、表格、幻灯片文字与备注）后渲染为样式化 HTML / 幻灯片卡片。
- 新增、修改、删除 `resources/knowledge/` 下任意文件后，通过"刷新"按钮可重建索引（耗时与文件数成正比）。
- 损坏文件或读取失败不阻塞其他资源，UI 显示明确的错误信息。
- Phase 1 所有验收项不退化（`npm run typecheck` / `npm test` / `npm run build` / `npm run tauri dev` 全部通过）。

**非目标（YAGNI）：**

- 用户自定义目录（留到 Phase 7）。
- 文件系统 watcher / 增量监听。
- PDF 全文检索、书签、注释。
- DOCX 主题/字体的像素级保真。
- 笔记关联与进度追踪（Phase 3、4）。
- AI 解读（Phase 5）。

## 2. 总体架构

```
┌────────────────────────────────────────────────────────────┐
│                  React/TS 前端 (src/)                      │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐ │
│  │ 阅读器   │ 笔记     │ 仪表盘   │ 知识库   │ 设置     │ │
│  │          │ (Phase3) │ (Phase4) │ Phase 2  │ (Phase1) │ │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘ │
│           Tauri IPC (invoke + events)                      │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────┴───────────────────────────────────┐
│            Rust 后端 (src-tauri/)                          │
│  ┌────────────┬────────────┬────────────┬────────────┐    │
│  │ 数据库     │ 扫描器     │ 阅读器     │ 迁移        │    │
│  │ (rusqlite) │ (walkdir)  │ (md/pdf/   │ (in-app)   │    │
│  │            │            │  docx/     │            │    │
│  │            │            │  pptx)     │            │    │
│  └────────────┴────────────┴────────────┴────────────┘    │
└────────────────────────┬───────────────────────────────────┘
                         │
        ┌────────────────┼──────────────────┐
        ▼                ▼                  ▼
   resources/       data/ai-stack.db    pdfium native
   knowledge/       (SQLite)           (PDF 渲染)
```

**关键边界：**

- 所有文件 I/O、SQL、二进制解析、PDF 栅格化均在 Rust 完成。
- 前端只消费 JSON：`{type, content}` 形态的结构化数据。**前端不直接持有 `data/ai-stack.db` 连接**。
- 不使用 `tauri-plugin-sql`：SQL 操作全部走自定义 Tauri command，避免插件版本与 rusqlite 版本冲突，并保证 schema 迁移与 Rust 代码同源。

## 3. 文件结构

```
src-tauri/src/
├── main.rs                  (Phase 1)
├── lib.rs                   (Phase 1, 注册新 commands)
├── menu.rs                  (Phase 1)
├── db.rs                    [新] 连接、迁移、CRUD
├── scanner.rs               [新] 文件系统遍历、索引
├── reader.rs                [新] 格式分发
├── readers/
│   ├── mod.rs
│   ├── markdown.rs          [新] comrak → HTML
│   ├── pdf.rs               [新] pdfium-render → pages (PNG base64)
│   ├── docx.rs              [新] docx-rs → 结构化 blocks
│   └── pptx.rs              [新] zip + quick-xml → slides
└── commands.rs              [新] Tauri 命令入口

src/
├── lib/
│   ├── tauri.ts             (Phase 1, 扩展)
│   └── library-api.ts       [新] invoke 包装、类型化返回
├── stores/
│   ├── theme.ts             (Phase 1)
│   └── library.ts           [新] Zustand: 扫描状态、树、选中
├── components/
│   ├── layout/              (Phase 1)
│   ├── ui/                 (Phase 1)
│   └── library/
│       ├── Tree.tsx         [新] 递归分类树
│       ├── ResourceList.tsx [新] 资源列表
│       ├── ScanProgress.tsx [新] 启动扫描进度条
│       └── reader/
│           ├── MarkdownReader.tsx
│           ├── PdfReader.tsx
│           ├── DocxReader.tsx
│           └── PptxReader.tsx
├── routes/
│   ├── Library.tsx          [改] 三栏布局
│   ├── Notes.tsx            (Phase 1, 不动)
│   ├── Dashboard.tsx        (Phase 1, 不动)
│   └── Settings.tsx         (Phase 1, 不动)
└── types/
    └── index.ts             [改] 新增 Library 类型

resources/knowledge/         (Phase 1, 仅添加测试样本文件)
tests/fixtures/              [新] 单元测试用 .md/.pdf/.docx/.pptx 小样本
docs/superpowers/
└── plans/
    └── 2026-09-29-ai-stack-phase-2-library.md   [后继, writing-plans 产出]
```

## 4. 数据模型

SQLite 表 schema（`db.rs` 内嵌迁移，逐版本号顺序应用）：

```sql
-- schema_version 表，迁移自校验
CREATE TABLE schema_version (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

-- 分类（来自目录树 + _index.md 第一行 # 标题）
CREATE TABLE categories (
  path TEXT PRIMARY KEY,           -- "03-large-language-models/rag"
  parent_path TEXT,                -- NULL 表示一级分类
  title TEXT NOT NULL,             -- 来自 _index.md 的 # heading，否则人化目录名
  sort_order INTEGER NOT NULL,     -- 解析自路径前缀 01-, 02-, ...
  FOREIGN KEY (parent_path) REFERENCES categories(path)
);
CREATE INDEX idx_categories_parent ON categories(parent_path);

-- 资源（来自扫描的文件）
CREATE TABLE resources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category_path TEXT NOT NULL,     -- 所属分类路径
  rel_path TEXT NOT NULL,          -- 相对 knowledge 根，例如 "transformers/attention-is-all-you-need.pdf"
  type TEXT NOT NULL,              -- 'markdown' | 'pdf' | 'docx' | 'pptx'
  title TEXT NOT NULL,             -- Markdown 取首个 # heading；其他格式取文件名 stem
  size_bytes INTEGER NOT NULL,
  hash TEXT NOT NULL,              -- sha256，用于变更检测
  indexed_at TEXT NOT NULL,        -- ISO8601 UTC
  page_count INTEGER,              -- PDF / PPTX 填充
  word_count INTEGER,              -- 段落级统计；Markdown/DOCX 填充
  UNIQUE(category_path, rel_path),
  FOREIGN KEY (category_path) REFERENCES categories(path)
);
CREATE INDEX idx_resources_category ON resources(category_path);
CREATE INDEX idx_resources_hash ON resources(hash);
```

**`_index.md` 不作为 resource 索引**：扫描器识别每个目录下 `_index.md` 时，仅用于解析该分类的 `title`，文件本身不入 `resources` 表。

## 5. 关键交互流程

### 5.1 应用启动

1. `App.tsx` 挂载 → 调 `scan_library({ force: false })`。
2. Rust 端：
   - 若 DB 文件不存在或 schema 版本低 → 运行迁移（创建表 + 插入 `schema_version`）。
   - 后台线程（`tauri::async_runtime::spawn_blocking`）遍历 `resources/knowledge/`。
   - 每处理一个文件 / 分类，向前端 `emit("scan_progress", {phase, current, total, current_path?})`。
   - 完成后返回 `{categories_count, resources_count, errors_count, duration_ms}`。
3. 前端 `library` store 接收 `scan_progress` 更新进度条；扫描完成后置 `status: 'ready'`，拉取分类树与初始资源。
4. 用户在扫描完成前进入 `/library` → 显示 `ScanProgress` 占位，不显示空树。

### 5.2 手动刷新

- `Library.tsx` 顶部"刷新"按钮 → 调 `scan_library({ force: true })`。
- `force: true` 时清空 `resources` / `categories` 表后重新插入；`force: false` 走 upsert（hash 不变跳过，hash 变更新，文件消失删除）。
- 扫描期间按钮置 `disabled` 并显示 spinner。

### 5.3 打开资源

1. 用户在 `ResourceList` 点击某资源 → store 记录 `selected_resource_id`。
2. 调 `read_resource(id)` → 返回 `ResourceContent`：
   ```ts
   type ResourceContent =
     | { type: 'markdown'; html: string; word_count: number }
     | { type: 'pdf'; pages: Array<{ index: number; data_url: string }>; page_count: number }
     | { type: 'docx'; blocks: DocxBlock[]; word_count: number }
     | { type: 'pptx'; slides: PptxSlide[]; slide_count: number };
   ```
3. `Library.tsx` 按 `type` 路由到对应 `Reader` 组件。
4. PDF 数据为 base64 PNG（单页 144 DPI，约 100-400 KB），首屏即时显示，其余分页懒加载（一次最多 10 页到内存）。

### 5.4 错误处理

| 场景 | 行为 |
|---|---|
| DB 打开失败 | 弹窗"无法打开数据库 <path>:<err>"，应用退出 |
| Schema 迁移失败 | 同上 |
| 单个文件扫描失败 | `eprintln!`，计入 `errors_count`，UI 显示黄色徽章"X 个文件跳过" |
| 单个文件读取失败 | Reader 显示错误卡"无法读取：<文件名>\n<原因>"，其他资源不受影响 |
| pdfium native 库缺失 | 启动时 `eprintln!` 警告，PDF 回退到 `pdf-extract` 纯文本模式，reader 显示提示 |
| SQLite 写失败 | Tauri 命令返回 `Err(String)`，前端 store 标 `status: 'error'`，显示重试按钮 |

所有 Tauri 命令返回 `Result<T, String>`（`String` 是错误信息），前端统一捕获并通过 toast 或 banner 提示。

## 6. Rust 依赖

`Cargo.toml` 新增：

```toml
rusqlite = { version = "0.32", features = ["bundled"] }   # 静态链接 SQLite，无需系统依赖
walkdir = "2"
sha2 = "0.10"
hex = "0.4"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde"] }
comrak = { version = "0.22", default-features = false, features = ["shortcodes"] }  # Markdown → HTML
pdfium-render = "0.8"                                     # PDF → pages
docx-rs = "0.4"                                            # DOCX → 结构
zip = "2"                                                  # PPTX 解包
quick-xml = { version = "0.36", features = ["serialize"] } # PPTX XML 解析
pulldown-cmark = "0.12"                                    # word_count 估算（兜底）
once_cell = "1"
base64 = "0.22"
anyhow = "1"
thiserror = "1"
```

`pdfium-render` 需要系统存在 pdfium 动态库（`pdfium.dll` on Windows）。在 `tauri.conf.json` 的 `bundle.resources` 中包含 `resources/bin/pdfium-windows-x64/pdfium.dll`；开发期由 README 提示用户从 pdfium-render releases 下载放置。

如果 pdfium 不可用，扫描阶段检测后通过 `READER_PDF_FALLBACK=true` 环境变量回退到 `pdf-extract` crate（仅文本）。

## 7. 前端依赖

`package.json` 不新增运行时依赖 —— Markdown 由 Rust 端 comrak 渲染为 HTML 字符串后直接注入 React（`MarkdownReader` 使用 `dangerouslySetInnerHTML`，仅对 comrak 输出信任；Phase 2 不引入 DOMPurify）。

不使用 mammoth.js / pdf.js / pptxjs（Rust 端已处理）。前端只渲染已结构化的数据。

仅在开发依赖新增 fixture 生成相关包（见 §10）。

## 8. 类型定义

`src/types/index.ts` 扩展：

```ts
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
  | { type: 'pdf'; pages: Array<{ index: number; dataUrl: string }>; pageCount: number }
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

## 9. 阅读器组件契约

| 组件 | 输入 | 行为 |
|---|---|---|
| `MarkdownReader` | `{ html: string }` | 渲染 comrak 输出（已为 HTML 字符串），应用 prose 类样式（Tailwind typography 或自写 markdown.css） |
| `PdfReader` | `{ pages, pageCount }` | 顶部页码选择器 + 当前页大图；左右键翻页；初始只显示第一页 |
| `DocxReader` | `{ blocks }` | 按 block kind 渲染对应 React 元素（heading、paragraph、list、table）；样式参考 Tailwind |
| `PptxReader` | `{ slides, slideCount }` | 横向卡片轮播：每张卡片显示 slide index、title、body、备注 |

所有 Reader 共用 `ReaderToolbar`（顶栏）：显示 `title`、类型徽章、字数 / 页数统计、"返回列表"按钮。

## 10. 测试

### Rust（`cargo test`）

```
src-tauri/src/scanner.rs        tests:
  - walk_known_tree              # 固定 fixture 树，断言 categories + resources 数量
  - index_incremental             # 二次扫描 hash 不变 → upsert 跳过
  - index_detects_deletion        # 删除 fixture 文件 → DB 行被清除
  - index_detects_modification    # 修改文件 hash 变 → 重新索引

src-tauri/src/db.rs             tests:
  - migrations_idempotent        # 重复运行迁移版本号仍正确
  - upsert_resource              # 同 (category_path, rel_path) 二次插入更新而非新增
  - list_by_category             # 按 category_path 查询返回正确顺序

src-tauri/src/readers/markdown.rs tests:
  - render_simple                # "# H\n\ntext" → "<h1>H</h1><p>text</p>"
  - render_gfm_table             # 表格语法 → 含 <table>

src-tauri/src/readers/docx.rs   tests:
  - extract_paragraphs           # fixture .docx → 段落数组
  - extract_headings             # Heading 1/2/3 区分

src-tauri/src/readers/pptx.rs   tests:
  - extract_slides               # fixture .pptx → slides 数组
  - extract_notes                # speaker notes 单独字段

src-tauri/src/readers/pdf.rs    tests:
  - rasterize_pages              # fixture .pdf → 至少 1 页 PNG bytes
  - fallback_text_only           # 模拟 pdfium 不可用 → 返回纯文本
```

测试 fixtures 在 `src-tauri/tests/fixtures/`：
- `knowledge-tree/`（微型目录树，含 `01-cat-a/01-sub/_index.md`、`01-cat-a/01-sub/note.md`）
- `sample.pdf`（PDFium 官方示例，~50KB）
- `sample.docx`（python-docx 脚本生成，包含 headings、paragraphs、table）
- `sample.pptx`（python-pptx 脚本生成，2 张 slide + notes）

fixtures 生成脚本：`scripts/gen-fixtures.mjs`（Node，使用 `docx` 和 `pptxgenjs` 仅作为 devDependency 生成 fixture；不打包到前端）。脚本写入：
- `src-tauri/tests/fixtures/knowledge-tree/`（微型目录树）
- `src-tauri/tests/fixtures/sample.pdf`（由仓库内置的 ~50KB 测试 PDF 复制）
- `src-tauri/tests/fixtures/sample.docx`
- `src-tauri/tests/fixtures/sample.pptx`

脚本运行方式：`npm run gen:fixtures`（package.json 新增该 script，调用 `node scripts/gen-fixtures.mjs`）。

### Vitest

```
src/stores/library.test.ts
  - scan state transitions: idle → scanning → ready
  - selected category updates tree selection
  - selected resource triggers content fetch

src/lib/library-api.test.ts
  - wrappers pass args through to invoke correctly
  - error mapping for Result<_, String>
```

### 手动验证（无自动化）

启动应用后：

1. 删除 `data/ai-stack.db`，重启 → 应用重建 DB，Library 显示 12 个一级分类。
2. 在 `resources/knowledge/02-deep-learning/transformers/` 复制一份样本 PDF → 点击"刷新" → 列表出现该 PDF → 打开 → 显示首页。
3. 修改某 `.md` 文件 → 刷新 → `indexed_at` 更新。
4. 删除某文件 → 刷新 → 列表中消失。
5. 关闭 `tauri.conf.json` 中的 pdfium 资源（模拟缺失）→ 启动 → 控制台有警告 → PDF 仍能打开（纯文本模式）。
6. 把一个损坏的 `.docx` 放入 → 刷新 → 列表中显示该文件但打开时显示错误卡。

## 11. 验收清单（Self-Review）

- [ ] `npm run typecheck` 0 error
- [ ] `npm test` 全部 PASS（Phase 1 + 新增）
- [ ] `cargo test` 全部 PASS（≥ 12 个用例）
- [ ] `npm run build` 成功
- [ ] `npm run tauri dev` 启动窗口，标题"AI Stack"
- [ ] 首次启动扫描完成（≤ 5 秒）后，Library 显示 12 个一级分类
- [ ] 侧边栏点击分类 → Library 中间栏列出该分类下资源
- [ ] 点击 `.md` 资源 → 显示排版良好的 HTML
- [ ] 点击 `.pdf` 资源 → 显示第一页图片，可翻页
- [ ] 点击 `.docx` 资源 → 显示段落结构
- [ ] 点击 `.pptx` 资源 → 显示幻灯片卡片
- [ ] "刷新"按钮触发重新扫描，期间显示进度
- [ ] 损坏文件不阻塞其他资源
- [ ] Phase 1 所有功能不退化（主题、设置、菜单）
- [ ] 删除 `data/ai-stack.db` 后重启应用，DB 重建成功
- [ ] pdfium 缺失时，PDF 仍能打开（文本回退）

## 12. 风险与缓解

| 风险 | 缓解 |
|---|---|
| pdfium 动态库跨平台分发复杂 | `tauri.conf.json` bundle.resources 包含 pdfium.dll；README 给出下载链接与回退方案（`pdf-extract`） |
| rusqlite `bundled` 增大二进制 | 可接受（增量约 1.5MB），避免 SQLite 系统依赖 |
| 大量小文件扫描慢 | scan 在 spawn_blocking 异步执行；emit progress 事件；后续可加并发 |
| DOCX 主题保真差 | 在 Reader 中显示"提示：DOCX 仅保留文本与结构，样式未完全还原" |
| PPTX 嵌入图片 / 图表 | Phase 2 仅提取文本与备注；图片忽略（TODO Phase 6 检索时再考虑） |
| 应用启动时阻塞太久 | 启动扫描由后端异步执行，前端 ScanProgress 显示进度；Library 路由切换不阻塞 |

## 13. 阶段拆分（高层，仅用于 writing-plans）

后续 writing-plans 阶段会基于本文产出 Task 1..N 的实施步骤，预计任务结构：

1. Rust 端：依赖、数据库迁移、扫描器、Markdown 阅读器、PDF 阅读器、DOCX 阅读器、PPTX 阅读器、Tauri 命令注册
2. 前端：API 包装、library store、Library 三栏布局、Tree、ResourceList、Reader 组件、ScanProgress
3. 测试：Rust fixture 生成与单测、Vitest store 测试、pdfium 回退验证
4. 文档：README 更新、phase-2 plan 文档

---

**待办：** 用户评审本文档 → 通过后转入 writing-plans，写 `2026-09-29-ai-stack-phase-2-library.md`。