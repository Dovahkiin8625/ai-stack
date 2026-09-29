# AI Stack

Windows 平台上的 AI 学习工作台。在同一个桌面应用里完成：
- 浏览 AI 知识库（Markdown / PDF / Word / PPTX）
- 记笔记、关联到知识点
- 调用云端 LLM 解读知识
- 追踪学习进度

## 状态

**Phase 1（当前）**：应用骨架 + 12 类 AI 知识目录占位。
后续阶段：见 [docs/roadmap.md](./docs/roadmap.md)。

## Phase 2（当前）—— 知识库资源管理

- 自动扫描 `resources/knowledge/`，索引到本地 SQLite（`data/ai-stack.db`）
- 三栏阅读：分类树 / 资源列表 / 阅读器
- 支持 Markdown / PDF / DOCX / PPTX 四种格式

### PDF 渲染前置依赖（pdfium）

PDF 阅读器依赖 `pdfium-render` crate，需要 `pdfium.dll`：

1. 从 https://github.com/nicklockwood/pdfium-binaries/releases 下载 `pdfium-windows-x64.zip`
2. 解压得到 `pdfium.dll`
3. 放置于 `src-tauri/resources/bin/pdfium-windows-x64/pdfium.dll`（`tauri.conf.json` 的 `bundle.resources` 已包含此路径）

如果 `pdfium.dll` 缺失，PDF 阅读器自动回退到纯文本提取（`pdf-extract` crate）。

## 技术栈

- [Tauri 2](https://v2.tauri.app/) + Rust
- React 19 + TypeScript + Vite
- TailwindCSS / React Router / Zustand
- SQLite（后续阶段）
- tauri-plugin-store / window-state / fs / dialog / sql

## 前置依赖（Windows）

⚠️ **必装：Rust 工具链**（[rustup.rs](https://rustup.rs/)）—— 阶段 1 当前无法在未安装 Rust 的环境运行桌面窗口验证。

- Node.js ≥ 20
- Rust ≥ 1.78（[rustup](https://rustup.rs/)）
- WebView2 Runtime（Windows 11 自带）
- Microsoft Visual Studio C++ Build Tools（[下载](https://visualstudio.microsoft.com/visual-cpp-build-tools/)）

## 开发

```bash
npm install
npm run tauri dev      # 启动桌面应用（首次 5-15 分钟编译）
```

## 构建

```bash
npm run tauri build    # 产出安装包到 src-tauri/target/release/bundle/
```

## 测试与检查

```bash
npm run typecheck      # TypeScript 类型检查
npm test               # 单元测试（vitest）
npm run build          # 前端构建
```

## 文档

- [架构总览](./docs/architecture.md)
- [路线图](./docs/roadmap.md)
- [设计文档](./docs/superpowers/specs/2026-09-29-ai-stack-design.md)
- [阶段 1 实施计划](./docs/superpowers/plans/2026-09-29-ai-stack-phase-1-skeleton.md)
