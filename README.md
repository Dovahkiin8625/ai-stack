# AI Stack

Windows + Android 平台上的 AI 学习工作台。在同一个桌面应用里完成：
- 浏览 AI 知识库（Markdown / PDF / Word / PPTX）
- 记笔记、关联到知识点
- 调用云端 LLM 解读知识
- 追踪学习进度

**知识库不再随安装包发布**（218 MB 太大）。桌面端读 `resources/knowledge/`，Android APK 内置 65 个分类骨架（仅 `_index.md`，~177 KB），其余按需从远端同步。详见下文「构建」与 [docs/architecture.md](./docs/architecture.md) 的「知识库分发」一节。

## 状态

**Phase 1（当前）**：应用骨架 + 12 类 AI 知识目录占位。
后续阶段：见 [docs/roadmap.md](./docs/roadmap.md)。

## Phase 2（当前）—— 知识库资源管理

- 自动扫描 `resources/knowledge/`，索引到本地 SQLite（`data/ai-stack.db`）
- 三栏阅读：分类树 / 资源列表 / 阅读器
- 支持 Markdown / PDF / DOCX / PPTX 四种格式

### PDF 渲染

PDF 页面渲染由前端 `pdf.js`（`pdfjs-dist`）承担，无 native 依赖；
后端 `pdf-extract` 只负责数页和文本提取。

## 技术栈

- [Tauri 2](https://v2.tauri.app/) + Rust
- React 19 + TypeScript + Vite
- TailwindCSS / React Router / Zustand
- SQLite（后续阶段）
- tauri-plugin-store / window-state / fs / dialog / sql

## 前置依赖（Windows）

⚠️ **必装：Rust 工具链**（[rustup.rs](https://rustup.rs/)）—— 阶段 1 当前无法在未安装 Rust 的环境运行桌面窗口验证。

- Node.js ≥ 20
- Rust ≥ 1.78（[rustup.rs](https://rustup.rs/)）
- WebView2 Runtime（Windows 11 自带）
- Microsoft Visual Studio C++ Build Tools（[下载](https://visualstudio.microsoft.com/visual-cpp-build-tools/)）

## 开发

```bash
npm install
npm run tauri dev      # 启动桌面应用（首次 5-15 分钟编译）
```

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
npx tauri android build --apk --target aarch64
```

产物：`src-tauri/gen/android/app/build/outputs/apk/universal/release/app-universal-release-unsigned.apk`

**签名**：本仓库不内置生产 keystore。sideload 装到自己的设备时，用 Android SDK 自带的 `apksigner` 工具 + `~/.android/debug.keystore`（密码 `android`，别名 `androiddebugkey`）签一次即可：

```cmd
"%LOCALAPPDATA%\Android\Sdk\build-tools\35.0.0\apksigner.bat" sign ^
  --ks %USERPROFILE%\.android\debug.keystore ^
  --ks-pass pass:android --key-pass pass:android ^
  --ks-key-alias androiddebugkey ^
  --out app-universal-release.apk app-universal-release-unsigned.apk
```

要正式分发时，生成自己的 keystore 并在 `src-tauri/tauri.conf.json` 的 `bundle.android.signing` 块里配好，下次构建会自动签名。

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
- [Android 支持与知识库远程分发计划](./docs/superpowers/plans/2026-10-08-android-support.md)
