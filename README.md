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

### 桌面端（Windows）

```bash
npm run build:windows            # 等价于 npx tauri build
```

产物（默认 release）：

| 类型 | 路径 |
|---|---|
| MSI 安装包 | `src-tauri/target/release/bundle/msi/*.msi` |
| NSIS 安装包 | `src-tauri/target/release/bundle/nsis/*-setup.exe` |
| 独立可执行 | `src-tauri/target/release/ai-stack.exe` |

加 `--debug` 出调试包。安装包**不再包含知识库**（218 MB 太大）—— 安装后首次启动需要知识库时，去设置页填「知识库同步地址」。

### Android

#### 一次性工具链搭建（手动）

在 Windows 上按顺序执行。命令在 PowerShell 5.1 / cmd 里都能跑。

**1. JDK 17**

浏览器打开 <https://adoptium.net/temurin/archive-17/?package=jdk>，下载 Windows x64 MSI，安装到默认路径 `C:\Program Files\Eclipse Adoptium\jdk-17.0.13.x-hotspot`。

新开 PowerShell，写用户级环境变量：

```powershell
[Environment]::SetEnvironmentVariable('JAVA_HOME', 'C:\Program Files\Eclipse Adoptium\jdk-17.0.13.11-hotspot', 'User')
```

**2. Android cmdline-tools**

下载 <https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip>，解压到 `%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\`（`latest` 这一层目录必须存在；解压后里面会再多一层 `cmdline-tools`，把里面的内容上移到 `latest/` 下）。

```powershell
[Environment]::SetEnvironmentVariable('ANDROID_HOME', "$env:LOCALAPPDATA\Android\Sdk", 'User')
[Environment]::SetEnvironmentVariable('ANDROID_SDK_ROOT', "$env:LOCALAPPDATA\Android\Sdk", 'User')
```

**3. 接受 license + 装平台组件**

新开 shell（让上面的环境变量生效）：

```cmd
%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\bin\sdkmanager --licenses
```

所有 license 问 `y/n` 时一路回车 `y`。

```cmd
%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\bin\sdkmanager "platform-tools"
%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\bin\sdkmanager "platforms;android-35"
%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\bin\sdkmanager "build-tools;35.0.0"
```

**4. NDK r27**

```cmd
%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\bin\sdkmanager "ndk;27.1.12297006"
[Environment]::SetEnvironmentVariable('NDK_HOME', "$env:LOCALAPPDATA\Android\Sdk\ndk\27.1.12297006", 'User')
```

把 NDK 的 clang 加到 PATH —— 这一步重要，否则 `ring` 依赖找不到工具链：

```powershell
[Environment]::SetEnvironmentVariable('Path', "$env:LOCALAPPDATA\Android\Sdk\ndk\27.1.12297006\toolchains\llvm\prebuilt\windows-x86_64\bin;$env:Path", 'User')
```

**5. Rust Android targets**

新开 shell：

```bash
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
```

**6. 验证**

```bash
cd C:\path\to\ai-stack\src-tauri
cargo check --target aarch64-linux-android
```

应当干净通过（除了可能的 Tauri capability 警告，见后文）。如果 `ring` 报 `clang.exe not found`，说明步骤 4 的 PATH 没生效，**新开 shell**。

#### 跑（dev）

USB 真机开调试后连电脑，或启动一个 AVD。`adb devices` 至少应看到一行 `device`：

```bash
npm run dev:android
```

这条等同 `npx tauri android dev`：Rust 编译一次后增量热重载，前端通过 Vite HMR 实时刷新。日志里会标出"Installing on device ... Started ..."以及 WebView 的远程调试 URL。

#### 打包（debug / 装到自己的设备）

```bash
npm run build:android -- --debug
```

debug 构建产物在 `src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-universal-debug.apk`（约 300 MB universal）。debug APK 已经是用 Android debug keystore 签过名的，可直接安装：

```bash
adb install -r "src-tauri\gen\android\app\build\outputs\apk\universal\debug\app-universal-debug.apk"
```

#### 打包（release / 分发）

```bash
npm run build:android
```

这条做的事：

1. `npx tauri android build --apk --target aarch64` —— Tauri 调 Gradle 出一个 **未签名** 的 `app-universal-release-unsigned.apk`（约 30-50 MB，arm64-only）
2. `scripts/build-android.mjs` 用 `apksigner` + `~/.android/debug.keystore`（密码 `android`，别名 `androiddebugkey`）把它签成 `app-universal-release.apk`

**debug keystore 自动生成的时机**：Android Gradle Plugin 在第一次构建 debug APK 时自动建好。直接跑 release 也行 —— `apksigner` 在签名前会自动去 debug 构建里调一次。

**正式分发时**，生成自己的 keystore：

```bash
keytool -genkey -v -keystore %USERPROFILE%\ai-stack-release.jks ^
  -keyalg RSA -keysize 2048 -validity 10000 -alias ai-stack
```

把口令填进 `src-tauri/tauri.conf.json` 的 `bundle.android.signing` 块，下次 `npm run build:android` 会自动签名。

### 知识库发布

```bash
npm run gen:manifest   # 输出 resources/dist/{manifest.json,knowledge/}
npm run serve:dist     # 本地静态服务器 http://127.0.0.1:8787/ 验证用
```

把 `resources/dist/` 整个传到任意静态托管（Nginx / 对象存储 / GitHub Release）。
设置页填的地址就是这个目录的 URL 根。

> ⚠️ Android 端访问服务端如果是明文 HTTP，需要在 `AndroidManifest.xml` 加 `android:usesCleartextTraffic="true"`（Tauri 2 当前模板已经在 debug 模式下默认允许，正式 release 需手动加）。

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
