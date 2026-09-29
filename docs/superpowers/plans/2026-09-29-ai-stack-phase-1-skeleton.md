# AI Stack — Phase 1 应用骨架 + 知识目录 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Windows 11 上构建 AI Stack 桌面应用的阶段 1 骨架：可启动的 Tauri + React/TS 桌面窗口、基础布局/路由/主题切换、12 类 AI 知识目录（空内容）。

**Architecture:** Tauri 2.x（Rust 后端）+ Vite + React 19 + TypeScript + TailwindCSS 4 + React Router + Zustand。Rust 负责窗口/菜单/插件，前端负责 UI/路由/主题。知识目录是纯文件系统占位，无内容。

**Tech Stack:**
- Tauri 2.x
- Vite 5+
- React 19 + TypeScript 5+
- TailwindCSS 4
- React Router 7+
- Zustand 5+
- lucide-react（图标）
- tauri-plugin-store / tauri-plugin-window-state / tauri-plugin-fs / tauri-plugin-dialog / tauri-plugin-sql

**Spec:** `docs/superpowers/specs/2026-09-29-ai-stack-design.md`

---

## Global Constraints

- 工作目录固定：`C:\project\ai-stack`
- 平台：Windows 11 (WebView2 必需)
- Node.js ≥ 20，Rust ≥ 1.78（参考 Tauri 官方要求）
- 应用 ID：`com.aistack.app`，应用名：`AI Stack`，窗口标题：`AI Stack`
- 路径命名：代码文件 kebab-case 或 PascalCase（按 React 习惯），目录用 kebab-case
- 所有提交信息使用中文或英文皆可，统一前缀：`feat:`、`chore:`、`docs:`
- 不在阶段 1 引入业务后端逻辑（数据库、AI 调用）
- `.gitignore` 必须忽略：`node_modules/`、`dist/`、`src-tauri/target/`、`data/`、`.env`

## Review Focus

阶段 1 是纯骨架，缺少业务逻辑，但下面五类输入/场景若阶段 1 处理不当，会拖慢后续阶段或埋雷，必须在对应任务里覆盖：

1. **首次启动无设置文件** — 应用应优雅降级到默认主题（亮色）、无 API Key 状态，而非崩溃（任务 5、7）
2. **Windows 11 上 WebView2 缺失或损坏** — 启动失败应有友好错误而非 panic（任务 1 文档需声明前置依赖）
3. **Tauri 菜单中文路径与 emoji 渲染** — 菜单文字含中文/符号，UI 不能乱码（任务 8）
4. **用户把窗口拉到极小尺寸** — 侧边栏与主区不能错位（任务 4 最小宽度约束）
5. **Rust 工具链未装或版本错** — 构建报错应可读，不是 linker 失踪类乱码（任务 1 文档 + 友好报错）

---

## File Structure (Phase 1)

```
ai-stack/
├── docs/
│   ├── superpowers/
│   │   ├── specs/2026-09-29-ai-stack-design.md     (已存在)
│   │   └── plans/2026-09-29-ai-stack-phase-1-skeleton.md   (本文件)
│   ├── architecture.md                              (Task 10)
│   └── roadmap.md                                   (Task 10)
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── index.css                                    (Tailwind + 主题变量)
│   ├── routes/
│   │   ├── Library.tsx
│   │   ├── Notes.tsx
│   │   ├── Dashboard.tsx
│   │   └── Settings.tsx
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Layout.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── Topbar.tsx
│   │   └── ui/
│   │       └── ThemeToggle.tsx
│   ├── stores/
│   │   └── theme.ts
│   ├── data/
│   │   └── categories.ts                            (12 分类定义)
│   ├── lib/
│   │   ├── tauri.ts                                 (插件 invoke 包装)
│   │   └── categories.test.ts                       (vitest)
│   └── types/
│       └── index.ts
├── src-tauri/
│   ├── src/
│   │   ├── main.rs
│   │   ├── lib.rs
│   │   └── menu.rs
│   ├── tauri.conf.json
│   ├── Cargo.toml
│   ├── capabilities/default.json
│   └── icons/                                       (Tauri 默认图标)
├── resources/
│   └── knowledge/                                   (Task 9 一次性创建)
├── public/
├── .gitignore
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── vitest.config.ts
└── README.md
```

每个文件单一职责。`categories.ts` 同时提供 TS 与测试数据，单一数据源；主题 store 单独成文件便于测试。

---

## Task 1: 项目初始化（Tauri + Vite + React + TS）

**Files:**
- Create: `package.json`, `vite.config.ts`, `tsconfig.json`, `tsconfig.node.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/vite-env.d.ts`, `.gitignore`
- Create: `src-tauri/Cargo.toml`, `src-tauri/tauri.conf.json`, `src-tauri/src/main.rs`, `src-tauri/src/lib.rs`, `src-tauri/build.rs`, `src-tauri/capabilities/default.json`, `src-tauri/icons/*`

**Interfaces:**
- Consumes: 无
- Produces: `npm run tauri dev` 能启动窗口，标题 `AI Stack`，窗口尺寸 1280×800

- [ ] **Step 1: 确认前置依赖**

在终端运行：
```bash
node --version    # 应 >= 20
npm --version
rustc --version   # 应 >= 1.78
cargo --version
```
若 Rust 未装：指引用户访问 https://rustup.rs/ 安装；WebView2 应已自带（Windows 11）。若 `cargo --version` 报错 PATH 不存在，重新打开终端或 `source $HOME/.cargo/env`。

- [ ] **Step 2: 创建项目根文件**

写入 `C:\project\ai-stack\package.json`：
```json
{
  "name": "ai-stack",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "tauri": "tauri",
    "typecheck": "tsc --noEmit",
    "test": "vitest run"
  },
  "dependencies": {
    "@tauri-apps/api": "^2.1.1",
    "@tauri-apps/plugin-dialog": "^2.0.1",
    "@tauri-apps/plugin-fs": "^2.0.3",
    "@tauri-apps/plugin-sql": "^2.0.2",
    "@tauri-apps/plugin-store": "^2.1.0",
    "@tauri-apps/plugin-window-state": "^2.0.1",
    "lucide-react": "^0.460.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-router-dom": "^7.0.2",
    "zustand": "^5.0.2"
  },
  "devDependencies": {
    "@tauri-apps/cli": "^2.1.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.4",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.49",
    "tailwindcss": "^3.4.16",
    "typescript": "^5.7.2",
    "vite": "^5.4.11",
    "vitest": "^2.1.8"
  }
}
```

写入 `C:\project\ai-stack\.gitignore`：
```
node_modules/
dist/
dist-ssr/
*.local
.env
.env.*
!.env.example

# Rust / Tauri
src-tauri/target/
src-tauri/Cargo.lock
src-tauri/gen/

# 用户数据
data/

# 编辑器
.vscode/*
!.vscode/extensions.json
.idea/
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?
```

写入 `C:\project\ai-stack\vite.config.ts`：
```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? { protocol: 'ws', host, port: 1421 }
      : undefined,
    watch: { ignored: ['**/src-tauri/**'] },
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: 'es2021',
    minify: 'esbuild',
    sourcemap: false,
  },
});
```

写入 `C:\project\ai-stack\tsconfig.json`：
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "Bundler",
    "allowImportingTsExtensions": false,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "types": ["vite/client", "vitest/globals"]
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

写入 `C:\project\ai-stack\tsconfig.node.json`：
```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
```

写入 `C:\project\ai-stack\index.html`（放在根，Tauri 读取此文件）：
```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI Stack</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

写入 `C:\project\ai-stack\src\main.tsx`：
```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

写入 `C:\project\ai-stack\src\App.tsx`（骨架，后续任务扩展）：
```tsx
export default function App() {
  return (
    <div className="flex h-screen items-center justify-center bg-white text-gray-900 dark:bg-gray-900 dark:text-gray-100">
      <h1 className="text-2xl font-semibold">AI Stack</h1>
    </div>
  );
}
```

写入 `C:\project\ai-stack\src\vite-env.d.ts`：
```ts
/// <reference types="vite/client" />
```

- [ ] **Step 3: 创建 Tauri 配置**

写入 `C:\project\ai-stack\src-tauri\Cargo.toml`：
```toml
[package]
name = "ai-stack"
version = "0.1.0"
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
```

写入 `C:\project\ai-stack\src-tauri\tauri.conf.json`：
```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "AI Stack",
  "version": "0.1.0",
  "identifier": "com.aistack.app",
  "build": {
    "frontendDist": "../dist",
    "devUrl": "http://localhost:1420",
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build"
  },
  "app": {
    "windows": [
      {
        "title": "AI Stack",
        "width": 1280,
        "height": 800,
        "minWidth": 900,
        "minHeight": 600,
        "resizable": true,
        "fullscreen": false,
        "center": true
      }
    ],
    "security": {
      "csp": null
    }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.ico"
    ]
  }
}
```

写入 `C:\project\ai-stack\src-tauri\build.rs`：
```rust
fn main() {
    tauri_build::build()
}
```

写入 `C:\project\ai-stack\src-tauri\src\main.rs`：
```rust
// Prevents additional console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    ai_stack_lib::run()
}
```

写入 `C:\project\ai-stack\src-tauri\src\lib.rs`：
```rust
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_sql::Builder::default().build())
        .run(tauri::generate_context!())
        .expect("error while running ai-stack application");
}
```

写入 `C:\project\ai-stack\src-tauri\capabilities\default.json`：
```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "默认权限",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "store:default",
    "window-state:default",
    "fs:default",
    "dialog:default",
    "sql:default"
  ]
}
```

- [ ] **Step 4: 创建占位图标**

Tauri 需要图标文件才能编译。运行：
```bash
cd C:\project\ai-stack
mkdir -p src-tauri/icons
# 这里使用 Tauri 提供的默认图标。最小化方式：让 Tauri CLI 自动生成。
npx @tauri-apps/cli icon
```
若交互式询问源图片，按 `Enter` 跳过（CLI 会在 `src-tauri/icons/` 生成全套占位图标）。

- [ ] **Step 5: 安装依赖**

```bash
cd C:\project\ai-stack
npm install
```
预期：安装完成，`package-lock.json` 生成，无 ERR。

- [ ] **Step 6: 验证类型检查与构建**

```bash
npm run typecheck
```
预期：0 error。

```bash
npm run build
```
预期：构建成功，`dist/` 目录生成。

- [ ] **Step 7: 验证 Tauri dev 启动**

```bash
npm run tauri dev
```
预期：首次会下载并编译 Rust 依赖（耗时长，5-15 分钟），编译完成后弹出名为 "AI Stack" 的窗口，窗口中央显示 "AI Stack" 文字，亮色背景。
关闭窗口后命令退出（Ctrl+C 终止后台进程）。

- [ ] **Step 8: 提交**

```bash
cd C:\project\ai-stack
git init
git add -A
git commit -m "chore: initialize tauri + react + ts skeleton"
```

---

## Task 2: Tailwind + 主题 CSS 变量

**Files:**
- Create: `tailwind.config.js`, `postcss.config.js`, `src/index.css`

**Interfaces:**
- Consumes: 无
- Produces: `dark` class 切换生效，CSS 变量定义主题色板

- [ ] **Step 1: 写入 Tailwind 配置**

写入 `C:\project\ai-stack\tailwind.config.js`：
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: 'var(--color-bg)',
        surface: 'var(--color-surface)',
        'surface-2': 'var(--color-surface-2)',
        border: 'var(--color-border)',
        text: 'var(--color-text)',
        'text-muted': 'var(--color-text-muted)',
        accent: 'var(--color-accent)',
        'accent-hover': 'var(--color-accent-hover)',
      },
    },
  },
  plugins: [],
};
```

写入 `C:\project\ai-stack\postcss.config.js`：
```js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

- [ ] **Step 2: 写入全局样式与 CSS 变量**

写入 `C:\project\ai-stack\src\index.css`：
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --color-bg: #ffffff;
  --color-surface: #f9fafb;
  --color-surface-2: #f3f4f6;
  --color-border: #e5e7eb;
  --color-text: #111827;
  --color-text-muted: #6b7280;
  --color-accent: #2563eb;
  --color-accent-hover: #1d4ed8;
}

.dark {
  --color-bg: #0b0f17;
  --color-surface: #111827;
  --color-surface-2: #1f2937;
  --color-border: #374151;
  --color-text: #f9fafb;
  --color-text-muted: #9ca3af;
  --color-accent: #3b82f6;
  --color-accent-hover: #60a5fa;
}

html, body, #root {
  height: 100%;
}

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei",
    Roboto, "Helvetica Neue", Arial, sans-serif;
  background-color: var(--color-bg);
  color: var(--color-text);
  -webkit-font-smoothing: antialiased;
}
```

- [ ] **Step 3: 验证构建**

```bash
npm run build
```
预期：CSS 正常产出（含 Tailwind 重置样式），无 warning。

- [ ] **Step 4: 视觉验证**

```bash
npm run tauri dev
```
手动验证：
1. 窗口背景为白色，文字深色。
2. 在 `src/App.tsx` 临时加 `className="dark"` 在 `<div>` 上，热重载应立刻切换为深色背景。验证后**移除**这个测试 class。

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: tailwind css + theme variables (light/dark)"
```

---

## Task 3: 12 类知识分类数据 + 单测

**Files:**
- Create: `src/data/categories.ts`, `src/lib/categories.test.ts`, `vitest.config.ts`

**Interfaces:**
- Consumes: 无
- Produces: `Category` 类型与 12 个一级分类的常量数组 `CATEGORIES`

- [ ] **Step 1: 写入 vitest 配置**

写入 `C:\project\ai-stack\vitest.config.ts`：
```ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
});
```

- [ ] **Step 2: 写入失败的单测（TDD）**

写入 `C:\project\ai-stack\src\lib\categories.test.ts`：
```ts
import { describe, it, expect } from 'vitest';
import { CATEGORIES, type Category } from '../data/categories';

describe('CATEGORIES', () => {
  it('至少包含 12 个一级分类', () => {
    expect(CATEGORIES.length).toBeGreaterThanOrEqual(12);
  });

  it('每个分类有 id、title、path、children 字段', () => {
    for (const c of CATEGORIES) {
      expect(c.id).toBeTruthy();
      expect(c.title).toBeTruthy();
      expect(c.path).toMatch(/^\d{2}-/);
      expect(Array.isArray(c.children)).toBe(true);
    }
  });

  it('id 全局唯一', () => {
    const ids = new Set<string>();
    for (const c of CATEGORIES) ids.add(c.id);
    expect(ids.size).toBe(CATEGORIES.length);
  });

  it('path 排序与 id 一致', () => {
    const sorted = [...CATEGORIES].sort((a, b) => a.path.localeCompare(b.path));
    expect(sorted.map((c) => c.path)).toEqual(CATEGORIES.map((c) => c.path));
  });
});

describe('Category 类型', () => {
  it('子分类可嵌套', () => {
    const sample: Category = {
      id: 'root',
      title: '根',
      path: '00-root',
      children: [],
    };
    expect(sample.children).toEqual([]);
  });
});
```

- [ ] **Step 3: 运行测试确认失败**

```bash
npm test
```
预期：FAIL，`Cannot find module '../data/categories'`。

- [ ] **Step 4: 写入分类数据**

写入 `C:\project\ai-stack\src\data\categories.ts`：
```ts
export interface Category {
  id: string;
  title: string;
  path: string;            // 资源目录下的相对路径
  icon?: string;           // lucide-react 图标名
  description?: string;
  children: Category[];
}

/**
 * 12 个一级分类，按学习路径排序。
 * path 与 resources/knowledge/ 下目录一一对应。
 */
export const CATEGORIES: Category[] = [
  {
    id: 'foundations',
    title: '基础理论',
    path: '01-foundations',
    icon: 'BookOpen',
    description: '数学、计算机科学、机器学习基础',
    children: [
      { id: 'mathematics', title: '数学基础', path: '01-foundations/01-mathematics', icon: 'Sigma', children: [] },
      { id: 'computer-science', title: '计算机基础', path: '01-foundations/02-computer-science', icon: 'Cpu', children: [] },
      { id: 'ml-basics', title: '机器学习基础', path: '01-foundations/03-ml-basics', icon: 'Brain', children: [] },
    ],
  },
  {
    id: 'deep-learning',
    title: '深度学习',
    path: '02-deep-learning',
    icon: 'Layers',
    description: '神经网络、CNN、RNN、Transformer',
    children: [
      { id: 'nn-fundamentals', title: '神经网络基础', path: '02-deep-learning/neural-network-fundamentals', children: [] },
      { id: 'cnn', title: 'CNN', path: '02-deep-learning/cnn', children: [] },
      { id: 'rnn-lstm', title: 'RNN / LSTM', path: '02-deep-learning/rnn-lstm', children: [] },
      { id: 'transformers', title: 'Transformer', path: '02-deep-learning/transformers', children: [] },
      { id: 'training-techniques', title: '训练技巧', path: '02-deep-learning/training-techniques', children: [] },
      { id: 'frameworks', title: '框架', path: '02-deep-learning/frameworks', children: [] },
    ],
  },
  {
    id: 'llm',
    title: '大语言模型',
    path: '03-large-language-models',
    icon: 'MessageSquareText',
    description: 'LLM 架构、训练、应用',
    children: [
      { id: 'arch-pretrain', title: '架构与预训练', path: '03-large-language-models/architecture-pretraining', children: [] },
      { id: 'fine-tuning', title: '微调', path: '03-large-language-models/fine-tuning', children: [] },
      { id: 'prompt-eng', title: '提示工程', path: '03-large-language-models/prompt-engineering', children: [] },
      { id: 'rag', title: 'RAG', path: '03-large-language-models/rag', children: [] },
      { id: 'llm-apps', title: 'LLM 应用', path: '03-large-language-models/llm-applications', children: [] },
      { id: 'llm-inference', title: '推理优化', path: '03-large-language-models/inference-optimization', children: [] },
      { id: 'multimodal-llm', title: '多模态 LLM', path: '03-large-language-models/multimodal-llm', children: [] },
    ],
  },
  {
    id: 'cv',
    title: '计算机视觉',
    path: '04-computer-vision',
    icon: 'Eye',
    description: '图像、视频、3D 视觉',
    children: [
      { id: 'img-cls', title: '图像分类', path: '04-computer-vision/image-classification', children: [] },
      { id: 'obj-det', title: '目标检测', path: '04-computer-vision/object-detection', children: [] },
      { id: 'seg', title: '分割', path: '04-computer-vision/segmentation', children: [] },
      { id: 'img-gen', title: '图像生成', path: '04-computer-vision/image-generation', children: [] },
      { id: 'video', title: '视频理解', path: '04-computer-vision/video-understanding', children: [] },
      { id: '3d-vision', title: '3D 视觉', path: '04-computer-vision/3d-vision', children: [] },
    ],
  },
  {
    id: 'nlp',
    title: '自然语言处理',
    path: '05-nlp',
    icon: 'Languages',
    description: '文本表示、生成、理解',
    children: [
      { id: 'text-rep', title: '文本表示', path: '05-nlp/text-representation', children: [] },
      { id: 'seq-label', title: '序列标注', path: '05-nlp/sequence-labeling', children: [] },
      { id: 'text-gen', title: '文本生成', path: '05-nlp/text-generation', children: [] },
      { id: 'mt', title: '机器翻译', path: '05-nlp/machine-translation', children: [] },
      { id: 'qa', title: '问答系统', path: '05-nlp/question-answering', children: [] },
      { id: 'ie', title: '信息抽取', path: '05-nlp/information-extraction', children: [] },
    ],
  },
  {
    id: 'speech',
    title: '语音与音频',
    path: '06-speech-audio',
    icon: 'Mic',
    description: 'ASR、TTS、音频生成',
    children: [
      { id: 'asr', title: '语音识别', path: '06-speech-audio/asr', children: [] },
      { id: 'tts', title: '语音合成', path: '06-speech-audio/tts', children: [] },
      { id: 'voice-clone', title: '声音克隆', path: '06-speech-audio/voice-cloning', children: [] },
      { id: 'audio-gen', title: '音频生成', path: '06-speech-audio/audio-generation', children: [] },
    ],
  },
  {
    id: 'ai-eng',
    title: 'AI 工程 / MLOps',
    path: '07-ai-engineering',
    icon: 'Wrench',
    description: '部署、监控、数据、实验',
    children: [
      { id: 'deploy', title: '模型部署', path: '07-ai-engineering/model-deployment', children: [] },
      { id: 'serving', title: '模型服务', path: '07-ai-engineering/model-serving', children: [] },
      { id: 'monitoring', title: '监控', path: '07-ai-engineering/monitoring', children: [] },
      { id: 'data-eng', title: '数据工程', path: '07-ai-engineering/data-engineering', children: [] },
      { id: 'exp-track', title: '实验追踪', path: '07-ai-engineering/experiment-tracking', children: [] },
      { id: 'vector-db', title: '向量数据库', path: '07-ai-engineering/vector-databases', children: [] },
      { id: 'agent-fw', title: 'Agent 框架', path: '07-ai-engineering/agent-frameworks', children: [] },
    ],
  },
  {
    id: 'agents',
    title: '智能体',
    path: '08-ai-agents',
    icon: 'Bot',
    description: 'Agent 架构、工具调用、规划',
    children: [
      { id: 'agent-arch', title: 'Agent 架构', path: '08-ai-agents/architectures', children: [] },
      { id: 'tool-use', title: '工具调用', path: '08-ai-agents/tool-use', children: [] },
      { id: 'plan-reason', title: '规划与推理', path: '08-ai-agents/planning-reasoning', children: [] },
      { id: 'multi-agent', title: '多智能体', path: '08-ai-agents/multi-agent', children: [] },
      { id: 'memory', title: '记忆系统', path: '08-ai-agents/memory-systems', children: [] },
    ],
  },
  {
    id: 'safety',
    title: 'AI 安全与对齐',
    path: '09-ai-safety',
    icon: 'Shield',
    description: '对齐、可解释性、红队',
    children: [
      { id: 'alignment', title: '对齐', path: '09-ai-safety/alignment', children: [] },
      { id: 'interp', title: '可解释性', path: '09-ai-safety/interpretability', children: [] },
      { id: 'red-team', title: '红队测试', path: '09-ai-safety/red-teaming', children: [] },
      { id: 'bias', title: '偏见与公平', path: '09-ai-safety/bias-fairness', children: [] },
      { id: 'privacy', title: '隐私', path: '09-ai-safety/privacy', children: [] },
    ],
  },
  {
    id: 'apps',
    title: 'AI 应用',
    path: '10-applications',
    icon: 'Sparkles',
    description: '行业落地案例',
    children: [
      { id: 'science', title: 'AI for Science', path: '10-applications/ai-for-science', children: [] },
      { id: 'code', title: 'AI for Code', path: '10-applications/ai-for-code', children: [] },
      { id: 'edu', title: 'AI for Education', path: '10-applications/ai-for-education', children: [] },
      { id: 'health', title: 'AI for Healthcare', path: '10-applications/ai-for-healthcare', children: [] },
      { id: 'finance', title: 'AI for Finance', path: '10-applications/ai-for-finance', children: [] },
      { id: 'robotics', title: '机器人', path: '10-applications/robotics', children: [] },
    ],
  },
  {
    id: 'tools',
    title: '工具与生态',
    path: '11-tools-ecosystem',
    icon: 'Package',
    description: '开发工具、云平台、模型、数据集',
    children: [
      { id: 'dev-tools', title: '开发工具', path: '11-tools-ecosystem/development-tools', children: [] },
      { id: 'cloud', title: '云平台', path: '11-tools-ecosystem/cloud-platforms', children: [] },
      { id: 'oss-models', title: '开源模型', path: '11-tools-ecosystem/open-source-models', children: [] },
      { id: 'datasets', title: '数据集', path: '11-tools-ecosystem/datasets', children: [] },
      { id: 'benchmarks', title: '基准测试', path: '11-tools-ecosystem/benchmarks', children: [] },
    ],
  },
  {
    id: 'trends',
    title: '行业与趋势',
    path: '12-industry-trends',
    icon: 'TrendingUp',
    description: '前沿论文、会议、新闻',
    children: [
      { id: 'papers', title: '前沿论文', path: '12-industry-trends/frontier-papers', children: [] },
      { id: 'reports', title: '行业报告', path: '12-industry-trends/industry-reports', children: [] },
      { id: 'conf', title: '学术会议', path: '12-industry-trends/conferences', children: [] },
      { id: 'news', title: '新闻动态', path: '12-industry-trends/news-updates', children: [] },
    ],
  },
];
```

- [ ] **Step 5: 运行测试确认通过**

```bash
npm test
```
预期：全部 PASS，5 个用例全绿。

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "feat: 12 knowledge categories data + unit tests"
```

---

## Task 4: 主题 Store

**Files:**
- Create: `src/stores/theme.ts`, `src/stores/theme.test.ts`

**Interfaces:**
- Consumes: 无
- Produces:
  - `useThemeStore()` 返回 `{ theme: 'light' | 'dark', toggle(), set(t) }`
  - 默认值 `'light'`，调用 `set` 时同步给 `<html>` 添加/移除 `dark` class

- [ ] **Step 1: 写入失败的单测**

写入 `C:\project\ai-stack\src\stores\theme.test.ts`：
```ts
import { describe, it, expect, beforeEach } from 'vitest';
import { useThemeStore } from './theme';

describe('theme store', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove('dark');
    useThemeStore.setState({ theme: 'light' });
  });

  it('默认主题为 light', () => {
    expect(useThemeStore.getState().theme).toBe('light');
  });

  it('toggle 切换 light <-> dark', () => {
    const { toggle } = useThemeStore.getState();
    toggle();
    expect(useThemeStore.getState().theme).toBe('dark');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    toggle();
    expect(useThemeStore.getState().theme).toBe('light');
    expect(document.documentElement.classList.contains('dark')).toBe(false);
  });

  it('set(dark) 立即应用 dark class', () => {
    useThemeStore.getState().set('dark');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });
});
```

- [ ] **Step 2: 运行测试确认失败**

```bash
npm test
```
预期：FAIL，`Cannot find module './theme'`。

- [ ] **Step 3: 写入实现**

写入 `C:\project\ai-stack\src\stores\theme.ts`：
```ts
import { create } from 'zustand';

export type Theme = 'light' | 'dark';

interface ThemeState {
  theme: Theme;
  toggle: () => void;
  set: (t: Theme) => void;
}

const STORAGE_KEY = 'ai-stack:theme';

function applyTheme(t: Theme) {
  const root = document.documentElement;
  if (t === 'dark') root.classList.add('dark');
  else root.classList.remove('dark');
}

function readInitial(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'dark' || stored === 'light') return stored;
  } catch {
    // localStorage 不可用时降级到 light（Review Focus #1）
  }
  return 'light';
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: readInitial(),
  toggle: () => {
    const next: Theme = get().theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem(STORAGE_KEY, next); } catch { /* ignore */ }
    set({ theme: next });
  },
  set: (t: Theme) => {
    applyTheme(t);
    try { localStorage.setItem(STORAGE_KEY, t); } catch { /* ignore */ }
    set({ theme: t });
  },
}));

// 首次加载时把当前主题应用到 DOM
applyTheme(useThemeStore.getState().theme);
```

- [ ] **Step 4: 运行测试确认通过**

```bash
npm test
```
预期：3 个用例全 PASS。

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: theme store with light/dark toggle"
```

---

## Task 5: 路由与四个占位页

**Files:**
- Create: `src/routes/Library.tsx`, `src/routes/Notes.tsx`, `src/routes/Dashboard.tsx`, `src/routes/Settings.tsx`
- Modify: `src/App.tsx`

**Interfaces:**
- Consumes: 无
- Produces: 路由 `/library`、`/notes`、`/dashboard`、`/settings`，默认重定向到 `/library`

- [ ] **Step 1: 写入四个占位页**

写入 `C:\project\ai-stack\src\routes\Library.tsx`：
```tsx
export default function Library() {
  return (
    <div className="p-6">
      <h2 className="text-xl font-semibold">知识库</h2>
      <p className="mt-2 text-text-muted">在此浏览 AI 知识分类与资源（阶段 2 实现）。</p>
    </div>
  );
}
```

写入 `C:\project\ai-stack\src\routes\Notes.tsx`：
```tsx
export default function Notes() {
  return (
    <div className="p-6">
      <h2 className="text-xl font-semibold">笔记</h2>
      <p className="mt-2 text-text-muted">在此管理学习笔记（阶段 3 实现）。</p>
    </div>
  );
}
```

写入 `C:\project\ai-stack\src\routes\Dashboard.tsx`：
```tsx
export default function Dashboard() {
  return (
    <div className="p-6">
      <h2 className="text-xl font-semibold">学习仪表盘</h2>
      <p className="mt-2 text-text-muted">在此查看学习进度与统计（阶段 4 实现）。</p>
    </div>
  );
}
```

写入 `C:\project\ai-stack\src\routes\Settings.tsx`：
```tsx
export default function Settings() {
  return (
    <div className="p-6">
      <h2 className="text-xl font-semibold">设置</h2>
      <p className="mt-2 text-text-muted">应用设置与 AI API Key（任务 7 完善）。</p>
    </div>
  );
}
```

- [ ] **Step 2: 重写 App.tsx 接入路由**

覆盖 `C:\project\ai-stack\src\App.tsx`：
```tsx
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Library from './routes/Library';
import Notes from './routes/Notes';
import Dashboard from './routes/Dashboard';
import Settings from './routes/Settings';

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/library" replace />} />
          <Route path="/library" element={<Library />} />
          <Route path="/notes" element={<Notes />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/library" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
```

- [ ] **Step 3: 占位 Layout 组件（任务 6 完整实现）**

写入 `C:\project\ai-stack\src\components\layout\Layout.tsx`（占位，任务 6 替换）：
```tsx
import { Outlet } from 'react-router-dom';

export default function Layout() {
  return (
    <div className="flex h-full flex-col">
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
```

- [ ] **Step 4: 验证类型 + 构建**

```bash
npm run typecheck
npm run build
```
预期：0 error。

- [ ] **Step 5: 视觉验证**

```bash
npm run tauri dev
```
手动验证：
1. 窗口显示 `/library` 内容（"知识库" 标题）。
2. 暂时在地址栏手动改成 `/#/settings`（开发期 WebView 不易改 URL，暂跳过此步骤；任务 6 完成后通过侧边栏跳转验证）。

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "feat: routing + four placeholder pages"
```

---

## Task 6: 布局组件（Sidebar + Topbar + Layout）

**Files:**
- Modify: `src/components/layout/Layout.tsx`
- Create: `src/components/layout/Sidebar.tsx`, `src/components/layout/Topbar.tsx`, `src/components/ui/ThemeToggle.tsx`, `src/types/index.ts`

**Interfaces:**
- Consumes: `CATEGORIES` from `src/data/categories.ts`, `useThemeStore`
- Produces: 左侧 240px 侧边栏显示一级分类；顶部栏含标题、搜索框占位、主题切换、设置入口；主内容区

- [ ] **Step 1: 写入类型定义**

写入 `C:\project\ai-stack\src\types\index.ts`：
```ts
export type RoutePath = '/library' | '/notes' | '/dashboard' | '/settings';

export interface NavItem {
  path: RoutePath;
  label: string;
  icon: string;
}
```

- [ ] **Step 2: 写入 ThemeToggle**

写入 `C:\project\ai-stack\src\components\ui\ThemeToggle.tsx`：
```tsx
import { Moon, Sun } from 'lucide-react';
import { useThemeStore } from '../../stores/theme';

export default function ThemeToggle() {
  const { theme, toggle } = useThemeStore();
  const isDark = theme === 'dark';
  return (
    <button
      onClick={toggle}
      aria-label={isDark ? '切换到亮色主题' : '切换到暗色主题'}
      className="rounded-md p-2 text-text-muted hover:bg-surface-2 hover:text-text"
    >
      {isDark ? <Sun size={18} /> : <Moon size={18} />}
    </button>
  );
}
```

- [ ] **Step 3: 写入 Sidebar**

写入 `C:\project\ai-stack\src\components\layout\Sidebar.tsx`：
```tsx
import { NavLink } from 'react-router-dom';
import { BookOpen, Layers, MessageSquareText, Eye, Languages, Mic,
         Wrench, Bot, Shield, Sparkles, Package, TrendingUp } from 'lucide-react';
import { CATEGORIES } from '../../data/categories';
import type { ComponentType } from 'react';

const ICONS: Record<string, ComponentType<{ size?: number }>> = {
  BookOpen, Layers, MessageSquareText, Eye, Languages, Mic,
  Wrench, Bot, Shield, Sparkles, Package, TrendingUp,
};

export default function Sidebar() {
  return (
    <aside
      aria-label="知识分类导航"
      className="flex h-full w-60 shrink-0 flex-col border-r border-border bg-surface"
    >
      <div className="flex h-14 items-center px-4 border-b border-border">
        <span className="text-base font-semibold tracking-tight">AI Stack</span>
      </div>
      <nav className="flex-1 overflow-y-auto p-2">
        <div className="mb-2 px-2 text-xs font-medium uppercase tracking-wider text-text-muted">
          分类
        </div>
        <ul className="space-y-0.5">
          {CATEGORIES.map((cat) => {
            const Icon = cat.icon && ICONS[cat.icon];
            return (
              <li key={cat.id}>
                <NavLink
                  to="/library"
                  state={{ focusCategory: cat.path }}
                  className={({ isActive }) =>
                    `flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors ${
                      isActive
                        ? 'bg-accent/10 text-accent'
                        : 'text-text hover:bg-surface-2'
                    }`
                  }
                >
                  {Icon && <Icon size={16} />}
                  <span className="truncate">{cat.title}</span>
                </NavLink>
              </li>
            );
          })}
        </ul>
      </nav>
      <div className="border-t border-border p-2 text-xs text-text-muted">
        v0.1.0 · Phase 1
      </div>
    </aside>
  );
}
```

- [ ] **Step 4: 写入 Topbar**

写入 `C:\project\ai-stack\src\components\layout\Topbar.tsx`：
```tsx
import { Link, useLocation } from 'react-router-dom';
import { Search, Settings as SettingsIcon } from 'lucide-react';
import ThemeToggle from '../ui/ThemeToggle';

const TITLES: Record<string, string> = {
  '/library': '知识库',
  '/notes': '笔记',
  '/dashboard': '学习仪表盘',
  '/settings': '设置',
};

export default function Topbar() {
  const { pathname } = useLocation();
  const title = TITLES[pathname] ?? 'AI Stack';

  return (
    <header
      role="banner"
      className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-surface px-4"
    >
      <h1 className="text-lg font-semibold">{title}</h1>

      <div className="flex items-center gap-2">
        <div className="relative">
          <Search
            size={16}
            className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-text-muted"
          />
          <input
            type="search"
            placeholder="搜索（阶段 6 实现）"
            disabled
            aria-label="搜索"
            className="w-64 rounded-md border border-border bg-bg py-1.5 pl-8 pr-3 text-sm text-text placeholder:text-text-muted disabled:opacity-50"
          />
        </div>

        <ThemeToggle />

        <Link
          to="/settings"
          aria-label="设置"
          className="rounded-md p-2 text-text-muted hover:bg-surface-2 hover:text-text"
        >
          <SettingsIcon size={18} />
        </Link>
      </div>
    </header>
  );
}
```

- [ ] **Step 5: 重写 Layout**

覆盖 `C:\project\ai-stack\src\components\layout\Layout.tsx`：
```tsx
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Topbar from './Topbar';

export default function Layout() {
  return (
    <div className="flex h-full min-w-[900px] bg-bg text-text">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: 验证**

```bash
npm run typecheck
npm run build
npm run tauri dev
```
手动验证：
1. 窗口左侧出现 12 个分类列表。
2. 顶部栏显示当前页标题、搜索框（禁用）、月亮/太阳图标、设置图标。
3. 窗口尺寸缩到 900×600 临界时布局不破（侧边栏 240 + 主区 660）。
4. 点击月亮图标 → 全局切深色；再点 → 切回亮色。
5. 点击侧边栏"基础理论" → 跳到 `/library`（后续阶段细化分类选中态）。
6. 点击右上设置图标 → 跳到 `/settings`。

- [ ] **Step 7: 提交**

```bash
git add -A
git commit -m "feat: layout components (sidebar + topbar + theme toggle)"
```

---

## Task 7: 设置页 — API Key 保存（tauri-plugin-store）

**Files:**
- Create: `src/lib/tauri.ts`, `src/components/settings/ApiKeyForm.tsx`
- Modify: `src/routes/Settings.tsx`

**Interfaces:**
- Consumes: `tauri-plugin-store`
- Produces: `Settings` 页可输入并保存 `apiKey`，刷新后仍在（store 文件持久化）

- [ ] **Step 1: 写入 tauri 插件包装**

写入 `C:\project\ai-stack\src\lib\tauri.ts`：
```ts
import { load, type Store } from '@tauri-apps/plugin-store';

/**
 * 应用全局 store：阶段 1 只存 API Key；
 * 后续阶段扩展 baseUrl、模型、温度等。
 */
const STORE_FILE = 'settings.json';
const KEY_API_KEY = 'apiKey';
const KEY_BASE_URL = 'baseUrl';
const KEY_MODEL = 'model';

export interface AppSettings {
  apiKey: string;
  baseUrl: string;
  model: string;
}

const DEFAULTS: AppSettings = {
  apiKey: '',
  baseUrl: 'https://api.anthropic.com',
  model: 'claude-sonnet-4-5',
};

let storePromise: Promise<Store> | null = null;

function getStore(): Promise<Store> {
  if (!storePromise) {
    storePromise = load(STORE_FILE, { autoSave: true });
  }
  return storePromise;
}

export async function getSettings(): Promise<AppSettings> {
  // Review Focus #1：首次启动无设置文件时优雅降级到默认值
  try {
    const store = await getStore();
    const apiKey = (await store.get<string>(KEY_API_KEY)) ?? DEFAULTS.apiKey;
    const baseUrl = (await store.get<string>(KEY_BASE_URL)) ?? DEFAULTS.baseUrl;
    const model = (await store.get<string>(KEY_MODEL)) ?? DEFAULTS.model;
    return { apiKey, baseUrl, model };
  } catch {
    return DEFAULTS;
  }
}

export async function saveSettings(s: Partial<AppSettings>): Promise<void> {
  const store = await getStore();
  if (s.apiKey !== undefined) await store.set(KEY_API_KEY, s.apiKey);
  if (s.baseUrl !== undefined) await store.set(KEY_BASE_URL, s.baseUrl);
  if (s.model !== undefined) await store.set(KEY_MODEL, s.model);
  await store.save();
}
```

- [ ] **Step 2: 写入 ApiKeyForm**

写入 `C:\project\ai-stack\src\components\settings\ApiKeyForm.tsx`：
```tsx
import { useEffect, useState } from 'react';
import { Eye, EyeOff, Save } from 'lucide-react';
import { getSettings, saveSettings, type AppSettings } from '../../lib/tauri';

export default function ApiKeyForm() {
  const [form, setForm] = useState<AppSettings>({
    apiKey: '', baseUrl: 'https://api.anthropic.com', model: 'claude-sonnet-4-5',
  });
  const [showKey, setShowKey] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'saved' | 'error'>('idle');

  useEffect(() => {
    setStatus('loading');
    getSettings()
      .then((s) => { setForm(s); setStatus('idle'); })
      .catch(() => setStatus('error'));
  }, []);

  async function onSave(e: React.FormEvent) {
    e.preventDefault();
    try {
      await saveSettings(form);
      setStatus('saved');
      setTimeout(() => setStatus('idle'), 1500);
    } catch {
      setStatus('error');
    }
  }

  return (
    <form onSubmit={onSave} className="space-y-4">
      <div>
        <label className="mb-1 block text-sm font-medium">API Base URL</label>
        <input
          type="url"
          value={form.baseUrl}
          onChange={(e) => setForm({ ...form, baseUrl: e.target.value })}
          className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
        />
      </div>

      <div>
        <label className="mb-1 block text-sm font-medium">Model</label>
        <input
          type="text"
          value={form.model}
          onChange={(e) => setForm({ ...form, model: e.target.value })}
          className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
        />
      </div>

      <div>
        <label className="mb-1 block text-sm font-medium">API Key</label>
        <div className="relative">
          <input
            type={showKey ? 'text' : 'password'}
            value={form.apiKey}
            onChange={(e) => setForm({ ...form, apiKey: e.target.value })}
            placeholder="sk-..."
            autoComplete="off"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 pr-10 text-sm"
          />
          <button
            type="button"
            onClick={() => setShowKey((v) => !v)}
            aria-label={showKey ? '隐藏' : '显示'}
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-text-muted hover:bg-surface-2"
          >
            {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        </div>
        <p className="mt-1 text-xs text-text-muted">
          仅保存在本机配置文件，不会上传。
        </p>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={status === 'loading'}
          className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
        >
          <Save size={16} />
          保存
        </button>
        {status === 'saved' && <span className="text-sm text-green-600">已保存</span>}
        {status === 'error' && <span className="text-sm text-red-600">保存失败</span>}
      </div>
    </form>
  );
}
```

- [ ] **Step 3: 重写 Settings 页**

覆盖 `C:\project\ai-stack\src\routes\Settings.tsx`：
```tsx
import ApiKeyForm from '../components/settings/ApiKeyForm';

export default function Settings() {
  return (
    <div className="mx-auto max-w-2xl p-6">
      <h2 className="text-xl font-semibold">设置</h2>
      <p className="mt-1 text-sm text-text-muted">
        配置 AI 云端 API。Key 仅存本地，不上传任何服务器。
      </p>

      <section className="mt-6">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-text-muted">
          AI 服务
        </h3>
        <ApiKeyForm />
      </section>
    </div>
  );
}
```

- [ ] **Step 4: 验证**

```bash
npm run typecheck
npm run build
npm run tauri dev
```
手动验证：
1. 导航到 `/settings`，输入任意 Base URL、Model、API Key，点保存 → 显示"已保存"。
2. 关闭应用并重新 `tauri dev`，再次到 `/settings` → 输入框保留之前的值。
3. 故意删 `%APPDATA%\com.aistack.app\settings.json`，重启 → 输入框回到默认值（不崩溃）。

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: settings page with api key persistence via tauri-plugin-store"
```

---

## Task 8: 应用菜单（中文/跨平台安全字符）

**Files:**
- Create: `src-tauri/src/menu.rs`, `src-tauri/src/lib.rs`
- Modify: `src-tauri/Cargo.toml`（按需）

**Interfaces:**
- Consumes: 无
- Produces: 应用菜单"文件 / 视图 / 帮助"，主题切换通过菜单也可触发

- [ ] **Step 1: 写入菜单模块**

写入 `C:\project\ai-stack\src-tauri\src\menu.rs`：
```rust
use tauri::menu::{Menu, MenuBuilder, MenuItem, MenuItemBuilder, Submenu, SubmenuBuilder};
use tauri::{AppHandle, Manager, Runtime};

pub const MENU_ID_TOGGLE_THEME: &str = "toggle_theme";
pub const MENU_ID_OPEN_FOLDER: &str = "open_folder";
pub const MENU_ID_NEW_NOTE: &str = "new_note";
pub const MENU_ID_ABOUT: &str = "about";
pub const MENU_ID_QUIT: &str = "quit";

pub fn build_menu<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    let open_folder = MenuItemBuilder::with_id(MENU_ID_OPEN_FOLDER, "打开文件夹…")
        .accelerator("CmdOrCtrl+O")
        .build(app)?;
    let new_note = MenuItemBuilder::with_id(MENU_ID_NEW_NOTE, "新建笔记")
        .accelerator("CmdOrCtrl+N")
        .build(app)?;
    let quit = MenuItemBuilder::with_id(MENU_ID_QUIT, "退出")
        .accelerator("CmdOrCtrl+Q")
        .build(app)?;

    let file: Submenu<R> = SubmenuBuilder::new(app, "文件")
        .item(&open_folder)
        .item(&new_note)
        .separator()
        .item(&quit)
        .build()?;

    let toggle_theme = MenuItemBuilder::with_id(MENU_ID_TOGGLE_THEME, "切换主题")
        .accelerator("CmdOrCtrl+T")
        .build(app)?;
    let view: Submenu<R> = SubmenuBuilder::new(app, "视图")
        .item(&toggle_theme)
        .build()?;

    let about = MenuItemBuilder::with_id(MENU_ID_ABOUT, "关于 AI Stack").build(app)?;
    let help: Submenu<R> = SubmenuBuilder::new(app, "帮助")
        .item(&about)
        .build()?;

    let menu = MenuBuilder::new(app)
        .item(&file)
        .item(&view)
        .item(&help)
        .build()?;
    Ok(menu)
}

/// 给前端抛事件用：列出菜单项 ID（仅用于文档/校验）
pub fn menu_item_ids() -> Vec<&'static str> {
    vec![
        MENU_ID_OPEN_FOLDER,
        MENU_ID_NEW_NOTE,
        MENU_ID_QUIT,
        MENU_ID_TOGGLE_THEME,
        MENU_ID_ABOUT,
    ]
}
```

- [ ] **Step 2: 修改 lib.rs 挂载菜单**

覆盖 `C:\project\ai-stack\src-tauri\src\lib.rs`：
```rust
mod menu;

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
            // 把菜单事件转发给前端，前端按 ID 决定动作
            app.on_menu_event(|app, event| {
                let _ = app.emit("menu", event.id().0.as_str());
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running ai-stack application");
}
```

- [ ] **Step 3: 前端订阅菜单事件（App.tsx）**

覆盖 `C:\project\ai-stack\src\App.tsx`：
```tsx
import { useEffect } from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import { listen } from '@tauri-apps/api/event';
import Layout from './components/layout/Layout';
import Library from './routes/Library';
import Notes from './routes/Notes';
import Dashboard from './routes/Dashboard';
import Settings from './routes/Settings';
import { useThemeStore } from './stores/theme';

export default function App() {
  const toggle = useThemeStore((s) => s.toggle);

  useEffect(() => {
    const unlisten = listen<string>('menu', (e) => {
      if (e.payload === 'toggle_theme') toggle();
      // 其他菜单项在后续阶段实现
    });
    return () => { unlisten.then((fn) => fn()); };
  }, [toggle]);

  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/library" replace />} />
          <Route path="/library" element={<Library />} />
          <Route path="/notes" element={<Notes />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/library" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
```

- [ ] **Step 4: 验证**

```bash
npm run typecheck
npm run build
npm run tauri dev
```
手动验证：
1. 窗口顶部菜单出现"文件 / 视图 / 帮助"。
2. 点击"视图 → 切换主题"或按 `Ctrl+T` → 主题切换。
3. 中文菜单文字正常显示（Review Focus #3），无乱码。
4. 点击"文件 → 退出"或按 `Ctrl+Q` → 应用退出。

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: application menu (file/view/help) with theme toggle shortcut"
```

---

## Task 9: 预置 AI 知识目录结构（仅目录 + _index.md 占位）

**Files:**
- Create: `resources/knowledge/<12 个一级目录>/<二级目录>/_index.md` 等约 200 个 `_index.md`

**Interfaces:**
- Consumes: 任务 3 的 `CATEGORIES`
- Produces: `resources/knowledge/` 下完整目录树，与 `CATEGORIES` 中 `path` 字段一一对应；每目录一个 `_index.md`

- [ ] **Step 1: 用脚本生成（避免手写 200 个文件）**

在仓库根新建 `scripts/scaffold-knowledge.mjs`：
```js
// scripts/scaffold-knowledge.mjs
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('.', import.meta.url));
const ROOT = join(here, '..', 'resources', 'knowledge');

// 与 src/data/categories.ts 中的 children 一一对应（仅目录，不含三级展开）
// 因为 categories.ts 中 children 数组对部分二级分类未展开三级，
// 这里以"每个二级目录"为单位生成 _index.md。
const TREE = {
  '01-foundations': ['01-mathematics', '02-computer-science', '03-ml-basics'],
  '02-deep-learning': [
    'neural-network-fundamentals', 'cnn', 'rnn-lstm',
    'transformers', 'training-techniques', 'frameworks',
  ],
  '03-large-language-models': [
    'architecture-pretraining', 'fine-tuning', 'prompt-engineering',
    'rag', 'llm-applications', 'inference-optimization', 'multimodal-llm',
  ],
  '04-computer-vision': [
    'image-classification', 'object-detection', 'segmentation',
    'image-generation', 'video-understanding', '3d-vision',
  ],
  '05-nlp': [
    'text-representation', 'sequence-labeling', 'text-generation',
    'machine-translation', 'question-answering', 'information-extraction',
  ],
  '06-speech-audio': ['asr', 'tts', 'voice-cloning', 'audio-generation'],
  '07-ai-engineering': [
    'model-deployment', 'model-serving', 'monitoring', 'data-engineering',
    'experiment-tracking', 'vector-databases', 'agent-frameworks',
  ],
  '08-ai-agents': [
    'architectures', 'tool-use', 'planning-reasoning',
    'multi-agent', 'memory-systems',
  ],
  '09-ai-safety': [
    'alignment', 'interpretability', 'red-teaming', 'bias-fairness', 'privacy',
  ],
  '10-applications': [
    'ai-for-science', 'ai-for-code', 'ai-for-education',
    'ai-for-healthcare', 'ai-for-finance', 'robotics',
  ],
  '11-tools-ecosystem': [
    'development-tools', 'cloud-platforms', 'open-source-models',
    'datasets', 'benchmarks',
  ],
  '12-industry-trends': [
    'frontier-papers', 'industry-reports', 'conferences', 'news-updates',
  ],
};

const indexContent = (topTitle, subTitle, topPath) => `# ${subTitle}

> 分类：**${topTitle}** → **${subTitle}**
> 路径：\`resources/knowledge/${topPath}/${subTitle.split(' / ').pop()}\`

本目录用于存放与「${subTitle}」相关的学习资料。

## 收录范围

- 教材与讲义（Markdown / PDF）
- 论文（PDF）
- 讲稿与笔记（Markdown / Word / PPTX）

## 命名约定

- 每个子主题一个文件夹，文件夹命名用 kebab-case。
- 每个文件夹下放一个 \`_index.md\` 作为目录索引（阶段 2 由索引生成器读取）。
- 文件名建议：\`YYYY-MM-DD-<title>.md\` 或原文件名。
`;

async function main() {
  for (const [top, subs] of Object.entries(TREE)) {
    await mkdir(join(ROOT, top), { recursive: true });
    for (const sub of subs) {
      const dir = join(ROOT, top, sub);
      await mkdir(dir, { recursive: true });
      const topTitle = ({  // 一级目录中文名
        '01-foundations': '基础理论',
        '02-deep-learning': '深度学习',
        '03-large-language-models': '大语言模型',
        '04-computer-vision': '计算机视觉',
        '05-nlp': '自然语言处理',
        '06-speech-audio': '语音与音频',
        '07-ai-engineering': 'AI 工程 / MLOps',
        '08-ai-agents': '智能体',
        '09-ai-safety': 'AI 安全与对齐',
        '10-applications': 'AI 应用',
        '11-tools-ecosystem': '工具与生态',
        '12-industry-trends': '行业与趋势',
      })[top] ?? top;
      const subTitle = sub.split('-').map((s) => s[0].toUpperCase() + s.slice(1)).join(' ');
      await writeFile(
        join(dir, '_index.md'),
        indexContent(topTitle, subTitle, top),
        'utf8',
      );
    }
  }
  console.log(`Created ${Object.values(TREE).flat().length} subdirectories under ${ROOT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
```

- [ ] **Step 2: 运行脚本**

```bash
cd C:\project\ai-stack
mkdir -p scripts
node scripts/scaffold-knowledge.mjs
```
预期输出：`Created 53 subdirectories under ...\resources\knowledge`。

- [ ] **Step 3: 验证目录结构**

```bash
ls resources/knowledge/
```
预期：12 个一级目录。

```bash
ls resources/knowledge/03-large-language-models/
```
预期：含 `architecture-pretraining/`、`fine-tuning/` 等 7 个二级目录。

```bash
ls resources/knowledge/03-large-language-models/fine-tuning/
```
预期：含 `_index.md`。

```bash
cat resources/knowledge/01-foundations/01-mathematics/_index.md
```
预期：显示占位 Markdown 内容。

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "feat: scaffold ai knowledge directory tree with _index.md placeholders"
```

---

## Task 10: 文档（README + architecture + roadmap）

**Files:**
- Create: `README.md`, `docs/architecture.md`, `docs/roadmap.md`

- [ ] **Step 1: 写入 README**

写入 `C:\project\ai-stack\README.md`：
```markdown
# AI Stack

Windows 平台上的 AI 学习工作台。在同一个桌面应用里完成：
- 浏览 AI 知识库（Markdown / PDF / Word / PPTX）
- 记笔记、关联到知识点
- 调用云端 LLM 解读知识
- 追踪学习进度

## 状态

**Phase 1（当前）**：应用骨架 + 12 类 AI 知识目录占位。
后续阶段：见 [docs/roadmap.md](./docs/roadmap.md)。

## 技术栈

- [Tauri 2](https://v2.tauri.app/) + Rust
- React 19 + TypeScript + Vite
- TailwindCSS / React Router / Zustand
- SQLite（后续阶段）
- tauri-plugin-store / window-state / fs / dialog / sql

## 前置依赖（Windows）

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
```

- [ ] **Step 2: 写入 architecture.md**

写入 `C:\project\ai-stack\docs\architecture.md`：
```markdown
# 架构总览

## 整体

```
┌────────────────────────────────────────────────────────────┐
│                  React/TS 前端 (src/)                      │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐ │
│  │ 阅读器   │ 笔记     │ 仪表盘   │ 知识库   │ 设置     │ │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘ │
│           Tauri IPC (commands / events)                    │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────┴───────────────────────────────────┐
│            Rust 后端 (src-tauri/)                          │
│  ┌────────────┬────────────┬────────────┬────────────┐    │
│  │ 文件系统   │ 数据库     │ 文档解析   │ AI 客户端  │    │
│  └────────────┴────────────┴────────────┴────────────┘    │
└────────────────────────────────────────────────────────────┘
```

## 目录职责

| 路径 | 职责 |
|---|---|
| `src/routes/` | 每个路由一个页面组件 |
| `src/components/layout/` | 全局布局（Sidebar/Topbar） |
| `src/components/<feature>/` | 功能专属组件（如 settings） |
| `src/stores/` | Zustand store |
| `src/data/` | 静态数据（分类树等） |
| `src/lib/` | 与后端/工具的交互包装 |
| `src-tauri/src/menu.rs` | 应用菜单 |
| `resources/knowledge/` | 预置 AI 知识目录（仓库内） |
| `data/` | 用户运行时数据（gitignored） |

## 阶段路线

- Phase 1：骨架 + 知识目录（当前）
- Phase 2：知识库扫描 + 阅读器
- Phase 3：笔记
- Phase 4：进度追踪
- Phase 5：AI 解读
- Phase 6：搜索
- Phase 7：打磨与发布
```

- [ ] **Step 3: 写入 roadmap.md**

写入 `C:\project\ai-stack\docs\roadmap.md`：
```markdown
# 路线图

## Phase 1：骨架 + 知识目录 ✅ 当前

- Tauri + React + TS 项目初始化
- 12 类 AI 知识目录占位（不含内容）
- 基础布局、路由、主题切换
- 设置页 API Key 占位（持久化到本地）
- 应用菜单（中文）

## Phase 2：知识库资源管理

- 文件夹扫描（`resources/knowledge/` 与用户目录）
- SQLite 索引（`data/ai-stack.db`）
- Markdown / PDF / Word / PPTX 阅读器
- 知识树浏览

## Phase 3：笔记系统

- Markdown 笔记编辑器（按知识项关联）
- 标签、全文搜索
- 导入/导出

## Phase 4：学习进度追踪

- 阅读时长、阅读位置
- 完成度、仪表盘可视化
- 学习目标

## Phase 5：AI 解读

- 多 provider 适配（Anthropic / OpenAI / 兼容协议）
- 针对知识项的解读、总结、问答
- 引用上下文

## Phase 6：搜索与发现

- FTS5 全文检索
- 相关推荐
- 收藏夹

## Phase 7：打磨与发布

- 主题、设置、托盘、快捷键
- 安装包（NSIS / MSI）
- 自动更新
- 用户文档
```

- [ ] **Step 4: 验证文档链接**

```bash
ls docs/
```
预期：`architecture.md` `roadmap.md` `superpowers/`。

```bash
ls docs/superpowers/
```
预期：`specs/` `plans/`。

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "docs: readme + architecture + roadmap"
```

---

## Self-Review Checklist

执行者（agent 或工程师）在跑完所有任务后，逐项核对：

- [ ] `npm run typecheck` 0 error
- [ ] `npm test` 全部通过（categories + theme store 至少 8 个用例）
- [ ] `npm run build` 成功，`dist/` 产出
- [ ] `npm run tauri dev` 启动窗口，标题"AI Stack"
- [ ] 侧边栏 12 个一级分类，点击切换路由
- [ ] 主题切换（顶栏按钮 + `Ctrl+T` 菜单）都生效，刷新后保持
- [ ] 设置页输入并保存 API Key，重启后仍在
- [ ] 菜单"文件 / 视图 / 帮助"中文正常显示
- [ ] 窗口最小 900×600 布局不破（Review Focus #4）
- [ ] `resources/knowledge/` 下 12 个一级目录 + 53 个二级目录，每目录有 `_index.md`
- [ ] README / architecture / roadmap 文档存在并链接正确
- [ ] 仓库有 ≥10 个 commit，每个任务对应一个

---

## 常见坑提示

1. **首次 `tauri dev` 极慢**：Rust 编译 tauri、serde 等依赖 5-15 分钟属正常，不要中途 Ctrl+C。
2. **`npx @tauri-apps/cli icon` 交互**：可能问源图，按提示跳过即可。
3. **WebView2 缺失**：Windows 11 自带，Windows 10 需手动装。
4. **中文菜单乱码**：确保 `src-tauri/src/menu.rs` 中菜单项用 UTF-8 字符串字面量，并按本计划"文件 → 视图 → 帮助"中文名填写。
5. **store 文件位置**：`%APPDATA%\com.aistack.app\settings.json`，删除可重置为默认。
6. **路径含空格**：若 `C:\project\ai-stack` 路径含空格，可能触发 Rust 编译路径错误，建议保持无空格。
