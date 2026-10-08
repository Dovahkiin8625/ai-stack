# 安卓支持与知识库远程分发 — 设计文档

日期：2026-10-08
状态：待审阅

## 1. 目标

为 AI Stack 增加 Android 支持并产出可安装的 APK，同时把知识库（218 MB / 380 个文件）从安装包内资源改为**首次启动后从远程静态托管拉取、按需缓存**。桌面端采用同一套逻辑，不再把知识库打进安装包。

成功标准：

- Android 端可构建出 arm64 APK，安装后能浏览、打开、笔记、调用 LLM 解读。
- 桌面端安装包不再包含知识库（体积减少 218 MB 原始 / 100 MB+ 压缩后），安装后首次启动从远程拉取。
- 两端共用同一套 `knowledge_root` 解析、同步、缓存逻辑；`scanner.rs` / `reader.rs` / `readers/*` 不因平台差异改动。
- 网络不可用时，App 仍能用已缓存内容完整工作。

## 2. 非目标

- 不做 iOS 支持（代码保持 iOS 可编译，但不构建、不验证）。
- 不做知识库的后台自动增量更新（用户手动触发「下载全部」）。
- 不做多知识库 / 用户自选目录（SAF）。
- 不引入 release 签名以外的发布渠道（应用商店、CI 自动发版）。

## 3. 平台约束（已验证）

1. **Android 上 `bundle.resources` 不是真实文件**。Tauri 2 把它们放进 APK assets，Rust 侧 `app.path().resolve(..., BaseDirectory::Resource)` 返回 `asset://localhost/...` URI，`std::fs` 无法读取，必须用 fs 插件的 `app.fs().read(...)`。
   参考：<https://tauri.ubitools.com/develop/resources>、<https://github.com/tauri-apps/tauri/blob/dev/crates/tauri-utils/src/platform.rs>
2. **`tauri-plugin-window-state` 不支持 Android/iOS**，注册与 Cargo 依赖都必须隔离到 desktop。
   参考：<https://docs.rs/crate/tauri-plugin-window-state/2.4.1/source/README.md>
3. Tauri 模板的 `AndroidManifest.xml` 默认**不带 `INTERNET` 权限**，远程拉取必须自行添加。

## 4. 运行时形态

```
APK / 安装包
├── assets/…/knowledge/        种子库：目录骨架 + 少量示例（< 1 MB）
│   └── seed-manifest.json     种子文件清单（相对路径 + 字节数）
└── files/ (app_data_dir)
    ├── ai-stack/ai-stack.db   SQLite（含 manifest 同步下来的完整资源清单）
    └── knowledge/             ← 真实文件系统目录，scanner/reader 只认这里
        └── <分类>/<子分类>/*.md|pdf|pptx|docx

远端静态托管（任意 Nginx / 对象存储 / GitHub Release）
└── manifest.json + 原样目录树
```

**核心不变量**：`knowledge/` 始终是真实文件系统目录。种子库首次启动释放进去，远端文件按需下载进同一目录。所有读取逻辑只认文件系统，"文件怎么到位"是唯一新增的层。

## 5. knowledge_root 解析规则

| 场景 | 解析结果 |
|---|---|
| 开发模式（cwd 向上能找到 `resources/knowledge/`） | 该目录（保留现有 `find_knowledge_root` 行为） |
| 已安装的桌面端 | `app_data_dir/knowledge` |
| Android | `app_data_dir/knowledge`，且启动时执行 `materialize_seed` |

原 `$RESOURCE` 回落分支与 `tauri.conf.json` 的 `bundle.resources` 一并删除。

**桌面端升级兼容**：检测到旧版本遗留的 bundle 资源目录非空时，首次启动一次性复制到 `app_data_dir/knowledge`，避免升级后知识库"消失"。

## 6. 种子库释放（materialize_seed）

仅 Android 需要。步骤：

1. 从 `BaseDirectory::Resource` 解析 `seed-manifest.json`（同样是 asset URI，走 `app.fs()`）。
2. 对每个条目：目标已存在且字节数匹配 → 跳过；否则 `app.fs().read(asset_uri)` → `std::fs::write` 落盘。
3. 过程中发 `seed://progress` 事件（已处理 / 总数）。

**幂等**：重复启动不重复写入。**兜底**：若实测 `app.fs()` 读 asset URI 不可行，退路是直接用 Android AssetManager（jni）读取 —— 仅影响 `materialize_seed` 一个函数，其余设计不变。该验证是实施的第一步，不推迟到最后。

## 7. 同步（sync 模块）

### manifest 格式

```json
{
  "version": "2026-10-08T12:00:00Z",
  "files": [
    { "path": "机器学习基础/监督学习/线性回归.md", "size": 18234, "sha256": "…" }
  ]
}
```

`path` 是相对 `knowledge/` 的路径，使用 `/` 分隔。

`version` 首版仅用于展示与日志，不做自动更新检测 —— 是否需要下载一律以"本地缺失或 size 不符"为准。

### 行为

1. `sync_manifest`（设置里 `syncBaseUrl` 非空时执行）：拉取 `manifest.json` → upsert 进 `resources` 表。
2. `resources` 表新增列：`remote_hash TEXT`、`present INTEGER NOT NULL DEFAULT 1`。manifest 里本地没有的文件也会插入行，`present=0`。**列表因此始终完整**。
3. `scan_library` 照常扫描本地目录，把命中的行置 `present=1`。
4. `read_resource` / `read_resource_bytes`：解析绝对路径 → 文件不存在且在 manifest 中 → 自动下载 → 重读。**用户点一下直接出内容**。
5. `download_all` / `download_resource`：带进度事件 `sync://progress`，可中断。`sync_status` 返回 `{ remote, present, total, cachedBytes, totalBytes }`。
6. 完整性校验首版用 `size`（`sha256` 写入 manifest 备用，后续启用）。

### title / type 派生复用

manifest upsert 复用 scanner 现有的派生规则（从 `_index.md` 的 H1 或文件名取标题、从扩展名取类型），不写第二套，避免规则漂移。实现方式：把 scanner 的派生函数抽为 `pub`。

## 8. 数据库

migration v2（当前为 v1）：

```sql
ALTER TABLE resources ADD COLUMN remote_hash TEXT;
ALTER TABLE resources ADD COLUMN present INTEGER NOT NULL DEFAULT 1;
```

`ResourceDto` 增加 `present: bool`；`sizeBytes` 优先取 manifest 的 size（未下载时才有意义）。

## 9. 前端

| 文件 | 改动 |
|---|---|
| `src/lib/sync.ts` **新增** | `syncManifest()` / `syncStatus()` / `downloadResource(id)` / `downloadAll()` / `stopAll()`，订阅 `sync://progress`、`seed://progress` |
| `src/lib/tauri.ts` | store 新增 `syncBaseUrl`（**留空 = 纯本地模式**，开发时 cwd 目录照旧可用） |
| `src/stores/library.ts` | 新增 `syncStatus`、`syncPhase`、每篇下载进度 |
| `SyncStatusBar.tsx` **新增** | Topbar 下方状态条：`已缓存 37/380 · 42 MB` +「下载全部」/「暂停」；`syncBaseUrl` 为空或全部缓存完成时不显示 |
| `CategoryPage.tsx` | 列表中 `present:false` 条目显示云下载图标 + 体积；点击后下载再打开（后端自动下载兜底） |
| `Settings.tsx` | 新增 `syncBaseUrl` 输入 + 说明 |
| `NotesPanel` / `SelectionMenu` | 桌面右侧栏 → 移动端底部抽屉；触屏划词菜单实测，不可用则降级为长按菜单 |

## 10. 响应式重排（断点 `md` = 768px，桌面端行为不变）

- `Layout.tsx`：移除 `min-w-[900px]`；`Sidebar` 在 `<md` 变遮罩抽屉，Topbar 汉堡按钮开合。
- `CategoryPage.tsx`：`<md` 时列表与阅读器不并排 —— 打开文章只显示阅读器，Topbar 出现返回按钮（回到该 subcategory 的列表路由）。
- 触控目标 ≥ 44px；`hover:` 反馈改 `active:`。

## 11. 平台隔离

- `lib.rs`：`tauri-plugin-window-state` 仅在 `#[cfg(desktop)]` 下注册。
- `Cargo.toml`：`tauri-plugin-window-state` 移入 desktop-only 的 `[target.'cfg(...)'.dependencies]`。
- `platform.rs`：按 `#[cfg(target_os = "android")]` 分支实现 `resolve_knowledge_root` 与 `materialize_seed`；桌面实现不含 seed 逻辑。
- `capabilities/default.json`：补 fs 资源读取 scope（`$RESOURCE/**`）。
- `AndroidManifest.xml`：加 `INTERNET` 权限；该文件位于 `src-tauri/gen/android/`，需从 `.gitignore` 排除并提交，否则 `tauri android init` 会丢失改动。

## 12. 构建与交付

1. 安装：JDK 17 → Android cmdline-tools / platform-tools / build-tools → NDK（版本实施时按 Tauri CLI 提示对齐）→ `rustup target add aarch64-linux-android armv7-linux-androideabi`。
2. `npx tauri android init` 生成 `src-tauri/gen/android`。
3. `npx tauri android build --debug --apk --target aarch64` → 装机验证。
4. `keytool` 生成 keystore → `npx tauri android build --apk` → 正式签名包。

**先 arm64 单架构**（覆盖绝大多数在用机型），编译更快 —— `rusqlite bundled` 每个 ABI 都要编一遍 SQLite C 代码。armv7 / x86_64 视需要追加。

新增脚本：

- `scripts/gen-remote-manifest.mjs`：扫描全量 `resources/knowledge/`，输出 manifest（含 sha256）。
- `scripts/serve-dist.mjs`：本地静态服务器，验证远端链路。

README 增加 Android 构建章节。

## 13. 测试

**自动**（Rust 单测）：manifest 解析、路径规范化、upsert 幂等、`ensure_local` 命中本地时不下载。下载逻辑抽成可注入 trait，测试用 fake 替身，不依赖真实网络。

**自动**（vitest）：`sync.test.ts` 覆盖 store 状态机，mock tauri api。

**人工验收**（真机或模拟器）：

1. 首启释放种子库
2. 拉到 manifest，列表显示 380 篇
3. 点未下载的文章，自动下载并打开
4. 断网后已缓存内容仍可读
5. 桌面端安装包不含知识库，安装后首次启动拉取成功

## 14. 风险

| 风险 | 缓解 |
|---|---|
| `app.fs()` 读不到 asset URI | 实施第一步即验证；兜底方案见 §6 |
| fs 插件 scope 不覆盖 `$RESOURCE` | 显式补 scope，仍不行则走 AssetManager |
| 380 文件索引 + 下载在低端机慢 | 索引照旧，下载并发限 3，逐篇懒加载 |
| 已安装的桌面端未配置 `syncBaseUrl` | 知识库为空且状态条不显示，属预期行为；README 说明需先配置，开发模式下不受影响 |
| `rusqlite bundled` 首次交叉编译慢 | 先编单 ABI，减少总编译量 |
| 老用户升级后知识库路径变化 | §5 的复制迁移 |
