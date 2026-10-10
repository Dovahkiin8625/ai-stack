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

## 知识库分发

218 MB / 380 个文件已经大到不适合打进安装包。所以**任何环境**（桌面安装后、Android APK）看到的 `knowledge/` 都是**真实的本地文件系统目录**，但内容来自两条不同的路：

### 桌面端（开发 / 安装后）

- 启动时 `build_state` 解析 `knowledge_root`，优先级：cwd 向上找 `resources/knowledge/`（**开发模式**）→ 否则 `app_data_dir/knowledge`（**已安装**）
- 已安装用户的 `app_data_dir/knowledge` 默认为空 —— 第一次跑（升级场景）会从旧安装包位置（`$RESOURCE/knowledge`，仅老版本存在）一次性拷贝过来，此后由 manifest 同步从远端补齐
- 写入 `~/.android/legacy-migrated` 标记文件保证这次迁移只跑一次（防止以后每次启动覆盖用户本地编辑）

### Android（APK）

- 首次启动：把 APK 内置的种子库（只含每个分类的 `_index.md`，约 177 KB）从 `asset://localhost/knowledge/` 释放到 `app_data_dir/knowledge/`
- 之后从「设置 → 知识库同步地址」填的远端拉 manifest 按需下载剩余文件
- 种子库与远端 manifest 是**同一份 shape**：`files` 数组、每条 `{path, size, sha256}`，由 `parse_seed_manifest` 消费
- 任意运行时下载文件都先落 `app_data_dir/.tmp/...`（**在 `knowledge/` 同级而不在里面**——避免被 scanner 当成假文章 / 假分类），大小校验通过后原子 rename 进 `knowledge/`

### manifest 落入 `resources` 表

- `sync_manifest` 把每条记录 upsert 进 `resources` 表（`present = 0`，`remote_hash` 存空串作为「受管」标记）
- 走相同路径的 `scan_library` 把 `app_data_dir/knowledge/` 里**已经在**的文件标成 `present = 1`
- 因此用户看到的列表是「完整」（manifest 里说有就有），但**已缓存**的会有 `present = true`、未缓存的有下载角标
- 列表读路径完全不知道远端；它只读 `resources` 表的 `present` 字段决定角标
- `delete_missing_resources` 删行时**绝不删** `remote_hash` 非空的行 —— 防止把没下完的远端条目误清

### `read_resource` 自动下载

- 桌面或 Android 用户点开一篇 `present = false` 的文章
- `read_resource` 命令查出资源 → 解析 `app_data_dir/knowledge/...` → 文件不存在 → 调 `ensure_local_at` 从远端拉 → 写盘 → 标 `present = 1` → 继续走 reader
- 失败给出可执行的中文报错（不是 raw rusqlite 文本）：
  - 没配 `sync_base_url`：「文章名 尚未缓存到本机，且未配置知识库同步地址（设置 → 知识库同步）」
  - 配了但网络不通：带 URL 的下载失败信息

## 阶段路线

- Phase 1：骨架 + 知识目录（当前）
- Phase 2：知识库扫描 + 阅读器
- Phase 3：笔记
- Phase 4：进度追踪
- Phase 5：AI 解读
- Phase 6：搜索
- Phase 7：打磨与发布
