# AI Stack — Windows AI 学习软件 设计文档

| 项 | 值 |
|---|---|
| 日期 | 2026-09-29 |
| 路径 | `C:\project\ai-stack` |
| 阶段 | 设计阶段 |
| 状态 | 待评审 |

## 1. 愿景与目标

打造一个 Windows 平台上的 **AI 学习工作台**，把"阅读 AI 知识、记笔记、AI 辅助解读、追踪学习进度"四件事统一在同一个桌面应用里。面向学习者（从入门到进阶）、研究者、教师与从业者。

**核心价值：**
- **本地优先**：所有学习资料、笔记、进度数据都在本地，AI 解读通过云端 API 即用即走。
- **结构化**：预置完整的 AI 知识体系目录，便于按体系化路径学习，而非零散阅读。
- **AI 增强**：对每个知识点/文档可一键调起 AI 解读、总结、问答、笔记润色。
- **跨格式**：原生支持 Markdown 博客、PDF 论文、Word 资料、PPTX 课件。

**非目标（YAGNI）：**
- 不做云同步、多人协作。
- 不做完整的在线课程平台。
- 不做自定义模型训练。

## 2. 总体架构

```
┌────────────────────────────────────────────────────────────┐
│                  React/TS 前端 (src/)                      │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐ │
│  │ 阅读器   │ 笔记编辑器│ 仪表盘   │ 知识库   │ 设置     │ │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘ │
│           Tauri IPC (commands / events)                    │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────┴───────────────────────────────────┐
│            Rust 后端 (src-tauri/)                          │
│  ┌────────────┬────────────┬────────────┬────────────┐    │
│  │ 文件系统   │ 数据库     │ 文档解析   │ AI 客户端  │    │
│  │ (扫描/读取)│ (SQLite)   │ (md/pdf/..)│ (HTTP)     │    │
│  └────────────┴────────────┴────────────┴────────────┘    │
└────────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   本地资源目录      SQLite DB       云端 LLM API
   (resources/,     (data/ai-stack.db)  (Claude/GPT/...)
    user-data/)
```

**前后端职责：**
- **前端 (React)**：UI、交互、富文本编辑器、Markdown 渲染、文档查看（PDF.js / mammoth.js / pptxjs）。
- **后端 (Rust)**：文件系统操作、SQLite 访问、文档元数据抽取、AI API 调用（持有 API Key）、后台索引。

**为什么这样分工：**
- 文件 I/O、PDF/DOCX/PPTX 二进制解析放 Rust，性能好且能流式处理。
- 渲染、编辑、Markdown 解析放前端，生态成熟。
- API Key 永远不出 Rust 进程，安全。

## 3. 项目拆解（按交付阶段）

每个阶段都是可独立交付、可演示的子项目。

| 阶段 | 名称 | 主要交付物 |
|---|---|---|
| **1** | **应用骨架 + 知识目录** | 可启动的空壳；12 个一级分类 + 完整子目录（空内容）；主题、布局、路由 |
| **2** | **知识库资源管理** | 文件夹扫描、SQLite 索引、按目录浏览、Markdown/PDF/DOCX/PPTX 阅读器 |
| **3** | **笔记系统** | Markdown 笔记编辑器、按知识项关联、标签、全文搜索、导入导出 |
| **4** | **学习进度追踪** | 阅读时长、阅读位置、进度条、完成度、仪表盘统计 |
| **5** | **AI 解读** | API Key 管理、对话 UI、针对知识项的解读/总结/问答、引用上下文 |
| **6** | **搜索与发现** | 全文检索（SQLite FTS5）、相关推荐、收藏夹 |
| **7** | **打磨与发布** | 主题、设置、托盘、快捷键、安装包、自动更新 |

**为什么先做骨架：** 即使后续阶段还没做，骨架也是真实可启动、能运行的桌面应用，给用户一个"东西真的存在"的可信交付，再逐层填充能力。

## 4. 阶段 1 详细范围

### 4.1 包含

1. **Tauri + React/TS 项目初始化**
   - `tauri create` + Vite + React + TypeScript
   - 启用 `tauri-plugin-sql`（后续阶段使用，先预留）
   - 启用 `tauri-plugin-fs`、`tauri-plugin-dialog`（后续阶段使用）
   - 启用 `tauri-plugin-store`（设置持久化）
   - 启用 `tauri-plugin-window-state`（窗口状态）

2. **基础布局**
   - 侧边栏（一级分类导航）
   - 顶部栏（标题、搜索框占位、设置入口）
   - 主内容区（路由出口）
   - 暗色/亮色主题切换（基础实现）

3. **路由**
   - `/library`：知识库浏览页（阶段 1 只展示分类树）
   - `/notes`：笔记列表（占位）
   - `/dashboard`：学习仪表盘（占位）
   - `/settings`：设置页（含 API Key 配置占位）

4. **应用菜单**
   - 文件：打开文件夹、新建笔记、退出
   - 视图：刷新、切换主题
   - 帮助：关于、文档

5. **窗口与图标**
   - 自定义应用图标占位
   - 窗口标题、最小尺寸
   - 启动时显示应用名与版本

6. **预置 AI 知识目录结构**
   - 在 `resources/knowledge/` 下创建完整分类树
   - 每个目录含 `_index.md`（占位说明文件，标注此分类用途）
   - 不含具体知识内容

7. **文档**
   - `README.md`：项目说明、开发命令
   - `docs/architecture.md`：架构总览
   - `docs/roadmap.md`：阶段路线图

### 4.2 不包含（明确推迟）

- 文件扫描与索引（阶段 2）
- 真实文档阅读器（阶段 2）
- 笔记编辑与存储（阶段 3）
- 进度数据采集（阶段 4）
- AI API 调用（阶段 5）
- 全文搜索（阶段 6）

### 4.3 验收标准

- [ ] `npm install` + `npm run tauri dev` 在 Windows 上能成功启动桌面窗口
- [ ] 应用窗口显示应用名"AI Stack"，标题栏、菜单正常
- [ ] 侧边栏展示 12 个一级分类，点击切换主区域文字
- [ ] 暗色/亮色主题切换生效
- [ ] 设置页可输入并保存 API Key（通过 `tauri-plugin-store` 存到本地 JSON 文件，不实际调用）
- [ ] `resources/knowledge/` 下有完整目录树，每个分类含 `_index.md`
- [ ] `README.md` 含开发、构建命令

## 5. AI 知识体系目录结构

放在 `resources/knowledge/`。一级 12 类、二级共约 50 个、三级共约 200 个（仅目录，不含内容）。

```
resources/knowledge/
├── 01-foundations/                 基础理论
│   ├── 01-mathematics/             数学基础
│   │   ├── linear-algebra/
│   │   ├── calculus/
│   │   ├── probability-statistics/
│   │   └── optimization/
│   ├── 02-computer-science/        计算机基础
│   │   ├── data-structures-algorithms/
│   │   ├── operating-systems/
│   │   └── computer-architecture/
│   └── 03-ml-basics/                机器学习基础
│       ├── supervised-learning/
│       ├── unsupervised-learning/
│       ├── reinforcement-learning/
│       └── model-evaluation/
├── 02-deep-learning/               深度学习
│   ├── neural-network-fundamentals/
│   ├── cnn/
│   ├── rnn-lstm/
│   ├── transformers/
│   ├── training-techniques/
│   └── frameworks/
│       ├── pytorch/
│       ├── tensorflow/
│       └── jax/
├── 03-large-language-models/      大语言模型
│   ├── architecture-pretraining/
│   ├── fine-tuning/
│   │   ├── sft/
│   │   ├── rlhf/
│   │   └── dpo/
│   ├── prompt-engineering/
│   ├── rag/
│   ├── llm-applications/
│   ├── inference-optimization/
│   └── multimodal-llm/
├── 04-computer-vision/            计算机视觉
│   ├── image-classification/
│   ├── object-detection/
│   ├── segmentation/
│   ├── image-generation/
│   ├── video-understanding/
│   └── 3d-vision/
├── 05-nlp/                        自然语言处理
│   ├── text-representation/
│   ├── sequence-labeling/
│   ├── text-generation/
│   ├── machine-translation/
│   ├── question-answering/
│   └── information-extraction/
├── 06-speech-audio/               语音与音频
│   ├── asr/
│   ├── tts/
│   ├── voice-cloning/
│   └── audio-generation/
├── 07-ai-engineering/             AI 工程与 MLOps
│   ├── model-deployment/
│   ├── model-serving/
│   ├── monitoring/
│   ├── data-engineering/
│   ├── experiment-tracking/
│   ├── vector-databases/
│   └── agent-frameworks/
├── 08-ai-agents/                  智能体
│   ├── architectures/
│   ├── tool-use/
│   ├── planning-reasoning/
│   ├── multi-agent/
│   └── memory-systems/
├── 09-ai-safety/                  AI 安全与对齐
│   ├── alignment/
│   ├── interpretability/
│   ├── red-teaming/
│   ├── bias-fairness/
│   └── privacy/
├── 10-applications/               AI 应用
│   ├── ai-for-science/
│   ├── ai-for-code/
│   ├── ai-for-education/
│   ├── ai-for-healthcare/
│   ├── ai-for-finance/
│   └── robotics/
├── 11-tools-ecosystem/            工具与生态
│   ├── development-tools/
│   ├── cloud-platforms/
│   ├── open-source-models/
│   ├── datasets/
│   └── benchmarks/
└── 12-industry-trends/            行业与趋势
    ├── frontier-papers/
    ├── industry-reports/
    ├── conferences/
    └── news-updates/
```

**分类原则：**
- 顺序反映典型学习路径：基础 → 深度学习 → LLM → 应用领域 → 工程 → 安全 → 行业。
- 编号前缀（01-）保证目录排序与展示顺序一致。
- 每目录一个 `_index.md`（占位），后续阶段填真实内容。
- 用户可后续在任意位置添加自己的资料文件夹（与应用目录分离，存于用户数据目录）。

## 6. 数据模型（阶段 1 仅声明，阶段 2 落地）

```sql
-- 资源（PDF/DOCX/PPTX/MD 等）
resources (
  id, type, path, title, category_path,
  size_bytes, hash, indexed_at,
  -- 后续阶段添加
  total_pages, last_position, last_read_at
)

-- 分类（来自目录扫描，可手动覆盖）
categories (path, parent_path, title, sort_order)

-- 笔记（阶段 3）
notes (id, resource_id, title, content_md, tags, created_at, updated_at)

-- 进度（阶段 4）
progress (resource_id, position, completed_pct, total_seconds)

-- AI 对话（阶段 5）
ai_conversations (id, resource_id, role, content, created_at)

-- AI 对话设置（阶段 5 落地；阶段 1 用 tauri-plugin-store 临时存）
ai_settings (key, value)  -- base_url, model, key_encrypted
```

## 7. 项目文件结构

```
ai-stack/
├── docs/
│   ├── superpowers/specs/             # 设计文档
│   ├── architecture.md                # 架构总览
│   └── roadmap.md                     # 阶段路线图
├── src/                                # React 前端
│   ├── main.tsx
│   ├── App.tsx
│   ├── routes/
│   ├── components/
│   │   ├── layout/                    # Sidebar, Topbar
│   │   └── ui/                        # Button, Card, ...
│   ├── stores/                        # Zustand stores
│   ├── styles/
│   ├── lib/                           # tauri api wrappers
│   └── types/
├── src-tauri/                          # Rust 后端
│   ├── src/
│   │   ├── main.rs
│   │   ├── commands.rs                # IPC commands
│   │   └── lib.rs
│   ├── tauri.conf.json
│   ├── Cargo.toml
│   └── icons/
├── resources/knowledge/                # 预置知识目录（阶段 1 创建）
├── data/                              # 用户数据（gitignored）
│   ├── ai-stack.db
│   └── notes/
├── public/
├── .gitignore
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js                 # 若用 Tailwind
└── README.md
```

## 8. 阶段 1 任务分解（高层）

1. 初始化 Tauri 项目（Vite + React + TS 模板）
2. 配置 `tauri.conf.json`（窗口、图标、应用元数据）
3. 引入基础依赖：React Router、Zustand、TailwindCSS、lucide-react 图标
4. 实现布局组件（Sidebar / Topbar / Layout）
5. 实现四个路由占位页
6. 实现主题切换（CSS 变量 + Zustand）
7. 集成 `tauri-plugin-store`，在设置页实现 API Key 保存
8. 创建应用菜单
9. 创建 12 大类知识目录（含 `_index.md` 占位）
10. 撰写 README、architecture、roadmap 文档
11. 在 Windows 11 上验证 `npm run tauri dev` 启动成功

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Tauri 在 Windows 上首次构建依赖多、慢 | 文档中说明 Rust 工具链、Visual Studio Build Tools、WebView2 前置条件 |
| 阶段 1 范围看似"空" | 明确告诉用户这是骨架，避免误解；后续阶段快速填充 |
| 知识目录命名中英文混杂 | 统一规则：目录用英文 kebab-case（路径稳定），`_index.md` 内可双语 |

## 10. 后续阶段预览（不展开）

- 阶段 2 引入 SQLite + 文件扫描 + 阅读器
- 阶段 3 引入富文本/Markdown 编辑器
- 阶段 4 引入进度采集与可视化
- 阶段 5 引入 AI 客户端（OpenAI 兼容协议，支持 Anthropic/OpenAI/国内代理）
- 阶段 6 引入 FTS5 + 标签
- 阶段 7 引入打包、安装、自动更新

---

**待办：** 用户评审此文档 → 通过后转入 writing-plans 阶段，写阶段 1 实施计划。
