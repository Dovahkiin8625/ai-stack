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
  indexedAt: string;
  pageCount: number | null;
  wordCount: number | null;
}

/**
 * `_index.md` 解析后的一条目：rel_path 与 Resource.relPath 一致（去掉 `./` 前缀），
 * 用来在目录列表里给每篇文档配一行简介。description 为 null 表示 _index.md
 * 里没写描述，UI 退化为只显示标题。
 */
export interface IndexEntry {
  relPath: string;
  title: string;
  description: string | null;
}

/**
 * `_index.md` 解析后的一个 ### 子分组。
 * - `subheading` 为 null 表示该 ## section 下没有 ### 子标题，entries 直接挂在 ## 下
 * - `entries` 是该子分组下的文章条目
 * - `rawMarkdown` 是该 ### 下的非条目 markdown（段落 / 非链接 bullet），原样渲染
 */
export interface ParsedGroup {
  subheading: string | null;
  entries: IndexEntry[];
  rawMarkdown: string;
}

/**
 * `_index.md` 解析后的一个 ## section。
 * - `heading` 是 ## 标题（原文照录，parser 不假设任何特定名字）
 * - `groups` 是该 section 下的 ### 子分组（无 ### 时为单个默认 group）
 * - `rawMarkdown` 是该 ## 下的非条目 markdown（说明性段落、非链接 bullet 等）
 */
export interface ParsedSection {
  heading: string;
  groups: ParsedGroup[];
  rawMarkdown: string;
}

/**
 * `_index.md` 完整解析结果 —— ArticleIndexView 严格按此结构渲染：
 * - 页面 H1 = `title`，H1~## 之间的前言 = `preamble`（含 blockquote）
 * - 每个 ## section 按出现顺序渲染；sections 内部按 ### 子分组渲染
 * - 用户编辑 _index.md 后重新调用 readSubcategoryIndex 即可看到变更（不监听文件变更）
 */
export interface SubcategoryIndex {
  /** # H1 标题；null 表示文件没 H1 */
  title: string | null;
  /** H1 与第一个 ## 之间的 markdown（含 blockquote / 段落） */
  preamble: string | null;
  /** 全部 ## sections（按文件顺序） */
  sections: ParsedSection[];
}

export type ResourceContent =
  | { type: 'markdown'; html: string; wordCount: number }
  | {
      type: 'pdf';
      /** 总页数（read_resource 一次性返回，避免栅格化所有页的 ~22s 延迟）。 */
      pageCount: number;
    }
  | {
      type: 'docx';
      /** 仅作统计/展示用 —— 实际渲染由前端 mammoth.js 完成，详见 DocxReader。 */
      wordCount: number;
    }
  | {
      type: 'pptx';
      /** 仅作统计/展示用 —— 实际渲染由前端 pptxviewjs 完成，详见 PptxReader。 */
      slideCount: number;
    };

export interface ScanSummary {
  categoriesCount: number;
  resourcesCount: number;
  errorsCount: number;
  durationMs: number;
}

export interface ScanProgress {
  phase: 'walking' | 'done';
  current: number;
  total: number;
  currentPath?: string;
}

// === Phase 3: notes ===
export type NoteSource = 'user' | 'ai';

export interface Note {
  id: number;
  resourceId: number;
  content: string;
  anchorText: string | null;
  anchorOccurrence: number;
  /**
   * 用户问 AI 的问题原文（仅"询问 AI"流程会写入）。
   * - AI 讲解场景永远为 null
   * - 笔记卡片只在 prompt 非空时渲染"❓ 提问"引用块，方便回看时知道当时问的是什么
   * - 用户编辑笔记时不影响 prompt（保留原始问题作为上下文）
   */
  prompt?: string | null;
  /** 笔记来源：'user' 人工添加 / 'ai' 大模型讲解 */
  source: NoteSource;
  createdAt: string;
  updatedAt: string;
}

export interface NewNoteInput {
  resourceId: number;
  content: string;
  anchorText: string | null;
  anchorOccurrence: number;
  /** 留空时后端默认 'user' */
  source?: NoteSource;
}

export interface NoteUpdateInput {
  id: number;
  content: string;
  anchorText: string | null;
  anchorOccurrence: number;
  /**
   * 编辑时如果需要把 AI 笔记降级为用户笔记，前端传 'user'；
   * 不传则保留原 source（向后兼容）。
   */
  source?: NoteSource;
}