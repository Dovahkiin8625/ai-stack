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
  phase: 'walking' | 'hashing' | 'inserting' | 'done';
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