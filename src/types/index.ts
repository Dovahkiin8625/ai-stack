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
      pages: Array<{ index: number; dataUrl: string }>;
      pageCount: number;
    }
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