import { create } from 'zustand';
import * as api from '../lib/library-api';
import type {
  Category,
  Resource,
  ResourceContent,
  ScanSummary,
} from '../types';

interface LibraryState {
  status: 'idle' | 'scanning' | 'ready' | 'error';
  summary?: ScanSummary;
  categories: Category[];
  selectedResourceId: number | null;
  resourceContent: ResourceContent | null;
  /** 按子分类路径懒加载并缓存的文章列表。Sidebar 展开子分类时填充。 */
  articlesByPath: Record<string, Resource[]>;
  error?: string;
  scan: (force?: boolean) => Promise<void>;
  selectResource: (id: number | null) => Promise<void>;
  loadArticles: (path: string) => Promise<void>;
  reset: () => void;
}

export const useLibraryStore = create<LibraryState>((set, get) => ({
  status: 'idle',
  categories: [],
  selectedResourceId: null,
  resourceContent: null,
  articlesByPath: {},

  scan: async (force = false) => {
    set({ status: 'scanning', error: undefined });
    try {
      const summary = await api.scanLibrary(force);
      const categories = await api.listCategories();
      // 扫描完成后再清空缓存：保留旧缓存直到新数据就绪，避免扫描窗口内闪烁
      set({ status: 'ready', summary, categories, articlesByPath: {} });
    } catch (e) {
      set({ status: 'error', error: String(e) });
    }
  },

  selectResource: async (id) => {
    set({ selectedResourceId: id, resourceContent: null });
    if (id == null) return;
    try {
      const content = await api.readResource(id);
      set({ resourceContent: content });
    } catch (e) {
      set({ error: String(e), resourceContent: null });
    }
  },

  loadArticles: async (path) => {
    // 命中缓存直接返回，不重复请求
    if (get().articlesByPath[path] !== undefined) return;
    try {
      const articles = await api.listResources(path);
      // 二次检查：请求飞行期间 scan() 可能已清空缓存；写入时保留其它已加载的子分类
      set((s) => ({
        articlesByPath: { ...s.articlesByPath, [path]: articles },
      }));
    } catch (e) {
      set({ error: String(e) });
    }
  },

  reset: () => set({
    status: 'idle',
    summary: undefined,
    categories: [],
    selectedResourceId: null,
    resourceContent: null,
    articlesByPath: {},
    error: undefined,
  }),
}));