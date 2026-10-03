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
  /** 当前是否有资源正在被读取（readResource 在飞行中）。
   *  CategoryPage 据此区分"加载中"和"出错"——
   *  之前只看 resourceContent 是否为 null，会把加载中的瞬间误显示成红色错误框。 */
  loadingResource: boolean;
  /** 按子分类路径懒加载并缓存的文章列表。Sidebar 展开子分类时填充。 */
  articlesByPath: Record<string, Resource[]>;
  error?: string;
  scan: (force?: boolean) => Promise<void>;
  selectResource: (id: number | null) => Promise<void>;
  loadArticles: (path: string) => Promise<void>;
  reset: () => void;
}

// 单调递增的"selectResource 调用序号"。每次调用自增；调用返回时
// 检查自己当时的 seq 是否还是最新值——不是的话说明有更新的请求
// 覆盖了这条，把结果丢弃，避免"A 慢响应覆盖 B 已加载内容" 的竞态。
let selectSeq = 0;

export const useLibraryStore = create<LibraryState>((set, get) => ({
  status: 'idle',
  categories: [],
  selectedResourceId: null,
  resourceContent: null,
  loadingResource: false,
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
    const seq = ++selectSeq;
    set({
      selectedResourceId: id,
      resourceContent: null,
      loadingResource: id != null,
      error: undefined,
    });
    if (id == null) {
      // id == null 时仍然标记这一轮结束，但仅当 seq 还是最新时写状态——
      // 否则是更新请求已经把 seq 推过去了，state 已经反映新内容，不该再改。
      if (seq === selectSeq) set({ loadingResource: false });
      return;
    }
    try {
      const content = await api.readResource(id);
      // 若飞行途中用户又点了别的资源，seq 已被新的 selectResource 推过去，
      // 这次的结果不要写进 state（否则会用旧 article 的内容覆盖新 article）。
      if (seq !== selectSeq) return;
      set({ resourceContent: content, loadingResource: false });
    } catch (e) {
      if (seq !== selectSeq) return;
      // 错误只在 selectResource 飞行结束后落地。飞行途中再次点击会覆盖这条，
      // 不会让旧错误卡在新内容上。
      set({ error: String(e), resourceContent: null, loadingResource: false });
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
    loadingResource: false,
    articlesByPath: {},
    error: undefined,
  }),
}));