import { create } from 'zustand';
import * as api from '../lib/library-api';
import type {
  Category,
  Resource,
  ResourceContent,
  ScanSummary,
  SubcategoryIndex,
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
  /**
   * 按子分类路径懒加载并缓存的 _index.md 解析结果（每条含描述）。
   *  - undefined：尚未触发读取
   *  - null：已尝试但文件不存在 / 解析失败（前端不要再 retry，避免噪声）
   *  - SubcategoryIndex：成功解析
   * 编辑 _index.md 后需要重新调用 loadIndex 才能看到变更（不监听文件变更）。
   */
  indexByPath: Record<string, SubcategoryIndex | null>;
  error?: string;
  scan: (force?: boolean) => Promise<void>;
  /**
   * 启动时立刻从 DB 读分类缓存 —— 不阻塞在扫描上，让侧栏在第一次扫描跑之前就能渲染。
   * 之前所有分类都靠 scan() 落库后再 setState，结果每次启动要等几秒扫描结束才出列表。
   * - 后续启动：DB 已有上次扫描结果 → 列表瞬间出现，扫描后台刷新
   * - 首次启动：DB 空 → status 保持 idle，分类为 []，跟"扫描中"是同样外观，
   *   但不会抛错，也不会把 status 推到 error（首次空 DB 是合法状态）
   */
  loadCachedCategories: () => Promise<void>;
  selectResource: (id: number | null) => Promise<void>;
  loadArticles: (path: string) => Promise<void>;
  loadIndex: (path: string) => Promise<void>;
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
  indexByPath: {},

  scan: async (force = false) => {
    set({ status: 'scanning', error: undefined });
    try {
      const summary = await api.scanLibrary(force);
      const categories = await api.listCategories();
      // 不重置 articlesByPath / indexByPath：
      // - articlesByPath 缓存（Resource[]）由 loadArticles 的 useEffect 自行管理——
      //   新挂载或缓存缺失时会自动重拉，scan 后保留旧值能避免已经渲染的 section 闪回 "加载中…"
      // - indexByPath（ParsedIndex）本来就不缓存（每次 mount 都重读 _index.md，见下）
      // 只更新 scan 真正改变的东西：status / summary / categories。
      set({ status: 'ready', summary, categories });
    } catch (e) {
      set({ status: 'error', error: String(e) });
    }
  },

  loadCachedCategories: async () => {
    try {
      const categories = await api.listCategories();
      // DB 有数据 → 直接 ready，跳过"等扫描跑完才出列表"那段空窗期。
      // 注意：不重置 articlesByPath / indexByPath —— 那些和 scan 无强绑定，
      // 留着可以省一次重复请求。
      if (categories.length > 0) {
        set({ status: 'ready', categories, error: undefined });
      }
      // 空 DB（首次启动）保持 idle / 空 categories —— 与"扫描中"视觉一致，
      // 此时 scan() 会紧接着被调起去落库，列表自然会出现。
    } catch (e) {
      // IPC 失败（如 DB 文件被锁）让 scan() 自己去报 —— 这里静默 no-op。
      console.warn('[library] loadCachedCategories failed:', e);
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

  /**
   * 每次调用都重新读 `_index.md` —— 不缓存。
   *
   * 原因：动态渲染是设计原则 —— 用户编辑 `_index.md` 后应能立刻看到。
   * `_index.md` 只有 ~1KB，解析毫秒级；缓存带来的失效问题比"省一次 IPC"严重得多。
   *
   * `articlesByPath`（DB 数据）仍然走缓存，因为它依赖 scan 重建，不靠单文件修改。
   *
   * 失败（文件不存在等）记 null 给 UI 渲染"该目录缺少 _index.md"提示，但**不阻止后续重试**——
   * 用户事后创建 `_index.md` 后再访问同一路径应能看到。
   */
  loadIndex: async (path) => {
    try {
      const idx = await api.readSubcategoryIndex(path);
      set((s) => ({
        indexByPath: { ...s.indexByPath, [path]: idx },
      }));
    } catch (e) {
      console.warn(`[library] loadIndex(${path}) failed:`, e);
      set((s) => ({
        indexByPath: { ...s.indexByPath, [path]: null },
      }));
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
    indexByPath: {},
    error: undefined,
  }),
}));