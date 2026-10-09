import { create } from 'zustand';
import * as api from '../lib/library-api';
import * as sync from '../lib/sync';
import type {
  Category,
  Resource,
  ResourceContent,
  ScanSummary,
  SubcategoryIndex,
  SyncStatus,
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
  // ---- 同步状态 ----
  /**
   * 后端 sync_status 拉到的统计；null = 还没拉过（冷启动）。
   * 拉失败时保留上次值（refreshSyncStatus 静默吞错），不阻塞 UI。
   */
  syncStatus: SyncStatus | null;
  /**
   * 同步阶段机。**与 status 解耦** —— library.status 表示"本地扫描"状态，
   * syncPhase 表示"远端同步"状态；改 syncPhase 不能影响 status，避免破坏
   * 首次启动"status=idle / 空 categories"的冷启动外观（与"扫描中"视觉一致）。
   */
  syncPhase: 'idle' | 'manifest' | 'downloading' | 'error';
  /** download_all 的最新进度；downloadAll 完成或被重置时清 null。 */
  downloadProgress: { done: number; total: number } | null;
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
  /**
   * write_resource 后端返的新 ResourceContent 直接塞进 store，不重走 selectResource。
   * 避免：编辑模式保存后 selectResource 把 resourceContent 置 null 再重读 → 编辑器闪一下空。
   * 也避免：selectResource 的 seq 检查误判为过期写入（编辑保存 → 重新 select 会被取消）。
   */
  updateResourceContent: (content: ResourceContent) => void;
  reset: () => void;
  // ---- 同步 actions ----
  /**
   * 拉远端 manifest 并落库。返回 Promise<boolean>：
   * - true  = 后端实际落了库（manifest 拿到了新条目），库已被内部 scan(false) 重建
   * - false = 后端报告 skipped（没配 base_url，纯本地模式）或抛错，**库未被重建**
   *
   * 启动序列据此决定要不要自己再扫一次：`if (!synced) await scan(false)`，
   * 避免"syncNow 内已扫 + 启动序列又扫一次"走两遍 walk。
   *
   * 错误处理故意只动 syncPhase / 共享 error 字段，**不动 status**——
   * status 是"本地扫描"状态机；syncNow 失败不该把冷启动的 `idle` 推到 error，
   * 否则用户首启看到红色空框。把这个不变量记进 reset 与每个 action。
   */
  syncNow: () => Promise<boolean>;
  /**
   * 单独拉一次 sync_status 写进 store。失败静默吞掉 —— 状态条是增强信息，
   * 拉不到就保持上次的值（或 null），不能让用户看到红条。
   */
  refreshSyncStatus: () => Promise<void>;
  /**
   * 触发后端 download_all。**进度通过 sync_progress 事件回传**，需要 UI 层在
   * mount 时 listen onSyncProgress 并写回 store.downloadProgress（这一步不在本 action 里）。
   * 本 action 只翻 syncPhase + 设初始 downloadProgress，监听责任分离避免循环依赖。
   */
  downloadAll: () => Promise<void>;
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
  syncStatus: null,
  syncPhase: 'idle',
  downloadProgress: null,

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

  updateResourceContent: (content) => {
    set({ resourceContent: content });
  },

  // ---- 同步 actions ----

  refreshSyncStatus: async () => {
    try {
      const s = await sync.syncStatus();
      set({ syncStatus: s });
    } catch {
      // 状态条是增强信息，拉不到就静默隐藏 —— 不抛到 error 字段，
      // 否则 UI 会出现"状态条红框"吓用户一跳。保留上次的 syncStatus（若有）。
    }
  },

  syncNow: async (): Promise<boolean> => {
    set({ syncPhase: 'manifest' });
    try {
      const r = await sync.syncManifest();
      if (r.skipped) {
        // 后端报告"未配置 base_url" —— 没拉到 manifest，DB 也没变化，
        // 没必要触发 scan；让 caller（启动序列）自己决定要不要扫。
        // 不动 status，保留冷启动外观。
        set({ syncPhase: 'idle' });
        return false;
      }
      // 拉到新条目并已落库 —— 重新 scan 让本地已有文件标 present=1。
      // 这一步是 RULING 1 的关键：synced 路径下库由 syncNow 自己重建，
      // caller 不应该再扫一次。
      await get().scan(false);
      await get().refreshSyncStatus();
      set({ syncPhase: 'idle' });
      return true;
    } catch (e) {
      // 错误只动 syncPhase / error。**不动 status** —— 否则冷启动 `idle` 被推到
      // `error`，UI 会渲染红色错误框（首次启动 DB 空 ≠ 错误）。
      set({ syncPhase: 'error', error: String(e) });
      return false;
    }
  },

  downloadAll: async () => {
    // 设 downloadProgress 为 (0, 0) 而不是 null：UI 据此区分"已开始但还没进度"
    // 与"还没开始"。sync_progress 第一个事件到来时直接替换 `done`。
    set({
      syncPhase: 'downloading',
      downloadProgress: { done: 0, total: 0 },
    });
    try {
      await sync.downloadAll();
      // 后端 fire-and-forget 后立刻返回 —— sync_phase 在 sync_progress 终事件
      // 到位时再翻回 idle（由 UI 层监听事件后切）。这里不动 syncPhase。
    } catch (e) {
      set({ syncPhase: 'error', error: String(e) });
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
    // 同步字段同样重置 —— 不重置会让"上一个会话/上一个账号"的同步状态泄漏到下次启动。
    syncStatus: null,
    syncPhase: 'idle',
    downloadProgress: null,
  }),
}));