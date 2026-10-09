import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../lib/library-api', () => ({
  scanLibrary: vi.fn(async () => ({ categoriesCount: 1, resourcesCount: 2, errorsCount: 0, durationMs: 5 })),
  listCategories: vi.fn(async () => [
    { path: 'a', parentPath: null, title: 'A', sortOrder: 1 },
  ]),
  listResources: vi.fn(async () => []),
  readResource: vi.fn(async () => ({ type: 'markdown', html: '<p>x</p>', wordCount: 1 })),
  readSubcategoryIndex: vi.fn(async () => ({ title: null, preamble: null, sections: [] })),
}));

vi.mock('../lib/sync', () => ({
  syncStatus: vi.fn(async () => ({
    total: 380,
    present: 37,
    totalBytes: 1000,
    cachedBytes: 100,
    configured: true,
  })),
  syncManifest: vi.fn(async () => ({ files: 380, indexes: 12, skipped: false })),
  setSyncBaseUrl: vi.fn(async () => undefined),
  downloadResource: vi.fn(async () => undefined),
  downloadAll: vi.fn(async () => undefined),
  onSyncProgress: vi.fn(async () => () => undefined),
  onSeedProgress: vi.fn(async () => () => undefined),
}));

import { useLibraryStore } from './library';

describe('library store', () => {
  beforeEach(async () => {
    useLibraryStore.setState({
      status: 'idle',
      categories: [],
      selectedResourceId: null,
      resourceContent: null,
      articlesByPath: {},
      indexByPath: {},
      syncStatus: null,
      syncPhase: 'idle',
      downloadProgress: null,
    });
    // 每个测试把 mock 实现恢复到顶部 mock() 工厂里定义的默认实现。
    // 否则 `mockResolvedValue / mockResolvedValueOnce` 会跨测试污染。
    const { scanLibrary, listCategories, listResources, readResource, readSubcategoryIndex } =
      await import('../lib/library-api');
    vi.mocked(scanLibrary).mockReset();
    vi.mocked(listCategories).mockReset();
    vi.mocked(listResources).mockReset();
    vi.mocked(readResource).mockReset();
    vi.mocked(readSubcategoryIndex).mockReset();
    // 重新设回默认行为（mockReset 会清掉实现）
    vi.mocked(scanLibrary).mockResolvedValue({
      categoriesCount: 1,
      resourcesCount: 2,
      errorsCount: 0,
      durationMs: 5,
    });
    vi.mocked(listCategories).mockResolvedValue([
      { path: 'a', parentPath: null, title: 'A', sortOrder: 1 },
    ]);
    vi.mocked(listResources).mockResolvedValue([]);
    vi.mocked(readResource).mockResolvedValue({
      type: 'markdown',
      html: '<p>x</p>',
      wordCount: 1,
    } as any);
    vi.mocked(readSubcategoryIndex).mockResolvedValue({
      title: null,
      preamble: null,
      sections: [],
    });
    // 同步 mock 同样 reset + 恢复默认（与 library-api 同等处理）
    const sync = await import('../lib/sync');
    vi.mocked(sync.syncStatus).mockReset();
    vi.mocked(sync.syncManifest).mockReset();
    vi.mocked(sync.downloadAll).mockReset();
    vi.mocked(sync.syncStatus).mockResolvedValue({
      total: 380,
      present: 37,
      totalBytes: 1000,
      cachedBytes: 100,
      configured: true,
    });
    vi.mocked(sync.syncManifest).mockResolvedValue({
      files: 380,
      indexes: 12,
      skipped: false,
    });
    vi.mocked(sync.downloadAll).mockResolvedValue(undefined);
  });

  it('initial status is idle', () => {
    expect(useLibraryStore.getState().status).toBe('idle');
  });

  it('loadCachedCategories sets ready from DB even before scan runs', async () => {
    // 模拟 DB 已经有上次扫描的分类；loadCachedCategories 应立刻把 status 置 ready，
    // 让侧栏在扫描跑之前就能渲染 —— 这是启动加速的关键路径。
    const { listCategories } = await import('../lib/library-api');
    const listMock = vi.mocked(listCategories);
    listMock.mockClear();
    listMock.mockResolvedValue([
      { path: 'a', parentPath: null, title: 'A', sortOrder: 1 },
      { path: 'a/sub', parentPath: 'a', title: 'Sub', sortOrder: 2 },
    ]);

    await useLibraryStore.getState().loadCachedCategories();
    const s = useLibraryStore.getState();
    expect(s.status).toBe('ready');
    expect(s.categories).toHaveLength(2);
    expect(listMock).toHaveBeenCalledTimes(1);
  });

  it('loadCachedCategories silently no-ops when DB is empty (first launch)', async () => {
    // 首次启动 DB 空 —— listCategories 返回 [] 是正常的；
    // 不应把 status 推到 'error'，也不应清掉已就绪的状态。
    const { listCategories } = await import('../lib/library-api');
    vi.mocked(listCategories).mockClear();
    vi.mocked(listCategories).mockResolvedValue([]);
    useLibraryStore.setState({ status: 'idle' });
    await useLibraryStore.getState().loadCachedCategories();
    const s = useLibraryStore.getState();
    expect(s.status).toBe('idle');
    expect(s.categories).toEqual([]);
  });

  it('scan transitions idle → scanning → ready and preserves articlesByPath cache', async () => {
    // scan 不应清空 articlesByPath / indexByPath —— 已经渲染的 section 不能
    // 因为 scan 完成而闪回 "加载中…"（loadIndex 的 useEffect 不依赖 index，
    // wipe 后不会重新触发，section 会永远卡在加载中）。
    useLibraryStore.setState({
      articlesByPath: { 'old/path': [] },
      indexByPath: { 'old/path': { title: null, preamble: null, sections: [] } },
    });
    const p = useLibraryStore.getState().scan();
    expect(useLibraryStore.getState().status).toBe('scanning');
    await p;
    const s = useLibraryStore.getState();
    expect(s.status).toBe('ready');
    expect(s.categories.length).toBe(1);
    expect(s.summary?.resourcesCount).toBe(2);
    expect(s.articlesByPath).toEqual({ 'old/path': [] });
    expect(s.indexByPath).toEqual({
      'old/path': { title: null, preamble: null, sections: [] },
    });
  });

  it('loadArticles fetches and caches on first call', async () => {
    const { listResources } = await import('../lib/library-api');
    const listResourcesMock = vi.mocked(listResources);
    listResourcesMock.mockImplementationOnce(async () => [
      // 模拟返回的资源：类型对齐 category, 父路径=目标 path
      { id: 1, categoryPath: 'a/1', relPath: 'x.md', type: 'markdown', title: 'X', sizeBytes: 0, indexedAt: '', pageCount: null, wordCount: null },
    ] as any);

    await useLibraryStore.getState().loadArticles('a/1');
    const articles2 = useLibraryStore.getState().articlesByPath;
    expect(articles2['a/1']).toHaveLength(1);
    expect(articles2['a/1'][0].id).toBe(1);
  });

  it('loadArticles is cached — does not re-fetch on second call with same path', async () => {
    const { listResources } = await import('../lib/library-api');
    const listResourcesMock = vi.mocked(listResources);
    listResourcesMock.mockClear();
    listResourcesMock.mockResolvedValue([]);

    await useLibraryStore.getState().loadArticles('a/1');
    await useLibraryStore.getState().loadArticles('a/1');
    expect(listResourcesMock).toHaveBeenCalledTimes(1);
  });

  it('loadIndex re-reads _index.md on every call (no stale cache)', async () => {
    const { readSubcategoryIndex } = await import('../lib/library-api');
    const readIndexMock = vi.mocked(readSubcategoryIndex);
    readIndexMock.mockClear();

    // 首次：含 entry A
    readIndexMock.mockResolvedValueOnce({
      title: 'Test',
      preamble: null,
      sections: [
        {
          heading: '文章目录',
          rawMarkdown: '',
          groups: [
            {
              subheading: null,
              entries: [{ relPath: 'a.md', title: 'A', description: null }],
              rawMarkdown: '',
            },
          ],
        },
      ],
    });
    await useLibraryStore.getState().loadIndex('cat/sub');
    const idx1 = useLibraryStore.getState().indexByPath['cat/sub']!;
    expect(idx1.sections[0].groups[0].entries).toHaveLength(1);

    // 用户编辑 _index.md 删除了 entry A —— 第二次调用应返回新内容
    readIndexMock.mockClear();
    readIndexMock.mockResolvedValueOnce({
      title: 'Test',
      preamble: null,
      sections: [
        {
          heading: '文章目录',
          rawMarkdown: '',
          groups: [{ subheading: null, entries: [], rawMarkdown: '' }],
        },
      ],
    });
    await useLibraryStore.getState().loadIndex('cat/sub');
    // 缓存被绕过：拿到的是新内容（空 entries）
    const idx2 = useLibraryStore.getState().indexByPath['cat/sub']!;
    expect(idx2.sections[0].groups[0].entries).toHaveLength(0);
    expect(readIndexMock).toHaveBeenCalledTimes(1);
  });

  it('loadIndex retries after failure (does not stick to null)', async () => {
    const { readSubcategoryIndex } = await import('../lib/library-api');
    const readIndexMock = vi.mocked(readSubcategoryIndex);
    readIndexMock.mockClear();
    readIndexMock.mockRejectedValueOnce(new Error('not found'));

    await useLibraryStore.getState().loadIndex('cat/missing');
    expect(useLibraryStore.getState().indexByPath['cat/missing']).toBeNull();

    // 用户事后创建了 _index.md —— 下一次调用应重新尝试并拿到内容
    readIndexMock.mockClear();
    readIndexMock.mockResolvedValueOnce({
      title: '现在有了',
      preamble: null,
      sections: [],
    });
    await useLibraryStore.getState().loadIndex('cat/missing');
    expect(useLibraryStore.getState().indexByPath['cat/missing']).toEqual({
      title: '现在有了',
      preamble: null,
      sections: [],
    });
    expect(readIndexMock).toHaveBeenCalledTimes(1);
  });

  it('selectResource triggers readResource', async () => {
    useLibraryStore.setState({ status: 'ready' });
    await useLibraryStore.getState().selectResource(7);
    const s = useLibraryStore.getState();
    expect(s.selectedResourceId).toBe(7);
    expect(s.resourceContent?.type).toBe('markdown');
  });

  // ---- 同步行为（RULING 1 + RULING 2） ----
  //
  // RULING 1：syncNow 必须返回 boolean，且 synced 路径内部 scan、skipped 路径不 scan，
  // 让 caller 的启动序列可以"恰好一次"扫描。
  //
  // RULING 2：sync_progress 事件 payload 是 {done, total}，整批失败时多了 error 字段。
  // 这里不直接测事件（事件 listener 由 UI 层负责），但 store 暴露的 downloadProgress
  // 必须能诚实地反映后端会送什么 —— 不会因为 store 假设"done==total 即成功"而误导 UI。

  it('syncNow returns true when manifest applied and calls scan(false) internally', async () => {
    // synced 路径：syncManifest 返回 skipped=false，syncNow 应当：
    // - 内部调 scan(false) 重建库（避免 caller 再扫一次）
    // - 内部调 refreshSyncStatus 拿最新 present/total —— "sync 成功"必然伴随
    //   最新 status 是 syncNow 对所有 caller（启动序列、SyncForm）的共同契约，
    //   移到这里之后 App.tsx 启动序列可以省掉重复的 IPC
    // - 翻回 idle 并返回 true
    const sync = await import('../lib/sync');
    vi.mocked(sync.syncManifest).mockResolvedValueOnce({
      files: 380,
      indexes: 12,
      skipped: false,
    });
    const api = await import('../lib/library-api');
    vi.mocked(api.scanLibrary).mockClear();

    const synced = await useLibraryStore.getState().syncNow();

    expect(synced).toBe(true);
    expect(api.scanLibrary).toHaveBeenCalledTimes(1);
    expect(api.scanLibrary).toHaveBeenCalledWith(false);
    // 这两条断言同时 pin 住"syncNow 内部 refresh"：任一失败都意味着实现丢了这个调用，
    // 而去掉内部 refreshSyncStatus 会让 store.syncStatus 在 synced 路径下保持 null，
    // SyncForm 等 caller 看到的 status 就是 stale 的。这两条必须双绿。
    expect(sync.syncStatus).toHaveBeenCalledTimes(1);
    expect(useLibraryStore.getState().syncStatus?.total).toBe(380);
    expect(useLibraryStore.getState().syncPhase).toBe('idle');
  });

  it('syncNow returns false when manifest is skipped and does NOT call scan', async () => {
    // skipped 路径：syncManifest 返回 skipped=true，syncNow 必须：
    // - **不**调 scan —— 启动序列的 caller 会自己扫
    // - 返回 false 让 caller 决定后续行为
    // - 把 syncPhase 翻回 idle（不是 error）
    const sync = await import('../lib/sync');
    vi.mocked(sync.syncManifest).mockResolvedValueOnce({
      files: 0,
      indexes: 0,
      skipped: true,
    });
    const api = await import('../lib/library-api');
    vi.mocked(api.scanLibrary).mockClear();

    const synced = await useLibraryStore.getState().syncNow();

    expect(synced).toBe(false);
    expect(api.scanLibrary).not.toHaveBeenCalled();
    expect(useLibraryStore.getState().syncPhase).toBe('idle');
    expect(useLibraryStore.getState().syncStatus).toBeNull();
  });

  it('syncNow on error returns false and does NOT clobber library status', async () => {
    // 关键不变量：syncNow 抛错时**只动 syncPhase / error**，不能把冷启动的 `status: 'idle'`
    // 推到 `status: 'error'`，否则首次启动看到红色空框。
    const sync = await import('../lib/sync');
    vi.mocked(sync.syncManifest).mockRejectedValueOnce(new Error('network down'));
    // 模拟冷启动：status='idle'，categories=[]
    useLibraryStore.setState({ status: 'idle', categories: [], error: undefined });

    const synced = await useLibraryStore.getState().syncNow();

    expect(synced).toBe(false);
    expect(useLibraryStore.getState().status).toBe('idle');
    expect(useLibraryStore.getState().categories).toEqual([]);
    expect(useLibraryStore.getState().syncPhase).toBe('error');
    expect(useLibraryStore.getState().error).toContain('network down');
  });

  it('refreshSyncStatus silently swallows errors and preserves prior syncStatus', async () => {
    // 状态条是增强信息 —— 拉不到就静默隐藏，不能让 UI 出红条吓用户。
    // 上次的 syncStatus 保留下来（避免"曾经有数据 → 拉失败 → UI 突然空白"的闪烁）。
    const sync = await import('../lib/sync');
    vi.mocked(sync.syncStatus).mockRejectedValueOnce(new Error('IPC boom'));
    useLibraryStore.setState({
      syncStatus: { total: 10, present: 5, totalBytes: 100, cachedBytes: 50, configured: true },
    });

    await useLibraryStore.getState().refreshSyncStatus();

    expect(useLibraryStore.getState().syncStatus).toEqual({
      total: 10,
      present: 5,
      totalBytes: 100,
      cachedBytes: 50,
      configured: true,
    });
  });

  it('downloadAll sets initial progress to (0, 0) so UI can render the bar before first event', async () => {
    // 后端 fire-and-forget 后立刻返回 —— 在 sync_progress 第一个事件到来前，
    // UI 应当已经能渲染进度条占位；(0, 0) 与 null 的区别就是"已开始 vs 还没开始"。
    await useLibraryStore.getState().downloadAll();
    const s = useLibraryStore.getState();
    expect(s.syncPhase).toBe('downloading');
    expect(s.downloadProgress).toEqual({ done: 0, total: 0 });
  });

  it('reset clears sync fields so the next session does not inherit stale state', async () => {
    // 不重置 syncStatus / syncPhase / downloadProgress 的话，登录态/账号切换时旧值会泄漏。
    useLibraryStore.setState({
      syncStatus: { total: 10, present: 5, totalBytes: 100, cachedBytes: 50, configured: true },
      syncPhase: 'downloading',
      downloadProgress: { done: 3, total: 10 },
    });

    useLibraryStore.getState().reset();

    const s = useLibraryStore.getState();
    expect(s.syncStatus).toBeNull();
    expect(s.syncPhase).toBe('idle');
    expect(s.downloadProgress).toBeNull();
  });
});