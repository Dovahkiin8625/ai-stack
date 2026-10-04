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
});