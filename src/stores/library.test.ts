import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../lib/library-api', () => ({
  scanLibrary: vi.fn(async () => ({ categoriesCount: 1, resourcesCount: 2, errorsCount: 0, durationMs: 5 })),
  listCategories: vi.fn(async () => [
    { path: 'a', parentPath: null, title: 'A', sortOrder: 1 },
  ]),
  listResources: vi.fn(async () => []),
  readResource: vi.fn(async () => ({ type: 'markdown', html: '<p>x</p>', wordCount: 1 })),
}));

import { useLibraryStore } from './library';

describe('library store', () => {
  beforeEach(() => {
    useLibraryStore.setState({
      status: 'idle',
      categories: [],
      selectedResourceId: null,
      resourceContent: null,
      articlesByPath: {},
    });
  });

  it('initial status is idle', () => {
    expect(useLibraryStore.getState().status).toBe('idle');
  });

  it('scan transitions idle → scanning → ready and clears articlesByPath', async () => {
    // 先填一个伪缓存，验证 scan 完成后会清空
    useLibraryStore.setState({ articlesByPath: { 'old/path': [] } });
    const p = useLibraryStore.getState().scan();
    expect(useLibraryStore.getState().status).toBe('scanning');
    await p;
    const s = useLibraryStore.getState();
    expect(s.status).toBe('ready');
    expect(s.categories.length).toBe(1);
    expect(s.summary?.resourcesCount).toBe(2);
    expect(s.articlesByPath).toEqual({});
  });

  it('loadArticles fetches and caches on first call', async () => {
    const { listResources } = await import('../lib/library-api');
    const listResourcesMock = vi.mocked(listResources);
    listResourcesMock.mockImplementationOnce(async () => [
      // 模拟返回的资源：类型对齐 category, 父路径=目标 path
      { id: 1, categoryPath: 'a/1', relPath: 'x.md', type: 'markdown', title: 'X', sizeBytes: 0, hash: '', indexedAt: '', pageCount: null, wordCount: null },
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

  it('selectResource triggers readResource', async () => {
    useLibraryStore.setState({ status: 'ready' });
    await useLibraryStore.getState().selectResource(7);
    const s = useLibraryStore.getState();
    expect(s.selectedResourceId).toBe(7);
    expect(s.resourceContent?.type).toBe('markdown');
  });
});