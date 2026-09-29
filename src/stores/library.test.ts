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
      selectedCategoryPath: null,
      resources: [],
      selectedResourceId: null,
      resourceContent: null,
    });
  });

  it('initial status is idle', () => {
    expect(useLibraryStore.getState().status).toBe('idle');
  });

  it('scan transitions idle → scanning → ready', async () => {
    const p = useLibraryStore.getState().scan();
    expect(useLibraryStore.getState().status).toBe('scanning');
    await p;
    const s = useLibraryStore.getState();
    expect(s.status).toBe('ready');
    expect(s.categories.length).toBe(1);
    expect(s.summary?.resourcesCount).toBe(2);
  });

  it('selectCategory triggers listResources', async () => {
    useLibraryStore.setState({ status: 'ready', categories: [{ path: 'a', parentPath: null, title: 'A', sortOrder: 1 }] });
    await useLibraryStore.getState().selectCategory('a');
    expect(useLibraryStore.getState().selectedCategoryPath).toBe('a');
    expect(useLibraryStore.getState().resources).toEqual([]);
  });

  it('selectResource triggers readResource', async () => {
    useLibraryStore.setState({ status: 'ready' });
    await useLibraryStore.getState().selectResource(7);
    const s = useLibraryStore.getState();
    expect(s.selectedResourceId).toBe(7);
    expect(s.resourceContent?.type).toBe('markdown');
  });
});