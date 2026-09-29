import { describe, it, expect, vi } from 'vitest';

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(async (cmd: string, _args?: unknown) => {
    if (cmd === 'list_categories') return [
      { path: 'a', parentPath: null, title: 'A', sortOrder: 1 },
    ];
    if (cmd === 'list_resources') return [];
    if (cmd === 'read_resource') throw new Error('boom');
    if (cmd === 'scan_library') return { categoriesCount: 0, resourcesCount: 0, errorsCount: 0, durationMs: 0 };
    return null;
  }),
}));

import { scanLibrary, listCategories, listResources, readResource } from './library-api';

describe('library-api', () => {
  it('scanLibrary returns summary', async () => {
    const s = await scanLibrary();
    expect(s.categoriesCount).toBe(0);
  });

  it('listCategories unwraps invoke result', async () => {
    const cats = await listCategories();
    expect(cats[0].title).toBe('A');
  });

  it('listResources returns empty array', async () => {
    const rs = await listResources('a');
    expect(rs).toEqual([]);
  });

  it('readResource propagates error as string', async () => {
    await expect(readResource(1)).rejects.toThrow(/boom/);
  });
});