import { describe, it, expect, vi } from 'vitest';

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(async (cmd: string, _args?: unknown) => {
    if (cmd === 'list_categories') return [
      { path: 'a', parentPath: null, title: 'A', sortOrder: 1 },
    ];
    if (cmd === 'list_resources') return [];
    if (cmd === 'read_resource') throw new Error('boom');
    if (cmd === 'scan_library') return { categoriesCount: 0, resourcesCount: 0, errorsCount: 0, durationMs: 0 };
    if (cmd === 'start_ai_qa') return {
      id: 99,
      resourceId: 1,
      content: '',
      anchorText: '选区',
      anchorOccurrence: 0,
      prompt: '用户问题',
      source: 'ai',
      createdAt: '2026-01-01T00:00:00Z',
      updatedAt: '2026-01-01T00:00:00Z',
    };
    return null;
  }),
}));

import { invoke } from '@tauri-apps/api/core';
import { scanLibrary, listCategories, listResources, readResource, startAiAsk } from './library-api';

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

describe('startAiAsk (选中询问)', () => {
  it('startAiAsk 调用 invoke(\'start_ai_qa\', { payload }) 并把 question 透传', async () => {
    const note = await startAiAsk({
      resourceId: 7,
      selectedText: '反向传播算法',
      contextBefore: '前文',
      contextAfter: '后文',
      sectionTitle: '章节',
      question: '为什么用 sigmoid?',
      baseUrl: 'https://example.com',
      performanceModel: 'claude-sonnet-4-5',
      apiKey: 'sk-test',
    });

    // invoke 必须用 start_ai_qa 命令名（与后端 commands.rs 注册名一致），
    // 并且 payload 平铺到根级 invoke args（与 startAiAnnotate 一致）
    expect(invoke).toHaveBeenCalledWith('start_ai_qa', {
      payload: {
        resourceId: 7,
        selectedText: '反向传播算法',
        contextBefore: '前文',
        contextAfter: '后文',
        sectionTitle: '章节',
        question: '为什么用 sigmoid?',
        baseUrl: 'https://example.com',
        performanceModel: 'claude-sonnet-4-5',
        apiKey: 'sk-test',
      },
    });

    // 后端返回的占位 note 必须包含 prompt 字段（用户问题）
    expect(note.prompt).toBe('用户问题');
    expect(note.source).toBe('ai');
  });
});