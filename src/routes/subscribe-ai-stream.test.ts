import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock @tauri-apps/api/event 让测试能捕获"监听了哪些事件"
vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(async (_eventName: string, _handler: unknown) => {
    // 返回一个 fake unlisten fn
    return vi.fn(async () => undefined);
  }),
}));

vi.mock('../stores/notes', () => ({
  useNotesStore: {
    getState: () => ({
      appendToNote: vi.fn(),
      setNoteContent: vi.fn(),
    }),
  },
}));

import { listen } from '@tauri-apps/api/event';
import { subscribeAiStream } from './subscribe-ai-stream';

describe('subscribeAiStream (AI 流式监听器)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('AI 讲解场景用 ai-annotate-* 事件名', async () => {
    const cleanup = await subscribeAiStream('ai-annotate', 'AI annotate error');
    expect(listen).toHaveBeenCalledWith('ai-annotate-chunk', expect.any(Function));
    expect(listen).toHaveBeenCalledWith('ai-annotate-done', expect.any(Function));
    expect(listen).toHaveBeenCalledWith('ai-annotate-error', expect.any(Function));
    await cleanup();
  });

  it('AI 问答场景用 ai-qa-* 事件名（与 ai-annotate 物理隔离）', async () => {
    const cleanup = await subscribeAiStream('ai-qa', 'AI ask error');
    expect(listen).toHaveBeenCalledWith('ai-qa-chunk', expect.any(Function));
    expect(listen).toHaveBeenCalledWith('ai-qa-done', expect.any(Function));
    expect(listen).toHaveBeenCalledWith('ai-qa-error', expect.any(Function));
    // 关键：不应监听 ai-annotate 系列 —— 否则两个流会互相串扰
    expect(listen).not.toHaveBeenCalledWith('ai-annotate-chunk', expect.any(Function));
    expect(listen).not.toHaveBeenCalledWith('ai-annotate-done', expect.any(Function));
    expect(listen).not.toHaveBeenCalledWith('ai-annotate-error', expect.any(Function));
    await cleanup();
  });

  it('返回一个 cleanup 函数，调用后会 unlisten 所有监听器', async () => {
    const unlistens: Array<() => Promise<void>> = [];
    vi.mocked(listen).mockImplementation(async () => {
      const u = vi.fn(async () => undefined);
      unlistens.push(u);
      return u;
    });

    const cleanup = await subscribeAiStream('ai-qa', 'AI ask error');
    // 三次 listen（chunk / done / error）应该都有对应的 unlisten 已被调用
    expect(unlistens.length).toBe(3);

    // cleanup 还没调用 → 没人 unlisten
    for (const u of unlistens) expect(u).not.toHaveBeenCalled();

    await cleanup();

    // cleanup 之后 → 三个都被 unlisten
    for (const u of unlistens) expect(u).toHaveBeenCalledTimes(1);
  });
});