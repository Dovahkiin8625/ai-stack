import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../lib/library-api', () => ({
  listNotes: vi.fn(async () => []),
  createNote: vi.fn(),
  updateNote: vi.fn(),
  deleteNote: vi.fn(),
}));

import * as api from '../lib/library-api';
import { useNotesStore } from './notes';

function makeNote(overrides: Partial<{
  id: number;
  resourceId: number;
  content: string;
  anchorText: string | null;
  anchorOccurrence: number;
  source: 'user' | 'ai';
  createdAt: string;
  updatedAt: string;
}> = {}) {
  return {
    id: 1,
    resourceId: 1,
    content: '',
    anchorText: null,
    anchorOccurrence: 0,
    source: 'user' as const,
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

describe('notes store — AI 标注编辑时降级为用户笔记', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useNotesStore.setState({
      resourceId: 1,
      notes: [],
      loading: false,
      panelOpen: false,
    });
  });

  it('update() 在传入 source 时把它传给后端并更新本地 store', async () => {
    const aiNote = makeNote({ id: 100, content: 'AI 原文', source: 'ai' });
    useNotesStore.setState({ notes: [aiNote] });

    const mockUpdate = vi.mocked(api.updateNote).mockResolvedValueOnce({
      ...aiNote,
      content: '用户改写',
      source: 'user',
    });

    await useNotesStore.getState().update(100, {
      content: '用户改写',
      anchorText: null,
      anchorOccurrence: 0,
      source: 'user',
    });

    // 后端调用必须带上 source='user'
    expect(mockUpdate).toHaveBeenCalledWith({
      id: 100,
      content: '用户改写',
      anchorText: null,
      anchorOccurrence: 0,
      source: 'user',
    });
    // 本地 store 的 source 必须从 'ai' 变为 'user'
    expect(useNotesStore.getState().notes[0].source).toBe('user');
    expect(useNotesStore.getState().notes[0].content).toBe('用户改写');
  });

  it('update() 不传 source 时不主动改 source（向后兼容）', async () => {
    const userNote = makeNote({ id: 200, content: 'hello', source: 'user' });
    useNotesStore.setState({ notes: [userNote] });

    const mockUpdate = vi.mocked(api.updateNote).mockResolvedValueOnce({
      ...userNote,
      content: 'hi',
    });

    await useNotesStore.getState().update(200, {
      content: 'hi',
      anchorText: null,
      anchorOccurrence: 0,
    });

    // 不传 source 时，调用 payload 里也不应出现 source 字段（或为 undefined）
    const callArg = mockUpdate.mock.calls[0][0];
    expect(callArg.source).toBeUndefined();
    expect(useNotesStore.getState().notes[0].source).toBe('user');
  });
});

describe('notes store — AI 讲解流式失败时让用户看见原因', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useNotesStore.setState({
      resourceId: 1,
      notes: [makeNote({ id: 300, content: '', source: 'ai' })],
      loading: false,
      panelOpen: false,
    });
  });

  it('setNoteContent 把占位的空字符串替换为后端写入的错误描述', () => {
    useNotesStore.getState().setNoteContent(
      300,
      'AI 讲解失败：HTTP 400: ...',
    );
    expect(useNotesStore.getState().notes[0].content).toBe(
      'AI 讲解失败：HTTP 400: ...',
    );
  });

  it('setNoteContent 不影响其它笔记', () => {
    useNotesStore.setState({
      notes: [
        makeNote({ id: 301, content: 'AI 占位 1', source: 'ai' }),
        makeNote({ id: 302, content: '用户笔记', source: 'user' }),
      ],
    });
    useNotesStore.getState().setNoteContent(301, 'AI 讲解失败：boom');
    const state = useNotesStore.getState();
    expect(state.notes[0].content).toBe('AI 讲解失败：boom');
    expect(state.notes[1].content).toBe('用户笔记');
  });
});

describe('notes store — 流式事件先于 addNote 落地的 race 修复', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useNotesStore.setState({
      resourceId: 1,
      notes: [],
      loading: false,
      panelOpen: false,
    });
  });

  it('appendToNote 在 note 尚未入 store 时排队，addNote 后一并 apply', () => {
    // 模拟后端在 addNote 前 emit 了一个 chunk（典型场景：401 错误事件抢先到达）
    useNotesStore.getState().appendToNote(500, '错误');
    // 这时候 store 里没有 500，append 不应该 throw，也不该产生空副作用
    expect(useNotesStore.getState().notes.find((n) => n.id === 500)).toBeUndefined();

    // addNote 后，排队的事件被 drain
    useNotesStore.getState().addNote(makeNote({ id: 500, content: '', source: 'ai' }));
    const note = useNotesStore.getState().notes.find((n) => n.id === 500)!;
    expect(note.content).toBe('错误');
  });

  it('多个 chunk 在 addNote 前累积，drain 时按入队顺序拼接', () => {
    useNotesStore.getState().appendToNote(501, '一');
    useNotesStore.getState().appendToNote(501, '二');
    useNotesStore.getState().appendToNote(501, '三');
    useNotesStore.getState().addNote(makeNote({ id: 501, content: '', source: 'ai' }));
    const note = useNotesStore.getState().notes.find((n) => n.id === 501)!;
    expect(note.content).toBe('一二三');
  });

  it('setNoteContent 在 note 尚未入 store 时排队，addNote 后覆盖占位空字符串', () => {
    // 模拟 error 事件先于 addNote：典型 401/404 极快失败
    useNotesStore.getState().setNoteContent(502, 'AI 讲解失败：HTTP 401');
    useNotesStore.getState().addNote(makeNote({ id: 502, content: '', source: 'ai' }));
    const note = useNotesStore.getState().notes.find((n) => n.id === 502)!;
    expect(note.content).toBe('AI 讲解失败：HTTP 401');
  });

  it('chunk 与 error 混合排队，addNote 后按入队顺序 apply —— error 覆盖 chunk', () => {
    // 实际场景：流式中途失败，error 事件携带后端写入 DB 的完整错误文本，
    // 用 setNoteContent 整体替换占位内容，与 DB 行为一致。
    // 这里的语义是"显示失败原因"，不是"在已有内容上追加"。
    useNotesStore.getState().appendToNote(503, '首段');
    useNotesStore.getState().setNoteContent(503, 'AI 讲解失败：超时');
    useNotesStore.getState().addNote(makeNote({ id: 503, content: '', source: 'ai' }));
    const note = useNotesStore.getState().notes.find((n) => n.id === 503)!;
    expect(note.content).toBe('AI 讲解失败：超时');
  });

  it('note 已经在 store 时，appendToNote 立即生效而不排队', () => {
    useNotesStore.setState({
      notes: [makeNote({ id: 504, content: '已有内容', source: 'ai' })],
    });
    useNotesStore.getState().appendToNote(504, ' 追加');
    const note = useNotesStore.getState().notes.find((n) => n.id === 504)!;
    expect(note.content).toBe('已有内容 追加');
  });
});