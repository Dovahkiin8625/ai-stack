import { create } from 'zustand';
import * as api from '../lib/library-api';
import type { Note, NoteSource } from '../types';

/**
 * AI 流式讲解的事件缓冲区（按 noteId 索引）。
 *
 * 解决 startAiAnnotate 后端立刻 spawn 流式任务、而前端还在 await NoteDto /
 * addNote / setPanelOpen 这一段的竞态：
 *   - 后端 spawn 后立刻向 Anthropic 发请求，可能很快返回 4xx
 *   - 错误事件可能在 frontend 完成 setup listener 之前就 emit
 *   - 直接丢弃会导致用户看到空白卡片
 *
 * 策略：listener 一律注册，但事件落到 noteId 时先看 store 里有没有这条
 * noteId 的笔记 —— 没有就排进这个 buffer；addNote 把对应 noteId 的 pending
 * 一并 apply。这样不论事件先到还是笔记先到，结果都是正确的。
 *
 * 缓冲区按 noteId 清理（addNote 后立即 drain），不会长期增长。
 */
const pendingByNoteId = new Map<number, Array<() => void>>();

function queueForNote(noteId: number, apply: () => void) {
  const list = pendingByNoteId.get(noteId);
  if (list) {
    list.push(apply);
  } else {
    pendingByNoteId.set(noteId, [apply]);
  }
}

function drainPendingForNote(noteId: number) {
  const list = pendingByNoteId.get(noteId);
  if (!list) return;
  pendingByNoteId.delete(noteId);
  for (const fn of list) fn();
}

function applyOrQueue(noteId: number, apply: () => void) {
  const inStore = useNotesStore.getState().notes.some((n) => n.id === noteId);
  if (inStore) apply();
  else queueForNote(noteId, apply);
}

interface NotesState {
  /** 当前展示的资源 ID（保证 store 内容始终只对应一个资源） */
  resourceId: number | null;
  notes: Note[];
  loading: boolean;
  /** 笔记抽屉是否展开。抽屉为 fixed 定位、默认隐藏，需要时通过 Topbar 按钮或正文中 mark / 添加讲解气泡触发。 */
  panelOpen: boolean;
  load: (resourceId: number) => Promise<void>;
  clear: () => void;
  togglePanel: () => void;
  setPanelOpen: (open: boolean) => void;
  create: (input: {
    content: string;
    anchorText: string | null;
    anchorOccurrence: number;
    source?: NoteSource;
    /** PDF 笔记用的 0-based 页码定位；markdown/DOCX 不传（保持 null）。 */
    pageIdx?: number | null;
  }) => Promise<Note | null>;
  /**
   * 把一条已存在的笔记直接放进 store（不调后端）。
   * 用于流式 AI 讲解：后端先建占位笔记、前端拿到 Note 后 addNote，
   * 后续 chunk 走 appendToNote —— 完全绕过普通 create 的网络往返。
   *
   * 同时清空 noteId 对应的 pendingByNoteId 队列，把 listener 在 addNote
   * 之前收到的事件（典型场景：流式任务极快失败，error 事件先于 addNote 到达）
   * 一并 apply，确保占位卡片立刻有内容。
   */
  addNote: (note: Note) => void;
  /**
   * 流式追加 note.content 末尾。AI 讲解专用，每来一个 chunk 就拼上一段。
   * 不调后端（后端自己同步写 DB），只更新本地 store 让 React 重渲染。
   * 若 note 还没入 store，则把这次 append 排进 pendingByNoteId，等 addNote 时 drain。
   */
  appendToNote: (noteId: number, text: string) => void;
  update: (
    id: number,
    input: {
      content: string;
      anchorText: string | null;
      anchorOccurrence: number;
      /** 编辑 AI 笔记时降级为用户笔记：不传则保留原 source */
      source?: NoteSource;
      /** PDF 笔记用的 0-based 页码定位；不传则保留原 pageIdx（翻页跳转不失效）。 */
      pageIdx?: number | null;
    },
  ) => Promise<void>;
  /**
   * AI 讲解流式失败时强制把 note.content 设为给定文本，
   * 让用户立即看到"AI 讲解失败：xxx"的提示，而不是空白卡片。
   * 若 note 还没入 store，同样排进 pending 队列。
   */
  setNoteContent: (noteId: number, content: string) => void;
  remove: (id: number) => Promise<void>;
}

export const useNotesStore = create<NotesState>((set, get) => ({
  resourceId: null,
  notes: [],
  loading: false,
  panelOpen: false,

  load: async (resourceId) => {
    set({ resourceId, notes: [], loading: true });
    try {
      const notes = await api.listNotes(resourceId);
      // 防止加载期间切换资源造成回写
      if (get().resourceId === resourceId) set({ notes, loading: false });
    } catch (e) {
      console.error('listNotes failed', e);
      if (get().resourceId === resourceId) set({ notes: [], loading: false });
    }
  },

  clear: () => set({ resourceId: null, notes: [], loading: false }),

  togglePanel: () => set((s) => ({ panelOpen: !s.panelOpen })),
  setPanelOpen: (open) => set({ panelOpen: open }),

  create: async ({ content, anchorText, anchorOccurrence, source, pageIdx }) => {
    const { resourceId } = get();
    if (resourceId == null) return null;
    try {
      const note = await api.createNote({
        resourceId,
        content,
        anchorText,
        anchorOccurrence,
        source,
        pageIdx,
      });
      set((s) => ({ notes: [...s.notes, note] }));
      return note;
    } catch (e) {
      console.error('createNote failed', e);
      return null;
    }
  },

  addNote: (note) => {
    set((s) => ({ notes: [...s.notes, note] }));
    // 笔记入 store 后，把 listener 在此之前排队的 chunk/error 全部 apply 一次
    drainPendingForNote(note.id);
  },

  appendToNote: (noteId, text) => {
    applyOrQueue(noteId, () => {
      set((s) => ({
        notes: s.notes.map((n) =>
          n.id === noteId ? { ...n, content: n.content + text } : n,
        ),
      }));
    });
  },

  update: async (id, { content, anchorText, anchorOccurrence, source, pageIdx }) => {
    try {
      const updated = await api.updateNote({
        id,
        content,
        anchorText,
        anchorOccurrence,
        source,
        pageIdx,
      });
      set((s) => ({ notes: s.notes.map((n) => (n.id === id ? updated : n)) }));
    } catch (e) {
      console.error('updateNote failed', e);
    }
  },

  setNoteContent: (noteId, content) => {
    applyOrQueue(noteId, () => {
      set((s) => ({
        notes: s.notes.map((n) => (n.id === noteId ? { ...n, content } : n)),
      }));
    });
  },

  remove: async (id) => {
    try {
      await api.deleteNote(id);
      set((s) => ({ notes: s.notes.filter((n) => n.id !== id) }));
      // 删除时清掉可能残留的 buffer
      pendingByNoteId.delete(id);
    } catch (e) {
      console.error('deleteNote failed', e);
    }
  },
}));
