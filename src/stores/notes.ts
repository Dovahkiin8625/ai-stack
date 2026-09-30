import { create } from 'zustand';
import * as api from '../lib/library-api';
import type { Note } from '../types';

interface NotesState {
  /** 当前展示的资源 ID（保证 store 内容始终只对应一个资源） */
  resourceId: number | null;
  notes: Note[];
  loading: boolean;
  load: (resourceId: number) => Promise<void>;
  clear: () => void;
  create: (input: {
    content: string;
    anchorText: string | null;
    anchorOccurrence: number;
  }) => Promise<Note | null>;
  update: (
    id: number,
    input: { content: string; anchorText: string | null; anchorOccurrence: number },
  ) => Promise<void>;
  remove: (id: number) => Promise<void>;
}

export const useNotesStore = create<NotesState>((set, get) => ({
  resourceId: null,
  notes: [],
  loading: false,

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

  create: async ({ content, anchorText, anchorOccurrence }) => {
    const { resourceId } = get();
    if (resourceId == null) return null;
    try {
      const note = await api.createNote({
        resourceId,
        content,
        anchorText,
        anchorOccurrence,
      });
      set((s) => ({ notes: [...s.notes, note] }));
      return note;
    } catch (e) {
      console.error('createNote failed', e);
      return null;
    }
  },

  update: async (id, { content, anchorText, anchorOccurrence }) => {
    try {
      const updated = await api.updateNote({
        id,
        content,
        anchorText,
        anchorOccurrence,
      });
      set((s) => ({ notes: s.notes.map((n) => (n.id === id ? updated : n)) }));
    } catch (e) {
      console.error('updateNote failed', e);
    }
  },

  remove: async (id) => {
    try {
      await api.deleteNote(id);
      set((s) => ({ notes: s.notes.filter((n) => n.id !== id) }));
    } catch (e) {
      console.error('deleteNote failed', e);
    }
  },
}));
