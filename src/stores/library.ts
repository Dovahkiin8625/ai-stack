import { create } from 'zustand';
import * as api from '../lib/library-api';
import type {
  Category,
  Resource,
  ResourceContent,
  ScanSummary,
} from '../types';

interface LibraryState {
  status: 'idle' | 'scanning' | 'ready' | 'error';
  summary?: ScanSummary;
  categories: Category[];
  selectedCategoryPath: string | null;
  resources: Resource[];
  selectedResourceId: number | null;
  resourceContent: ResourceContent | null;
  error?: string;
  scan: (force?: boolean) => Promise<void>;
  selectCategory: (path: string | null) => Promise<void>;
  selectResource: (id: number | null) => Promise<void>;
  reset: () => void;
}

export const useLibraryStore = create<LibraryState>((set, _get) => ({
  status: 'idle',
  categories: [],
  selectedCategoryPath: null,
  resources: [],
  selectedResourceId: null,
  resourceContent: null,

  scan: async (force = false) => {
    set({ status: 'scanning', error: undefined });
    try {
      const summary = await api.scanLibrary(force);
      const categories = await api.listCategories();
      set({ status: 'ready', summary, categories });
    } catch (e) {
      set({ status: 'error', error: String(e) });
    }
  },

  selectCategory: async (path) => {
    set({ selectedCategoryPath: path, selectedResourceId: null, resourceContent: null });
    if (!path) {
      set({ resources: [] });
      return;
    }
    try {
      const resources = await api.listResources(path);
      set({ resources });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  selectResource: async (id) => {
    set({ selectedResourceId: id, resourceContent: null });
    if (id == null) return;
    try {
      const content = await api.readResource(id);
      set({ resourceContent: content });
    } catch (e) {
      set({ error: String(e), resourceContent: null });
    }
  },

  reset: () => set({
    status: 'idle',
    summary: undefined,
    categories: [],
    selectedCategoryPath: null,
    resources: [],
    selectedResourceId: null,
    resourceContent: null,
    error: undefined,
  }),
}));