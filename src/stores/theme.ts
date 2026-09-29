import { create } from 'zustand';

export type Theme = 'light' | 'dark';

interface ThemeState {
  theme: Theme;
  toggle: () => void;
  set: (t: Theme) => void;
}

const STORAGE_KEY = 'ai-stack:theme';

function applyTheme(t: Theme) {
  const root = document.documentElement;
  if (t === 'dark') root.classList.add('dark');
  else root.classList.remove('dark');
}

function readInitial(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'dark' || stored === 'light') return stored;
  } catch {
    // localStorage 不可用时降级到 light（Review Focus #1）
  }
  return 'light';
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: readInitial(),
  toggle: () => {
    const next: Theme = get().theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem(STORAGE_KEY, next); } catch { /* ignore */ }
    set({ theme: next });
  },
  set: (t: Theme) => {
    applyTheme(t);
    try { localStorage.setItem(STORAGE_KEY, t); } catch { /* ignore */ }
    set({ theme: t });
  },
}));

// 首次加载时把当前主题应用到 DOM
applyTheme(useThemeStore.getState().theme);
