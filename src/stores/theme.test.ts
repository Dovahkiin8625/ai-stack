import { describe, it, expect, beforeEach } from 'vitest';
import { useThemeStore } from './theme';

describe('theme store', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove('dark');
    useThemeStore.setState({ theme: 'light' });
  });

  it('默认主题为 light', () => {
    expect(useThemeStore.getState().theme).toBe('light');
  });

  it('toggle 切换 light <-> dark', () => {
    const { toggle } = useThemeStore.getState();
    toggle();
    expect(useThemeStore.getState().theme).toBe('dark');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    toggle();
    expect(useThemeStore.getState().theme).toBe('light');
    expect(document.documentElement.classList.contains('dark')).toBe(false);
  });

  it('set(dark) 立即应用 dark class', () => {
    useThemeStore.getState().set('dark');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });
});
