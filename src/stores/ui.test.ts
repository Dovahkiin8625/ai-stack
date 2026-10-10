import { describe, it, expect, beforeEach } from 'vitest';
import { useUiStore } from './ui';

describe('ui store', () => {
  beforeEach(() => useUiStore.setState({ navOpen: false }));
  it('opens and closes the nav drawer', () => {
    useUiStore.getState().openNav();
    expect(useUiStore.getState().navOpen).toBe(true);
    useUiStore.getState().closeNav();
    expect(useUiStore.getState().navOpen).toBe(false);
  });
  it('toggles', () => {
    useUiStore.getState().toggleNav();
    expect(useUiStore.getState().navOpen).toBe(true);
    useUiStore.getState().toggleNav();
    expect(useUiStore.getState().navOpen).toBe(false);
  });
});
