import { describe, it, expect, vi, beforeEach } from 'vitest';

const state = new Map<string, unknown>();
vi.mock('@tauri-apps/plugin-store', () => ({
  load: vi.fn(async () => ({
    get: async (k: string) => state.get(k),
    set: async (k: string, v: unknown) => { state.set(k, v); },
    save: async () => {},
  })),
}));

import { getSettings, saveSettings, DEFAULTS } from './tauri';

describe('syncBaseUrl', () => {
  beforeEach(() => state.clear());

  it('defaults to empty string', async () => {
    expect((await getSettings()).syncBaseUrl).toBe('');
  });

  it('round-trips', async () => {
    await saveSettings({ syncBaseUrl: 'https://x.com/kb' });
    expect((await getSettings()).syncBaseUrl).toBe('https://x.com/kb');
  });

  it('DEFAULTS contains syncBaseUrl', () => {
    expect(DEFAULTS.syncBaseUrl).toBe('');
  });
});
