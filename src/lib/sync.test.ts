import { describe, it, expect, vi } from 'vitest';

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(async (cmd: string) => {
    if (cmd === 'sync_status')
      return { total: 380, present: 37, totalBytes: 1000, cachedBytes: 100, configured: true };
    if (cmd === 'sync_manifest') return { files: 380, indexes: 12, skipped: false };
    if (cmd === 'set_sync_base_url') return null;
    if (cmd === 'download_resource') return null;
    if (cmd === 'download_all') return null;
    return null;
  }),
}));

import { syncStatus, syncManifest, setSyncBaseUrl } from './sync';

describe('sync', () => {
  it('syncStatus returns counts', async () => {
    const s = await syncStatus();
    expect(s.total).toBe(380);
    expect(s.present).toBe(37);
    expect(s.configured).toBe(true);
  });
  it('syncManifest returns file counts', async () => {
    const m = await syncManifest();
    expect(m.files).toBe(380);
    expect(m.skipped).toBe(false);
  });
  it('setSyncBaseUrl trims trailing slash', async () => {
    const { invoke } = await import('@tauri-apps/api/core');
    await setSyncBaseUrl('https://x.com/kb/');
    expect(invoke).toHaveBeenCalledWith('set_sync_base_url', { url: 'https://x.com/kb/' });
  });
});