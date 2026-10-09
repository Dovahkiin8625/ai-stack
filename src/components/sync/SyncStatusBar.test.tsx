// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import SyncStatusBar from './SyncStatusBar';
import { useLibraryStore } from '../../stores/library';

// onSyncProgress 内部 listen() 来自 @tauri-apps/api/event —— 在 jsdom 下没有真事件系统，
// stub 出一个立刻 resolve 的 unlisten 函数，组件挂载时不会炸。
vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(async () => () => undefined),
}));

describe('SyncStatusBar', () => {
  beforeEach(() => {
    // 每个用例前把 store 拉回初始 —— 状态条的几个用例会塞不同的 syncStatus，
    // 不重置会污染下一个用例。
    useLibraryStore.setState({
      syncStatus: null,
      syncPhase: 'idle',
      downloadProgress: null,
    });
  });

  it('未配置远端时不渲染（developer/纯本地模式）', () => {
    // configured: false → 不渲染条 —— 一个永远 "0/0 downloaded" 的条是噪音
    useLibraryStore.setState({
      syncStatus: { total: 0, present: 0, totalBytes: 0, cachedBytes: 0, configured: false },
    });
    const { container } = render(<SyncStatusBar />);
    expect(container.firstChild).toBeNull();
  });

  it('显示已缓存数量并提供「下载全部」', () => {
    useLibraryStore.setState({
      syncStatus: { total: 380, present: 37, totalBytes: 1000, cachedBytes: 100, configured: true },
      syncPhase: 'idle',
      downloadProgress: null,
    });
    render(<SyncStatusBar />);
    expect(screen.getByText(/37 \/ 380/)).toBeTruthy();
    expect(screen.getByRole('button', { name: '下载全部' })).toBeTruthy();
  });

  it('全部缓存后不渲染（无意义的"全完成"长条）', () => {
    // present >= total → 不渲染条；这是 RULING 2/3 的边界 —— 已经全缓存了
    // 状态条就该消失，而不是永久显示 "已缓存 380 / 380"。
    useLibraryStore.setState({
      syncStatus: { total: 380, present: 380, totalBytes: 1000, cachedBytes: 1000, configured: true },
    });
    const { container } = render(<SyncStatusBar />);
    expect(container.firstChild).toBeNull();
  });

  it('下载中显示进度而非缓存统计', () => {
    useLibraryStore.setState({
      syncStatus: { total: 380, present: 37, totalBytes: 1000, cachedBytes: 100, configured: true },
      syncPhase: 'downloading',
      downloadProgress: { done: 12, total: 343 },
    });
    render(<SyncStatusBar />);
    expect(screen.getByText(/12 \/ 343/)).toBeTruthy();
    // 下载中不应再渲染「下载全部」按钮（RULING 2：disable 而不是装饰 暂停）
    expect(screen.queryByRole('button', { name: '下载全部' })).toBeNull();
  });
});