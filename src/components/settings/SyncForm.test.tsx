// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import SyncForm from './SyncForm';
import { useLibraryStore } from '../../stores/library';

// 隔离外部命令：SyncForm 触发 拉取清单 会同时调用
// tauri.saveSettings 和 sync.setSyncBaseUrl，再走 useLibraryStore.syncNow()
// （最终到 sync.syncManifest + 内部 scan(false) 走 library-api）。
// 全部相关 path 都得打桩，jsdom 下 IPC 抛 "command not found" 让 promise reject。
vi.mock('../../lib/tauri', () => ({
  saveSettings: vi.fn(async () => undefined),
}));
vi.mock('../../lib/sync', () => ({
  syncStatus: vi.fn(async () => ({
    total: 0, present: 0, totalBytes: 0, cachedBytes: 0, configured: true,
  })),
  syncManifest: vi.fn(async () => ({ files: 1, indexes: 0, skipped: false })),
  setSyncBaseUrl: vi.fn(async () => undefined),
  downloadResource: vi.fn(async () => undefined),
  downloadAll: vi.fn(async () => undefined),
  onSyncProgress: vi.fn(async () => () => undefined),
  onSeedProgress: vi.fn(async () => () => undefined),
}));
// syncNow 内部 scan(false) 需要 library-api 全套打桩。
vi.mock('../../lib/library-api', () => ({
  scanLibrary: vi.fn(async () => ({ categoriesCount: 0, resourcesCount: 0, errorsCount: 0, durationMs: 0 })),
  listCategories: vi.fn(async () => []),
  listResources: vi.fn(async () => []),
  readResource: vi.fn(async () => ({ type: 'markdown', html: '', wordCount: 0, markdown: '' })),
  readSubcategoryIndex: vi.fn(async () => ({ title: null, preamble: null, sections: [] })),
  readResourceBytes: vi.fn(async () => new Uint8Array()),
  writeResource: vi.fn(async () => ({ type: 'markdown', html: '', wordCount: 0, markdown: '' })),
}));

import * as tauri from '../../lib/tauri';
import * as syncLib from '../../lib/sync';

describe('SyncForm', () => {
  beforeEach(() => {
    // 用例之间必须清掉 store 状态 —— 否则上一轮的 syncPhase / downloadProgress 会
    // 渗到下一轮（"按钮不该禁用却禁用" 这类 flaky 通常由此而来）。
    useLibraryStore.setState({
      syncPhase: 'idle',
      downloadProgress: null,
      syncStatus: null,
    });
    vi.mocked(tauri.saveSettings).mockReset();
    vi.mocked(syncLib.setSyncBaseUrl).mockReset();
  });

  it('初始态：拉取清单按钮可见、开始同步按钮禁用（未拉过清单）', () => {
    render(<SyncForm url="https://x.com/kb" onUrlChange={() => {}} />);
    expect(screen.getByRole('button', { name: /拉取清单/ })).toBeTruthy();
    // 「开始同步」按钮在没拉过清单时必须禁用 —— 不然用户没拉取就去同步
    // 会落到 download_all 上却没有任何 manifest 数据，浪费时间且误导。
    const syncBtn = screen.getByRole('button', { name: /开始同步/ });
    expect((syncBtn as HTMLButtonElement).disabled).toBe(true);
  });

  it('拉取清单成功后：开始同步按钮变为可用', async () => {
    useLibraryStore.setState({
      syncPhase: 'idle',
      downloadProgress: null,
    });
    render(<SyncForm url="https://x.com/kb" onUrlChange={() => {}} />);
    // 触发 拉取清单 —— mock 走 useLibraryStore.syncNow() 的默认成功路径
    // （test setup 里没 override，所以 syncPhase 应该保持 idle 后被内部 scan 套用）。
    fireEvent.click(screen.getByRole('button', { name: /拉取清单/ }));

    // 让内部 await 都 settle
    await vi.waitFor(() =>
      expect(
        (screen.getByRole('button', { name: /开始同步/ }) as HTMLButtonElement).disabled
      ).toBe(false)
    );
  });

  it('URL 改动后：再次禁用开始同步（强制重新拉取）', async () => {
    const onChange = vi.fn();
    const { rerender } = render(<SyncForm url="https://x.com/kb" onUrlChange={onChange} />);
    // 先点 拉取清单 → 等可用
    fireEvent.click(screen.getByRole('button', { name: /拉取清单/ }));
    await vi.waitFor(() =>
      expect(
        (screen.getByRole('button', { name: /开始同步/ }) as HTMLButtonElement).disabled
      ).toBe(false)
    );

    // 用户改了 URL —— 必须重新拉取才能保证同步到的是新 URL 下的内容。
    // 这里通过 rerender 模拟受控切换；onChange 也会被 input 触发，
    // 但组件内部只比较 fetchedUrl 与当前 url，rendr 的关键在于 prop 变化。
    rerender(<SyncForm url="https://other.com/kb" onUrlChange={onChange} />);
    expect(
      (screen.getByRole('button', { name: /开始同步/ }) as HTMLButtonElement).disabled
    ).toBe(true);
  });

  it('downloading 阶段显示进度条', () => {
    useLibraryStore.setState({
      syncPhase: 'downloading',
      downloadProgress: { done: 3, total: 10 },
    });
    const { container } = render(
      <SyncForm url="https://x.com/kb" onUrlChange={() => {}} />
    );
    // 进度文案：3 / 10 + 30%
    expect(screen.getByText(/3 \/ 10/)).toBeTruthy();
    expect(screen.getByText(/30%/)).toBeTruthy();
    // 进度条 DOM：宽度样式为 "30%"
    const bar = container.querySelector('[style*="30%"]');
    expect(bar).toBeTruthy();
  });

  it('非 downloading 阶段：不显示进度条', () => {
    // syncPhase 是 'manifest' 但不是 'downloading' —— 这种情况显示"清单拉取中"
    // 但**不应**显示"正在同步 X / Y"——那条文案专属于 downloading 阶段。
    useLibraryStore.setState({
      syncPhase: 'manifest',
      downloadProgress: { done: 1, total: 100 },
    });
    render(<SyncForm url="https://x.com/kb" onUrlChange={() => {}} />);
    expect(screen.queryByText(/\/ 100/)).toBeNull();
  });

  it('拉取清单失败时把真实错误回传显示（不吞掉）', async () => {
    // 关键：用户报告"拉取清单失败"时，必须让他看到**为什么**失败——
    // 旧的版本一律返回"请检查地址是否可达..."，他根本不知道是 404 / 解析错 / 网络。
    // 这里让底层 sync.syncManifest 抛出，走 store.syncNow 的 catch 把真实错写入
    // store.error，组件再去读它 —— 这样测的就是真实路径而不是 fake syncNow。
    const { syncManifest } = await import('../../lib/sync');
    vi.mocked(syncManifest).mockRejectedValueOnce(
      new Error('拉取 http://x.com/kb/manifest.json 失败：HTTP 404')
    );

    render(<SyncForm url="http://x.com/kb" onUrlChange={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /拉取清单/ }));

    // 用户现在看到的就是后端原文，能直接判断是 404 / 解析错 / 网络不通。
    const err = await screen.findByText(/HTTP 404/);
    expect(err).toBeTruthy();
    expect(screen.getByText(/拉取清单失败：.*HTTP 404/)).toBeTruthy();
  });
});
