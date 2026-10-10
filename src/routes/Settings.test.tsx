// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

// 路由层必须包一层 MemoryRouter，否则 useNavigate 抛"无法在 Router 外使用"。
const mockedNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockedNavigate,
  };
});

vi.mock('../components/settings/ApiKeyForm', () => ({
  default: () => <div data-testid="api-form-stub" />,
}));
vi.mock('../components/settings/SyncForm', () => ({
  default: () => <div data-testid="sync-form-stub" />,
}));

vi.mock('../lib/tauri', () => ({
  getSettings: vi.fn(async () => ({
    apiKey: 'sk-existing',
    baseUrl: 'https://api.anthropic.com',
    lightweightModel: 'claude-haiku-4-5-20251001',
    performanceModel: 'claude-sonnet-4-5',
    syncBaseUrl: 'https://x.com/kb',
  })),
  saveSettings: vi.fn(async () => undefined),
  DEFAULTS: {
    apiKey: '',
    baseUrl: 'https://api.anthropic.com',
    lightweightModel: 'claude-haiku-4-5-20251001',
    performanceModel: 'claude-sonnet-4-5',
    syncBaseUrl: '',
  },
  testModel: vi.fn(),
}));
vi.mock('../lib/sync', () => ({
  setSyncBaseUrl: vi.fn(async () => undefined),
}));

import Settings from './Settings';
import * as tauri from '../lib/tauri';
import * as syncLib from '../lib/sync';

function renderSettings() {
  return render(
    <MemoryRouter initialEntries={['/settings']}>
      <Settings />
    </MemoryRouter>
  );
}

describe('Settings 页面', () => {
  beforeEach(() => {
    mockedNavigate.mockClear();
    vi.mocked(tauri.saveSettings).mockClear();
    vi.mocked(syncLib.setSyncBaseUrl).mockClear();
  });

  it('mount 后只渲染一个「保存」和一个「保存并关闭」按钮（form 内没有保存按钮）', async () => {
    renderSettings();
    // 等 getSettings() resolve + 切到 ready
    await screen.findByRole('button', { name: /^保存$/ });
    // 「保存并关闭」必须紧贴同一个 footer，不能藏在 form 里
    await screen.findByRole('button', { name: /保存并关闭/ });

    // 检查每个子组件都没有"保存并同步"或"保存"按钮 —— 否则就成了底部+表单双套保存
    expect(screen.queryByRole('button', { name: /保存并同步/ })).toBeNull();
  });

  it('点击底部「保存」会调用 saveSettings + setSyncBaseUrl，但不导航', async () => {
    renderSettings();
    await screen.findByRole('button', { name: /^保存$/ });
    fireEvent.click(screen.getByRole('button', { name: /^保存$/ }));

    await vi.waitFor(() => {
      expect(tauri.saveSettings).toHaveBeenCalled();
      expect(syncLib.setSyncBaseUrl).toHaveBeenCalledWith('https://x.com/kb');
    });
    // 单纯保存不跳转 —— 用户可能还要继续编辑或点击"测试"
    expect(mockedNavigate).not.toHaveBeenCalled();
  });

  it('点击「保存并关闭」后 navigate(-1) 回到上一页', async () => {
    renderSettings();
    await screen.findByRole('button', { name: /保存并关闭/ });
    fireEvent.click(screen.getByRole('button', { name: /保存并关闭/ }));

    await vi.waitFor(() => expect(mockedNavigate).toHaveBeenCalledWith(-1));
    // 关闭前也要持久化 —— 否则用户以为保存了
    expect(tauri.saveSettings).toHaveBeenCalled();
    expect(syncLib.setSyncBaseUrl).toHaveBeenCalledWith('https://x.com/kb');
  });
});
