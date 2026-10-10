// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import NotesPanel, { PANEL_CLASSES } from './NotesPanel';

// Mock 整个 notes store —— NotesPanel 通过 useNotesStore(selector) 读取所有状态，
// 真实 store 会 import library-api → invoke('plugin:...') 在 jsdom 下抛错。
vi.mock('../../stores/notes', () => ({
  useNotesStore: vi.fn(),
}));

import { useNotesStore } from '../../stores/notes';

const mockedUseNotesStore = vi.mocked(useNotesStore);

/** 伪造 store 状态：每个 selector 拿到的都是这一份快照。 */
const fakeState: any = {
  notes: [],
  loading: false,
  create: vi.fn(),
  update: vi.fn(),
  remove: vi.fn(),
  panelOpen: false,
  setPanelOpen: vi.fn(),
};

beforeEach(() => {
  mockedUseNotesStore.mockReset();
  mockedUseNotesStore.mockImplementation(((selector: any) => selector(fakeState)) as any);
  fakeState.panelOpen = false;
});

describe('NotesPanel 移动端定位', () => {
  it('PANEL_CLASSES — 窄屏是底部抽屉', () => {
    expect(PANEL_CLASSES.mobile).toContain('max-md:bottom-0');
    expect(PANEL_CLASSES.mobile).toContain('max-md:h-[70vh]');
  });

  it('PANEL_CLASSES — 关闭态把面板移出视口（Review Focus #5）', () => {
    expect(PANEL_CLASSES.closed).toContain('max-md:translate-y-full');
    expect(PANEL_CLASSES.closed).toContain('pointer-events-none');
  });

  it('PANEL_CLASSES — 桌面仍是右侧固定栏', () => {
    expect(PANEL_CLASSES.desktop).toContain('md:w-[320px]');
    expect(PANEL_CLASSES.desktop).toContain('md:right-0');
  });

  it('关闭态渲染：className 包含 off-viewport 类，不含 open', () => {
    fakeState.panelOpen = false;
    const { container } = render(<NotesPanel />);
    const aside = container.querySelector('aside');
    expect(aside).not.toBeNull();
    const cls = aside!.className;
    // 移出视口（窄屏专用类）必须存在
    expect(cls).toContain('max-md:translate-y-full');
    // 桌面关闭态也要保留：右侧滑出
    expect(cls).toContain('md:translate-x-full');
    // open 类的字符串不能出现 —— 一旦恒用 open class，这个测试立刻挂
    expect(cls).not.toContain(PANEL_CLASSES.open);
  });

  it('开启态渲染：className 包含 open，不含 off-viewport', () => {
    fakeState.panelOpen = true;
    const { container } = render(<NotesPanel />);
    const aside = container.querySelector('aside');
    expect(aside).not.toBeNull();
    const cls = aside!.className;
    expect(cls).toContain(PANEL_CLASSES.open);
    expect(cls).not.toContain('max-md:translate-y-full');
    expect(cls).not.toContain('md:translate-x-full');
  });
});
