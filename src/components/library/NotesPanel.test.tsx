import { describe, it, expect } from 'vitest';
import { PANEL_CLASSES } from './NotesPanel';

describe('NotesPanel 移动端定位', () => {
  it('窄屏是底部抽屉', () => {
    expect(PANEL_CLASSES.mobile).toContain('max-md:bottom-0');
    expect(PANEL_CLASSES.mobile).toContain('max-md:h-[70vh]');
  });
  it('关闭态把面板移出视口，不只是变透明（Review Focus #5）', () => {
    expect(PANEL_CLASSES.closed).toContain('max-md:translate-y-full');
    expect(PANEL_CLASSES.closed).toContain('pointer-events-none');
  });
  it('桌面仍是右侧固定栏', () => {
    expect(PANEL_CLASSES.desktop).toContain('md:w-[320px]');
    expect(PANEL_CLASSES.desktop).toContain('md:right-0');
  });
});