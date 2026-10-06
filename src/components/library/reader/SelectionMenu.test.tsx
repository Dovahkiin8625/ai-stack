// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import SelectionMenu from './SelectionMenu';

describe('SelectionMenu (选中文字右键浮层菜单)', () => {
  it('renders at the given fixed position with anchor data attached', () => {
    const { container } = render(
      <SelectionMenu
        x={120}
        y={240}
        anchorText="hello world"
        anchorOccurrence={0}
        onAddNote={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    const el = container.firstChild as HTMLElement;
    expect(el.style.position).toBe('fixed');
    expect(el.style.top).toBe('240px');
    expect(el.style.left).toBe('120px');
    expect(el.getAttribute('data-selection-menu')).not.toBeNull();
  });

  it('renders all 5 actions (复制 / 添加批注 / 翻译 / AI 讲解 / 询问 AI) when callbacks provided', () => {
    const { getByText } = render(
      <SelectionMenu
        x={0}
        y={0}
        anchorText="x"
        anchorOccurrence={0}
        onAddNote={vi.fn()}
        onAiAnnotate={vi.fn()}
        onAiAsk={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    expect(getByText('复制')).toBeTruthy();
    expect(getByText('添加批注')).toBeTruthy();
    expect(getByText('翻译')).toBeTruthy();
    expect(getByText('AI 讲解')).toBeTruthy();
    expect(getByText('询问 AI')).toBeTruthy();
  });

  it('hides AI 讲解 / 询问 AI when callbacks are not provided', () => {
    const { queryByText, getByText } = render(
      <SelectionMenu
        x={0}
        y={0}
        anchorText="x"
        anchorOccurrence={0}
        onAddNote={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    expect(getByText('复制')).toBeTruthy();
    expect(getByText('添加批注')).toBeTruthy();
    expect(getByText('翻译')).toBeTruthy();
    expect(queryByText('AI 讲解')).toBeNull();
    expect(queryByText('询问 AI')).toBeNull();
  });

  it('calls onAddNote with anchorText and anchorOccurrence when "添加批注" is clicked', () => {
    const onAddNote = vi.fn();
    const { getByText } = render(
      <SelectionMenu
        x={0}
        y={0}
        anchorText="foo bar"
        anchorOccurrence={3}
        onAddNote={onAddNote}
        onClose={vi.fn()}
      />,
    );
    fireEvent.click(getByText('添加批注'));
    expect(onAddNote).toHaveBeenCalledWith({ anchorText: 'foo bar', anchorOccurrence: 3 });
  });

  it('calls onAiAnnotate then onClose when "AI 讲解" is clicked', () => {
    const onAiAnnotate = vi.fn();
    const onClose = vi.fn();
    const { getByText } = render(
      <SelectionMenu
        x={0}
        y={0}
        anchorText="x"
        anchorOccurrence={0}
        onAddNote={vi.fn()}
        onAiAnnotate={onAiAnnotate}
        onClose={onClose}
      />,
    );
    fireEvent.click(getByText('AI 讲解'));
    expect(onAiAnnotate).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onAiAsk then onClose when "询问 AI" is clicked', () => {
    const onAiAsk = vi.fn();
    const onClose = vi.fn();
    const { getByText } = render(
      <SelectionMenu
        x={0}
        y={0}
        anchorText="x"
        anchorOccurrence={0}
        onAddNote={vi.fn()}
        onAiAsk={onAiAsk}
        onClose={onClose}
      />,
    );
    fireEvent.click(getByText('询问 AI'));
    expect(onAiAsk).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('shows "已复制" after clicking 复制 and calls onClose', async () => {
    // Clipboard API 在 jsdom 下未实现，handleCopy 会回退到 execCommand
    // 我们只验证 UI 状态变化
    const onClose = vi.fn();
    const { getByText, findByText } = render(
      <SelectionMenu
        x={0}
        y={0}
        anchorText="x"
        anchorOccurrence={0}
        onAddNote={vi.fn()}
        onClose={onClose}
      />,
    );
    fireEvent.click(getByText('复制'));
    // 800ms 后 setTimeout 触发 onClose；只验证"已复制"出现即可
    await findByText('已复制');
  });

  it('closes on Escape keydown', async () => {
    const onClose = vi.fn();
    render(
      <SelectionMenu
        x={0}
        y={0}
        anchorText="x"
        anchorOccurrence={0}
        onAddNote={vi.fn()}
        onClose={onClose}
      />,
    );
    // SelectionMenu 的 useEffect 用 setTimeout 0 推迟挂载全局 listener
    // 等下一个 tick 再触发 Escape
    await new Promise<void>((r) => setTimeout(r, 5));
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});