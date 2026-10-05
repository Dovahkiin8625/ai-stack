// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import AskInputBox from './AskInputBox';

describe('AskInputBox (选中询问浮层输入框)', () => {
  it('渲染位置由 position prop 决定 (fixed 定位)', () => {
    const { container } = render(
      <AskInputBox
        position={{ x: 120, y: 240 }}
        onSubmit={vi.fn()}
        onCancel={vi.fn()}
      />,
    );
    const el = container.firstChild as HTMLElement;
    expect(el.style.position).toBe('fixed');
    expect(el.style.top).toBe('240px');
    expect(el.style.left).toBe('120px');
  });

  it('placeholder 默认是 "向 AI 提问…"', () => {
    const { getByRole } = render(
      <AskInputBox
        position={{ x: 0, y: 0 }}
        onSubmit={vi.fn()}
        onCancel={vi.fn()}
      />,
    );
    const input = getByRole('textbox') as HTMLInputElement;
    expect(input.placeholder).toBe('向 AI 提问…');
  });

  it('Enter 提交时调用 onSubmit(去除首尾空白的问题)', () => {
    const onSubmit = vi.fn();
    const { getByRole } = render(
      <AskInputBox
        position={{ x: 0, y: 0 }}
        onSubmit={onSubmit}
        onCancel={vi.fn()}
      />,
    );
    const input = getByRole('textbox') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '  为什么用 sigmoid?  ' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith('为什么用 sigmoid?');
  });

  it('空问题 / 纯空白 不允许提交（onSubmit 不被调用）', () => {
    const onSubmit = vi.fn();
    const { getByRole } = render(
      <AskInputBox
        position={{ x: 0, y: 0 }}
        onSubmit={onSubmit}
        onCancel={vi.fn()}
      />,
    );
    const input = getByRole('textbox') as HTMLInputElement;

    // 完全空
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSubmit).not.toHaveBeenCalled();

    // 只有空白
    fireEvent.change(input, { target: { value: '   ' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('发送按钮在问题为空时 disabled', () => {
    const { getByRole } = render(
      <AskInputBox
        position={{ x: 0, y: 0 }}
        onSubmit={vi.fn()}
        onCancel={vi.fn()}
      />,
    );
    const sendBtn = getByRole('button', { name: /发送/ }) as HTMLButtonElement;
    expect(sendBtn.disabled).toBe(true);

    const input = getByRole('textbox') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '为什么用 sigmoid?' } });
    expect(sendBtn.disabled).toBe(false);
  });

  it('Esc 取消：调用 onCancel', () => {
    const onCancel = vi.fn();
    const { getByRole } = render(
      <AskInputBox
        position={{ x: 0, y: 0 }}
        onSubmit={vi.fn()}
        onCancel={onCancel}
      />,
    );
    const input = getByRole('textbox') as HTMLInputElement;
    fireEvent.keyDown(input, { key: 'Escape' });
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('点击取消按钮也调用 onCancel', () => {
    const onCancel = vi.fn();
    const { getByRole } = render(
      <AskInputBox
        position={{ x: 0, y: 0 }}
        onSubmit={vi.fn()}
        onCancel={onCancel}
      />,
    );
    fireEvent.click(getByRole('button', { name: /取消/ }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('点击发送按钮也走 onSubmit（与 Enter 等价）', () => {
    const onSubmit = vi.fn();
    const { getByRole } = render(
      <AskInputBox
        position={{ x: 0, y: 0 }}
        onSubmit={onSubmit}
        onCancel={vi.fn()}
      />,
    );
    const input = getByRole('textbox') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '为什么用 sigmoid?' } });
    fireEvent.click(getByRole('button', { name: /发送/ }));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith('为什么用 sigmoid?');
  });

  it('输入框挂载时自动获得焦点（用户点完菜单后无需再点击）', () => {
    const { getByRole } = render(
      <AskInputBox
        position={{ x: 0, y: 0 }}
        onSubmit={vi.fn()}
        onCancel={vi.fn()}
      />,
    );
    const input = getByRole('textbox') as HTMLInputElement;
    expect(document.activeElement).toBe(input);
  });
});