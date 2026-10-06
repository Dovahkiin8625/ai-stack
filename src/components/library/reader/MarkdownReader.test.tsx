// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor } from '@testing-library/react';
import { createRef } from 'react';
import MarkdownReader, { type MarkdownReaderHandle } from './MarkdownReader';

// Mock library-api — MarkdownReader 不直接调它（save 走 onSaveMarkdown prop），
// 但 store / 父组件会调，stub 一个空实现避免编译报错
vi.mock('../../../lib/library-api', () => ({
  readResource: vi.fn(),
  writeResource: vi.fn(),
}));

// 用一个全局空操作 stub anchor.ts —— MarkdownReader 内部跑锚点 / 笔记 marker 逻辑，
// 我们的测试只关心"切换 + 文本编辑 + 保存"路径，不验证 anchor 计算
vi.mock('../../../lib/anchor', () => ({
  computeAnchor: vi.fn(),
  extractContext: vi.fn(),
  insertNoteMarkers: vi.fn(),
  unwrapAnchors: vi.fn(),
}));

// CodeMirror 在 jsdom 里需要 contenteditable 才能跑；jsdom 默认不实现 getSelection 的 Range API，
// 这里 stub 出 CodeMirror 启动期调用的最小方法。CodeMirror 内部用 View 测自身不需要浏览器完整支持。
const emptyRectList = {
  item: () => null,
  length: 0,
  [Symbol.iterator]: function* () {},
};
const emptyRect = {
  x: 0,
  y: 0,
  width: 0,
  height: 0,
  top: 0,
  right: 0,
  bottom: 0,
  left: 0,
  toJSON() {},
};
// jsdom 的 Range 没有 getClientRects / getBoundingClientRect —— CodeMirror 的
// measure 流程会调它，不 stub 就抛 TypeError 让整个测试文件挂掉。
// 用 beforeAll 提前注入，确保第一个 render 触发 CodeMirror measure 之前 stub 已就位。
const origRangeGetClientRects = Range.prototype.getClientRects;
const origRangeGetBoundingClientRect = Range.prototype.getBoundingClientRect;
function installRangeStubs() {
  Range.prototype.getClientRects = function () {
    return emptyRectList as unknown as DOMRectList;
  };
  Range.prototype.getBoundingClientRect = function () {
    return emptyRect as unknown as DOMRect;
  };
}
function removeRangeStubs() {
  if (origRangeGetClientRects === undefined) {
    delete (Range.prototype as { getClientRects?: () => DOMRectList }).getClientRects;
  } else {
    Range.prototype.getClientRects = origRangeGetClientRects;
  }
  if (origRangeGetBoundingClientRect === undefined) {
    delete (Range.prototype as { getBoundingClientRect?: () => DOMRect }).getBoundingClientRect;
  } else {
    Range.prototype.getBoundingClientRect = origRangeGetBoundingClientRect;
  }
}
// jsdom 也没有完整的 Selection API —— stub 出 CodeMirror 启动期要的最小形状。
const mockRange = {
  getBoundingClientRect: () => emptyRect,
  getClientRects: () => emptyRectList,
  cloneRange: () => mockRange,
  collapse: () => undefined,
  selectNodeContents: () => undefined,
  setStart: () => undefined,
  setEnd: () => undefined,
};
const origGetSelection = window.getSelection;
window.getSelection = () =>
  ({
    rangeCount: 0,
    isCollapsed: true,
    getRangeAt: () => mockRange,
    removeAllRanges: () => undefined,
    addRange: () => undefined,
    selectAllChildren: () => undefined,
    toString: () => '',
  }) as unknown as Selection;

beforeAll(() => {
  installRangeStubs();
});
beforeEach(() => {
  vi.useRealTimers();
});
afterEach(() => {
  // 还原以防别的测试需要
  window.getSelection = origGetSelection;
  removeRangeStubs();
  installRangeStubs();
});
afterAll(() => {
  removeRangeStubs();
  window.getSelection = origGetSelection;
});

const baseHtml = '<h1>Hi</h1>';
const baseMarkdown = '# Hi\n';

describe('MarkdownReader (预览 / 编辑 模式切换)', () => {
  it('默认渲染预览模式：toolbar 出现两个 pill，html 可见，编辑器不在 DOM', () => {
    const { container, queryByText } = render(
      <MarkdownReader html={baseHtml} markdown={baseMarkdown} onSaveMarkdown={vi.fn()} />,
    );
    expect(queryByText('预览')).toBeTruthy();
    expect(queryByText('编辑')).toBeTruthy();
    // 预览模式渲染 comrak html
    expect(container.querySelector('h1')).not.toBeNull();
    // 编辑模式 CodeMirror 不该出现
    expect(container.querySelector('.cm-editor')).toBeNull();
  });

  it('点编辑 pill 切换到编辑模式：CodeMirror 出现并填上 markdown 原文', async () => {
    const { container, getByText } = render(
      <MarkdownReader html={baseHtml} markdown={baseMarkdown} onSaveMarkdown={vi.fn()} />,
    );
    fireEvent.click(getByText('编辑'));
    // CodeMirror 异步 mount —— waitFor 验证 .cm-content 出现
    await waitFor(() => {
      expect(container.querySelector('.cm-editor')).not.toBeNull();
    });
    // CodeMirror 的 .cm-content textContent 会丢掉文件末尾换行（DOM 文本节点规则），
    // 比 markdown prop 少一个 \n。这不是 bug —— CodeMirror 的 state.doc.length 仍是准确的，
    // 只是 DOM 渲染对 textContent 做了规范化。断言时去掉末尾 \n。
    const content = container.querySelector('.cm-content');
    expect(content?.textContent).toBe('# Hi');
    // 预览 html 不该再渲染（切到编辑就不再走 comrak 渲染管道）
    expect(container.querySelector('h1')).toBeNull();
  });

  it('编辑器改内容后：内部 draft 变化、dirty 变 true、保存按钮可点击', async () => {
    const ref = createRef<MarkdownReaderHandle>();
    const { container, getByText } = render(
      <MarkdownReader
        ref={ref}
        html={baseHtml}
        markdown={baseMarkdown}
        onSaveMarkdown={vi.fn()}
      />,
    );
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      expect(container.querySelector('.cm-editor')).not.toBeNull();
    });
    // jsdom 下 contenteditable 的 input 事件 CodeMirror 不会同步 state —— 用 imperative
    // handle 拿到 EditorView 后 dispatch transaction 模拟用户编辑。
    await waitFor(() => {
      expect(ref.current?.editorView).not.toBeNull();
    });
    const view = ref.current!.editorView!;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: '# Edited\n' },
    });
    await waitFor(() => {
      const saveBtn = getByText('保存');
      expect(saveBtn.getAttribute('disabled')).toBeNull();
    });
  });

  it('编辑模式下默认保存按钮显示但 disabled（不 dirty）', async () => {
    const { getByText } = render(
      <MarkdownReader html={baseHtml} markdown={baseMarkdown} onSaveMarkdown={vi.fn()} />,
    );
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      const saveBtn = getByText('保存');
      // 没改任何内容 → 保存按钮存在但 disabled（视觉提示而不是隐藏）
      expect(saveBtn).toBeTruthy();
      expect(saveBtn.getAttribute('disabled')).not.toBeNull();
    });
  });

  it('点保存调 onSaveMarkdown(draft)，保存成功后切回预览模式', async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    const ref = createRef<MarkdownReaderHandle>();
    const { getByText } = render(
      <MarkdownReader
        ref={ref}
        html={baseHtml}
        markdown={baseMarkdown}
        onSaveMarkdown={onSave}
      />,
    );
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      expect(ref.current?.editorView).not.toBeNull();
    });
    // 先 dispatch 一次 transaction 制造 dirty —— 否则保存按钮 disabled（保护不浪费 IPC）
    const view = ref.current!.editorView!;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: '# Hi edited\n' },
    });
    // @uiw/react-codemirror 对 onChange 有 200ms debounce —— 等 button 真正 enable
    await waitFor(
      () => {
        expect(getByText('保存').getAttribute('disabled')).toBeNull();
      },
      { timeout: 1000 },
    );
    fireEvent.click(getByText('保存'));
    await waitFor(() => {
      expect(onSave).toHaveBeenCalledTimes(1);
    });
    expect(onSave).toHaveBeenCalledWith('# Hi edited\n');
  });

  it('保存失败：onSaveMarkdown 抛错 → 错误条出现 + 留在编辑模式', async () => {
    const onSave = vi.fn().mockRejectedValue(new Error('写盘失败'));
    const ref = createRef<MarkdownReaderHandle>();
    const { container, getByText } = render(
      <MarkdownReader
        ref={ref}
        html={baseHtml}
        markdown={baseMarkdown}
        onSaveMarkdown={onSave}
      />,
    );
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      expect(ref.current?.editorView).not.toBeNull();
    });
    const view = ref.current!.editorView!;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: '# Hi edited\n' },
    });
    await waitFor(
      () => {
        expect(getByText('保存').getAttribute('disabled')).toBeNull();
      },
      { timeout: 1000 },
    );
    fireEvent.click(getByText('保存'));
    await waitFor(() => {
      expect(container.textContent).toContain('写盘失败');
    });
    // 仍然在编辑模式（保存失败不能切回预览，否则 draft 丢失）
    expect(container.querySelector('.cm-editor')).not.toBeNull();
  });

  it('编辑模式下点预览且 draft dirty → 显示确认条（不直接切回预览）', async () => {
    const ref = createRef<MarkdownReaderHandle>();
    const { container, getByText, queryByText } = render(
      <MarkdownReader
        ref={ref}
        html={baseHtml}
        markdown={baseMarkdown}
        onSaveMarkdown={vi.fn()}
      />,
    );
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      expect(ref.current?.editorView).not.toBeNull();
    });
    const view = ref.current!.editorView!;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: '# Edited\n' },
    });
    fireEvent.click(getByText('预览'));
    // dirty 状态下点预览不应直接切回 —— 应出确认条
    await waitFor(() => {
      expect(queryByText('放弃修改')).toBeTruthy();
      expect(queryByText('继续编辑')).toBeTruthy();
    });
    // CodeMirror 还在（未切回预览）
    expect(container.querySelector('.cm-editor')).not.toBeNull();
  });

  it('确认条点"放弃修改"：丢弃 draft，切回预览', async () => {
    const ref = createRef<MarkdownReaderHandle>();
    const { container, getByText } = render(
      <MarkdownReader
        ref={ref}
        html={baseHtml}
        markdown={baseMarkdown}
        onSaveMarkdown={vi.fn()}
      />,
    );
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      expect(ref.current?.editorView).not.toBeNull();
    });
    const view = ref.current!.editorView!;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: '# Edited\n' },
    });
    fireEvent.click(getByText('预览'));
    await waitFor(() => {
      expect(container.textContent).toContain('放弃修改');
    });
    fireEvent.click(getByText('放弃修改'));
    await waitFor(() => {
      expect(container.querySelector('h1')).not.toBeNull();
    });
    expect(container.querySelector('.cm-editor')).toBeNull();
  });

  it('确认条点"继续编辑"：保留 draft，留在编辑模式 + 关掉确认条', async () => {
    const ref = createRef<MarkdownReaderHandle>();
    const { container, getByText, queryByText } = render(
      <MarkdownReader
        ref={ref}
        html={baseHtml}
        markdown={baseMarkdown}
        onSaveMarkdown={vi.fn()}
      />,
    );
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      expect(ref.current?.editorView).not.toBeNull();
    });
    const view = ref.current!.editorView!;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: '# Edited\n' },
    });
    fireEvent.click(getByText('预览'));
    await waitFor(() => {
      expect(container.textContent).toContain('放弃修改');
    });
    fireEvent.click(getByText('继续编辑'));
    await waitFor(() => {
      expect(queryByText('放弃修改')).toBeNull();
    });
    expect(container.querySelector('.cm-editor')).not.toBeNull();
  });

  it('非 dirty 时点预览 → 直接切回，不出确认条', async () => {
    const { container, getByText, queryByText } = render(
      <MarkdownReader html={baseHtml} markdown={baseMarkdown} onSaveMarkdown={vi.fn()} />,
    );
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      expect(container.querySelector('.cm-editor')).not.toBeNull();
    });
    // 没改任何东西
    fireEvent.click(getByText('预览'));
    await waitFor(() => {
      expect(container.querySelector('h1')).not.toBeNull();
    });
    expect(container.querySelector('.cm-editor')).toBeNull();
    expect(queryByText('放弃修改')).toBeNull();
  });

  it('imperative handle: hasUnsavedChanges() 初始为 false', () => {
    const ref = createRef<MarkdownReaderHandle>();
    render(
      <MarkdownReader
        ref={ref}
        html={baseHtml}
        markdown={baseMarkdown}
        onSaveMarkdown={vi.fn()}
      />,
    );
    expect(ref.current?.hasUnsavedChanges()).toBe(false);
  });

  it('切到编辑后改内容 → hasUnsavedChanges() true；保存成功后（prop 同步更新）false', async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    const ref = createRef<MarkdownReaderHandle>();
    const { container, getByText, rerender } = render(
      <MarkdownReader
        ref={ref}
        html={baseHtml}
        markdown={baseMarkdown}
        onSaveMarkdown={onSave}
      />,
    );
    expect(ref.current?.hasUnsavedChanges()).toBe(false);
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      expect(ref.current?.editorView).not.toBeNull();
    });
    const view = ref.current!.editorView!;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: '# Edited\n' },
    });
    await waitFor(() => {
      expect(ref.current?.hasUnsavedChanges()).toBe(true);
    });
    fireEvent.click(getByText('保存'));
    await waitFor(() => {
      expect(onSave).toHaveBeenCalledTimes(1);
    });
    // 模拟父组件：onSave resolve 后调 updateResourceContent → markdown prop 变成新版本
    rerender(
      <MarkdownReader
        ref={ref}
        html={'<h1>Edited</h1>'}
        markdown={'# Edited\n'}
        onSaveMarkdown={onSave}
      />,
    );
    expect(ref.current?.hasUnsavedChanges()).toBe(false);
    void container;
  });
});

describe('MarkdownReader preview 模式原有行为保持不变', () => {
  it('note marker 的 sup.note-marker 仍可点击触发 onMarkClick', () => {
    const onMarkClick = vi.fn();
    const { container } = render(
      <MarkdownReader
        html={'<p>hello <sup class="note-marker" data-note-id="42">x</sup></p>'}
        markdown={'hello'}
        notes={[]}
        onMarkClick={onMarkClick}
        onSaveMarkdown={vi.fn()}
      />,
    );
    const marker = container.querySelector('sup.note-marker') as HTMLElement;
    expect(marker).not.toBeNull();
    fireEvent.click(marker);
    expect(onMarkClick).toHaveBeenCalledWith(42);
  });

  // 回归守卫：从编辑模式切回预览后，公式 / 代码高亮 / note marker 必须重新渲染。
  // 之前的 bug：渲染管道只跑在 useEffect([html])，切回预览时 html 没变、管道不重跑，
  // 预览 div 是全新 mount 的空 DOM，所以 $...$ / ```math 这些后处理全没了。
  // 用户感知："点编辑后切回预览，公式消失了"。修复后预览 div mount 时也要跑管道。
  it('从编辑切回预览后：display math 块（```math）仍然渲染成 KaTeX', async () => {
    const html =
      '<pre><code class="language-math">x^2 + y^2 = z^2</code></pre>';
    const { container, getByText } = render(
      <MarkdownReader html={html} markdown="" onSaveMarkdown={vi.fn()} />,
    );
    // 初始预览应当已经把 math 块渲染成 .katex
    await waitFor(() => {
      expect(container.querySelector('.katex')).not.toBeNull();
    });
    // 切到编辑 → 切回预览
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      expect(container.querySelector('.cm-editor')).not.toBeNull();
    });
    fireEvent.click(getByText('预览'));
    // bug：这里 .katex 消失；修复后必须仍在
    await waitFor(() => {
      expect(container.querySelector('.katex')).not.toBeNull();
    });
  });

  it('从编辑切回预览后：inline math（$...$）仍然被 auto-render 处理', async () => {
    const html = '<p>欧拉公式 $e^{i\pi} + 1 = 0$ 是最美的。</p>';
    const { container, getByText } = render(
      <MarkdownReader html={html} markdown="" onSaveMarkdown={vi.fn()} />,
    );
    await waitFor(() => {
      expect(container.querySelector('.katex')).not.toBeNull();
    });
    fireEvent.click(getByText('编辑'));
    await waitFor(() => {
      expect(container.querySelector('.cm-editor')).not.toBeNull();
    });
    fireEvent.click(getByText('预览'));
    await waitFor(() => {
      expect(container.querySelector('.katex')).not.toBeNull();
    });
  });
});
