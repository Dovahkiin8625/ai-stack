// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, act, waitFor } from '@testing-library/react';
import { createRef } from 'react';

// jsdom 没有 ResizeObserver —— PdfReader 容器监听依赖它。给个最小桩：
// 进入即立刻给一次尺寸 800，让容器宽度立刻非 0，能进 render effect。
class StubResizeObserver {
  cb: ResizeObserverCallback;
  constructor(cb: ResizeObserverCallback) { this.cb = cb; }
  observe(_el: Element) {
    // 异步触发一次，给个固定宽度
    queueMicrotask(() => {
      this.cb(
        [{ contentRect: { width: 800, height: 600, top: 0, left: 0, right: 800, bottom: 600, x: 0, y: 0 } } as ResizeObserverEntry],
        this as unknown as ResizeObserver,
      );
    });
  }
  unobserve() {}
  disconnect() {}
}
(globalThis as any).ResizeObserver = StubResizeObserver;

// canvas.getContext('2d') 在 jsdom 里返回 null —— PdfReader render 路径需要
// 但我们的 StubTextLayer 不画 canvas，stub 一个最小 ctx。
HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  // pdf.js render() 实际不需要画布做任何事（Promise 直接 resolve）
})) as any;

/**
 * Stub pdfjs-dist，避免在 jsdom 下跑真实的 canvas / Worker。
 * 我们的桩只支持 PdfReader 笔记流程用到的 DOM 行为：
 * - getDocument 返回 PDFDocumentProxy
 * - PDFDocumentProxy.numPages / getPage / destroy
 * - PDFPageProxy.getViewport / getTextContent / render
 *
 * render effect 完成后，往 .textLayer 注入 span 列表，
 * 模拟 pdf.js TextLayer 把字符摆到坐标的 DOM 结构。
 */

// vi.mock 工厂会被 hoist 到文件顶部。状态用 vi.hoisted 暴露。
const pdfState = vi.hoisted(() => {
  const spans: Array<{ text: string; x: number; y: number }> = [];
  const instances: Array<{ rendered: Promise<void>; spans: Array<{ text: string }>; container: HTMLElement }> = [];
  return {
    getSpans: () => spans,
    setSpans: (s: typeof spans) => { spans.length = 0; spans.push(...s); },
    getInstances: () => instances,
    pageWidth: 600,
    pageHeight: 800,
  };
});

vi.mock('pdfjs-dist', () => {
  class StubTextLayer {
    container: HTMLElement;
    private resolveRender!: () => void;
    rendered: Promise<void>;
    spans: Array<{ text: string }>;
    constructor(init: { container: HTMLElement }) {
      this.container = init.container;
      this.rendered = new Promise((r) => (this.resolveRender = r));
      this.spans = [];
    }
    async render() {
      for (const s of pdfState.getSpans()) {
        const span = document.createElement('span');
        span.textContent = s.text;
        span.style.position = 'absolute';
        span.style.left = `${s.x}px`;
        span.style.top = `${s.y}px`;
        this.container.appendChild(span);
        this.spans.push({ text: s.text });
      }
      this.resolveRender();
    }
    cancel() {}
  }

  return {
    GlobalWorkerOptions: { workerSrc: '' },
    TextLayer: StubTextLayer,
    getDocument: () => ({
      promise: Promise.resolve({
        numPages: 2,
        async getPage(_pageNum: number) {
          return {
            async getViewport({ scale }: { scale: number }) {
              return { width: pdfState.pageWidth * scale, height: pdfState.pageHeight * scale };
            },
            async getTextContent() {
              return { items: pdfState.getSpans().map((s) => ({ str: s.text })) };
            },
            render() {
              return { promise: Promise.resolve() };
            },
          };
        },
        async destroy() {},
      }),
    }),
  };
});

// Stub library-api 的 readResourceBytes，返回空 buffer 让 PdfReader 走完 load effect
vi.mock('../../../lib/library-api', () => ({
  readResourceBytes: async () => new Uint8Array(),
}));

import PdfReader, { type PdfReaderHandle } from './PdfReader';

beforeEach(() => {
  pdfState.setSpans([{ text: 'hello world', x: 0, y: 0 }]);
  pdfState.pageWidth = 600;
  pdfState.pageHeight = 800;
});

describe('PdfReader (PDF 阅读器 — 笔记集成)', () => {
  it('renders the toolbar with current page indicator', async () => {
    const { container } = render(<PdfReader resourceId={1} pageCount={2} />);
    await waitFor(() => {
      expect(container.textContent).toContain('第 1 / 2 页');
    });
  });

  it('opens SelectionMenu on right-click when text is selected within the textLayer', async () => {
    const { container, getByText } = render(<PdfReader resourceId={1} pageCount={2} />);
    // 等第一个 textLayer 渲染完成：loading 解除，spans 出现
    await waitFor(() => {
      const tl = container.querySelector('.textLayer');
      expect(tl).not.toBeNull();
      expect(tl!.querySelector('span')).not.toBeNull();
    });
    const textLayer = container.querySelector('.textLayer')!;
    const span = textLayer.querySelector('span')!;
    // 模拟 selection：在 span 上设置选区
    const range = document.createRange();
    range.selectNodeContents(span);
    const sel = window.getSelection();
    sel?.removeAllRanges();
    sel?.addRange(range);
    fireEvent.contextMenu(textLayer, {
      clientX: 100,
      clientY: 200,
      bubbles: true,
    });
    await waitFor(() => {
      expect(getByText('添加批注')).toBeTruthy();
    });
  });

  it('does not open SelectionMenu on right-click when selection is empty', async () => {
    const { container, queryByText } = render(<PdfReader resourceId={1} pageCount={2} />);
    await waitFor(() => {
      expect(container.querySelector('.textLayer')).not.toBeNull();
    });
    fireEvent.contextMenu(container.querySelector('.textLayer')!, {
      clientX: 100,
      clientY: 200,
      bubbles: true,
    });
    expect(queryByText('添加批注')).toBeNull();
  });

  it('inserts <sup.note-marker> elements when notes prop is provided', async () => {
    const { container } = render(
      <PdfReader
        resourceId={1}
        pageCount={2}
        notes={[
          {
            id: 100,
            resourceId: 1,
            content: 'x',
            anchorText: 'hello world',
            anchorOccurrence: 0,
            source: 'user',
            createdAt: '2026-01-01T00:00:00Z',
            updatedAt: '2026-01-01T00:00:00Z',
            pageIdx: 0,
          },
        ]}
      />,
    );
    await waitFor(() => {
      const tl = container.querySelector('.textLayer')!;
      const marker = tl.querySelector('sup.note-marker');
      expect(marker).not.toBeNull();
      expect(marker!.getAttribute('data-note-id')).toBe('100');
    });
  });

  it('scrollToAnchor turns to the note page', async () => {
    pdfState.setSpans([{ text: 'page2 text', x: 0, y: 0 }]);
    const ref = createRef<PdfReaderHandle>();
    const { container } = render(
      <PdfReader
        ref={ref}
        resourceId={1}
        pageCount={2}
        notes={[
          {
            id: 200,
            resourceId: 1,
            content: 'x',
            anchorText: 'page2 text',
            anchorOccurrence: 0,
            source: 'user',
            createdAt: '2026-01-01T00:00:00Z',
            updatedAt: '2026-01-01T00:00:00Z',
            pageIdx: 1,
          },
        ]}
      />,
    );
    await waitFor(() => {
      expect(container.textContent).toContain('第 1 / 2 页');
    });
    act(() => {
      ref.current?.scrollToAnchor(200);
    });
    await waitFor(() => {
      expect(container.textContent).toContain('第 2 / 2 页');
    });
  });
});