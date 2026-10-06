import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import * as pdfjsLib from 'pdfjs-dist';
import type { PDFDocumentProxy, PDFPageProxy } from 'pdfjs-dist';
import type { TextLayer } from 'pdfjs-dist/types/src/display/text_layer';
// Vite 把 worker 当资源处理，打包时复制到 dist 并返回最终 URL
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import * as api from '../../../lib/library-api';
import type { Note } from '../../../types';
import {
  computeAnchor,
  extractContext,
  insertNoteMarkers,
  unwrapAnchors,
} from '../../../lib/anchor';
import AskInputBox from './AskInputBox';
import SelectionMenu from './SelectionMenu';

export interface PdfReaderHandle {
  scrollToAnchor: (noteId: number) => void;
}

interface Props {
  /** 资源 ID：用于 readResourceBytes 加载 PDF。 */
  resourceId: number;
  /** 总页数，由 read_resource 提前返回（让 UI 立即显示页码）。 */
  pageCount: number;
  /** 资源下的笔记列表；用于插入角标。 */
  notes?: Note[];
  /** 用户点击 <sup.note-marker> 时通知父组件。 */
  onMarkClick?: (noteId: number) => void;
  /** "添加批注"：reader 算好 anchor 后通知父组件。父组件去 createNote + 打开抽屉。 */
  onAddNoteAtSelection?: (input: {
    anchorText: string;
    anchorOccurrence: number;
    pageIdx: number;
  }) => void;
  /** "AI 讲解"：reader 算好上下文 + pageIdx，丢给父组件调 LLM。 */
  onAiAnnotate?: (input: {
    selectedText: string;
    contextBefore: string;
    contextAfter: string;
    sectionTitle: string;
    anchorText: string;
    anchorOccurrence: number;
    pageIdx: number;
  }) => Promise<void>;
  /** "询问 AI"：reader 算好上下文 + pageIdx，开 AskInputBox 让用户输入问题。 */
  onAiAsk?: (input: {
    selectedText: string;
    contextBefore: string;
    contextAfter: string;
    sectionTitle: string;
    anchorText: string;
    anchorOccurrence: number;
    pageIdx: number;
    question: string;
  }) => Promise<void>;
}

const PREFETCH_RADIUS = 2;
const FALLBACK_DPR = 1;
const MIN_TARGET_WIDTH = 480;
const MAX_TARGET_WIDTH = 2400;

/** 当单页可见文本 < 这个阈值时，整页直接作为 contextBefore + contextAfter，
 *  避免给 LLM 喂半页造成上下文缺失。 */
const WHOLE_PAGE_CONTEXT_THRESHOLD = 300;

// 在模块加载时设置一次 worker；同一个 webview 生命周期内不会变
pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

/**
 * pdf.js v4+ 文字层 CSS：浏览器选中要靠这些规则把 span 摆到 canvas 对应坐标。
 *
 * 关键点：容器和 span 的尺寸/位置都引用 `--scale-factor` CSS 变量，
 * 而该变量要在 new TextLayer 之前通过 JS 设到容器 style 上 —— 不设的话
 * 浏览器选中时拿到的 span 和视觉位置完全错位（"选择 hello 复制出来是 world"）。
 */
const TEXT_LAYER_CSS = `
.textLayer {
  position: absolute;
  text-align: initial;
  inset: 0;
  overflow: clip;
  opacity: 1;
  line-height: 1;
  -webkit-text-size-adjust: none;
  -moz-text-size-adjust: none;
  text-size-adjust: none;
  forced-color-adjust: none;
  transform-origin: 0 0;
  caret-color: CanvasText;
  z-index: 2;
  user-select: text;
  -webkit-user-select: text;
}
.textLayer :is(span, br) {
  color: transparent;
  position: absolute;
  white-space: pre;
  cursor: text;
  transform-origin: 0% 0%;
}
.textLayer > :not(.markedContent),
.textLayer .markedContent span:not(.markedContent) {
  z-index: 1;
}
.textLayer span.markedContent {
  top: 0;
  height: 0;
}
.textLayer span[role="img"] {
  -webkit-user-select: none;
  -moz-user-select: none;
  user-select: none;
  cursor: default;
}
.textLayer ::selection {
  background: rgba(0 0 255 / 0.25);
  background: color-mix(in srgb, AccentColor, transparent 75%);
}
.textLayer br::selection {
  background: transparent;
}
.textLayer .endOfContent {
  display: block;
  position: absolute;
  inset: 100% 0 0;
  z-index: -1;
  cursor: default;
  -webkit-user-select: none;
  -moz-user-select: none;
  user-select: none;
}
`;

/**
 * PDF 阅读器 —— 用 pdf.js 直接渲染，自带 text layer 支持选中/复制。
 *
 * 设计要点：
 * 1. 启动时一次性 `getDocument({data})` 把整个 PDF load 到内存（pdf.js 内部流式解析）
 * 2. 当前页 = canvas（视觉）+ 透明 textLayerDiv（按坐标定位的字符 spans），浏览器原生选中
 * 3. 视口跟随容器宽度：`viewport = page.getViewport({scale: containerWidth / nativeWidth})`
 * 4. 预取相邻 ±2 页：拿 PDFPageProxy 缓存起来；canvas/textLayer 不预渲染（耗内存）
 * 5. 翻页/卸载调 `page.cleanup()` + `textLayer.cancel()` 释放 pdf.js 内部资源
 *
 * 笔记功能（Phase 6）：
 * 6. 容器 onContextMenu：在 textLayer 内右键选中 → 算 anchor → 开 SelectionMenu
 * 7. 每次重画当前页 → unwrapAnchors + insertNoteMarkers 注回当前页的 textLayer
 * 8. forwardRef.scrollToAnchor(noteId)：切到 note.pageIdx，然后 scrollIntoView + flash
 */
export default forwardRef<PdfReaderHandle, Props>(function PdfReader(
  { resourceId, pageCount, notes = [], onMarkClick, onAddNoteAtSelection, onAiAnnotate, onAiAsk },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const textLayerElRef = useRef<HTMLDivElement | null>(null);
  const [idx, setIdx] = useState(0);
  const [containerWidth, setContainerWidth] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  // 真实页数（pdf.js 解析权威值）。初始用 prop 的 hint，加载完用 pdf.numPages 覆盖
  const [actualPageCount, setActualPageCount] = useState<number | null>(pageCount);
  // 渲染时使用的页数：优先 pdf.js 权威值，回退到后端 hint
  const effectivePageCount = actualPageCount ?? pageCount;
  // 取消过期请求
  const seqRef = useRef(0);
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const textLayerRef = useRef<TextLayer | null>(null);
  const currentPageRef = useRef<PDFPageProxy | null>(null);
  const cssInjectedRef = useRef(false);

  // 浮层状态：右键唤起的 SelectionMenu；以及"询问 AI"浮层输入框
  const [menu, setMenu] = useState<{
    x: number;
    y: number;
    anchorText: string;
    anchorOccurrence: number;
  } | null>(null);
  const [askBox, setAskBox] = useState<{
    x: number;
    y: number;
    selectedText: string;
    contextBefore: string;
    contextAfter: string;
    sectionTitle: string;
    anchorText: string;
    anchorOccurrence: number;
  } | null>(null);

  // 注入 text layer CSS 到 document head
  useEffect(() => {
    if (cssInjectedRef.current) return;
    cssInjectedRef.current = true;
    const style = document.createElement('style');
    style.textContent = TEXT_LAYER_CSS;
    document.head.appendChild(style);
  }, []);

  // 切资源时彻底重置
  useEffect(() => {
    setIdx(0);
    setContainerWidth(0);
    seqRef.current += 1;
    setLoading(true);
    setLoadError(null);
    // 清理当前页的 textLayer
    if (textLayerRef.current) {
      textLayerRef.current.cancel();
      textLayerRef.current = null;
    }
    currentPageRef.current = null;
    textLayerElRef.current = null;
    // 关掉旧 PDF（destroy 会顺带释放所有内部缓存的 page proxy）
    if (pdfRef.current) {
      void pdfRef.current.destroy();
      pdfRef.current = null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resourceId]);

  // 监听容器尺寸
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const w = Math.round(entry.contentRect.width);
      setContainerWidth((prev) => (prev === w ? prev : w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // 加载 PDF 文档
  useEffect(() => {
    let cancelled = false;
    const seq = ++seqRef.current;
    setLoading(true);
    setLoadError(null);
    (async () => {
      try {
        const raw = await api.readResourceBytes(resourceId);
        if (cancelled || seq !== seqRef.current) return;
        const data = normalizeBytes(raw);
        if (cancelled || seq !== seqRef.current) return;
        const task = pdfjsLib.getDocument({ data });
        const pdf = await task.promise;
        if (cancelled || seq !== seqRef.current) {
          void pdf.destroy();
          return;
        }
        pdfRef.current = pdf;
        // 用 pdf.js 自己的 numPages 覆盖后端给的 hint —— 后端用 pdf-extract 数页，
        // 对很多 PDF 漏掉 \x0c 分页符会返回 1。pdf.js 解析 PDF 内部 page tree 是权威值。
        if (pdf.numPages !== pageCount) {
          setActualPageCount(pdf.numPages);
        }
        setLoading(false);
      } catch (e) {
        if (cancelled || seq !== seqRef.current) return;
        setLoadError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resourceId]);

  const targetWidthPx = useMemo(
    () => (containerWidth > 0 ? computeTargetWidthPx(containerWidth) : 0),
    [containerWidth],
  );

  // 渲染当前页：container width 已知、PDF 已加载、idx 变化时触发
  useEffect(() => {
    const pdf = pdfRef.current;
    // 加载中（loading=true 是 PDF 还在加载或当前正在栅格化），且不是新加载触发——
    // 不要在此 effect 中做实际渲染，避免和加载 effect 抢。
    if (!pdf || loading) return;
    if (idx < 0 || idx >= effectivePageCount) return;
    if (targetWidthPx === 0) return;
    const host = hostRef.current;
    if (!host) return;
    const seq = ++seqRef.current;

    // 翻页前先清掉旧 textLayer（不调 page.cleanup —— 会让缓存里的 page 变脏，
    // 且 pdf.js 内部已经缓存了 page proxy，我们再管一份纯属添乱）。
    if (textLayerRef.current) {
      textLayerRef.current.cancel();
      textLayerRef.current = null;
    }
    currentPageRef.current = null;
    textLayerElRef.current = null;

    (async () => {
      // 直接让 pdf.js 给 page proxy —— 它自己内部有 _pagePromises 缓存。
      // pdf.js getPage 是 1-indexed（不是 0-indexed），而我们的 idx 是 0-indexed。
      let page: PDFPageProxy;
      try {
        page = await pdf.getPage(idx + 1);
      } catch (e) {
        if (seq !== seqRef.current) return;
        console.error(`[pdf] getPage(${idx + 1}) failed`, e);
        return;
      }
      if (seq !== seqRef.current) return;

      // viewport
      const baseViewport = page.getViewport({ scale: 1 });
      const scale = targetWidthPx / baseViewport.width;
      const viewport = page.getViewport({ scale });

      // 清空 host
      host.replaceChildren();
      host.style.width = `${viewport.width}px`;
      host.style.height = `${viewport.height}px`;

      // canvas
      const canvas = document.createElement('canvas');
      canvas.width = Math.floor(viewport.width);
      canvas.height = Math.floor(viewport.height);
      canvas.className = 'block';
      host.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // textLayer div
      const textLayerDiv = document.createElement('div');
      textLayerDiv.className = 'textLayer';
      // pdf.js v4 的 TextLayer 内部用 `calc(var(--scale-factor) * Xpx)` 算每个
      // span 的 left/top/fontSize；canvas 和文本层都用同一 scale。变量要在
      // TextLayer 构造前设到容器上（构造里 setLayerDimensions 读了 pageWidth/Height
      // 直接乘 var(--scale-factor) 给容器 width/height）。
      textLayerDiv.style.setProperty('--scale-factor', String(scale));
      host.appendChild(textLayerDiv);
      textLayerElRef.current = textLayerDiv;

      // text layer（要先于 canvas 渲染，因为 layout 尺寸由它撑开）
      const textContent = await page.getTextContent();
      if (seq !== seqRef.current) return;
      const textLayer = new pdfjsLib.TextLayer({
        textContentSource: textContent,
        container: textLayerDiv,
        viewport,
      });
      textLayerRef.current = textLayer;

      // canvas 渲染
      const renderTask = page.render({ canvasContext: ctx, viewport });

      try {
        await Promise.all([
          textLayer.render(),
          renderTask.promise.catch((e: unknown) => {
            // 翻页触发 cancel 是正常情况
            if (e instanceof Error && /cancelled/i.test(e.message)) return;
            throw e;
          }),
        ]);
      } catch (e) {
        if (seq !== seqRef.current) return;
        console.error(`[pdf] render page ${idx} failed`, e);
        return;
      }
      if (seq !== seqRef.current) return;

      currentPageRef.current = page;
      setLoading(false);
      // 当前页重画完 → 插入笔记角标
      insertNoteMarkers(textLayerDiv, notes);
    })();
    // deps 含 loading（取反）：PDF 从 null 变 ready 时也必须重跑 —— 仅靠 setLoading
    // 副作用让 React 知道该重画一次。其它依赖（idx、targetWidthPx、pageCount）
    // 仍然是触发翻页/缩放重渲染的关键。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx, targetWidthPx, effectivePageCount, loading]);

  // notes 变化时，只在当前页 textLayer 里重做角标插入（不重画 PDF）。
  // 翻页 / 重画会在上面那个 effect 末尾自己跑 insertNoteMarkers，这里只用 effect 接 notes 变更。
  useEffect(() => {
    const textLayer = textLayerElRef.current;
    if (!textLayer) return;
    unwrapAnchors(textLayer);
    insertNoteMarkers(textLayer, notes);
  }, [notes, idx]);

  // 预取相邻 ±PREFETCH_RADIUS 页：触发 pdf.js 内部缓存命中，
  // 让用户翻页时 `pdf.getPage(idx+1)` 是 0ms。我们不存外部缓存，
  // 避免 page proxy 生命周期混乱（曾踩过 cleanup 后变脏的坑）。
  useEffect(() => {
    const pdf = pdfRef.current;
    if (!pdf || loading || targetWidthPx === 0) return;
    for (let off = 1; off <= PREFETCH_RADIUS; off++) {
      for (const target of [idx + off, idx - off]) {
        if (target < 0 || target >= effectivePageCount) continue;
        // idx 是 0-indexed（前端 UI），pdf.js getPage 是 1-indexed
        void pdf.getPage(target + 1).catch(() => {
          /* 预取失败静默 */
        });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx, targetWidthPx, effectivePageCount, loading]);

  const goPrev = () => setIdx((i) => Math.max(0, i - 1));
  const goNext = () => setIdx((i) => Math.min(effectivePageCount - 1, i + 1));

  // 右键唤起浮层菜单：选中文本后右键出现 SelectionMenu（5 项）。
  // 没选中或选区不在当前页 textLayer 内 → 不 preventDefault，让浏览器显示默认菜单。
  const handleContextMenu = (e: React.MouseEvent) => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    const textLayer = textLayerElRef.current;
    if (!textLayer || !textLayer.contains(range.commonAncestorContainer)) return;

    const text = range.toString().trim();
    if (text.length < 2 || text.length > 500) return;

    const { anchorText, anchorOccurrence } = computeAnchor(textLayer, range);
    if (!anchorText) return;

    e.preventDefault();
    setMenu({ x: e.clientX, y: e.clientY, anchorText, anchorOccurrence });
  };

  // PDF 单页 没有外可见 h1/h2/h3：把"页码"作为 sectionTitle，章节定位的退化。
  // 整页文本 < 阈值时，直接当 contextBefore + contextAfter，避免给 LLM 喂半页。
  const computeContext = (textLayer: HTMLElement, range: Range) => {
    const ctx = extractContext(textLayer, range);
    const wholePageText = textLayer.textContent?.replace(/\s+/g, ' ').trim() ?? '';
    if (wholePageText.length < WHOLE_PAGE_CONTEXT_THRESHOLD) {
      return {
        ...ctx,
        sectionTitle: `第 ${idx + 1} 页`,
        contextBefore: wholePageText,
        contextAfter: wholePageText,
      };
    }
    return { ...ctx, sectionTitle: `第 ${idx + 1} 页` };
  };

  // SelectionMenu 上的"AI 讲解"按钮：从当前选区取"第 N 页"上下文（章节定位退化），
  // 丢给父组件去调 LLM 并建笔记。父组件拿到占位笔记后立刻打开抽屉显示。
  const handleAiAnnotate = () => {
    if (!onAiAnnotate || !menu) return;
    const sel = window.getSelection();
    const textLayer = textLayerElRef.current;
    if (!sel || sel.rangeCount === 0 || !textLayer) return;
    const range = sel.getRangeAt(0);
    if (!textLayer.contains(range.commonAncestorContainer)) return;
    const selectedText = sel.toString().trim();
    if (!selectedText) return;
    const ctx = computeContext(textLayer, range);
    void onAiAnnotate({
      selectedText,
      contextBefore: ctx.contextBefore,
      contextAfter: ctx.contextAfter,
      sectionTitle: ctx.sectionTitle,
      anchorText: menu.anchorText,
      anchorOccurrence: menu.anchorOccurrence,
      pageIdx: idx,
    }).catch((e) => console.error('[pdf] AI explain failed', e));
  };

  // SelectionMenu 上的"询问 AI"：关菜单 → 开 AskInputBox 让用户输入问题 → 提交。
  const handleAiAskOpen = () => {
    if (!onAiAsk || !menu) return;
    const sel = window.getSelection();
    const textLayer = textLayerElRef.current;
    if (!sel || sel.rangeCount === 0 || !textLayer) return;
    const range = sel.getRangeAt(0);
    if (!textLayer.contains(range.commonAncestorContainer)) return;
    const selectedText = sel.toString().trim();
    if (!selectedText) return;
    const ctx = computeContext(textLayer, range);
    setAskBox({
      x: menu.x,
      y: menu.y,
      selectedText,
      contextBefore: ctx.contextBefore,
      contextAfter: ctx.contextAfter,
      sectionTitle: ctx.sectionTitle,
      anchorText: menu.anchorText,
      anchorOccurrence: menu.anchorOccurrence,
    });
  };

  const handleAiAskSubmit = (question: string) => {
    if (!onAiAsk || !askBox) return;
    const a = askBox;
    setAskBox(null);
    void onAiAsk({
      selectedText: a.selectedText,
      contextBefore: a.contextBefore,
      contextAfter: a.contextAfter,
      sectionTitle: a.sectionTitle,
      anchorText: a.anchorText,
      anchorOccurrence: a.anchorOccurrence,
      pageIdx: idx,
      question,
    }).catch((e) => console.error('[pdf] AI ask failed', e));
  };

  const handleAiAskCancel = () => setAskBox(null);

  useEffect(() => {
    if (!askBox) return;
    const close = () => setAskBox(null);
    const onMouseDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest('[data-ask-input-box]')) return;
      close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    const id = window.setTimeout(() => {
      document.addEventListener('mousedown', onMouseDown);
      document.addEventListener('keydown', onKey);
    }, 0);
    return () => {
      clearTimeout(id);
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [askBox]);

  // 点击 <sup.note-marker> → 通知父组件
  const handleClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement | null;
    const marker = target?.closest('sup.note-marker') as HTMLElement | null;
    if (!marker) return;
    const id = Number(marker.dataset.noteId);
    if (!Number.isFinite(id)) return;
    onMarkClick?.(id);
  };

  // 跳转到某条笔记：切到对应页 → scrollIntoView → flash 动画
  useImperativeHandle(ref, () => ({
    scrollToAnchor(noteId: number) {
      const note = notes.find((n) => n.id === noteId);
      if (!note || note.pageIdx == null) return;
      const targetPage = note.pageIdx;
      if (targetPage !== idx) {
        setIdx(targetPage);
        // 翻页后 textLayer 才刚被 render effect 重建，marker 还没插入。
        // 等 idx 变更触发重画完，再 scrollIntoView —— 用 setTimeout 0 等同 tick。
        // 这里不能直接 querySelector，因为翻页未完成。
        setTimeout(() => {
          const root = textLayerElRef.current;
          const marker = root?.querySelector<HTMLElement>(
            `sup.note-marker[data-note-id="${noteId}"]`,
          );
          if (!marker) return;
          marker.scrollIntoView({ behavior: 'smooth', block: 'center' });
          marker.classList.remove('note-flash');
          void marker.offsetWidth;
          marker.classList.add('note-flash');
        }, 50);
        return;
      }
      const root = textLayerElRef.current;
      const marker = root?.querySelector<HTMLElement>(
        `sup.note-marker[data-note-id="${noteId}"]`,
      );
      if (!marker) return;
      marker.scrollIntoView({ behavior: 'smooth', block: 'center' });
      marker.classList.remove('note-flash');
      void marker.offsetWidth;
      marker.classList.add('note-flash');
    },
  }));

  return (
    <div className="flex h-full flex-col">
      <Toolbar idx={idx} pageCount={effectivePageCount} onPrev={goPrev} onNext={goNext} />
      <div
        ref={containerRef}
        className="relative flex-1 overflow-auto bg-surface-2"
        onContextMenu={handleContextMenu}
        onClick={handleClick}
      >
        {/* pdf.js canvas + textLayer 都挂到这个 host 上，host 尺寸由 viewport 决定 */}
        <div className="mx-auto my-4 block w-fit">
          <div ref={hostRef} className="relative inline-block shadow" />
        </div>
        {loading && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-text-muted">
            <Loader2 size={16} className="mr-2 animate-spin" />
            正在渲染第 {idx + 1} / {effectivePageCount} 页…
          </div>
        )}
        {loadError && (
          <div className="m-4 rounded border border-red-300 bg-red-50 p-3 text-sm text-red-700">
            PDF 加载失败：{loadError}
          </div>
        )}
      </div>
      {menu && (
        <SelectionMenu
          x={menu.x}
          y={menu.y}
          anchorText={menu.anchorText}
          anchorOccurrence={menu.anchorOccurrence}
          onAddNote={(input) => {
            onAddNoteAtSelection?.({ ...input, pageIdx: idx });
          }}
          onAiAnnotate={onAiAnnotate ? handleAiAnnotate : undefined}
          onAiAsk={onAiAsk ? handleAiAskOpen : undefined}
          onClose={() => setMenu(null)}
        />
      )}
      {askBox && (
        <AskInputBox
          position={{ x: askBox.x, y: askBox.y }}
          onSubmit={handleAiAskSubmit}
          onCancel={handleAiAskCancel}
        />
      )}
    </div>
  );
});

interface ToolbarProps {
  idx: number;
  pageCount: number;
  onPrev: () => void;
  onNext: () => void;
}

function Toolbar({ idx, pageCount, onPrev, onNext }: ToolbarProps) {
  return (
    <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-2 text-sm">
      <button
        onClick={onPrev}
        disabled={idx === 0}
        className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
      >
        <ChevronLeft size={14} /> 上一页
      </button>
      <span>第 {idx + 1} / {pageCount} 页</span>
      <button
        onClick={onNext}
        disabled={idx >= pageCount - 1}
        className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
      >
        下一页 <ChevronRight size={14} />
      </button>
    </div>
  );
}

/** 容器 CSS 宽度 → 渲染宽度（设备像素）。 */
function computeTargetWidthPx(containerWidth: number): number {
  const dpr =
    typeof window !== 'undefined' && typeof window.devicePixelRatio === 'number'
      ? window.devicePixelRatio
      : FALLBACK_DPR;
  const raw = Math.round(containerWidth * dpr);
  return Math.max(MIN_TARGET_WIDTH, Math.min(MAX_TARGET_WIDTH, raw));
}

/** Tauri 2 IPC 字节归一化：Uint8Array / ArrayBuffer / number[] → Uint8Array。 */
function normalizeBytes(raw: Uint8Array | ArrayBuffer | number[]): Uint8Array {
  if (raw instanceof Uint8Array) return raw;
  if (raw instanceof ArrayBuffer) return new Uint8Array(raw);
  if (Array.isArray(raw)) return new Uint8Array(raw);
  throw new Error(`unexpected bytes type: ${typeof raw}`);
}