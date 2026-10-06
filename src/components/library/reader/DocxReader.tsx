import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';
import mammoth from 'mammoth';
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

export interface DocxReaderHandle {
  scrollToAnchor: (noteId: number) => void;
}

interface Props {
  /** 资源 ID：用于 readResourceBytes 加载 docx，然后交给 mammoth 转 HTML。 */
  resourceId: number;
  /** 后端 word 数（仅展示用，与 mammoth 转换独立）。 */
  wordCount: number;
  /** 资源下的笔记列表；用于插入角标。 */
  notes?: Note[];
  onMarkClick?: (noteId: number) => void;
  onAddNoteAtSelection?: (input: {
    anchorText: string;
    anchorOccurrence: number;
  }) => void;
  onAiAnnotate?: (input: {
    selectedText: string;
    contextBefore: string;
    contextAfter: string;
    sectionTitle: string;
    anchorText: string;
    anchorOccurrence: number;
  }) => Promise<void>;
  onAiAsk?: (input: {
    selectedText: string;
    contextBefore: string;
    contextAfter: string;
    sectionTitle: string;
    anchorText: string;
    anchorOccurrence: number;
    question: string;
  }) => Promise<void>;
}

/**
 * DOCX 阅读器 —— 后端只数 word 数，原文件 raw bytes 交给前端 mammoth.js
 * 转成 semantic HTML（含 bold/italic/headings/lists/tables/images）。
 *
 * 设计要点：
 * - 拿到 bytes 后 `new Uint8Array(...)` → `.buffer` 转 ArrayBuffer 喂给
 *   mammoth（浏览器侧 mammoth 走 ArrayBufferInput）
 * - 图片走 `mammoth.images.dataUri` —— 把 docx 内嵌图片转 data URI，
 *   这样不依赖外部资源，PPT 离线状态下也能完整渲染
 * - mammoth 输出非信任 HTML（警告明确写在 mammoth README），
 *   不过我们 sources 是用户自己放的 knowledge 目录，且 tauri webview
 *   默认 sandbox 隔离前端，与 markdown reader 同样不做 DOMPurify
 *
 * 老实现：后端用 docx-rs 抽 text → 前端只渲染 heading/p/table 的纯文本。
 * 现在前端 mammoth 直接处理原始 docx，版式 100% 来自 docx 本身。
 *
 * 笔记功能（Phase 6）：
 * - mammoth 渲染完后 useEffect 跑 insertNoteMarkers，把 <sup.note-marker> 插入 host
 * - 容器 onContextMenu：选中文本后右键 → 算 anchor → 开 SelectionMenu
 * - forwardRef.scrollToAnchor(noteId)：直接 scrollIntoView + flash
 */
export default forwardRef<DocxReaderHandle, Props>(function DocxReader(
  { resourceId, wordCount, notes = [], onMarkClick, onAddNoteAtSelection, onAiAnnotate, onAiAsk },
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    let cancelled = false;
    const host = hostRef.current;
    if (!host) return;
    // 切资源时先清空旧内容
    host.replaceChildren();
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const raw = await api.readResourceBytes(resourceId);
        if (cancelled) return;
        const ab = toArrayBuffer(raw);
        const { value, messages } = await mammoth.convertToHtml(
          { arrayBuffer: ab },
          {
            // 把内嵌图片（png/jpeg 等）转 data URI，
            // 让 docx 里的图在 webview 里直接显示，不用走文件系统。
            convertImage: mammoth.images.dataUri,
          },
        );
        if (cancelled) return;
        // mammoth 输出是 HTML 字符串；用 prose 样式接管段落 / 表格 / 标题的视觉
        host.innerHTML = value;
        if (messages.length > 0) {
          // 转换期间的 warning/error 不致命 —— 记录到控制台，方便调试
          // 个别 docx 不规范的元素 mammoth 会跳过（不影响主体渲染）。
          console.warn(
            `[docx ${resourceId}] mammoth 转换产生 ${messages.length} 条消息：`,
            messages,
          );
        }
        setLoading(false);
        // 当前 docx HTML 写完 → 重新插入笔记角标（覆盖旧 DOM）
        unwrapAnchors(host);
        insertNoteMarkers(host, notes);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
    // 依赖 notes：notes 变化时也要重做 marker pipeline（但 mammoth 不会重跑）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resourceId, notes]);

  // 右键唤起浮层菜单：选中文本后右键出现 SelectionMenu。
  // 没选中文本则不 preventDefault，让浏览器显示默认菜单。
  const handleContextMenu = (e: React.MouseEvent) => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    const host = hostRef.current;
    if (!host || !host.contains(range.commonAncestorContainer)) return;

    const text = range.toString().trim();
    if (text.length < 2 || text.length > 500) return;

    const { anchorText, anchorOccurrence } = computeAnchor(host, range);
    if (!anchorText) return;

    e.preventDefault();
    setMenu({ x: e.clientX, y: e.clientY, anchorText, anchorOccurrence });
  };

  // SelectionMenu 上的"AI 讲解"按钮：从当前选区取"局部上下文"（章节 + 前后窗），
  // 丢给父组件去调 LLM 并建笔记。
  const handleAiAnnotate = () => {
    if (!onAiAnnotate || !menu) return;
    const sel = window.getSelection();
    const host = hostRef.current;
    if (!sel || sel.rangeCount === 0 || !host) return;
    const range = sel.getRangeAt(0);
    if (!host.contains(range.commonAncestorContainer)) return;
    const selectedText = sel.toString().trim();
    if (!selectedText) return;
    const ctx = extractContext(host, range);
    void onAiAnnotate({
      selectedText,
      contextBefore: ctx.contextBefore,
      contextAfter: ctx.contextAfter,
      sectionTitle: ctx.sectionTitle,
      anchorText: menu.anchorText,
      anchorOccurrence: menu.anchorOccurrence,
    }).catch((e) => console.error('[docx] AI explain failed', e));
  };

  // SelectionMenu 上的"询问 AI"
  const handleAiAskOpen = () => {
    if (!onAiAsk || !menu) return;
    const sel = window.getSelection();
    const host = hostRef.current;
    if (!sel || sel.rangeCount === 0 || !host) return;
    const range = sel.getRangeAt(0);
    if (!host.contains(range.commonAncestorContainer)) return;
    const selectedText = sel.toString().trim();
    if (!selectedText) return;
    const ctx = extractContext(host, range);
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
      question,
    }).catch((e) => console.error('[docx] AI ask failed', e));
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

  useImperativeHandle(ref, () => ({
    scrollToAnchor(noteId: number) {
      const host = hostRef.current;
      if (!host) return;
      const marker = host.querySelector<HTMLElement>(
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
    <div
      className="relative h-full overflow-auto bg-surface-2"
      onContextMenu={handleContextMenu}
      onClick={handleClick}
    >
      <div
        ref={hostRef}
        className="prose prose-sm mx-auto my-4 max-w-3xl rounded-lg border border-border bg-surface p-6 shadow-sm dark:prose-invert"
      />
      {loading && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-text-muted">
          <Loader2 size={16} className="mr-2 animate-spin" />
          正在解析 DOCX…
        </div>
      )}
      {error && (
        <div className="m-4 rounded border border-red-300 bg-red-50 p-3 text-sm text-red-700">
          DOCX 加载失败：{error}
          <div className="mt-1 text-xs text-red-600">
            文件大小约 {wordCount.toLocaleString()} 词
          </div>
        </div>
      )}
      {menu && (
        <SelectionMenu
          x={menu.x}
          y={menu.y}
          anchorText={menu.anchorText}
          anchorOccurrence={menu.anchorOccurrence}
          onAddNote={(input) => onAddNoteAtSelection?.(input)}
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

/** Tauri 2 IPC 字节归一化：Uint8Array / ArrayBuffer / number[] → ArrayBuffer。 */
function toArrayBuffer(raw: Uint8Array | ArrayBuffer | number[]): ArrayBuffer {
  if (raw instanceof ArrayBuffer) return raw;
  if (raw instanceof Uint8Array) {
    // 切片以拿到 underlying ArrayBuffer，避免和 Uint8Array 共享同一 buffer
    // 时 mammoth 读到原始 buffer 长度而不是我们传进去的字节数。
    return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  }
  if (Array.isArray(raw)) return new Uint8Array(raw).buffer;
  throw new Error(`unexpected bytes type: ${typeof raw}`);
}