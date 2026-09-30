import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { MessageSquarePlus } from 'lucide-react';
import katex from 'katex';
import renderMathInElement from 'katex/contrib/auto-render';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import go from 'highlight.js/lib/languages/go';
import json from 'highlight.js/lib/languages/json';
import bash from 'highlight.js/lib/languages/bash';
import rust from 'highlight.js/lib/languages/rust';
import typescript from 'highlight.js/lib/languages/typescript';
import javascript from 'highlight.js/lib/languages/javascript';
import xml from 'highlight.js/lib/languages/xml';
import css from 'highlight.js/lib/languages/css';
import type { Note } from '../../../types';

export interface MarkdownReaderHandle {
  scrollToAnchor: (noteId: number) => void;
}

interface Props {
  html: string;
  notes?: Note[];
  onMarkClick?: (noteId: number) => void;
  onAddNoteAtSelection?: (input: {
    anchorText: string;
    anchorOccurrence: number;
  }) => void;
}

hljs.registerLanguage('python', python);
hljs.registerLanguage('go', go);
hljs.registerLanguage('json', json);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('rust', rust);
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('xml', xml);
hljs.registerLanguage('css', css);

/**
 * comrak 不识别 ```math 围栏也不处理 $...$，所以在 frontend 兜底：
 * 1) 把 <pre><code class="language-math">…</code></pre> 用 KaTeX 渲染
 * 2) 给其它围栏调用 highlight.js 加语法高亮，并补上 data-lang 供 CSS 显示语言标签
 * 3) 高亮每条带 anchor 的笔记（按 Nth occurrence 包裹成 <mark data-note-id="…">）
 * 4) 让 auto-render 扫一遍内联 $...$ / $$...$$ / \(...\) / \[...\]
 */
function renderMathBlocks(root: HTMLElement) {
  const blocks = root.querySelectorAll<HTMLPreElement>('pre > code.language-math');
  blocks.forEach((code) => {
    const pre = code.parentElement;
    if (!pre) return;
    const tex = decodeEntities(code.textContent ?? '');
    const span = document.createElement('div');
    span.className = 'my-4 overflow-x-auto';
    try {
      katex.render(tex, span, { displayMode: true, throwOnError: false });
    } catch {
      span.textContent = tex;
    }
    pre.replaceWith(span);
  });
}

function highlightCodeBlocks(root: HTMLElement) {
  const blocks = root.querySelectorAll<HTMLElement>('pre > code[class*="language-"]:not(.language-math)');
  blocks.forEach((code) => {
    const pre = code.parentElement as HTMLPreElement | null;
    const lang = [...code.classList]
      .find((c) => c.startsWith('language-'))
      ?.slice('language-'.length);
    if (lang) {
      try {
        if (hljs.getLanguage(lang)) {
          hljs.highlightElement(code);
        } else {
          code.classList.add('hljs');
        }
      } catch {
        code.classList.add('hljs');
      }
      if (pre) pre.dataset.lang = lang;
    } else {
      code.classList.add('hljs');
    }
  });
}

/**
 * 把先前 highlightAnchors 留下的 <mark.note-anchor> 解包回纯文本。
 * 这是 idempotent 的前提：每次重跑前必须先清理，否则会把已有锚点
 * 当作"不可见"而把同一段文字的另一次出现误判成目标 → 重复包裹。
 */
function unwrapAnchors(root: HTMLElement) {
  const marks = root.querySelectorAll('mark.note-anchor');
  marks.forEach((m) => {
    const parent = m.parentNode;
    if (!parent) return;
    while (m.firstChild) parent.insertBefore(m.firstChild, m);
    parent.removeChild(m);
    parent.normalize();
  });
}

/**
 * 把所有带 anchor 的笔记在正文中高亮。
 * 对每个 note，按 anchorOccurrence（0-based）找到第 N+1 次出现并包成 <mark>。
 * 长文本优先匹配，避免短文本把长文本截断。
 * 跳过已经被 KaTeX 渲染、hljs 高亮或代码块的文本节点，
 * 避免在数学符号 / 代码 token 里塞 mark 把它们拆坏。
 */
const ANCHOR_SKIP_SELECTOR =
  'mark.note-anchor, pre, code, .katex, .katex-mathml, .katex-html, .hljs';

function highlightAnchors(root: HTMLElement, notes: Note[]) {
  const anchors = notes
    .filter((n) => n.anchorText && n.anchorText.trim().length > 0)
    .sort((a, b) => (b.anchorText!.length - a.anchorText!.length));
  if (anchors.length === 0) return;

  for (const note of anchors) {
    const needle = note.anchorText!;
    let remaining = note.anchorOccurrence;
    let matched = false;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const t = node.textContent ?? '';
        if (!t.includes(needle)) return NodeFilter.FILTER_REJECT;
        const parent = (node as Text).parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;
        // 关键：跳过 mark / katex / hljs / pre / code 内的文本
        if (parent.closest(ANCHOR_SKIP_SELECTOR)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    walker.currentNode = root;
    const candidates: Text[] = [];
    let cur: Node | null = walker.nextNode();
    while (cur) {
      candidates.push(cur as Text);
      cur = walker.nextNode();
    }
    for (const textNode of candidates) {
      const idx = (textNode.textContent ?? '').indexOf(needle);
      if (idx < 0) continue;
      if (remaining > 0) {
        remaining -= 1;
        continue;
      }
      // 命中：拆 textNode，包裹匹配段
      const before = textNode.textContent!.slice(0, idx);
      const after = textNode.textContent!.slice(idx + needle.length);
      const beforeNode = document.createTextNode(before);
      const afterNode = document.createTextNode(after);
      const mark = document.createElement('mark');
      mark.className = 'note-anchor';
      mark.dataset.noteId = String(note.id);
      mark.title = '点击查看笔记';
      mark.textContent = needle;
      const parent = textNode.parentNode!;
      parent.insertBefore(beforeNode, textNode);
      parent.insertBefore(mark, textNode);
      parent.insertBefore(afterNode, textNode);
      parent.removeChild(textNode);
      matched = true;
      break;
    }
    // 没匹配上时静默忽略（笔记保留在右侧面板）
    void matched;
  }
}

/** 计算选中文字在 root 内的 0-based 出现位置（用于 anchorOccurrence）。 */
function occurrenceOfSelection(root: HTMLElement, needle: string, range: Range): number {
  if (!needle) return 0;
  // 克隆一个 Range：从 root 起点到 range.startContainer/startOffset
  const probe = document.createRange();
  probe.setStart(root, 0);
  probe.setEnd(range.startContainer, range.startOffset);
  const before = probe.toString();
  let count = 0;
  let from = 0;
  while (true) {
    const i = before.indexOf(needle, from);
    if (i < 0) break;
    count += 1;
    from = i + needle.length;
  }
  return count;
}

function decodeEntities(s: string): string {
  return s
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

export default forwardRef<MarkdownReaderHandle, Props>(function MarkdownReader(
  { html, notes = [], onMarkClick, onAddNoteAtSelection },
  ref,
) {
  const innerRef = useRef<HTMLDivElement>(null);
  const [bubble, setBubble] = useState<{
    x: number;
    y: number;
    anchorText: string;
    anchorOccurrence: number;
  } | null>(null);

  useImperativeHandle(ref, () => ({
    scrollToAnchor(noteId: number) {
      const root = innerRef.current;
      if (!root) return;
      const mark = root.querySelector<HTMLElement>(
        `mark.note-anchor[data-note-id="${noteId}"]`,
      );
      if (!mark) return;
      mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
      mark.classList.remove('note-flash');
      // 触发 reflow 重新跑动画
      void mark.offsetWidth;
      mark.classList.add('note-flash');
    },
  }));

  // 文章内容（html）变化时：跑一次完整的渲染管道
  // - 解包旧锚点 / 处理 math 块 / 高亮代码 / 包新锚点 / 内联 math
  // 这样每次切文章都能从干净状态开始
  useEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    unwrapAnchors(el);
    renderMathBlocks(el);
    highlightCodeBlocks(el);
    highlightAnchors(el, notes);
    renderMathInElement(el, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
        { left: '\\[', right: '\\]', display: true },
      ],
      throwOnError: false,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [html]);

  // 笔记变化时：只重做锚点包裹，不动 math/code/inlineMath
  useEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    unwrapAnchors(el);
    highlightAnchors(el, notes);
  }, [notes]);

  // 选中文字 → 显示浮动气泡
  const handleMouseUp = () => {
    if (!onAddNoteAtSelection) return;
    const el = innerRef.current;
    if (!el) return;
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
      setBubble(null);
      return;
    }
    const range = sel.getRangeAt(0);
    if (!el.contains(range.commonAncestorContainer)) {
      setBubble(null);
      return;
    }
    const text = sel.toString().trim();
    if (text.length < 2 || text.length > 500) {
      setBubble(null);
      return;
    }
    const rect = range.getBoundingClientRect();
    const occurrence = occurrenceOfSelection(el, text, range);
    setBubble({
      x: rect.left + rect.width / 2,
      y: rect.top - 8,
      anchorText: text,
      anchorOccurrence: occurrence,
    });
  };

  const handleClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement | null;
    const mark = target?.closest('mark.note-anchor') as HTMLElement | null;
    if (!mark) return;
    const id = Number(mark.dataset.noteId);
    if (!Number.isFinite(id)) return;
    onMarkClick?.(id);
  };

  return (
    <div className="relative h-full overflow-auto">
      <div
        className="prose prose-sm max-w-none p-6 dark:prose-invert"
        ref={innerRef}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        // comrak 输出受信任；Phase 2 不引入 DOMPurify
        dangerouslySetInnerHTML={{ __html: html }}
      />
      {bubble && (
        <button
          type="button"
          onMouseDown={(e) => {
            // mousedown 会清掉 selection，提前吃掉事件
            e.preventDefault();
            onAddNoteAtSelection?.({
              anchorText: bubble.anchorText,
              anchorOccurrence: bubble.anchorOccurrence,
            });
            setBubble(null);
            window.getSelection()?.removeAllRanges();
          }}
          style={{
            position: 'fixed',
            top: bubble.y,
            left: bubble.x,
            transform: 'translate(-50%, -100%)',
          }}
          className="z-50 inline-flex items-center gap-1 rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-white shadow-lg hover:bg-accent-hover"
        >
          <MessageSquarePlus size={12} />
          添加批注
        </button>
      )}
    </div>
  );
});