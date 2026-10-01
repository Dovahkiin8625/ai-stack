import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { Check, Copy, Loader2, Languages, MessageSquarePlus, Sparkles } from 'lucide-react';
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
import { translateText as apiTranslateText } from '../../../lib/library-api';
import { getSettings } from '../../../lib/tauri';
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
  /** 右键菜单"AI 讲解"：把选区上下文交给父组件，父组件去调 LLM 然后建笔记（source='ai'）。
   *  MarkdownReader 只负责"提取上下文 + 显示进度/错误"，不直接管笔记持久化。 */
  onAiAnnotate?: (input: {
    selectedText: string;
    contextBefore: string;
    contextAfter: string;
    sectionTitle: string;
    anchorText: string;
    anchorOccurrence: number;
  }) => Promise<void>;
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
 * 3) 给每条带 anchor 的笔记在匹配文本末尾插入角标 <sup class="note-marker" data-note-id="…">
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
 * 移除先前 insertNoteMarkers 留下的 <sup.note-marker>。
 * 这是 idempotent 的前提：每次重跑前必须先清理，否则会把已有角标
 * 当作"不可见"而把同一段文字的另一次出现误判成目标 → 重复插入。
 */
function unwrapAnchors(root: HTMLElement) {
  const markers = root.querySelectorAll('sup.note-marker');
  markers.forEach((m) => {
    m.remove();
  });
  root.normalize();
}

/**
 * 正文里的"匹配单元"。按文档顺序遍历：
 * - 普通文本节点：每个 Text 节点一个单元
 * - 公式 (.katex)：整个公式一个单元，文本是可见部分 (.katex-html) 的 textContent
 *   排除 MathML 副本、pre/code/hljs，避免重复计数 / 拆坏代码块。
 * - 代码块 / 行内代码 (pre/code)：整体一个单元，文本是 code 元素的可见 textContent。
 *   el 是角标要"插到它之后"的元素：代码块 → <pre>；行内代码 → <code>。
 */
type AnchorUnit =
  | { kind: 'text'; node: Text; text: string }
  | { kind: 'katex'; el: Element; text: string }
  | { kind: 'code'; el: HTMLElement; text: string };

function collectAnchorUnits(root: HTMLElement): AnchorUnit[] {
  const units: AnchorUnit[] = [];
  const seenKatex = new WeakSet<Element>();

  // 先收集代码块（<pre><code>）：renderMathBlocks 已经把 math 块变成 <div>，
  // 剩下的 <pre> 都是真代码块，不会和 katex unit 冲突。
  // el 用 <pre> 而不是 <code>，这样 insertBefore(marker, pre.nextSibling) 把角标
  // 钉在整块代码块末尾，而不是夹在 </code> 和 </pre> 之间造成怪异空白。
  root.querySelectorAll('pre').forEach((pre) => {
    const code = pre.querySelector('code');
    if (!code) return;
    const text = (code.textContent ?? '').replace(/\s+/g, ' ').trim();
    if (!text) return;
    units.push({ kind: 'code', el: pre as HTMLPreElement, text });
  });

  // 再收集行内代码（<code>，但不在 <pre> 内）：inline <code> 用 <code> 自己作 el。
  root.querySelectorAll('code').forEach((code) => {
    if (code.closest('pre')) return;
    const text = (code.textContent ?? '').trim();
    if (!text) return;
    units.push({ kind: 'code', el: code, text });
  });

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = (node as Text).parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      // 跳过 mathml 副本（hidden）、代码块、hljs 高亮 token
      if (parent.closest('.katex-mathml, pre, code, .hljs')) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let cur: Node | null = walker.nextNode();
  while (cur) {
    const textNode = cur as Text;
    const katex = textNode.parentElement?.closest('.katex') ?? null;
    if (katex) {
      if (!seenKatex.has(katex)) {
        seenKatex.add(katex);
        // 只用可见部分 (katex-html)，避免 MathML 副本造成重复计数
        const visible = katex.querySelector('.katex-html');
        units.push({ kind: 'katex', el: katex, text: visible?.textContent ?? '' });
      }
    } else {
      units.push({ kind: 'text', node: textNode, text: textNode.textContent ?? '' });
    }
    cur = walker.nextNode();
  }

  return units;
}

function createNoteMarker(note: Note, markerNumber: number): HTMLElement {
  const marker = document.createElement('sup');
  marker.className = 'note-marker';
  marker.dataset.noteId = String(note.id);
  marker.textContent = `[${markerNumber}]`;
  marker.title = '点击查看笔记';
  return marker;
}

/**
 * 把所有带 anchor 的笔记在正文中插入角标。
 * 角标序号：按笔记在 notes 数组中的顺序，1, 2, 3... 动态编号（仅统计带 anchor 的笔记）。
 * 对每个 note，按 anchorOccurrence（0-based）找到第 N+1 次出现：
 *   - 普通文本：拆 textNode，在匹配段末尾插入角标
 *   - 公式 (.katex)：在 .katex 元素之后插入角标（不破坏 KaTeX 内部 DOM）
 * 原文不做任何包裹/高亮处理，保持阅读体验。
 * 长文本优先匹配，避免短 anchor 把长 anchor 截断。
 */
function insertNoteMarkers(root: HTMLElement, notes: Note[]) {
  // 1) 先按 notes 数组原序分配角标序号（与 NotesPanel 显示顺序一致）
  const numberById = new Map<number, number>();
  let anchoredIdx = 0;
  for (const n of notes) {
    if (n.anchorText && n.anchorText.trim().length > 0) {
      anchoredIdx += 1;
      numberById.set(n.id, anchoredIdx);
    }
  }

  // 2) 按 anchor 长度降序匹配，避免短 anchor 抢先
  const anchors = notes
    .filter((n) => n.anchorText && n.anchorText.trim().length > 0)
    .sort((a, b) => (b.anchorText!.length - a.anchorText!.length));
  if (anchors.length === 0) return;

  const units = collectAnchorUnits(root);

  for (const note of anchors) {
    const needle = note.anchorText!;
    let remaining = note.anchorOccurrence;
    let matched = false;
    const markerNumber = numberById.get(note.id) ?? note.id;

    for (const unit of units) {
      const idx = unit.text.indexOf(needle);
      if (idx < 0) continue;
      if (remaining > 0) {
        remaining -= 1;
        continue;
      }

      const marker = createNoteMarker(note, markerNumber);

      if (unit.kind === 'katex') {
        // 公式：在 .katex 之后插入角标，不动 KaTeX 内部结构
        const parent = unit.el.parentNode;
        if (parent) {
          parent.insertBefore(marker, unit.el.nextSibling);
          matched = true;
        }
      } else if (unit.kind === 'code') {
        // 代码块 / 行内代码：在 unit.el（<pre> 或 <code>）之后插入角标，
        // 不动 code 内部 DOM，避免破坏 hljs 高亮或 inline 渲染。
        const parent = unit.el.parentNode;
        if (parent) {
          parent.insertBefore(marker, unit.el.nextSibling);
          matched = true;
        }
      } else {
        // 普通文本：拆 textNode，在匹配段末尾插入角标
        const before = unit.text.slice(0, idx + needle.length);
        const after = unit.text.slice(idx + needle.length);
        const beforeNode = document.createTextNode(before);
        const afterNode = document.createTextNode(after);
        const textParent = unit.node.parentNode!;
        textParent.insertBefore(beforeNode, unit.node);
        textParent.insertBefore(marker, unit.node);
        textParent.insertBefore(afterNode, unit.node);
        textParent.removeChild(unit.node);
        matched = true;
      }
      break;
    }
    // 没匹配上时静默忽略（笔记保留在右侧面板）
    void matched;
  }
}

/** 计算选中文字在 root 内的 0-based 出现位置（用于 anchorOccurrence）。
 *
 * 注意：KaTeX 在 .katex 下同时挂 .katex-mathml（隐藏 MathML 副本）和 .katex-html（可见部分），
 * 两者的 textContent 完全相同。如果直接用 range.toString() 计数，会把 MathML 副本也包含进去，
 * 跟 collectAnchorUnits 中"只看可见部分"的匹配规则不一致 → 公式上 occurrence 偏高、
 * 角标被错位跳过。这里用一个跳过 .katex-mathml 的 TreeWalker 自己拼"before"文本。
 */
function occurrenceOfSelection(root: HTMLElement, needle: string, range: Range): number {
  if (!needle) return 0;
  const target = range.startContainer;

  let before = '';
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = (node as Text).parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      // 与 collectAnchorUnits 对齐：跳过 katex 副本 + pre + code，
      // 否则 occurrence 会把代码块/行内代码里的文本也算进去，超出
      // insertNoteMarkers 实际能匹配的 unit 范围 → 静默失败。
      if (parent.closest('.katex-mathml, pre, code')) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let cur: Node | null = walker.nextNode();
  while (cur) {
    const textNode = cur as Text;
    if (textNode === target) {
      // 到达选区起点：只截到 startOffset
      before += (textNode.textContent ?? '').slice(0, range.startOffset);
      break;
    }
    before += textNode.textContent ?? '';
    cur = walker.nextNode();
  }

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

/**
 * 给 AI 讲解用的"局部上下文"：
 * - sectionTitle：选区上方最近的 h1/h2/h3，告诉 LLM 这段属于哪个章节
 * - contextBefore / contextAfter：选区两侧各 ~600 字的可见文本，做空白规整
 *
 * 设计取舍：仅取章节标题 + 局部窗口，不发整篇文章。
 * - 文章脉络（属于哪个章节）→ 章节标题
 * - 邻接语境（前后在讲什么）→ 600 字窗口
 * - 主题细节 → 用户自己选的那段
 * 这样 prompt 长度可控（通常 < 2k tokens），又能给出足够定位信息，
 * 避免粗暴地把整篇 markdown 塞给 LLM 造成 token 浪费和无关干扰。
 */
const AI_CONTEXT_WINDOW = 600;

function extractAiContext(
  root: HTMLElement,
  range: Range,
): { sectionTitle: string; contextBefore: string; contextAfter: string } {
  // 最近的章节标题：遍历所有 heading，找最后一个"在 range 起点之前"的
  let sectionTitle = '';
  const startNode = range.startContainer;
  const headings = root.querySelectorAll('h1, h2, h3');
  for (const h of Array.from(headings)) {
    const pos = startNode.compareDocumentPosition(h);
    if (pos & Node.DOCUMENT_POSITION_PRECEDING) {
      sectionTitle = (h.textContent ?? '').trim();
    } else if (pos & Node.DOCUMENT_POSITION_FOLLOWING) {
      break;
    }
  }

  // 窗口文本：先用一个覆盖全文的 range，再把端点收缩到选区起点/终点，toString 即为侧文
  const preRange = document.createRange();
  preRange.selectNodeContents(root);
  preRange.setEnd(range.startContainer, range.startOffset);
  const beforeRaw = preRange.toString();

  const postRange = document.createRange();
  postRange.selectNodeContents(root);
  postRange.setStart(range.endContainer, range.endOffset);
  const afterRaw = postRange.toString();

  // 空白规整 + 截尾/截首
  const norm = (s: string) => s.replace(/\s+/g, ' ').trim();
  const beforeTrim = norm(beforeRaw);
  const afterTrim = norm(afterRaw);
  const contextBefore =
    beforeTrim.length > AI_CONTEXT_WINDOW
      ? beforeTrim.slice(-AI_CONTEXT_WINDOW)
      : beforeTrim;
  const contextAfter =
    afterTrim.length > AI_CONTEXT_WINDOW
      ? afterTrim.slice(0, AI_CONTEXT_WINDOW)
      : afterTrim;
  return { sectionTitle, contextBefore, contextAfter };
}

export default forwardRef<MarkdownReaderHandle, Props>(function MarkdownReader(
  { html, notes = [], onMarkClick, onAddNoteAtSelection, onAiAnnotate },
  ref,
) {
  const innerRef = useRef<HTMLDivElement>(null);
  /** 右键唤起的上下文菜单：用户选中文本后右键打开。
   *  - menu 形态：列出"添加批注 / 翻译"两项
   *  - 翻译流程会把 menu 复用成 loading / success / error 的展示面板，
   *    因此同时持有 translation 状态机
   */
  const [menu, setMenu] = useState<{
    x: number;
    y: number;
    anchorText: string;
    anchorOccurrence: number;
  } | null>(null);
  /** 复制成功后的瞬时反馈：true 时菜单只显示一行"已复制"，800ms 后自动关闭 */
  const [copied, setCopied] = useState(false);

  /**
   * 选区菜单上的"翻译"流程：idle → loading → success/error。
   * 跟随 menu 位置显示；菜单关闭时由 useEffect 的 close handler 重置为 idle。
   */
  const [translation, setTranslation] = useState<
    | { status: 'idle' }
    | { status: 'loading' }
    | { status: 'success'; text: string }
    | { status: 'error'; message: string }
  >({ status: 'idle' });

  useImperativeHandle(ref, () => ({
    scrollToAnchor(noteId: number) {
      const root = innerRef.current;
      if (!root) return;
      const marker = root.querySelector<HTMLElement>(
        `sup.note-marker[data-note-id="${noteId}"]`,
      );
      if (!marker) return;
      marker.scrollIntoView({ behavior: 'smooth', block: 'center' });
      marker.classList.remove('note-flash');
      // 触发 reflow 重新跑动画
      void marker.offsetWidth;
      marker.classList.add('note-flash');
    },
  }));

  // 文章内容（html）变化时：跑一次完整的渲染管道
  // - 解包旧锚点 / 处理 math 块 / 高亮代码 / 插入新角标 / 内联 math
  // 这样每次切文章都能从干净状态开始
  useEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    unwrapAnchors(el);
    renderMathBlocks(el);
    highlightCodeBlocks(el);
    insertNoteMarkers(el, notes);
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

  // 笔记变化时：只重做角标插入，不动 math/code/inlineMath
  useEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    unwrapAnchors(el);
    insertNoteMarkers(el, notes);
  }, [notes]);

  // 右键唤起上下文菜单：选中文本后右键出现"添加批注 / 翻译"。
  // 没选中文本则不 preventDefault，让浏览器显示默认菜单。
  const handleContextMenu = (e: React.MouseEvent) => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    const el = innerRef.current;
    if (!el || !el.contains(range.commonAncestorContainer)) return;

    const text = range.toString().trim();
    if (text.length < 2 || text.length > 500) return;

    // 选区起点若落在 .katex 公式内：把锚点钉在该公式末尾（.katex 之后），
    // 避免角标插入到 KaTeX 内部 DOM 导致渲染异常 / 角标丢失。
    // - anchorText 改为整段公式的可见文本（与 katex 单元的 .katex-html textContent 相等）
    // - occurrence 统计 DOM 中位于目标公式之前的"同文本公式"数量
    // insertNoteMarkers 的 katex 分支会在第 N 个匹配的 katex 单元之后插角标。
    //
    // 同样的逻辑适用于 <code>：无论是 <pre><code> 代码块还是行内 `<code>`，
    // 选区起点落在里面时，anchor = 整段 code 的可见文本，occurrence 统计
    // DOM 中位于目标 code 之前的"同文本 code"数量。insertNoteMarkers 的 code
    // 分支会把角标插到 code 元素之后，不破坏 hljs 高亮 / inline 渲染。
    const startContainer = range.startContainer;
    const startEl =
      startContainer.nodeType === Node.ELEMENT_NODE
        ? (startContainer as Element)
        : startContainer.parentElement;
    const katexEl = startEl?.closest('.katex') ?? null;
    const codeEl = startEl?.closest('code') ?? null;

    let anchorText: string;
    let anchorOccurrence: number;

    if (katexEl) {
      const formulaText = (katexEl.querySelector('.katex-html')?.textContent ?? '').trim();
      anchorText = formulaText;
      let prior = 0;
      const all = el.querySelectorAll('.katex');
      for (const k of Array.from(all)) {
        if (k === katexEl) break;
        if ((k.querySelector('.katex-html')?.textContent ?? '').trim() === formulaText) {
          prior += 1;
        }
      }
      anchorOccurrence = prior;
    } else if (codeEl) {
      const codeText = (codeEl.textContent ?? '').replace(/\s+/g, ' ').trim();
      anchorText = codeText;
      let prior = 0;
      const all = el.querySelectorAll('code');
      for (const c of Array.from(all)) {
        if (c === codeEl) break;
        if ((c.textContent ?? '').replace(/\s+/g, ' ').trim() === codeText) {
          prior += 1;
        }
      }
      anchorOccurrence = prior;
    } else {
      anchorText = text;
      anchorOccurrence = occurrenceOfSelection(el, text, range);
    }

    e.preventDefault();
    setMenu({
      x: e.clientX,
      y: e.clientY,
      anchorText,
      anchorOccurrence,
    });
    // 新选区：重置翻译结果，避免把上一段的翻译显示在新的菜单上
    setTranslation({ status: 'idle' });
  };

  // 菜单打开时挂全局监听：点击外部 / Esc 关闭
  useEffect(() => {
    if (!menu) return;
    const close = () => {
      setMenu(null);
      setTranslation({ status: 'idle' });
      setCopied(false);
    };
    const onMouseDown = (e: MouseEvent) => {
      // 点击在菜单内部 → 由按钮自己的 handler 接管；这里只负责关闭"外部点击"
      const target = e.target as HTMLElement | null;
      if (target?.closest('[data-selection-menu]')) return;
      close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    // setTimeout 0 跳过本次右键触发的 mousedown，避免菜单"刚开就被关"
    const id = window.setTimeout(() => {
      document.addEventListener('mousedown', onMouseDown);
      document.addEventListener('keydown', onKey);
    }, 0);
    return () => {
      clearTimeout(id);
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [menu]);

  const handleClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement | null;
    const marker = target?.closest('sup.note-marker') as HTMLElement | null;
    if (!marker) return;
    const id = Number(marker.dataset.noteId);
    if (!Number.isFinite(id)) return;
    onMarkClick?.(id);
  };

  // 菜单上的"复制"按钮：写入剪贴板并闪一个"已复制"反馈。
  // 我们的自定义菜单 preventDefault 了浏览器默认右键菜单，会把"复制"等基础动作吞掉，
  // 所以必须在菜单里显式补一份 —— 否则用户连复制选区都没法做。
  // 优先用异步 Clipboard API；Tauri WebView 老版本或非安全上下文会失败，回退到 execCommand。
  const handleCopy = async () => {
    if (!menu) return;
    const text = menu.anchorText;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      try {
        document.execCommand('copy');
      } catch {
        // 两种 API 都失败（极少见，权限被禁），静默处理 —— 不阻塞菜单关闭
      }
    }
    setCopied(true);
    window.setTimeout(() => {
      setCopied(false);
      setMenu(null);
      setTranslation({ status: 'idle' });
    }, 800);
  };

  // 菜单上的"翻译"按钮：拉设置 → 调 LLM → 把结果放回菜单位置。
  // 翻译走"轻量模型"（不启用 thinking），由后端强制不携带 thinking 字段。
  const handleTranslate = async () => {
    if (!menu) return;
    const text = menu.anchorText.trim();
    setTranslation({ status: 'loading' });
    try {
      const settings = await getSettings();
      const result = await apiTranslateText({
        text,
        baseUrl: settings.baseUrl,
        lightweightModel: settings.lightweightModel,
        apiKey: settings.apiKey,
      });
      setTranslation({ status: 'success', text: result });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setTranslation({ status: 'error', message: msg });
    }
  };

  // 菜单上的"AI 讲解"按钮：从当前选区取出"局部上下文"（章节标题 + 前后窗），
  // 丢给父组件去调 LLM 并建笔记。这里不直接管笔记持久化，
  // 让 CategoryPage 用统一的 startAiAnnotate 流程保证笔记面板渲染一致。
  //
  // **不显示 loading 气泡**：父组件拿到占位笔记后立刻打开抽屉显示，
  // 流式 chunk 通过 Tauri 事件追加到笔记内容 —— 用户在抽屉里看到文字边生成边流入。
  // 这里只负责关菜单 + 把上下文扔出去，fire-and-forget。
  const handleAiAnnotate = () => {
    if (!menu || !onAiAnnotate) return;
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    const el = innerRef.current;
    if (!el || !el.contains(range.commonAncestorContainer)) return;
    const selectedText = sel.toString().trim();
    if (!selectedText) return;

    const ctx = extractAiContext(el, range);
    // 立即关菜单，避免遮挡笔记抽屉
    setMenu(null);
    setTranslation({ status: 'idle' });
    void onAiAnnotate({
      selectedText,
      contextBefore: ctx.contextBefore,
      contextAfter: ctx.contextAfter,
      sectionTitle: ctx.sectionTitle,
      anchorText: menu.anchorText,
      anchorOccurrence: menu.anchorOccurrence,
    }).catch((e) => console.error('AI explain failed', e));
  };

  return (
    <div className="relative h-full overflow-auto">
      <div
        className="prose prose-sm max-w-none p-6 dark:prose-invert"
        ref={innerRef}
        onContextMenu={handleContextMenu}
        onClick={handleClick}
        // comrak 输出受信任；Phase 2 不引入 DOMPurify
        dangerouslySetInnerHTML={{ __html: html }}
      />
      {menu && (
        <div
          data-selection-menu
          style={{
            position: 'fixed',
            top: menu.y,
            left: menu.x,
          }}
          // 容器本身接管 mousedown，避免被全局 close handler 当成"外部点击"误关
          onMouseDown={(e) => e.stopPropagation()}
          className="z-50 min-w-[160px] overflow-hidden rounded-lg border border-border bg-surface py-1 shadow-lg"
        >
          {translation.status === 'loading' && (
            <div className="flex items-center gap-2 px-3 py-2 text-xs text-text-muted">
              <Loader2 size={12} className="animate-spin" />
              翻译中...
            </div>
          )}
          {translation.status === 'success' && (
            <div className="px-3 py-2">
              <div className="mb-1 text-xs uppercase tracking-wider text-text-muted">
                翻译
              </div>
              <div className="max-w-md whitespace-pre-wrap text-sm text-text">
                {translation.text}
              </div>
              <button
                type="button"
                onClick={() => {
                  setMenu(null);
                  setTranslation({ status: 'idle' });
                }}
                className="mt-2 text-xs text-text-muted hover:text-text"
              >
                关闭
              </button>
            </div>
          )}
          {translation.status === 'error' && (
            <div className="px-3 py-2">
              <div className="mb-1 text-xs uppercase tracking-wider text-red-600">
                翻译失败
              </div>
              <div className="max-w-md whitespace-pre-wrap text-xs text-red-600">
                {translation.message}
              </div>
              <button
                type="button"
                onClick={() => {
                  setMenu(null);
                  setTranslation({ status: 'idle' });
                }}
                className="mt-2 text-xs text-text-muted hover:text-text"
              >
                关闭
              </button>
            </div>
          )}
          {translation.status === 'idle' && (
            <>
              {copied ? (
                <div className="flex items-center gap-2 px-3 py-2 text-sm text-text-muted">
                  <Check size={14} className="text-green-600" />
                  已复制
                </div>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => void handleCopy()}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-text hover:bg-surface-2"
                  >
                    <Copy size={14} className="text-text-muted" />
                    复制
                  </button>
                  <div className="my-1 border-t border-border" />
                  <button
                    type="button"
                    onClick={() => {
                      onAddNoteAtSelection?.({
                        anchorText: menu.anchorText,
                        anchorOccurrence: menu.anchorOccurrence,
                      });
                      setMenu(null);
                      setTranslation({ status: 'idle' });
                      window.getSelection()?.removeAllRanges();
                    }}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-text hover:bg-surface-2"
                  >
                    <MessageSquarePlus size={14} className="text-text-muted" />
                    添加批注
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleTranslate()}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-text hover:bg-surface-2"
                  >
                    <Languages size={14} className="text-text-muted" />
                    翻译
                  </button>
                  {onAiAnnotate && (
                    <>
                      <div className="my-1 border-t border-border" />
                      <button
                        type="button"
                        onClick={() => void handleAiAnnotate()}
                        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-text hover:bg-surface-2"
                      >
                        <Sparkles size={14} className="text-text-muted" />
                        AI 讲解
                      </button>
                    </>
                  )}
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
});