import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
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
import CodeMirror from '@uiw/react-codemirror';
import { markdown as cmMarkdown, markdownLanguage } from '@codemirror/lang-markdown';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorView } from '@codemirror/view';
import type { Note } from '../../../types';
import {
  computeAnchor,
  extractContext,
  insertNoteMarkers,
  unwrapAnchors,
  type AnchorConfig,
} from '../../../lib/anchor';
import AskInputBox from './AskInputBox';
import SelectionMenu from './SelectionMenu';

export interface MarkdownReaderHandle {
  scrollToAnchor: (noteId: number) => void;
  /**
   * 当前是否有未保存的 markdown 编辑。CategoryPage 在用户切换到其它资源时
   * 先调一次，避免直接 selectResource 把 textarea 内容顶没了。
   * 只有 mode === 'edit' 且 draft 与磁盘上的 markdown 不同时才返回 true。
   */
  hasUnsavedChanges: () => boolean;
  /**
   * 编辑模式时拿到底层 CodeMirror EditorView，用于在编辑器里做程序化操作
   * （光标定位、插入 snippet）。当前主要被测试用来 dispatch transaction 模拟编辑
   * —— 因为 jsdom 下 contenteditable 的 input 事件 CodeMirror 不会自动同步 state。
   * 非编辑模式返回 null。
   */
  editorView: EditorView | null;
}

interface Props {
  html: string;
  /** 编辑模式 textarea 的初始值。资源切换 / 保存成功后由父组件更新。 */
  markdown: string;
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
    /** PDF 笔记用的 0-based 页码定位；markdown/DOCX 始终为 undefined。 */
    pageIdx?: number;
  }) => Promise<void>;
  /** 右键菜单"询问 AI"：把选区上下文 + 用户问题交给父组件，父组件去调 LLM 然后建笔记
   * （source='ai'，prompt=用户问题）。与 onAiAnnotate 平行：两者走同一个笔记抽屉/流式机制，
   * 但 prompt 是否为空决定了笔记卡片是否渲染"❓ 提问"引用块。 */
  onAiAsk?: (input: {
    selectedText: string;
    contextBefore: string;
    contextAfter: string;
    sectionTitle: string;
    anchorText: string;
    anchorOccurrence: number;
    question: string;
    /** PDF 笔记用的 0-based 页码定位；markdown/DOCX 始终为 undefined。 */
    pageIdx?: number;
  }) => Promise<void>;
  /**
   * 保存按钮点击时调：把当前 draft 交给父组件，父组件调 write_resource 并把
   * 返回的新 ResourceContent updateResourceContent 进 store。Promise 失败时
   * MarkdownReader 显示错误条并不切回预览（让用户能继续编辑）。
   */
  onSaveMarkdown: (markdown: string) => Promise<void>;
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
 * Markdown reader 用 anchor.ts 的"原子单元"配置：
 * - .katex 公式整体一个单元
 * - <code>（行内 / 代码块共用）一个单元；marker 插在 <pre> 后或 <code> 后
 * - occurrenceSkipSelector 让 plain text occurrence 不被代码块/公式副本干扰
 */
const MARKDOWN_CONFIG: AnchorConfig = {
  atoms: [
    {
      matcher: (el) => el.matches('.katex'),
      textFrom: (el) => (el.querySelector('.katex-html')?.textContent ?? '').trim(),
    },
    {
      matcher: (el) => el.matches('code'),
      textFrom: (el) => (el.textContent ?? '').replace(/\s+/g, ' ').trim(),
      insertTarget: (el) => el.closest('pre') ?? el,
    },
  ],
  occurrenceSkipSelector: '.katex-mathml, pre, code',
};

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

function decodeEntities(s: string): string {
  return s
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

export default forwardRef<MarkdownReaderHandle, Props>(function MarkdownReader(
  { html, markdown, notes = [], onMarkClick, onAddNoteAtSelection, onAiAnnotate, onAiAsk, onSaveMarkdown },
  ref,
) {
  const innerRef = useRef<HTMLDivElement>(null);
  /** 右键唤起的上下文菜单：用户选中文本后右键打开。
   *  SelectionMenu 自身管翻译状态机 + outside-click close，父组件只持有位置和 anchor。 */
  const [menu, setMenu] = useState<{
    x: number;
    y: number;
    anchorText: string;
    anchorOccurrence: number;
  } | null>(null);

  // === 编辑模式状态 ===
  // mode 决定渲染预览（HTML）还是编辑（textarea）。draft 是 textarea 的当前值。
  // 切到编辑时初始化为 markdown prop；markdown prop 变化（资源切换 / 保存后父组件换 store）
  // 时由下面的 useEffect 重置 draft + 强制回 preview。
  const [mode, setMode] = useState<'preview' | 'edit'>('preview');
  const [draft, setDraft] = useState<string>(markdown);
  const [confirmDiscard, setConfirmDiscard] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  // dirty 用 ref 而不是 state —— 只在渲染时需要（"保存"按钮显示），
  // hasUnsavedChanges imperative API 也只需要读它一次，不必触发渲染
  const dirtyRef = useRef<boolean>(false);
  const draftRef = useRef<string>(markdown);

  /** "询问 AI" 浮层输入框的位置。null 表示未打开。
   *  - 打开流程：右键菜单 → 点"询问 AI" → 关闭菜单 → 在原菜单位置显示 AskInputBox
   *  - 关闭流程：提交后（onSubmit）/ 取消按钮 / Esc / 外部点击
   *  同时缓存 anchor 上下文（选中文本 + 章节 + 前后窗 + anchor 信息），
   *  提交时直接复用，不再依赖 window.getSelection() —— 浮层聚焦到输入框后
   *  原选区可能已被浏览器清掉，重新取会拿到空。 */
  const [askBox, setAskBox] = useState<
    | {
        x: number;
        y: number;
        selectedText: string;
        contextBefore: string;
        contextAfter: string;
        sectionTitle: string;
        anchorText: string;
        anchorOccurrence: number;
      }
    | null
  >(null);

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
    hasUnsavedChanges() {
      return mode === 'edit' && dirtyRef.current;
    },
    get editorView() {
      // 预览模式 / 还没 mount 时返回 null —— 调用前应先确认 mode === 'edit'。
      // 这里返回 getter 让组件重渲染时也拿到最新 view
      return mode === 'edit' ? editorViewRef.current : null;
    },
  }));

  // markdown prop 变化（资源切换 / 保存成功后父组件 updateResourceContent）
  // → 重置 draft、强制回 preview、清掉错误。这是单一同步点 —— 父组件换 store
  // 后本组件就"从干净状态开始"，draft = 新版 markdown，dirty = false。
  useEffect(() => {
    draftRef.current = markdown;
    setDraft(markdown);
    dirtyRef.current = false;
    setMode('preview');
    setConfirmDiscard(false);
    setSaveError(null);
    setSaving(false);
    // 切到预览 → 清掉 EditorView 引用。CodeMirror 这时已经 unmounted，
    // 留着旧 view 会导致 imperative 调用命中已销毁的 EditorView 抛错。
    editorViewRef.current = null;
  }, [markdown]);

  // 文章内容（html）变化或模式切回预览时：跑一次完整的渲染管道
  // - 解包旧锚点 / 处理 math 块 / 高亮代码 / 插入新角标 / 内联 math
  //
  // 依赖加 mode 的原因 —— 之前 bug：切到编辑再切回预览时，html prop 没变 → 管道不重跑，
  // 预览 div 是 React 全新 mount 的空 DOM，$...$ / ```math 等后处理全没了。
  // 现在 mode 进入 'preview' 也触发管道跑一次，覆盖 edit → preview 切换。
  useEffect(() => {
    if (mode !== 'preview') return;
    const el = innerRef.current;
    if (!el) return;
    unwrapAnchors(el);
    renderMathBlocks(el);
    highlightCodeBlocks(el);
    insertNoteMarkers(el, notes, MARKDOWN_CONFIG);
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
  }, [html, mode]);

  // 笔记变化时：只重做角标插入，不动 math/code/inlineMath
  useEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    unwrapAnchors(el);
    insertNoteMarkers(el, notes, MARKDOWN_CONFIG);
  }, [notes]);

  // 右键唤起上下文菜单：选中文本后右键出现"添加批注 / 翻译"。
  // 没选中文本则不 preventDefault，让浏览器显示默认菜单。
  // 选区起点在 .katex / <code> 内时走 computeAnchor 的 atom 路径，
  // 否则走 plain text 路径——anchor 计算逻辑统一在 anchor.ts 里。
  const handleContextMenu = (e: React.MouseEvent) => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    const el = innerRef.current;
    if (!el || !el.contains(range.commonAncestorContainer)) return;

    const text = range.toString().trim();
    if (text.length < 2 || text.length > 500) return;

    const { anchorText, anchorOccurrence } = computeAnchor(el, range, MARKDOWN_CONFIG);
    if (!anchorText) return;

    e.preventDefault();
    setMenu({ x: e.clientX, y: e.clientY, anchorText, anchorOccurrence });
  };

  const handleClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement | null;
    const marker = target?.closest('sup.note-marker') as HTMLElement | null;
    if (!marker) return;
    const id = Number(marker.dataset.noteId);
    if (!Number.isFinite(id)) return;
    onMarkClick?.(id);
  };

  // 菜单上的"AI 讲解"按钮：从当前选区取出"局部上下文"（章节标题 + 前后窗），
  // 丢给父组件去调 LLM 并建笔记。这里不直接管笔记持久化，
  // 让 CategoryPage 用统一的 startAiAnnotate 流程保证笔记面板渲染一致。
  //
  // **不显示 loading 气泡**：父组件拿到占位笔记后立刻打开抽屉显示，
  // 流式 chunk 通过 Tauri 事件追加到笔记内容 —— 用户在抽屉里看到文字边生成边流入。
  // 这里只负责把上下文扔出去，fire-and-forget。菜单本身已被 SelectionMenu.onClose 关闭。
  const handleAiAnnotate = () => {
    if (!onAiAnnotate) return;
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    const el = innerRef.current;
    if (!el || !el.contains(range.commonAncestorContainer)) return;
    const selectedText = sel.toString().trim();
    if (!selectedText) return;

    const ctx = extractContext(el, range);
    void onAiAnnotate({
      selectedText,
      contextBefore: ctx.contextBefore,
      contextAfter: ctx.contextAfter,
      sectionTitle: ctx.sectionTitle,
      anchorText: menu!.anchorText,
      anchorOccurrence: menu!.anchorOccurrence,
      // markdown 没有 pageIdx
    }).catch((e) => console.error('AI explain failed', e));
  };

  // 右键菜单的"询问 AI"：关菜单 → 弹出 mini 输入框 → 用户输入问题 → 提交。
  // 在打开瞬间把选区上下文全部缓存进 askBox —— 浮层聚焦后浏览器会清掉原 selection，
  // 提交时再取就拿不到了，所以一次性把 selection 衍生信息全存好。
  const handleAiAskOpen = () => {
    if (!onAiAsk || !menu) return;
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    const el = innerRef.current;
    if (!el || !el.contains(range.commonAncestorContainer)) return;
    const selectedText = sel.toString().trim();
    if (!selectedText) return;

    const ctx = extractContext(el, range);
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

  // 用户提交问题：把缓存的选区上下文 + 问题一起交给父组件去调 LLM 建笔记。
  const handleAiAskSubmit = (question: string) => {
    if (!onAiAsk || !askBox) return;
    setAskBox(null);
    void onAiAsk({
      selectedText: askBox.selectedText,
      contextBefore: askBox.contextBefore,
      contextAfter: askBox.contextAfter,
      sectionTitle: askBox.sectionTitle,
      anchorText: askBox.anchorText,
      anchorOccurrence: askBox.anchorOccurrence,
      question,
      // markdown 没有 pageIdx
    }).catch((e) => console.error('AI ask failed', e));
  };

  const handleAiAskCancel = () => {
    setAskBox(null);
  };

  // 浮层打开时挂全局监听：点击 AskInputBox 外部 / Esc 关闭浮层。
  // AskInputBox 自身接管 mousedown（stopPropagation），所以这里只关心"外部点击"。
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
    // setTimeout 0 跳过触发本次打开的 mousedown，避免"刚开就被关"
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

  // === 编辑模式 handlers ===
  const handleEditClick = () => {
    // 从预览切到编辑：把 draft 同步成 markdown prop（首次进入；或保存后 props 已变）
    // —— 避免 textarea 看到陈旧值
    draftRef.current = markdown;
    setDraft(markdown);
    dirtyRef.current = false;
    setConfirmDiscard(false);
    setSaveError(null);
    setMode('edit');
  };

  const handlePreviewClick = () => {
    if (dirtyRef.current) {
      // dirty 状态点预览不直接切：显示确认条，让用户主动选"放弃"还是"继续编辑"。
      // 这是保护未保存编辑的唯一护栏（不要静默丢弃）。
      setConfirmDiscard(true);
      return;
    }
    // 非 dirty：直接切回。draft 跟 markdown 一致，没东西可丢。
    setMode('preview');
  };

  const handleDraftChange = (next: string) => {
    // @uiw/react-codemirror 把当前编辑器内容（完整 doc 字符串）作为第一参；
    // 跟 textarea 的 onChange(e.target.value) 等价。CodeMirror 每次 transaction
    // （输入 / undo / 选区替换）都触发 —— debounce 在框架层就够了。
    draftRef.current = next;
    setDraft(next);
    dirtyRef.current = next !== markdown;
  };

  const handleDiscardConfirm = () => {
    // 放弃修改：清掉 draft + dirty，回到预览。markdown prop 未变，下一次点编辑会重新初始化。
    draftRef.current = markdown;
    setDraft(markdown);
    dirtyRef.current = false;
    setConfirmDiscard(false);
    setSaveError(null);
    setMode('preview');
  };

  const handleContinueEditing = () => {
    setConfirmDiscard(false);
  };

  const handleSaveClick = async () => {
    if (saving) return;
    setSaving(true);
    setSaveError(null);
    try {
      await onSaveMarkdown(draftRef.current);
      // 保存成功后父组件会 updateResourceContent → markdown prop 变 → useEffect 重置 draft/mode。
      // 这里不主动切预览，否则会出现"先切预览（用旧 html）→ props 变 → useEffect 再切预览"的两次切换。
    } catch (e) {
      // 失败：留在编辑模式（切回预览会丢 draft），显示错误条
      const msg = e instanceof Error ? e.message : String(e);
      setSaveError(msg);
    } finally {
      setSaving(false);
    }
  };

  // dirty 是从 draft / markdown 派生的，每渲染重算 —— 比维护 ref 更直观，
  // 且父组件渲染代价可忽略（mode toggle 重渲染不频繁）
  const dirty = draft !== markdown;

  // 持有 EditorView 引用以便 imperative handle 暴露（测试用 + 未来按需程序化操作）
  const editorViewRef = useRef<EditorView | null>(null);

  // CodeMirror 编辑器扩展：markdown 语言 + 跟随当前主题（light/dark）。
  // - 跟当前 document.documentElement.dark class 走（theme store 同步切换 html.dark），
  //   这里只读 classList，避免在 MarkdownReader 里订阅主题 store 引入额外依赖
  // - 用 useMemo 防止每次 render 都重建扩展数组（会触发 CodeMirror 重建 EditorState）
  const isDark = document.documentElement.classList.contains('dark');
  const editorExtensions = useMemo(
    () => [
      // markdown() 启用了完整 markdown 高亮（headings / bold / italic / code / links / lists）
      // markdownLanguage 提供代码块内的语法识别 —— 但不自动嵌套 hljs，需要 codeMirrorLanguageData
      // 注意：必须 alias 为 cmMarkdown —— 本组件 props 里有同名参数 markdown，shadow 后会拿不到 import
      cmMarkdown({ base: markdownLanguage, codeLanguages: () => null }),
      isDark ? oneDark : [],
      EditorView.lineWrapping,
      EditorView.theme({
        '&': { height: '100%', fontSize: '13px' },
        '.cm-scroller': { fontFamily: 'inherit' },
        '.cm-content': { padding: '16px' },
      }),
    ],
    [isDark],
  );

  return (
    <div className="relative flex h-full flex-col">
      {/* Toolbar：模式切换 pill + 保存按钮。
          sticky 在 reader 顶部，滚动正文时 toolbar 不消失，方便随时切模式 / 保存。 */}
      <div
        data-md-toolbar
        className="sticky top-0 z-10 flex items-center gap-2 border-b border-border bg-surface px-3"
      >
        <div className="flex items-center rounded-md border border-border bg-surface-2 p-0.5 text-xs">
          <button
            type="button"
            onClick={handlePreviewClick}
            aria-pressed={mode === 'preview'}
            className={`rounded px-3 py-1 transition-colors ${
              mode === 'preview'
                ? 'bg-surface text-text shadow-sm'
                : 'text-text-muted hover:text-text'
            }`}
          >
            预览
          </button>
          <button
            type="button"
            onClick={handleEditClick}
            aria-pressed={mode === 'edit'}
            className={`rounded px-3 py-1 transition-colors ${
              mode === 'edit'
                ? 'bg-surface text-text shadow-sm'
                : 'text-text-muted hover:text-text'
            }`}
          >
            编辑
          </button>
        </div>
        {/* 保存按钮：编辑模式始终可见（discoverability 优先）；dirty=false 时 disabled，
            dirty=true 时可点，保存中禁用 + loading 文字。 */}
        {mode === 'edit' && (
          <button
            type="button"
            onClick={handleSaveClick}
            disabled={saving || !dirty}
            className="rounded-md border border-border bg-surface-2 px-3 py-1 text-xs text-text hover:bg-surface disabled:opacity-50"
          >
            {saving ? '保存中…' : '保存'}
          </button>
        )}
        {saveError && (
          <div
            role="alert"
            className="ml-2 rounded-md border border-red-300 bg-red-50 px-2 py-1 text-xs text-red-700"
          >
            保存失败：{saveError}
          </div>
        )}
      </div>

      {/* dirty 状态下点预览 → 确认条，不直接切。
          浮在 toolbar 下方，避免遮挡正文滚动。 */}
      {confirmDiscard && (
        <div
          role="dialog"
          aria-label="放弃未保存的修改"
          className="sticky top-[41px] z-10 flex items-center gap-3 border-b border-border bg-amber-50 px-3 py-2 text-xs text-amber-900"
        >
          <span>当前有未保存的修改，是否放弃？</span>
          <button
            type="button"
            onClick={handleContinueEditing}
            className="rounded border border-amber-300 bg-white px-2 py-0.5 hover:border-amber-500"
          >
            继续编辑
          </button>
          <button
            type="button"
            onClick={handleDiscardConfirm}
            className="rounded border border-red-300 bg-white px-2 py-0.5 text-red-700 hover:border-red-500"
          >
            放弃修改
          </button>
        </div>
      )}

      <div className="relative flex-1 overflow-auto">
        {mode === 'preview' ? (
          <div
            className="prose prose-sm max-w-none p-6 dark:prose-invert"
            ref={innerRef}
            onContextMenu={handleContextMenu}
            onClick={handleClick}
            // comrak 输出受信任；Phase 2 不引入 DOMPurify
            dangerouslySetInnerHTML={{ __html: html }}
          />
        ) : (
          <CodeMirror
            value={draft}
            // onChange 在 CodeMirror 内部每次 transaction 后触发；
            // 这里同步 draft / dirty 状态。
            onChange={handleDraftChange}
            extensions={editorExtensions}
            // 把 EditorView 存到 ref —— imperative handle 通过 getter 暴露
            // （测试需要 dispatch transaction；未来也可以用来程序化插入内容）
            onCreateEditor={(view) => {
              editorViewRef.current = view;
            }}
            basicSetup={{
              lineNumbers: true,
              foldGutter: false,
              highlightActiveLine: true,
              highlightSelectionMatches: false,
            }}
            theme="none"
            // 让 .cm-editor 填满 flex-1 容器（CodeMirror 默认按内容高度走，会在 sticky toolbar 容器里塌陷）
            className="h-full"
          />
        )}
        {menu && (
          <SelectionMenu
            x={menu.x}
            y={menu.y}
            anchorText={menu.anchorText}
            anchorOccurrence={menu.anchorOccurrence}
            onAddNote={(input) => {
              onAddNoteAtSelection?.(input);
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
    </div>
  );
});