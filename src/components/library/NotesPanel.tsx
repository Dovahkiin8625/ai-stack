import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { MessageSquarePlus, Pencil, Trash2, X, Check, StickyNote, Sparkles, HelpCircle } from 'lucide-react';
import type { Note } from '../../types';
import { useNotesStore } from '../../stores/notes';

export interface NotesPanelHandle {
  focusNote: (noteId: number) => void;
}

interface Props {
  /** 点击笔记卡里的锚点摘要 → 通知父级去文章里定位 */
  onAnchorClick?: (noteId: number) => void;
}

/** 笔记面板定位：桌面 = 右侧固定栏；窄屏 = 底部抽屉。
 *  关闭态必须是 transform 移出视口，不能只改透明度 —— 否则面板会永久遮住正文。 */
export const PANEL_CLASSES = {
  base: 'fixed z-30 flex flex-col border-border bg-surface',
  desktop:
    'md:right-0 md:top-14 md:bottom-0 md:w-[320px] md:border-l md:shadow-[-4px_0_12px_rgba(0,0,0,0.04)]',
  mobile: 'max-md:inset-x-0 max-md:bottom-0 max-md:h-[70vh] max-md:rounded-t-2xl max-md:border-t',
  motion: 'transition-transform duration-200 ease-out motion-reduce:transition-none',
  open: 'translate-x-0',
  closed: 'max-md:translate-y-full pointer-events-none md:translate-x-full',
};

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    const pad = (n: number) => n.toString().padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  } catch {
    return iso;
  }
}

const NotesPanel = forwardRef<NotesPanelHandle, Props>(function NotesPanel({ onAnchorClick }, ref) {
  const notes = useNotesStore((s) => s.notes);
  const loading = useNotesStore((s) => s.loading);
  const create = useNotesStore((s) => s.create);
  const update = useNotesStore((s) => s.update);
  const remove = useNotesStore((s) => s.remove);
  const panelOpen = useNotesStore((s) => s.panelOpen);
  const setPanelOpen = useNotesStore((s) => s.setPanelOpen);

  const [composer, setComposer] = useState('');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingDraft, setEditingDraft] = useState('');
  const [focusedId, setFocusedId] = useState<number | null>(null);

  const composerRef = useRef<HTMLTextAreaElement>(null);
  const itemRefs = useRef<Map<number, HTMLLIElement>>(new Map());

  useImperativeHandle(ref, () => ({
    focusNote(id: number) {
      const el = itemRefs.current.get(id);
      if (!el) return;
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      setFocusedId(id);
      window.setTimeout(() => {
        setFocusedId((cur) => (cur === id ? null : cur));
      }, 1400);
    },
  }));

  // 抽屉打开时 focus composer
  useEffect(() => {
    if (panelOpen) composerRef.current?.focus();
  }, [panelOpen]);

  // Escape 关闭抽屉
  useEffect(() => {
    if (!panelOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setPanelOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [panelOpen, setPanelOpen]);

  const handleAdd = async () => {
    const content = composer.trim();
    if (!content) return;
    const note = await create({ content, anchorText: null, anchorOccurrence: 0 });
    if (note) {
      setComposer('');
      composerRef.current?.focus();
    }
  };

  const startEdit = (n: Note) => {
    setEditingId(n.id);
    setEditingDraft(n.content);
  };

  const commitEdit = async () => {
    if (editingId == null) return;
    const content = editingDraft.trim();
    if (!content) return;
    const target = notes.find((n) => n.id === editingId);
    if (!target) return;
    // AI 笔记被用户编辑后降级为用户笔记——不再展示 AI 标识。
    // 用户笔记保持不变，避免无意义的 source 字段写回。
    const source = target.source === 'ai' ? 'user' : undefined;
    await update(editingId, {
      content,
      anchorText: target.anchorText,
      anchorOccurrence: target.anchorOccurrence,
      source,
    });
    setEditingId(null);
    setEditingDraft('');
  };

  return (
    <aside
      aria-label="笔记"
      aria-hidden={!panelOpen}
      className={`${PANEL_CLASSES.base} ${PANEL_CLASSES.desktop} ${PANEL_CLASSES.mobile} ${PANEL_CLASSES.motion} ${
        panelOpen ? PANEL_CLASSES.open : PANEL_CLASSES.closed
      }`}
    >
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <StickyNote size={16} className="text-accent" />
        <span className="text-sm font-medium">笔记</span>
        <span className="ml-auto text-xs text-text-muted">
          {notes.length} 条
        </span>
        <button
          type="button"
          onClick={() => setPanelOpen(false)}
          aria-label="关闭笔记面板"
          className="rounded p-1 text-text-muted hover:bg-surface-2 hover:text-text"
        >
          <X size={14} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3">
        {loading && notes.length === 0 ? (
          <div className="text-sm text-text-muted">加载中...</div>
        ) : notes.length === 0 ? (
          <div className="rounded border border-dashed border-border p-4 text-center text-xs text-text-muted">
            暂无笔记。
            <br />
            在左侧文章中选中文字，或直接在下方输入。
          </div>
        ) : (
          <ul className="space-y-2">
            {notes.map((n) => {
              const isFocused = focusedId === n.id;
              return (
                <li
                  key={n.id}
                  ref={(el) => {
                    if (el) itemRefs.current.set(n.id, el);
                    else itemRefs.current.delete(n.id);
                  }}
                  className={isFocused ? 'note-flash' : undefined}
                >
                  <div className="group rounded-md border border-border bg-bg p-2 text-sm shadow-sm hover:border-accent/50">
                    {n.anchorText && (
                      <button
                        type="button"
                        onClick={() => onAnchorClick?.(n.id)}
                        className="mb-1 block w-full cursor-pointer rounded bg-yellow-100 px-1.5 py-1 text-left text-xs italic text-yellow-900 hover:bg-yellow-200 dark:bg-yellow-900/40 dark:text-yellow-100 dark:hover:bg-yellow-900/60"
                        title="跳转到正文位置"
                      >
                        “{n.anchorText}”
                      </button>
                    )}
                    {/* "询问 AI" 流程有用户问题 prompt；AI 讲解流程没有，prompt 为 null/undefined/空串。
                       只在 prompt 真的有内容时渲染"❓ 提问"引用块 —— 让回看时知道这条答案是回答什么问题的。 */}
                    {n.prompt && n.prompt.trim().length > 0 && (
                      <div
                        className="mb-1 rounded border-l-2 border-blue-300 bg-blue-50 px-2 py-1 text-xs text-blue-900 dark:border-blue-700 dark:bg-blue-900/30 dark:text-blue-100"
                        title="用户当时向 AI 提的问题"
                      >
                        <div className="mb-0.5 flex items-center gap-1 font-medium opacity-80">
                          <HelpCircle size={11} />
                          提问
                        </div>
                        <div className="whitespace-pre-wrap break-words">{n.prompt}</div>
                      </div>
                    )}
                    {editingId === n.id ? (
                      <textarea
                        value={editingDraft}
                        onChange={(e) => setEditingDraft(e.target.value)}
                        className="w-full resize-none rounded border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent"
                        rows={3}
                        autoFocus
                      />
                    ) : n.source === 'ai' && n.content === '' ? (
                      // AI 讲解占位卡片：在首个 chunk 到达前（含 extended thinking
                      // 与网络往返）显示友好动画，避免用户误以为卡住。
                      // 首个 appendToNote 写入非空内容后会自然切回正常 <p>。
                      <div
                        className="flex items-center gap-2 py-0.5 text-text-muted"
                        role="status"
                        aria-live="polite"
                      >
                        <Sparkles size={14} className="animate-pulse text-accent" />
                        <span className="text-xs">AI 正在讲解</span>
                        <span className="flex gap-0.5" aria-hidden>
                          <span className="h-1 w-1 animate-bounce rounded-full bg-text-muted [animation-delay:0ms]" />
                          <span className="h-1 w-1 animate-bounce rounded-full bg-text-muted [animation-delay:150ms]" />
                          <span className="h-1 w-1 animate-bounce rounded-full bg-text-muted [animation-delay:300ms]" />
                        </span>
                      </div>
                    ) : (
                      <p className="whitespace-pre-wrap break-words text-text">
                        {n.content}
                      </p>
                    )}
                    <div className="mt-1.5 flex items-center justify-between text-[11px] text-text-muted">
                      <div className="flex items-center gap-1.5">
                        <span>{formatTime(n.updatedAt)}</span>
                        {n.source === 'ai' && (
                          <span
                            className="inline-flex items-center gap-0.5 rounded bg-accent/10 px-1.5 py-0.5 text-[10px] font-medium text-accent"
                            title="由 AI 生成"
                          >
                            <Sparkles size={10} />
                            AI
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                        {editingId === n.id ? (
                          <>
                            <button
                              type="button"
                              onClick={commitEdit}
                              className="rounded p-1 hover:bg-surface-2"
                              title="保存"
                            >
                              <Check size={12} />
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                setEditingId(null);
                                setEditingDraft('');
                              }}
                              className="rounded p-1 hover:bg-surface-2"
                              title="取消"
                            >
                              <X size={12} />
                            </button>
                          </>
                        ) : (
                          <>
                            <button
                              type="button"
                              onClick={() => startEdit(n)}
                              className="rounded p-1 hover:bg-surface-2"
                              title="编辑"
                            >
                              <Pencil size={12} />
                            </button>
                            <button
                              type="button"
                              onClick={() => void remove(n.id)}
                              className="rounded p-1 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30"
                              title="删除"
                            >
                              <Trash2 size={12} />
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="border-t border-border p-3">
        <textarea
          ref={composerRef}
          value={composer}
          onChange={(e) => setComposer(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
              void handleAdd();
            }
          }}
          placeholder="写一条笔记（⌘/Ctrl + Enter 发送）"
          rows={2}
          className="w-full resize-none rounded-md border border-border bg-bg px-2 py-1.5 text-sm outline-none focus:border-accent"
        />
        <div className="mt-2 flex justify-end">
          <button
            type="button"
            onClick={() => void handleAdd()}
            disabled={!composer.trim()}
            className="inline-flex items-center gap-1 rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-white hover:bg-accent-hover disabled:opacity-40"
          >
            <MessageSquarePlus size={12} />
            添加笔记
          </button>
        </div>
      </div>
    </aside>
  );
});

export default NotesPanel;
