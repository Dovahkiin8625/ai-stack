import { forwardRef, useImperativeHandle, useRef, useState } from 'react';
import { MessageSquarePlus, Pencil, Trash2, X, Check, StickyNote } from 'lucide-react';
import type { Note } from '../../types';
import { useNotesStore } from '../../stores/notes';

export interface NotesPanelHandle {
  focusNote: (noteId: number) => void;
}

interface Props {
  /** 点击笔记卡里的锚点摘要 → 通知父级去文章里定位 */
  onAnchorClick?: (noteId: number) => void;
}

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
    await update(editingId, {
      content,
      anchorText: target.anchorText,
      anchorOccurrence: target.anchorOccurrence,
    });
    setEditingId(null);
    setEditingDraft('');
  };

  return (
    <div className="flex h-full w-[320px] shrink-0 flex-col border-l border-border bg-surface">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <StickyNote size={16} className="text-accent" />
        <span className="text-sm font-medium">笔记</span>
        <span className="ml-auto text-xs text-text-muted">
          {notes.length} 条
        </span>
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
                    {editingId === n.id ? (
                      <textarea
                        value={editingDraft}
                        onChange={(e) => setEditingDraft(e.target.value)}
                        className="w-full resize-none rounded border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent"
                        rows={3}
                        autoFocus
                      />
                    ) : (
                      <p className="whitespace-pre-wrap break-words text-text">
                        {n.content}
                      </p>
                    )}
                    <div className="mt-1.5 flex items-center justify-between text-[11px] text-text-muted">
                      <span>{formatTime(n.updatedAt)}</span>
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
    </div>
  );
});

export default NotesPanel;
