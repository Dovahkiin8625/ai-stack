import { useMemo } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Settings as SettingsIcon, StickyNote } from 'lucide-react';
import ThemeToggle from '../ui/ThemeToggle';
import { useNotesStore } from '../../stores/notes';
import { useLibraryStore } from '../../stores/library';
import type { ResourceType } from '../../types';

const STATIC_TITLES: Record<string, string> = {
  '/settings': '设置',
};

const TYPE_LABELS: Record<ResourceType, string> = {
  markdown: 'Markdown',
  pdf: 'PDF',
  docx: 'Word',
  pptx: 'PPT',
};

export default function Topbar() {
  const { pathname } = useLocation();

  const selectedResourceId = useLibraryStore((s) => s.selectedResourceId);
  const articlesByPath = useLibraryStore((s) => s.articlesByPath);

  // 从 store 里找出当前打开的文章：标题、类型、统计都来自同一个 Resource，
  // 避免上提到 store 增加同步成本。
  const article = useMemo(() => {
    if (selectedResourceId == null) return null;
    for (const list of Object.values(articlesByPath)) {
      const hit = list.find((r) => r.id === selectedResourceId);
      if (hit) return hit;
    }
    return null;
  }, [selectedResourceId, articlesByPath]);

  const panelOpen = useNotesStore((s) => s.panelOpen);
  const togglePanel = useNotesStore((s) => s.togglePanel);

  return (
    <header
      role="banner"
      className="relative z-40 flex h-14 shrink-0 items-center justify-between border-b border-border bg-surface px-4"
    >
      <div className="flex min-w-0 items-center gap-3">
        {article ? (
          <>
            <span className="shrink-0 rounded bg-surface-2 px-2 py-0.5 text-xs font-semibold uppercase tracking-wider text-text-muted">
              {TYPE_LABELS[article.type]}
            </span>
            <span className="truncate text-base font-semibold">
              {article.title}
            </span>
            {article.pageCount != null && (
              <span className="shrink-0 text-xs text-text-muted">
                {article.pageCount} 页
              </span>
            )}
            {article.wordCount != null && (
              <span className="shrink-0 text-xs text-text-muted">
                {article.wordCount} 字
              </span>
            )}
          </>
        ) : (
          <h1 className="truncate text-lg font-semibold">
            {STATIC_TITLES[pathname] ?? ''}
          </h1>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-2">
        <button
          type="button"
          onClick={togglePanel}
          aria-label={panelOpen ? '关闭笔记面板' : '打开笔记面板'}
          aria-pressed={panelOpen}
          className={`rounded-md p-2 transition-colors ${
            panelOpen
              ? 'bg-accent/10 text-accent'
              : 'text-text-muted hover:bg-surface-2 hover:text-text'
          }`}
        >
          <StickyNote size={18} />
        </button>

        <ThemeToggle />

        <Link
          to="/settings"
          aria-label="设置"
          className="rounded-md p-2 text-text-muted hover:bg-surface-2 hover:text-text"
        >
          <SettingsIcon size={18} />
        </Link>
      </div>
    </header>
  );
}
