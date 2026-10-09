import { useMemo } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Settings as SettingsIcon, StickyNote, ChevronLeft, Menu } from 'lucide-react';
import ThemeToggle from '../ui/ThemeToggle';
import { useNotesStore } from '../../stores/notes';
import { useLibraryStore } from '../../stores/library';
import { useUiStore } from '../../stores/ui';
import { useIsMobile } from '../../lib/useIsMobile';
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

  const navigate = useNavigate();
  const navOpen = useUiStore((s) => s.navOpen);
  const toggleNav = useUiStore((s) => s.toggleNav);
  // 窄屏且打开了某篇文章时，返回按钮回到该子分类的列表
  const isMobile = useIsMobile();
  const articleRoute = /^\/library\/[^/]+\/[^/]+\/[^/]+/.test(pathname);
  const listRoute = articleRoute
    ? `/${pathname.split('/').slice(0, 4).join('/')}`
    : null;

  return (
    <header
      role="banner"
      className="relative z-40 flex h-14 shrink-0 items-center justify-between border-b border-border bg-surface px-4"
    >
      <div className="flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={toggleNav}
          aria-label={navOpen ? '关闭分类导航' : '打开分类导航'}
          aria-expanded={navOpen}
          aria-controls="primary-nav-drawer"
          className="-ml-2 rounded-md p-2 text-text-muted hover:bg-surface-2 md:hidden"
        >
          <Menu size={18} />
        </button>
        {isMobile && listRoute && (
          <button
            type="button"
            onClick={() => navigate(listRoute)}
            aria-label="返回列表"
            className="-ml-2 rounded-md p-2 text-text-muted hover:bg-surface-2 md:hidden"
          >
            <ChevronLeft size={18} />
          </button>
        )}
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
