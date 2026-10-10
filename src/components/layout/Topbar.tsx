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
  const isMobile = useIsMobile();
  // 文章页需要返回按钮（回到它所在的子分类列表）
  const articleRoute = /^\/library\/[^/]+\/[^/]+\/[^/]+/.test(pathname);
  // 窄屏下任何非根级路由都需要一个返回按钮。navigate(-1) 走浏览器历史，
  // 历史为空时停留在原地——比硬编码回 /library 更符合用户预期。
  const canGoBack = articleRoute || pathname === '/settings';

  return (
    <header
      role="banner"
      className="relative z-40 flex min-h-14 shrink-0 items-center justify-between border-b border-border bg-surface px-4 pt-[env(safe-area-inset-top)]"
    >
      <div className="flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={toggleNav}
          aria-label={navOpen ? '关闭分类导航' : '打开分类导航'}
          aria-expanded={navOpen}
          aria-controls="primary-nav-drawer"
          className="-ml-2 rounded-md p-2 text-text-muted hover:bg-surface-2 active:bg-surface-2/70 md:hidden"
        >
          <Menu size={18} />
        </button>
        {isMobile && canGoBack && (
          <button
            type="button"
            onClick={() => navigate(-1)}
            aria-label={articleRoute ? '返回列表' : '返回'}
            className="-ml-2 rounded-md p-2 text-text-muted hover:bg-surface-2 active:bg-surface-2/70 md:hidden"
          >
            <ChevronLeft size={18} />
          </button>
        )}
        {article ? (
          <>
            {/* 类型徽章在小屏也保留（只占 ~50px），标题/页数/字数挪到正文区，避免顶栏过挤 */}
            <span className="shrink-0 rounded bg-surface-2 px-2 py-0.5 text-xs font-semibold uppercase tracking-wider text-text-muted">
              {TYPE_LABELS[article.type]}
            </span>
            <span className="hidden truncate text-base font-semibold md:inline">
              {article.title}
            </span>
            {article.pageCount != null && (
              <span className="hidden shrink-0 text-xs text-text-muted md:inline">
                {article.pageCount} 页
              </span>
            )}
            {article.wordCount != null && (
              <span className="hidden shrink-0 text-xs text-text-muted md:inline">
                {article.wordCount} 字
              </span>
            )}
          </>
        ) : (
          <h1 className="hidden truncate text-lg font-semibold md:block">
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
              : 'text-text-muted hover:bg-surface-2 active:bg-surface-2/70 hover:text-text'
          }`}
        >
          <StickyNote size={18} />
        </button>

        <ThemeToggle />

        <Link
          to="/settings"
          aria-label="设置"
          className="rounded-md p-2 text-text-muted hover:bg-surface-2 active:bg-surface-2/70 hover:text-text"
        >
          <SettingsIcon size={18} />
        </Link>
      </div>
    </header>
  );
}
