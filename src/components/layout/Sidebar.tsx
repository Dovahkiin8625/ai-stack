import { useEffect, useMemo, useState } from 'react';
import { NavLink, useLocation, useMatch } from 'react-router-dom';
import {
  BookOpen, Layers, MessageSquareText, Eye, Languages, Mic,
  Wrench, Bot, Shield, Sparkles, Package, TrendingUp,
  ChevronRight, ChevronDown,
  FileText, FileType, Presentation, RefreshCw, CloudDownload,
} from 'lucide-react';
import type { ComponentType } from 'react';
import { iconForCategoryPath, type LucideIconName } from '../../data/categories';
import { useLibraryStore } from '../../stores/library';
import { useUiStore } from '../../stores/ui';
import type { Category, ResourceType } from '../../types';

const TOP_ICON_MAP: Record<LucideIconName, ComponentType<{ size?: number }>> = {
  BookOpen, Layers, MessageSquareText, Eye, Languages, Mic,
  Wrench, Bot, Shield, Sparkles, Package, TrendingUp,
};

const RESOURCE_ICONS: Record<ResourceType, ComponentType<{ size?: number }>> = {
  markdown: FileText,
  pdf: BookOpen,
  docx: FileType,
  pptx: Presentation,
};

export default function Sidebar() {
  const location = useLocation();
  // Match /library/:category, /library/:category/:subPath, /library/:category/:subPath/:article
  const libraryMatch = useMatch('/library/:category/*');
  const activeCategory = libraryMatch?.params.category ?? null;
  const activeSubSlug = (() => {
    if (!libraryMatch) return null;
    const rest = location.pathname.replace(/^\/library\/[^/]+/, '').replace(/^\//, '');
    if (!rest) return null;
    return rest.split('/')[0] || null;
  })();
  const activeArticleId = (() => {
    const m = location.pathname.match(/^\/library\/[^/]+\/[^/]+\/(\d+)/);
    return m ? Number(m[1]) : null;
  })();
  const activeSubPath =
    activeCategory && activeSubSlug ? `${activeCategory}/${activeSubSlug}` : null;

  // DB 驱动 —— scanner 从 resources/knowledge/ 实际目录 + _index.md 派生
  const categories = useLibraryStore((s) => s.categories);
  const articlesByPath = useLibraryStore((s) => s.articlesByPath);
  const loadArticles = useLibraryStore((s) => s.loadArticles);
  // 启动时第一次扫描仍在跑（status='scanning'），但侧栏已经用缓存分类渲染出来 —— 这时给个低调指示。
  // 首次启动 DB 空时不显示 —— 侧栏本就空，加 badge 反显突兀。
  const status = useLibraryStore((s) => s.status);
  const showSyncBadge = status === 'scanning' && categories.length > 0;

  /**
   * 顶层分类 = parentPath 为 null 的（src-tauri/src/scanner.rs 里也是这样写的）。
   * 子分类按 parentPath 索引，避免每个顶层循环里再 O(n) 查找。
   */
  const { topCategories, childrenByParent } = useMemo(() => {
    const tops: Category[] = [];
    const byParent = new Map<string, Category[]>();
    for (const c of categories) {
      if (c.parentPath === null) tops.push(c);
      else {
        const arr = byParent.get(c.parentPath) ?? [];
        arr.push(c);
        byParent.set(c.parentPath, arr);
      }
    }
    return { topCategories: tops, childrenByParent: byParent };
  }, [categories]);

  const [expanded, setExpanded] = useState<Set<string>>(() => {
    const initial = new Set<string>();
    if (activeCategory) initial.add(activeCategory);
    if (activeSubPath) initial.add(activeSubPath);
    return initial;
  });

  // URL 变化时自动展开 active category + active subcategory（修复重构后深度链接不展开的回归）
  useEffect(() => {
    if (!activeCategory) return;
    setExpanded((prev) => {
      let changed = false;
      const next = new Set(prev);
      if (!next.has(activeCategory)) {
        next.add(activeCategory);
        changed = true;
      }
      if (activeSubPath && !next.has(activeSubPath)) {
        next.add(activeSubPath);
        changed = true;
      }
      return changed ? next : prev;
    });
  }, [activeCategory, activeSubPath]);

  // 懒加载：所有已展开但尚未缓存的子分类
  useEffect(() => {
    for (const path of expanded) {
      if (!path.includes('/')) continue; // 顶层 category 不调
      if (articlesByPath[path] === undefined) {
        void loadArticles(path);
      }
    }
  }, [expanded, articlesByPath, loadArticles]);

  const toggleExpand = (path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const navOpen = useUiStore((s) => s.navOpen);
  const closeNav = useUiStore((s) => s.closeNav);
  // 路由变化时关闭抽屉 —— 抽屉里点完分类，立刻看到正文
  useEffect(() => {
    useUiStore.getState().closeNav();
  }, [location.pathname]);

  // Escape 关闭抽屉 —— 抽屉打开了却只接受点击遮罩或再次点击汉堡，
  // 等于把键盘用户堵死。open 期间挂 keydown、关闭即解绑。
  useEffect(() => {
    if (!navOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeNav();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [navOpen, closeNav]);

  // 抽屉里点同一个 active 分类时，pathname 不变 → 上面的 useEffect 不触发；
  // 但用户预期是"选了就走"。在 <nav> 上捕一下，点 <a> 时也关闭抽屉。
  const handleNavClick = (e: React.MouseEvent<HTMLElement>) => {
    const target = e.target as HTMLElement | null;
    if (target && target.closest('a')) closeNav();
  };

  return (
    <>
      <aside
        id="primary-nav-drawer"
        aria-label="知识分类导航"
        className={`flex h-full w-72 shrink-0 flex-col border-r border-border bg-surface
          transition-transform duration-200 ease-out motion-reduce:transition-none
          max-md:fixed max-md:top-14 max-md:bottom-0 max-md:left-0 max-md:z-50
          ${navOpen ? 'max-md:translate-x-0' : 'max-md:-translate-x-full'}`}
      >
        <div className="flex h-14 items-center px-4 border-b border-border">
          <span className="text-base font-semibold tracking-tight">AI Stack</span>
        </div>
        {showSyncBadge && (
          <div
            className="flex items-center gap-1.5 border-b border-border px-4 py-1.5 text-xs text-text-muted"
            aria-live="polite"
          >
            <RefreshCw size={11} className="animate-spin" />
            <span>正在同步索引…</span>
          </div>
        )}
        <nav className="flex-1 overflow-y-auto p-2" onClick={handleNavClick}>
          <ul className="space-y-0.5">
            {topCategories.map((cat) => {
              const iconName = iconForCategoryPath(cat.path);
              const Icon = iconName ? TOP_ICON_MAP[iconName] : undefined;
              const isActive = activeCategory === cat.path;
              const isOpen = expanded.has(cat.path);
              const children = childrenByParent.get(cat.path) ?? [];
              const hasChildren = children.length > 0;
              return (
                <li key={cat.path}>
                  <div
                    className={`group flex items-center rounded-md text-sm transition-colors ${
                      isActive
                        ? 'bg-accent/10 text-accent'
                        : 'text-text hover:bg-surface-2'
                    }`}
                  >
                    <NavLink
                      to={`/library/${cat.path}`}
                      className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1.5"
                    >
                      {Icon && <Icon size={16} />}
                      <span className="truncate">{cat.title}</span>
                    </NavLink>
                    {hasChildren && (
                      <button
                        type="button"
                        aria-label={isOpen ? '折叠子分类' : '展开子分类'}
                        aria-expanded={isOpen}
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          toggleExpand(cat.path);
                        }}
                        className="mr-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-text-muted hover:bg-surface-2 hover:text-text"
                      >
                        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                      </button>
                    )}
                  </div>
                  {hasChildren && isOpen && (
                    <ul className="ml-2 mt-0.5 space-y-0.5 border-l border-border pl-2">
                      {children.map((sub) => {
                        // sub.path 形如 "02-deep-learning/transformers"，路由里只取尾段
                        const subSlug = sub.path.slice(cat.path.length + 1);
                        const isSubOpen = expanded.has(sub.path);
                        const isSubActive =
                          activeCategory === cat.path && activeSubSlug === subSlug;
                        const subArticles = articlesByPath[sub.path];
                        return (
                          <li key={sub.path}>
                            <div className="group flex items-center rounded-md text-sm">
                              <NavLink
                                to={`/library/${cat.path}/${subSlug}`}
                                className={() =>
                                  `flex min-w-0 flex-1 items-center rounded-md px-2 py-1 transition-colors ${
                                    isSubActive
                                      ? 'bg-accent/10 text-accent'
                                      : 'text-text-muted hover:bg-surface-2 hover:text-text'
                                  }`
                                }
                              >
                                <span className="truncate">{sub.title}</span>
                              </NavLink>
                              <button
                                type="button"
                                aria-label={isSubOpen ? '折叠文章列表' : '展开文章列表'}
                                aria-expanded={isSubOpen}
                                onClick={() => toggleExpand(sub.path)}
                                className="mr-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-text-muted hover:bg-surface-2 hover:text-text"
                              >
                                {isSubOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                              </button>
                            </div>
                            {isSubOpen && (
                              <ul className="ml-2 mt-0.5 space-y-0.5 border-l border-border pl-2">
                                {subArticles === undefined ? (
                                  <li className="px-2 py-1 text-xs text-text-muted">
                                    加载中…
                                  </li>
                                ) : subArticles.length === 0 ? (
                                  <li className="px-2 py-1 text-xs text-text-muted">
                                    暂无文章
                                  </li>
                                ) : (
                                  subArticles.map((r) => {
                                    const AIcon = RESOURCE_ICONS[r.type];
                                    const isArticleActive = activeArticleId === r.id;
                                    return (
                                      <li key={r.id}>
                                        <NavLink
                                          to={`/library/${cat.path}/${subSlug}/${r.id}`}
                                          className={() =>
                                            `flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors ${
                                              isArticleActive
                                                ? 'bg-accent/10 text-accent'
                                                : 'text-text-muted hover:bg-surface-2 hover:text-text'
                                            }`
                                          }
                                        >
                                          {AIcon && <AIcon size={12} />}
                                          <span className="truncate">{r.title}</span>
                                          {/* 未缓存角标：紧跟标题右侧；affordance 而非 alarm —— 后端
                                              read_resource 会自动下载，点开照样能读。 */}
                                          {!r.present && (
                                            <CloudDownload
                                              size={10}
                                              className="shrink-0 text-text-muted"
                                              aria-label="未缓存"
                                            />
                                          )}
                                        </NavLink>
                                      </li>
                                    );
                                  })
                                )}
                              </ul>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </li>
              );
            })}
          </ul>
        </nav>
      </aside>
      {navOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/40 md:hidden"
          aria-hidden="true"
          onClick={closeNav}
        />
      )}
    </>
  );
}
