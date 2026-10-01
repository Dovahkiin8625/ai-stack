import { useEffect, useState } from 'react';
import { NavLink, useLocation, useMatch } from 'react-router-dom';
import {
  BookOpen, Layers, MessageSquareText, Eye, Languages, Mic,
  Wrench, Bot, Shield, Sparkles, Package, TrendingUp,
  ChevronRight, ChevronDown,
  FileText, FileType, Presentation,
} from 'lucide-react';
import type { ComponentType } from 'react';
import { CATEGORIES } from '../../data/categories';
import { useLibraryStore } from '../../stores/library';
import type { ResourceType } from '../../types';

const TOP_ICONS: Record<string, ComponentType<{ size?: number }>> = {
  BookOpen, Layers, MessageSquareText, Eye, Languages, Mic,
  Wrench, Bot, Shield, Sparkles, Package, TrendingUp,
};

const RESOURCE_ICONS: Record<ResourceType, ComponentType<{ size?: number }>> = {
  markdown: FileText,
  pdf: BookOpen,
  docx: FileType,
  pptx: Presentation,
};

function categoryPathToSubSlug(category: string, fullPath: string): string {
  if (fullPath.startsWith(`${category}/`)) return fullPath.slice(category.length + 1);
  return fullPath;
}

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

  const articlesByPath = useLibraryStore((s) => s.articlesByPath);
  const loadArticles = useLibraryStore((s) => s.loadArticles);

  /**
   * 不变量：顶层 category path 不含 `/`，subcategory path 必含 `/`。
   * 若未来引入三层分类，需要拆分为两个 Set。
   */
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

  return (
    <aside
      aria-label="知识分类导航"
      className="flex h-full w-72 shrink-0 flex-col border-r border-border bg-surface"
    >
      <div className="flex h-14 items-center px-4 border-b border-border">
        <span className="text-base font-semibold tracking-tight">AI Stack</span>
      </div>
      <nav className="flex-1 overflow-y-auto p-2">
        <ul className="space-y-0.5">
          {CATEGORIES.map((cat) => {
            const Icon = cat.icon && TOP_ICONS[cat.icon];
            const isActive = activeCategory === cat.path;
            const isOpen = expanded.has(cat.path);
            const hasChildren = cat.children.length > 0;
            return (
              <li key={cat.id}>
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
                      className="mr-1 flex h-6 w-6 shrink-0 items-center justify-center rounded text-text-muted hover:bg-surface-2 hover:text-text"
                    >
                      {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                    </button>
                  )}
                </div>
                {hasChildren && isOpen && (
                  <ul className="ml-2 mt-0.5 space-y-0.5 border-l border-border pl-2">
                    {cat.children.map((sub) => {
                      const subSlug = categoryPathToSubSlug(cat.path, sub.path);
                      const isSubOpen = expanded.has(sub.path);
                      const isSubActive =
                        activeCategory === cat.path && activeSubSlug === subSlug;
                      const subArticles = articlesByPath[sub.path];
                      return (
                        <li key={sub.id}>
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
                              className="mr-1 flex h-6 w-6 shrink-0 items-center justify-center rounded text-text-muted hover:bg-surface-2 hover:text-text"
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
  );
}