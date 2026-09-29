import { useEffect, useState } from 'react';
import { NavLink, useLocation, useMatch } from 'react-router-dom';
import {
  BookOpen, Layers, MessageSquareText, Eye, Languages, Mic,
  Wrench, Bot, Shield, Sparkles, Package, TrendingUp,
  ChevronRight, ChevronDown,
} from 'lucide-react';
import { CATEGORIES } from '../../data/categories';
import type { ComponentType } from 'react';

const ICONS: Record<string, ComponentType<{ size?: number }>> = {
  BookOpen, Layers, MessageSquareText, Eye, Languages, Mic,
  Wrench, Bot, Shield, Sparkles, Package, TrendingUp,
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

  const [expanded, setExpanded] = useState<Set<string>>(() => {
    const initial = new Set<string>();
    if (activeCategory) initial.add(activeCategory);
    return initial;
  });

  // Auto-expand the row matching the current URL on mount and when URL changes
  useEffect(() => {
    if (activeCategory) {
      setExpanded((prev) => {
        if (prev.has(activeCategory)) return prev;
        const next = new Set(prev);
        next.add(activeCategory);
        return next;
      });
    }
  }, [activeCategory]);

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
      className="flex h-full w-60 shrink-0 flex-col border-r border-border bg-surface"
    >
      <div className="flex h-14 items-center px-4 border-b border-border">
        <span className="text-base font-semibold tracking-tight">AI Stack</span>
      </div>
      <nav className="flex-1 overflow-y-auto p-2">
        <div className="mb-2 px-2 text-xs font-medium uppercase tracking-wider text-text-muted">
          分类
        </div>
        <ul className="space-y-0.5">
          {CATEGORIES.map((cat) => {
            const Icon = cat.icon && ICONS[cat.icon];
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
                      const isSubActive = activeCategory === cat.path && activeSubSlug === subSlug;
                      return (
                        <li key={sub.id}>
                          <NavLink
                            to={`/library/${cat.path}/${subSlug}`}
                            className={() =>
                              `flex items-center gap-2 rounded-md px-2 py-1 text-sm transition-colors ${
                                isSubActive
                                  ? 'bg-accent/10 text-accent'
                                  : 'text-text-muted hover:bg-surface-2 hover:text-text'
                              }`
                            }
                          >
                            <span className="truncate">{sub.title}</span>
                          </NavLink>
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
      <div className="border-t border-border p-2 text-xs text-text-muted">
        v0.1.0 · Phase 2.5
      </div>
    </aside>
  );
}