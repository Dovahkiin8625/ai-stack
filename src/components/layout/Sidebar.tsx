import { NavLink, useLocation } from 'react-router-dom';
import { BookOpen, Layers, MessageSquareText, Eye, Languages, Mic,
         Wrench, Bot, Shield, Sparkles, Package, TrendingUp } from 'lucide-react';
import { CATEGORIES } from '../../data/categories';
import type { ComponentType } from 'react';

const ICONS: Record<string, ComponentType<{ size?: number }>> = {
  BookOpen, Layers, MessageSquareText, Eye, Languages, Mic,
  Wrench, Bot, Shield, Sparkles, Package, TrendingUp,
};

interface LocationState { focusCategory?: string }

export default function Sidebar() {
  const location = useLocation();
  const focused = (location.state as LocationState | null)?.focusCategory;

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
            // 高亮规则：在 /library 路由且 state.focusCategory 与本分类 path 匹配
            const isFocused = location.pathname === '/library' && focused === cat.path;
            return (
              <li key={cat.id}>
                <NavLink
                  to="/library"
                  state={{ focusCategory: cat.path }}
                  className={() =>
                    `flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors ${
                      isFocused
                        ? 'bg-accent/10 text-accent'
                        : 'text-text hover:bg-surface-2'
                    }`
                  }
                >
                  {Icon && <Icon size={16} />}
                  <span className="truncate">{cat.title}</span>
                </NavLink>
              </li>
            );
          })}
        </ul>
      </nav>
      <div className="border-t border-border p-2 text-xs text-text-muted">
        v0.1.0 · Phase 1
      </div>
    </aside>
  );
}
