import { Link, useLocation } from 'react-router-dom';
import { Search, Settings as SettingsIcon } from 'lucide-react';
import ThemeToggle from '../ui/ThemeToggle';

const TITLES: Record<string, string> = {
  '/library': '知识库',
  '/notes': '笔记',
  '/dashboard': '学习仪表盘',
  '/settings': '设置',
};

export default function Topbar() {
  const { pathname } = useLocation();
  const title = TITLES[pathname] ?? 'AI Stack';

  return (
    <header
      role="banner"
      className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-surface px-4"
    >
      <h1 className="text-lg font-semibold">{title}</h1>

      <div className="flex items-center gap-2">
        <div className="relative">
          <Search
            size={16}
            className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-text-muted"
          />
          <input
            type="search"
            placeholder="搜索（阶段 6 实现）"
            disabled
            aria-label="搜索"
            className="w-64 rounded-md border border-border bg-bg py-1.5 pl-8 pr-3 text-sm text-text placeholder:text-text-muted disabled:opacity-50"
          />
        </div>

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
