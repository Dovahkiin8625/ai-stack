import { Moon, Sun } from 'lucide-react';
import { useThemeStore } from '../../stores/theme';

export default function ThemeToggle() {
  const { theme, toggle } = useThemeStore();
  const isDark = theme === 'dark';
  return (
    <button
      onClick={toggle}
      aria-label={isDark ? '切换到亮色主题' : '切换到暗色主题'}
      className="rounded-md p-2 text-text-muted hover:bg-surface-2 active:bg-surface-2/70 hover:text-text"
    >
      {isDark ? <Sun size={18} /> : <Moon size={18} />}
    </button>
  );
}
