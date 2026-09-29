import type { Category } from '../../data/categories';

interface Props {
  subCategories: Category[];
  selectedPath: string | null;
  onSelect: (path: string) => void;
}

export default function SubCategoryTabs({ subCategories, selectedPath, onSelect }: Props) {
  if (subCategories.length === 0) {
    return (
      <div className="px-3 py-2 text-xs text-text-muted">
        该分类下暂无子分类
      </div>
    );
  }
  return (
    <div className="flex flex-wrap gap-1 border-b border-border bg-surface px-2 py-2">
      {subCategories.map((sub) => {
        const active = selectedPath === sub.path;
        return (
          <button
            key={sub.id}
            onClick={() => onSelect(sub.path)}
            className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
              active
                ? 'bg-accent/10 text-accent'
                : 'text-text-muted hover:bg-surface-2 hover:text-text'
            }`}
          >
            {sub.title}
          </button>
        );
      })}
    </div>
  );
}