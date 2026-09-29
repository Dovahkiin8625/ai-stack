import { FileText, FileType, Presentation, BookOpen } from 'lucide-react';
import type { Resource, ResourceType } from '../../types';

const ICONS: Record<ResourceType, React.ComponentType<{ size?: number }>> = {
  markdown: FileText,
  pdf: BookOpen,
  docx: FileType,
  pptx: Presentation,
};

interface Props {
  resources: Resource[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export default function ResourceList({ resources, selectedId, onSelect }: Props) {
  if (resources.length === 0) {
    return <div className="p-3 text-sm text-text-muted">暂无资源</div>;
  }
  return (
    <ul className="space-y-0.5 overflow-y-auto">
      {resources.map((r) => {
        const Icon = ICONS[r.type];
        const active = selectedId === r.id;
        return (
          <li key={r.id}>
            <button
              onClick={() => onSelect(r.id)}
              className={`flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm ${
                active ? 'bg-accent/10 text-accent' : 'hover:bg-surface-2'
              }`}
            >
              <Icon size={14} />
              <span className="truncate">{r.title}</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}