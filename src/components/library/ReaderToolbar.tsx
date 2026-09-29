import { ArrowLeft } from 'lucide-react';
import type { ResourceType } from '../../types';

const TYPE_LABELS: Record<ResourceType, string> = {
  markdown: 'Markdown',
  pdf: 'PDF',
  docx: 'Word',
  pptx: 'PPT',
};

interface Props {
  title: string;
  type: ResourceType;
  pageCount?: number;
  wordCount?: number;
  onBack: () => void;
}

export default function ReaderToolbar({ title, type, pageCount, wordCount, onBack }: Props) {
  return (
    <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-2">
      <button
        onClick={onBack}
        className="inline-flex items-center gap-1 rounded px-2 py-1 text-sm text-text-muted hover:bg-surface-2"
      >
        <ArrowLeft size={14} />
        返回列表
      </button>
      <div className="flex items-center gap-3 text-sm">
        <span className="rounded bg-surface-2 px-2 py-0.5 text-xs uppercase tracking-wider text-text-muted">
          {TYPE_LABELS[type]}
        </span>
        <span className="truncate font-medium">{title}</span>
        {pageCount != null && <span className="text-text-muted">{pageCount} 页</span>}
        {wordCount != null && <span className="text-text-muted">{wordCount} 字</span>}
      </div>
    </div>
  );
}