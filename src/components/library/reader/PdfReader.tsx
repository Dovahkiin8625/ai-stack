import { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface Props {
  pages: Array<{ index: number; dataUrl: string }>;
  pageCount: number;
}

export default function PdfReader({ pages, pageCount }: Props) {
  const [idx, setIdx] = useState(0);
  if (pages.length === 0) {
    return (
      <div className="p-6 text-sm text-text-muted">
        该 PDF 暂无可视页面（pdfium 渲染不可用，已回退到纯文本）。
        页数估算：{pageCount}
      </div>
    );
  }
  const page = pages[idx] ?? pages[0];
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-2 text-sm">
        <button
          onClick={() => setIdx((i) => Math.max(0, i - 1))}
          disabled={idx === 0}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
        >
          <ChevronLeft size={14} /> 上一页
        </button>
        <span>
          第 {idx + 1} / {pages.length} 页{pageCount > pages.length && `（共 ${pageCount} 页）`}
        </span>
        <button
          onClick={() => setIdx((i) => Math.min(pages.length - 1, i + 1))}
          disabled={idx >= pages.length - 1}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
        >
          下一页 <ChevronRight size={14} />
        </button>
      </div>
      <div className="flex-1 overflow-auto bg-surface-2 p-4">
        <img src={page.dataUrl} alt={`page ${idx + 1}`} className="mx-auto max-w-full rounded shadow" />
      </div>
    </div>
  );
}