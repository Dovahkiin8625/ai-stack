import { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface Slide {
  index: number;
  title: string | null;
  body: string[];
  notes: string | null;
}

interface Props {
  slides: Slide[];
  slideCount: number;
}

export default function PptxReader({ slides }: Props) {
  const [idx, setIdx] = useState(0);
  if (slides.length === 0) {
    return <div className="p-6 text-sm text-text-muted">无可用幻灯片</div>;
  }
  const s = slides[idx];
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-2 text-sm">
        <button
          onClick={() => setIdx((i) => Math.max(0, i - 1))}
          disabled={idx === 0}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
        >
          <ChevronLeft size={14} /> 上一张
        </button>
        <span>第 {idx + 1} / {slides.length} 张</span>
        <button
          onClick={() => setIdx((i) => Math.min(slides.length - 1, i + 1))}
          disabled={idx >= slides.length - 1}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
        >
          下一张 <ChevronRight size={14} />
        </button>
      </div>
      <div className="flex-1 overflow-auto p-6">
        <article className="mx-auto max-w-3xl rounded-lg border border-border bg-surface p-6 shadow-sm">
          <div className="mb-3 text-xs uppercase tracking-wider text-text-muted">
            Slide {s.index + 1}
          </div>
          {s.title && <h2 className="mb-4 text-2xl font-semibold">{s.title}</h2>}
          <ul className="list-disc space-y-1 pl-6 text-sm">
            {s.body.map((line, i) => <li key={i}>{line}</li>)}
          </ul>
          {s.notes && (
            <div className="mt-6 rounded border border-border bg-surface-2 p-3 text-sm">
              <div className="mb-1 text-xs uppercase tracking-wider text-text-muted">
                Speaker Notes
              </div>
              <p className="whitespace-pre-wrap text-text-muted">{s.notes}</p>
            </div>
          )}
        </article>
      </div>
    </div>
  );
}