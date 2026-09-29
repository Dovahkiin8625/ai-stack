import { useEffect, useState } from 'react';
import { useLibraryStore } from '../stores/library';
import { listen } from '@tauri-apps/api/event';
import type { ScanProgress as ScanProgressEvt } from '../types';
import ScanProgress from '../components/library/ScanProgress';
import Tree from '../components/library/Tree';
import ResourceList from '../components/library/ResourceList';
import { AlertCircle } from 'lucide-react';
import ReaderToolbar from '../components/library/ReaderToolbar';
import MarkdownReader from '../components/library/reader/MarkdownReader';
import PdfReader from '../components/library/reader/PdfReader';
import DocxReader from '../components/library/reader/DocxReader';
import PptxReader from '../components/library/reader/PptxReader';

export default function Library() {
  const status = useLibraryStore((s) => s.status);
  const summary = useLibraryStore((s) => s.summary);
  const error = useLibraryStore((s) => s.error);
  const scan = useLibraryStore((s) => s.scan);
  const categories = useLibraryStore((s) => s.categories);
  const selectedCategoryPath = useLibraryStore((s) => s.selectedCategoryPath);
  const selectCategory = useLibraryStore((s) => s.selectCategory);
  const resources = useLibraryStore((s) => s.resources);
  const selectedResourceId = useLibraryStore((s) => s.selectedResourceId);
  const selectResource = useLibraryStore((s) => s.selectResource);
  const content = useLibraryStore((s) => s.resourceContent);
  const current = resources.find((r) => r.id === useLibraryStore.getState().selectedResourceId);
  const [progress, setProgress] = useState<ScanProgressEvt | undefined>(undefined);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    (async () => {
      unlisten = await listen<ScanProgressEvt>('scan_progress', (e) => setProgress(e.payload));
      if (status === 'idle') await scan(false);
    })();
    return () => { unlisten?.(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <header className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">知识库</h2>
        <button
          onClick={() => scan(true)}
          disabled={status === 'scanning'}
          className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm hover:bg-surface-2 disabled:opacity-50"
        >
          刷新
        </button>
      </header>
      <ScanProgress status={status} progress={progress} summary={summary} error={error} />
      <div className="grid flex-1 grid-cols-[240px_320px_1fr] gap-4 overflow-hidden">
        <aside className="overflow-y-auto rounded-md border border-border bg-surface p-2">
          <Tree
            categories={categories}
            selectedPath={selectedCategoryPath}
            onSelect={(p) => selectCategory(p)}
          />
        </aside>
        <section className="overflow-y-auto rounded-md border border-border bg-surface p-2">
          <ResourceList
            resources={resources}
            selectedId={selectedResourceId}
            onSelect={(id) => selectResource(id)}
          />
        </section>
        <section className="flex flex-col overflow-hidden rounded-md border border-border bg-bg">
          {current ? (
            <>
              <ReaderToolbar
                title={current.title}
                type={current.type}
                pageCount={current.pageCount ?? undefined}
                wordCount={current.wordCount ?? undefined}
                onBack={() => selectResource(null)}
              />
              <div className="flex-1 overflow-auto">
                {content ? (
                  <>
                    {content.type === 'markdown' && <MarkdownReader html={content.html} />}
                    {content.type === 'pdf' && (
                      <PdfReader pages={content.pages} pageCount={content.pageCount} />
                    )}
                    {content.type === 'docx' && (
                      <DocxReader blocks={content.blocks} wordCount={content.wordCount} />
                    )}
                    {content.type === 'pptx' && (
                      <PptxReader slides={content.slides} slideCount={content.slideCount} />
                    )}
                  </>
                ) : (
                  <div className="m-4 flex items-start gap-2 rounded-md border border-red-300 bg-red-50 p-4 text-sm text-red-700">
                    <AlertCircle size={16} className="mt-0.5 shrink-0" />
                    <div>
                      <div className="font-medium">无法读取：{current.title}</div>
                      <div className="mt-1 text-xs text-red-600">{error ?? '请尝试刷新索引或更换文件'}</div>
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex h-full items-center justify-center p-6 text-sm text-text-muted">
              {status === 'ready' ? '请选择左侧资源' : '等待索引完成'}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}