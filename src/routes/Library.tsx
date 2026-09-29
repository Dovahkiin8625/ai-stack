import { useEffect, useState } from 'react';
import { useLibraryStore } from '../stores/library';
import { listen } from '@tauri-apps/api/event';
import type { ScanProgress as ScanProgressEvt } from '../types';
import ScanProgress from '../components/library/ScanProgress';
import Tree from '../components/library/Tree';
import ResourceList from '../components/library/ResourceList';

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
        <section className="rounded-md border border-border bg-surface p-2 text-sm text-text-muted">
          阅读器（任务 12）
        </section>
      </div>
    </div>
  );
}