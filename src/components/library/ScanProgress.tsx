import { Loader2 } from 'lucide-react';
import type { ScanProgress as ScanProgressEvt, ScanSummary } from '../../types';

interface Props {
  status: 'idle' | 'scanning' | 'ready' | 'error';
  progress?: ScanProgressEvt;
  summary?: ScanSummary;
  error?: string;
}

export default function ScanProgress({ status, progress, summary, error }: Props) {
  if (status === 'ready' && summary) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-border bg-surface p-3 text-sm text-text-muted">
        <span className="text-text">
          索引完成 · {summary.categoriesCount} 个分类 · {summary.resourcesCount} 个资源
          {summary.errorsCount > 0 && (
            <span className="ml-2 text-amber-600">（{summary.errorsCount} 个文件跳过）</span>
          )}
        </span>
      </div>
    );
  }
  if (status === 'error') {
    return (
      <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700">
        索引失败：{error ?? '未知错误'}
      </div>
    );
  }
  if (status === 'scanning') {
    return (
      <div className="flex items-center gap-2 rounded-md border border-border bg-surface p-3 text-sm">
        <Loader2 size={16} className="animate-spin text-accent" />
        <span className="text-text-muted">
          正在索引 {progress?.phase ?? '...'}
          {progress?.currentPath && ` (${progress.currentPath})`}
        </span>
      </div>
    );
  }
  return null;
}