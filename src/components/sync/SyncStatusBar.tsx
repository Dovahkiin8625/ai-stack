import { useEffect } from 'react';
import { CloudDownload, Loader2 } from 'lucide-react';
import { useLibraryStore } from '../../stores/library';
import { onSyncProgress } from '../../lib/sync';

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

/**
 * Topbar 之下的远端同步状态条。
 *
 * 渲染规则：
 * 1. 未配置远端（configured: false） → 不渲染 —— 纯本地模式不该有噪音条
 * 2. 已全缓存（present >= total） → 不渲染 —— "已缓存 380 / 380" 的条没意义
 * 3. 下载中（syncPhase='downloading'） → 显示进度，隐藏「下载全部」按钮
 *    （RULING 2：disable 而不是假装 暂停 —— 后端 fire-and-forget 无法中断）
 * 4. 同步出错（syncPhase='error'） → 按钮变「重试」，由 sync_progress 事件的
 *    `error` 字段或 downloadAll action 自身抛错触发（RULING 3：error 字段驱动）
 * 5. 其余 → 显示「已缓存 X / Y」+「下载全部」按钮
 *
 * 不论如何，本组件**绝不**断言"全部完成"：terminal 事件 {done=total} 与"整批全部
 * 404"在形状上不可分（RULING 3）—— 终止后只去 refreshSyncStatus 拿最新数据。
 */
export default function SyncStatusBar() {
  const status = useLibraryStore((s) => s.syncStatus);
  const phase = useLibraryStore((s) => s.syncPhase);
  const progress = useLibraryStore((s) => s.downloadProgress);
  const downloadAll = useLibraryStore((s) => s.downloadAll);

  useEffect(() => {
    // RULING 4：mount 时订阅、unmount 时取消 —— 组件重挂时旧回调泄漏是个真实的内存/逻辑漏洞。
    let unlisten: (() => void) | undefined;
    void onSyncProgress((p) => {
      // 始终把最新 done/total 写回；store 内部据此决定要不要切 phase。
      useLibraryStore.setState({ downloadProgress: { done: p.done, total: p.total } });

      if (p.error) {
        // 整批失败 —— RULING 3：仅当 error 字段存在时切到 error。
        // 注意：单条失败不在这儿处理 —— 后端 batch loop 吞单条错，仅在所有都 4xx/5xx
        // 时才送 `{done:0, total:0, error: ...}` 出来，与正常 shape 靠 error 字段区分。
        useLibraryStore.setState({
          syncPhase: 'error',
          error: p.error,
          downloadProgress: null,
        });
      } else if (p.total > 0 && p.done >= p.total) {
        // 正常结束 —— 不假定成功；翻回 idle，去问 sync_status 拿最新 present。
        useLibraryStore.setState({ syncPhase: 'idle', downloadProgress: null });
        void useLibraryStore.getState().refreshSyncStatus();
      }
    }).then((u) => {
      unlisten = u;
    });
    return () => {
      unlisten?.();
    };
  }, []);

  if (!status || !status.configured) return null;
  if (status.total === 0 || status.present >= status.total) return null;

  const downloading = phase === 'downloading';
  const errored = phase === 'error';

  return (
    <div
      className="flex items-center gap-3 border-b border-border bg-surface-2 px-4 py-1.5 text-xs text-text-muted"
      role="status"
      aria-live="polite"
    >
      <CloudDownload size={13} className="shrink-0" />
      <span className="shrink-0">
        {downloading && progress && progress.total > 0
          ? `正在下载 ${progress.done} / ${progress.total}`
          : `已缓存 ${status.present} / ${status.total}`}
        {' · '}
        {fmtBytes(status.cachedBytes)} / {fmtBytes(status.totalBytes)}
      </span>
      {downloading && (
        <Loader2 size={13} className="ml-auto shrink-0 animate-spin" aria-label="下载中" />
      )}
      {!downloading && (
        <button
          type="button"
          onClick={() => void downloadAll()}
          className="ml-auto shrink-0 rounded px-2 py-1 text-accent hover:bg-accent/10"
        >
          {errored ? '重试' : '下载全部'}
        </button>
      )}
    </div>
  );
}