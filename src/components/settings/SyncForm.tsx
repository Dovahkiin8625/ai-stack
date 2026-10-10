import { useState } from 'react';
import { CloudDownload, Loader2, RefreshCw } from 'lucide-react';
import { saveSettings } from '../../lib/tauri';
import { setSyncBaseUrl } from '../../lib/sync';
import { useLibraryStore } from '../../stores/library';

interface Props {
  /** 当前同步地址（受控，来自 Settings 父组件，让最底部的"保存"能写到同步 URL）。 */
  url: string;
  onUrlChange: (v: string) => void;
}

/**
 * 知识库同步设置表单。
 *
 * - 受控组件：URL 由 Settings 父组件持有，与 AI 配置共用同一份"保存"按钮落地。
 * - 「拉取清单」：保存 URL（前端 plugin-store + 后端 app_config）→ 调 syncNow() 拉远端
 *   manifest.json 并落库。注意 saveSettings 和 setSyncBaseUrl 必须**都写**：前者给前端
 *   getSettings 用，后者给后端 sync_manifest 命令读（它们落两个不同的 store，故意保留）。
 *   拉取成功后把 URL 记为 fetchedUrl，使「开始同步」按钮可用。
 * - 「开始同步」：仅在 fetchedUrl === url 时启用（URL 改了就强制重新拉取），
 *   点击后调 libraryStore.downloadAll() —— 实际进度走 store 的 downloadProgress，
 *   本组件订阅渲染进度条。
 */
export default function SyncForm({ url, onUrlChange }: Props) {
  /** 拉过清单的那一版 URL；与当前 url 不一致时「开始同步」强制禁用。 */
  const [fetchedUrl, setFetchedUrl] = useState<string | null>(null);
  const [fetchState, setFetchState] = useState<
    { kind: 'idle' } | { kind: 'fetching' } | { kind: 'error'; message: string }
  >({ kind: 'idle' });

  const syncNow = useLibraryStore((s) => s.syncNow);
  const downloadAll = useLibraryStore((s) => s.downloadAll);
  const phase = useLibraryStore((s) => s.syncPhase);
  const progress = useLibraryStore((s) => s.downloadProgress);

  async function onFetch() {
    if (!url.trim()) {
      setFetchState({ kind: 'error', message: '请先填写同步地址' });
      return;
    }
    setFetchState({ kind: 'fetching' });
    // 清掉 store 里的旧 error —— 否则上一次失败的 error 会跟这一次的语义混杂，
    // 用户看到的可能是旧原因而非本次真实失败。
    useLibraryStore.setState({ error: undefined });
    try {
      // 见文档：先写两个 store + 后端 config，sync_manifest 才能读到新 URL。
      await saveSettings({ syncBaseUrl: url });
      await setSyncBaseUrl(url);
      const ok = await syncNow();
      if (ok) {
        setFetchedUrl(url);
        setFetchState({ kind: 'idle' });
        return;
      }
      // syncNow 返回 false：可能是 skipped（无 base_url）也可能是抛了错。
      // 真实错误已写入 store.error。这里读出来给用户 —— 之前一律说"拉取清单失败"
      // 太泛，用户看不出是连不上、找不到文件、还是 manifest 解析失败。
      const storeErr = useLibraryStore.getState().error;
      setFetchState({
        kind: 'error',
        message: storeErr
          ? `拉取清单失败：${storeErr}`
          : '拉取清单失败：未配置同步地址或已跳过（请检查 URL 后再试）',
      });
    } catch (e) {
      setFetchState({ kind: 'error', message: `拉取清单失败：${String(e)}` });
    }
  }

  async function onStartSync() {
    await downloadAll();
  }

  /** URL 改动 / 刚开始都没有"已拉清单" → 同步按钮禁用。 */
  const isDownloading = phase === 'downloading';
  const canSync = fetchedUrl === url && url.trim().length > 0 && !isDownloading;

  /** 进度条展示：仅在 downloading 阶段 + 后端给了真实数字时显示。 */
  const showProgress = isDownloading && progress != null && progress.total > 0;
  const pct =
    progress && progress.total > 0
      ? Math.min(100, Math.round((progress.done / progress.total) * 100))
      : 0;

  return (
    <div className="space-y-3">
      <div>
        <label className="mb-1 block text-sm font-medium">知识库同步地址</label>
        <input
          type="url"
          value={url}
          onChange={(e) => onUrlChange(e.target.value)}
          placeholder="https://your-host/kb"
          className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
        />
        <p className="mt-1 text-xs text-text-muted">
          指向静态托管的发布根目录（与 <code>manifest.json</code> 同级，
          内容是 <code>&lt;分类&gt;/&lt;文件&gt;</code>，不再带 <code>knowledge/</code>）。
          留空则只用本机已缓存的内容（开发模式下仍读取仓库里的 resources/knowledge）。
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onFetch}
          disabled={fetchState.kind === 'fetching' || !url.trim()}
          className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-2 text-sm font-medium hover:bg-surface-2 disabled:opacity-50"
        >
          {fetchState.kind === 'fetching' ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <RefreshCw size={16} />
          )}
          拉取清单
        </button>
        <button
          type="button"
          onClick={onStartSync}
          disabled={!canSync}
          className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
        >
          {isDownloading ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <CloudDownload size={16} />
          )}
          开始同步
        </button>
        {fetchState.kind === 'idle' && fetchedUrl === url && (
          <span className="text-xs text-green-600">清单已就绪</span>
        )}
        {fetchState.kind === 'error' && (
          <span className="text-xs text-red-600">{fetchState.message}</span>
        )}
      </div>

      {showProgress && progress && (
        <div className="space-y-1" role="status" aria-live="polite">
          <div className="flex items-center justify-between text-xs text-text-muted">
            <span>
              正在同步 {progress.done} / {progress.total}
            </span>
            <span>{pct}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
            <div
              className="h-full bg-accent transition-[width] duration-200"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
