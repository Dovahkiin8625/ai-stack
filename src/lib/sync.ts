import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type { SyncProgress, SyncStatus, SyncManifestResult } from '../types';

export type { SyncProgress, SyncStatus, SyncManifestResult };

/**
 * 知识库远端同步 transport 层。
 *
 * 与 library-api 平行 —— 后端命令族 `sync_*` / `download_*` / `set_sync_base_url`
 * 不属于"扫读本地"这条线，单独放这个文件，避免 library-api 在 remote sync 加入后
 * 既管 domain surface 又管 infra transport。
 *
 * 进度通过 Tauri 事件回传：
 * - `sync_progress`：download_all 的批量进度（payload 见上方 RULING）
 * - `seed_progress`：首次启动从安装包释放种子库的进度（同一份 SyncProgress shape）
 *
 * **重要（事件契约）**：download_all 在没有 pending 时**完全不 emit 事件**就返回，
 * "未配置 base_url" 时则**抛错**（不是 skipped）；seed_progress 则是 materialize_seed
 * 内部按文件递增 emit。所以 SyncProgress 不能被当作"完成 = 成功"用。
 */

export async function syncStatus(): Promise<SyncStatus> {
  return invoke<SyncStatus>('sync_status');
}

/**
 * 拉远端 manifest.json → 解析 → 落库（upsert_manifest + upsert_index_files）。
 * - `skipped: true` 表示后端没配 base_url（纯本地模式），不要当成失败
 * - `skipped: false` 表示实际拉到了新条目并已落库，调用方此时可以走 scan() 重新标 present
 */
export async function syncManifest(): Promise<SyncManifestResult> {
  return invoke<SyncManifestResult>('sync_manifest');
}

/**
 * 把同步源 URL 写入 app_config —— 后端 set_config 已经 trim 过，前端不重复处理，
 * 否则 trim 不一致会让"用户输入带斜杠"和"程序写入带斜杠"两种路径走不同 key。
 */
export async function setSyncBaseUrl(url: string): Promise<void> {
  await invoke('set_sync_base_url', { url });
}

/** 单资源显式下载（前端「下载」按钮触发）。 */
export async function downloadResource(id: number): Promise<void> {
  await invoke('download_resource', { id });
}

/**
 * 后台批量下载 —— fire-and-forget，立即返回；进度通过 `sync_progress` 事件回传。
 * 见顶部关于事件契约的警告：什么都没得下时不会 emit，整批失败的事件 shape 不同。
 */
export async function downloadAll(): Promise<void> {
  await invoke('download_all');
}

/** 监听 download_all 的 sync_progress 事件。返回的 unlisten 必须在 useEffect cleanup 里调。 */
export async function onSyncProgress(cb: (p: SyncProgress) => void): Promise<UnlistenFn> {
  return listen<SyncProgress>('sync_progress', (e) => cb(e.payload));
}

/** 监听首启释放种子库的 seed_progress 事件（payload 与 sync_progress 同形）。 */
export async function onSeedProgress(cb: (p: SyncProgress) => void): Promise<UnlistenFn> {
  return listen<SyncProgress>('seed_progress', (e) => cb(e.payload));
}