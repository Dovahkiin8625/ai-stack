import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { useNotesStore } from '../stores/notes';

/**
 * 注册 AI 流式事件的通用监听器。
 *
 * 为什么抽出来：
 * - AI 讲解（ai-annotate-*）和 AI 问答（ai-qa-*）走完全相同的协议：
 *   - chunk: { noteId, text } → appendToNote
 *   - done: noteId            → cleanup
 *   - error: { noteId, message, finalContent } → setNoteContent + cleanup
 * - 唯一差异是事件名前缀和错误日志 tag
 * - 不抽出来就得 copy-paste 一遍 ~50 行 + 容易漂移
 *
 * activeNoteId 锁：保证同一时刻只有一条流的 chunk 在写 store，
 * 旧 listener 不会串扰到下一轮（用户在 AI 讲解跑着时点"AI 讲解"又来一次）。
 *
 * 为什么不在 startAiAsk 之前 await：subscribe 必须在 invoke 之前完成注册，
 * 否则后端极快失败时（如 401）的 error 事件会"先到没 listener"被丢弃。
 * notes store 的 applyOrQueue 兜底"事件先于 addNote"的竞态。
 *
 * 返回 cleanup 函数：调用后解除全部监听 + 清掉 activeNoteId。
 * 调用方应在 done/error 触发后调用一次，组件卸载时兜底调用一次。
 */
export async function subscribeAiStream(
  eventPrefix: 'ai-annotate' | 'ai-qa',
  errorLogTag: string,
): Promise<() => Promise<void>> {
  const unlistens: UnlistenFn[] = [];
  let activeNoteId: number | null = null;

  const uChunk = await listen<{ noteId: number; text: string }>(
    `${eventPrefix}-chunk`,
    (e) => {
      if (activeNoteId == null) activeNoteId = e.payload.noteId;
      if (e.payload.noteId !== activeNoteId) return;
      useNotesStore.getState().appendToNote(activeNoteId, e.payload.text);
    },
  );
  unlistens.push(uChunk);

  const cleanup = async () => {
    for (const u of unlistens) await u();
    activeNoteId = null;
  };

  const uDone = await listen<number>(`${eventPrefix}-done`, (e) => {
    if (activeNoteId == null) activeNoteId = e.payload;
    if (e.payload !== activeNoteId) return;
    void cleanup();
  });
  unlistens.push(uDone);

  const uError = await listen<{
    noteId: number;
    message: string;
    finalContent: string;
  }>(`${eventPrefix}-error`, (e) => {
    if (activeNoteId == null) activeNoteId = e.payload.noteId;
    if (e.payload.noteId !== activeNoteId) return;
    console.error(`${errorLogTag}:`, e.payload.message);
    // 同步本地 store：后端已经把错误描述写进 DB，但前端占位卡片仍是空字符串。
    // 把 final_content 套回去，让用户在面板里立刻看到失败原因（而不是一个空卡片）。
    useNotesStore.getState().setNoteContent(activeNoteId, e.payload.finalContent);
    void cleanup();
  });
  unlistens.push(uError);

  return cleanup;
}