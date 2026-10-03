import { useEffect, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';
import mammoth from 'mammoth';
import * as api from '../../../lib/library-api';

interface Props {
  /** 资源 ID：用于 readResourceBytes 加载 docx，然后交给 mammoth 转 HTML。 */
  resourceId: number;
  /** 后端 word 数（仅展示用，与 mammoth 转换独立）。 */
  wordCount: number;
}

/**
 * DOCX 阅读器 —— 后端只数 word 数，原文件 raw bytes 交给前端 mammoth.js
 * 转成 semantic HTML（含 bold/italic/headings/lists/tables/images）。
 *
 * 设计要点：
 * - 拿到 bytes 后 `new Uint8Array(...)` → `.buffer` 转 ArrayBuffer 喂给
 *   mammoth（浏览器侧 mammoth 走 ArrayBufferInput）
 * - 图片走 `mammoth.images.dataUri` —— 把 docx 内嵌图片转 data URI，
 *   这样不依赖外部资源，PPT 离线状态下也能完整渲染
 * - mammoth 输出非信任 HTML（警告明确写在 mammoth README），
 *   不过我们 sources 是用户自己放的 knowledge 目录，且 tauri webview
 *   默认 sandbox 隔离前端，与 markdown reader 同样不做 DOMPurify
 *
 * 老实现：后端用 docx-rs 抽 text → 前端只渲染 heading/p/table 的纯文本。
 * 现在前端 mammoth 直接处理原始 docx，版式 100% 来自 docx 本身。
 */
export default function DocxReader({ resourceId, wordCount }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const host = hostRef.current;
    if (!host) return;
    // 切资源时先清空旧内容
    host.replaceChildren();
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const raw = await api.readResourceBytes(resourceId);
        if (cancelled) return;
        const ab = toArrayBuffer(raw);
        const { value, messages } = await mammoth.convertToHtml(
          { arrayBuffer: ab },
          {
            // 把内嵌图片（png/jpeg 等）转 data URI，
            // 让 docx 里的图在 webview 里直接显示，不用走文件系统。
            convertImage: mammoth.images.dataUri,
          },
        );
        if (cancelled) return;
        // mammoth 输出是 HTML 字符串；用 prose 样式接管段落 / 表格 / 标题的视觉
        host.innerHTML = value;
        if (messages.length > 0) {
          // 转换期间的 warning/error 不致命 —— 记录到控制台，方便调试
          // 个别 docx 不规范的元素 mammoth 会跳过（不影响主体渲染）。
          console.warn(
            `[docx ${resourceId}] mammoth 转换产生 ${messages.length} 条消息：`,
            messages,
          );
        }
        setLoading(false);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [resourceId]);

  return (
    <div className="relative h-full overflow-auto bg-surface-2">
      <div
        ref={hostRef}
        className="prose prose-sm mx-auto my-4 max-w-3xl rounded-lg border border-border bg-surface p-6 shadow-sm dark:prose-invert"
      />
      {loading && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-text-muted">
          <Loader2 size={16} className="mr-2 animate-spin" />
          正在解析 DOCX…
        </div>
      )}
      {error && (
        <div className="m-4 rounded border border-red-300 bg-red-50 p-3 text-sm text-red-700">
          DOCX 加载失败：{error}
          <div className="mt-1 text-xs text-red-600">
            文件大小约 {wordCount.toLocaleString()} 词
          </div>
        </div>
      )}
    </div>
  );
}

/** Tauri 2 IPC 字节归一化：Uint8Array / ArrayBuffer / number[] → ArrayBuffer。 */
function toArrayBuffer(raw: Uint8Array | ArrayBuffer | number[]): ArrayBuffer {
  if (raw instanceof ArrayBuffer) return raw;
  if (raw instanceof Uint8Array) {
    // 切片以拿到 underlying ArrayBuffer，避免和 Uint8Array 共享同一 buffer
    // 时 mammoth 读到原始 buffer 长度而不是我们传进去的字节数。
    return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  }
  if (Array.isArray(raw)) return new Uint8Array(raw).buffer;
  throw new Error(`unexpected bytes type: ${typeof raw}`);
}