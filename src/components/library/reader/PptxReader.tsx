import { useEffect, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { PPTXViewer } from 'pptxviewjs';
import * as api from '../../../lib/library-api';

interface Props {
  /** 资源 ID：用于 readResourceBytes 加载 pptx，然后交给 pptxviewjs 渲染。 */
  resourceId: number;
  /** 后端给的 slide 数（pptxviewjs 加载完会用自身权威值）。 */
  slideCount: number;
}

/**
 * PPTX 阅读器 —— 后端只数 slide 数，原文件 raw bytes 交给前端 pptxviewjs
 * 用 HTML5 Canvas 渲染（按 slide 翻页，完整保留版式 / 图表 / 表格 / 图片）。
 *
 * 设计要点：
 * - pptxviewjs 自身按 slide 翻页（nextSlide / previousSlide / goToSlide），
 *   但它没内置翻页 UI；我们自己写一个 toolbar，用其 API 同步当前页码
 * - viewer.loadFile(File) 接受浏览器 File 对象；后端给的是 Uint8Array，
 *   包成 Blob → File 喂进去。ArrayBuffer 也行，但 File 是更明确的上传语义
 * - 切资源时 `viewer.destroy()` 释放旧 viewer 内部缓存 / canvas，
 *   否则 pptxviewjs 会持有 WebGL/canvas context 跨多次加载导致资源累积
 *
 * 老实现：后端 quick-xml 抽 text run → 前端只显示文本列表，
 * 完全没有版式（位置/颜色/字号全没了）。
 */
export default function PptxReader({ resourceId, slideCount }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<PPTXViewer | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // 实际 slide 数（pptxviewjs 解析权威值）；初始用 prop 的 hint，
  // 加载完用 viewer.getSlideCount() 覆盖
  const [actualCount, setActualCount] = useState<number>(slideCount);
  const [idx, setIdx] = useState(0);

  // 加载 PPTX 并初始化 viewer
  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas) return;
    setLoading(true);
    setError(null);
    setIdx(0);
    setActualCount(slideCount);

    (async () => {
      try {
        const bytes = await api.readResourceBytes(resourceId);
        if (cancelled) return;

        // 包成 File 让 pptxviewjs 拿到正确 MIME（也避免它从 buffer 推断类型出错）
        const file = new File(
          [toArrayBuffer(bytes) as ArrayBuffer],
          'presentation.pptx',
          { type: 'application/vnd.openxmlformats-officedocument.presentationml.presentation' },
        );

        const viewer = new PPTXViewer({ canvas });
        viewerRef.current = viewer;

        viewer.on('loadComplete', (...args: unknown[]) => {
          if (cancelled) return;
          // pptxviewjs 的 callback 签名是 (...args: unknown[]) => void；
          // 真实 payload 形如 { slideCount: number }，按位断言即可。
          const data = args[0] as { slideCount?: number } | undefined;
          if (data?.slideCount !== undefined) {
            setActualCount(data.slideCount);
          }
        });
        viewer.on('renderComplete', () => {
          if (cancelled) return;
          setLoading(false);
        });

        await viewer.loadFile(file);
        if (cancelled) return;
        await viewer.render();
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      // 释放旧 viewer（pptxviewjs 内部会 cache 各种资源；不显式 destroy 会越攒越多）
      const v = viewerRef.current;
      viewerRef.current = null;
      if (v) {
        try {
          v.destroy();
        } catch {
          /* 静默忽略：用户已经切走，老 viewer 报不报无所谓 */
        }
      }
    };
  }, [resourceId, slideCount]);

  // 翻页：viewer 自带 nextSlide/previousSlide；翻完会触发 renderComplete
  // → 上面的 listener 把 loading 置 false（仅在错误路径上用，PPT 实际很快）
  const goPrev = () => {
    const v = viewerRef.current;
    if (!v) return;
    void v.previousSlide().then(() => {
      const cur = v.getCurrentSlideIndex();
      // viewer.getCurrentSlideIndex 返回 1-based；前端 UI 是 0-based
      setIdx(Math.max(0, cur - 1));
    });
  };
  const goNext = () => {
    const v = viewerRef.current;
    if (!v) return;
    void v.nextSlide().then(() => {
      const cur = v.getCurrentSlideIndex();
      setIdx(Math.min(actualCount - 1, cur - 1));
    });
  };

  const effectiveCount = actualCount || slideCount;
  const canPrev = idx > 0;
  const canNext = effectiveCount > 0 && idx < effectiveCount - 1;

  return (
    <div className="flex h-full flex-col">
      <Toolbar idx={idx} slideCount={effectiveCount} canPrev={canPrev} canNext={canNext} onPrev={goPrev} onNext={goNext} />
      <div className="relative flex-1 overflow-auto bg-surface-2">
        <div className="mx-auto my-4 block w-fit">
          {/* pptxviewjs 直接在 canvas 上画 —— 我们给 canvas 一个固定 aspect ratio
              占位（16:9），viewer 会按这个尺寸内部 scale slides。
              不预设尺寸会让 pptxviewjs 拿到一个 0 高度的 canvas 报错。 */}
          <canvas
            ref={canvasRef}
            width={1280}
            height={720}
            className="block rounded-lg border border-border bg-white shadow"
          />
        </div>
        {loading && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-text-muted">
            <Loader2 size={16} className="mr-2 animate-spin" />
            正在加载 PPTX…
          </div>
        )}
        {error && (
          <div className="m-4 rounded border border-red-300 bg-red-50 p-3 text-sm text-red-700">
            PPTX 加载失败：{error}
            <div className="mt-1 text-xs text-red-600">
              共 {slideCount.toLocaleString()} 张幻灯片
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface ToolbarProps {
  idx: number;
  slideCount: number;
  canPrev: boolean;
  canNext: boolean;
  onPrev: () => void;
  onNext: () => void;
}

function Toolbar({ idx, slideCount, canPrev, canNext, onPrev, onNext }: ToolbarProps) {
  return (
    <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-2 text-sm">
      <button
        onClick={onPrev}
        disabled={!canPrev}
        className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
      >
        <ChevronLeft size={14} /> 上一张
      </button>
      <span>第 {idx + 1} / {slideCount} 张</span>
      <button
        onClick={onNext}
        disabled={!canNext}
        className="inline-flex items-center gap-1 rounded px-2 py-1 text-text-muted hover:bg-surface-2 disabled:opacity-40"
      >
        下一张 <ChevronRight size={14} />
      </button>
    </div>
  );
}

/** Tauri 2 IPC 字节归一化：Uint8Array / ArrayBuffer / number[] → ArrayBuffer。 */
function toArrayBuffer(raw: Uint8Array | ArrayBuffer | number[]): ArrayBuffer {
  if (raw instanceof ArrayBuffer) return raw;
  if (raw instanceof Uint8Array) {
    return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  }
  if (Array.isArray(raw)) return new Uint8Array(raw).buffer;
  throw new Error(`unexpected bytes type: ${typeof raw}`);
}