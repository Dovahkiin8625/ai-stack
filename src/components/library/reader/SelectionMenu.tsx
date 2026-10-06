import { useEffect, useState } from 'react';
import { Check, Copy, Languages, Loader2, MessageSquarePlus, Sparkles, HelpCircle } from 'lucide-react';
import { translateText as apiTranslateText } from '../../../lib/library-api';
import { getSettings } from '../../../lib/tauri';

/**
 * 选中文字右键唤起的浮层菜单。reader-agnostic：
 * - MarkdownReader / PdfReader / DocxReader 都在右键时计算 anchor，
 *   然后开一个 <SelectionMenu anchor={...} onXxx={...} />
 * - 菜单本身管 "复制 / 添加批注 / 翻译 / AI 讲解 / 询问 AI" 5 项 + 翻译状态机
 * - outside click / Escape 自动关闭（useEffect 在菜单打开时挂全局监听）
 *
 * 翻译流程：用户点"翻译" → 显示 loading → 调 LLM（轻量模型）→ 显示成功/失败
 * → 用户点"关闭"或外部点击。状态机由本组件持有。
 */
export interface SelectionMenuProps {
  x: number;
  y: number;
  anchorText: string;
  anchorOccurrence: number;
  /** 添加批注 —— reader 端实现 onAddNoteAtSelection（含 anchor 写入 + 抽屉展开）。 */
  onAddNote: (input: { anchorText: string; anchorOccurrence: number }) => void;
  /** AI 讲解 —— reader 把"局部上下文"准备好后调 onAiAnnotate；菜单本身只管把 menu 关掉。 */
  onAiAnnotate?: () => void;
  /** 询问 AI —— reader 会自己读 selection 上下文，开 AskInputBox。这里只关菜单。 */
  onAiAsk?: () => void;
  /** 关闭菜单（提交"翻译"成功/失败时外部点击）。 */
  onClose: () => void;
}

export default function SelectionMenu({
  x,
  y,
  anchorText,
  anchorOccurrence,
  onAddNote,
  onAiAnnotate,
  onAiAsk,
  onClose,
}: SelectionMenuProps) {
  const [copied, setCopied] = useState(false);
  const [translation, setTranslation] = useState<
    | { status: 'idle' }
    | { status: 'loading' }
    | { status: 'success'; text: string }
    | { status: 'error'; message: string }
  >({ status: 'idle' });

  // 菜单打开时挂全局监听：点击外部 / Esc 关闭
  useEffect(() => {
    const close = () => {
      setTranslation({ status: 'idle' });
      setCopied(false);
      onClose();
    };
    const onMouseDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest('[data-selection-menu]')) return;
      close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    // setTimeout 0 跳过本次右键触发的 mousedown，避免"刚开就被关"
    const id = window.setTimeout(() => {
      document.addEventListener('mousedown', onMouseDown);
      document.addEventListener('keydown', onKey);
    }, 0);
    return () => {
      clearTimeout(id);
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [onClose]);

  // "复制"按钮：写入剪贴板并闪一个"已复制"反馈，800ms 后自动关闭。
  // 自定义菜单 preventDefault 了浏览器默认右键菜单，必须显式补上"复制"。
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(anchorText);
    } catch {
      try {
        document.execCommand('copy');
      } catch {
        // 两种 API 都失败（极少见），静默
      }
    }
    setCopied(true);
    window.setTimeout(() => {
      setCopied(false);
      onClose();
    }, 800);
  };

  // "翻译"：调 LLM（轻量模型），结果展示在原菜单位置
  const handleTranslate = async () => {
    const text = anchorText.trim();
    setTranslation({ status: 'loading' });
    try {
      const settings = await getSettings();
      const result = await apiTranslateText({
        text,
        baseUrl: settings.baseUrl,
        lightweightModel: settings.lightweightModel,
        apiKey: settings.apiKey,
      });
      setTranslation({ status: 'success', text: result });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setTranslation({ status: 'error', message: msg });
    }
  };

  const handleAddNote = () => {
    onAddNote({ anchorText, anchorOccurrence });
    onClose();
    window.getSelection()?.removeAllRanges();
  };

  const handleAiAnnotate = () => {
    onAiAnnotate?.();
    onClose();
  };

  const handleAiAsk = () => {
    onAiAsk?.();
    onClose();
  };

  return (
    <div
      data-selection-menu
      style={{
        position: 'fixed',
        top: y,
        left: x,
      }}
      // 容器自身接管 mousedown，避免全局 close handler 误关
      onMouseDown={(e) => e.stopPropagation()}
      className="z-50 min-w-[160px] overflow-hidden rounded-lg border border-border bg-surface py-1 shadow-lg"
    >
      {translation.status === 'loading' && (
        <div className="flex items-center gap-2 px-3 py-2 text-xs text-text-muted">
          <Loader2 size={12} className="animate-spin" />
          翻译中...
        </div>
      )}
      {translation.status === 'success' && (
        <div className="px-3 py-2">
          <div className="mb-1 text-xs uppercase tracking-wider text-text-muted">
            翻译
          </div>
          <div className="max-w-md whitespace-pre-wrap text-sm text-text">
            {translation.text}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="mt-2 text-xs text-text-muted hover:text-text"
          >
            关闭
          </button>
        </div>
      )}
      {translation.status === 'error' && (
        <div className="px-3 py-2">
          <div className="mb-1 text-xs uppercase tracking-wider text-red-600">
            翻译失败
          </div>
          <div className="max-w-md whitespace-pre-wrap text-xs text-red-600">
            {translation.message}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="mt-2 text-xs text-text-muted hover:text-text"
          >
            关闭
          </button>
        </div>
      )}
      {translation.status === 'idle' && (
        <>
          {copied ? (
            <div className="flex items-center gap-2 px-3 py-2 text-sm text-text-muted">
              <Check size={14} className="text-green-600" />
              已复制
            </div>
          ) : (
            <>
              <button
                type="button"
                onClick={() => void handleCopy()}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-text hover:bg-surface-2"
              >
                <Copy size={14} className="text-text-muted" />
                复制
              </button>
              <div className="my-1 border-t border-border" />
              <button
                type="button"
                onClick={handleAddNote}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-text hover:bg-surface-2"
              >
                <MessageSquarePlus size={14} className="text-text-muted" />
                添加批注
              </button>
              <button
                type="button"
                onClick={() => void handleTranslate()}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-text hover:bg-surface-2"
              >
                <Languages size={14} className="text-text-muted" />
                翻译
              </button>
              {onAiAnnotate && (
                <>
                  <div className="my-1 border-t border-border" />
                  <button
                    type="button"
                    onClick={handleAiAnnotate}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-text hover:bg-surface-2"
                  >
                    <Sparkles size={14} className="text-text-muted" />
                    AI 讲解
                  </button>
                </>
              )}
              {onAiAsk && (
                <button
                  type="button"
                  onClick={handleAiAsk}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-text hover:bg-surface-2"
                >
                  <HelpCircle size={14} className="text-text-muted" />
                  询问 AI
                </button>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}