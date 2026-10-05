import { useEffect, useRef, useState } from 'react';

/**
 * 选中询问浮层：紧贴选区出现的 mini 输入框。
 *
 * 设计取舍：
 * - 展示逻辑全部内聚在这一个组件里（无业务依赖、无 store 依赖）
 *   → MarkdownReader 只管"什么时候打开 + 定位在哪 + 提交后干什么"
 * - 验证逻辑在内部：空 / 纯空白不允许提交（按钮 disabled + Enter 不触发）
 * - 提交时 trim 输入再透传给 onSubmit，避免空白带进去污染 prompt
 *
 * 由 MarkdownReader 通过 onAiAsk 透传 answer + 不在这里维护笔记状态。
 */
export interface AskInputBoxProps {
  /** 浮层锚点位置（视口坐标），由父组件根据选区 getBoundingClientRect() 算出 */
  position: { x: number; y: number };
  /** 用户在输入框中提交 trim 后的问题文本 */
  onSubmit: (question: string) => void;
  /** 用户取消输入（Esc / 取消按钮 / 失焦） */
  onCancel: () => void;
}

export default function AskInputBox({
  position,
  onSubmit,
  onCancel,
}: AskInputBoxProps) {
  const [text, setText] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const trimmed = text.trim();
  const canSubmit = trimmed.length > 0;

  // 挂载即获焦：用户点完菜单后无需再点输入框
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = () => {
    if (!canSubmit) return;
    onSubmit(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onCancel();
    }
  };

  return (
    <div
      data-ask-input-box
      style={{
        position: 'fixed',
        top: position.y,
        left: position.x,
      }}
      // 容器接管 mousedown，避免 MarkdownReader 的"外部点击关闭菜单"误关
      onMouseDown={(e) => e.stopPropagation()}
      className="z-50 flex items-center gap-1 rounded-lg border border-border bg-surface px-1.5 py-1 shadow-lg"
    >
      <input
        ref={inputRef}
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="向 AI 提问…"
        maxLength={500}
        className="w-56 rounded border border-transparent bg-bg px-2 py-1 text-sm text-text outline-none focus:border-accent"
        aria-label="向 AI 提问"
      />
      <button
        type="button"
        onClick={onCancel}
        className="rounded px-1.5 py-0.5 text-xs text-text-muted hover:bg-surface-2 hover:text-text"
        title="取消（Esc）"
      >
        取消
      </button>
      <button
        type="button"
        onClick={handleSubmit}
        disabled={!canSubmit}
        className="rounded bg-accent px-2 py-0.5 text-xs font-medium text-white hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed"
        title="发送（Enter）"
      >
        发送
      </button>
    </div>
  );
}