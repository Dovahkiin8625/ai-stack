import { useState } from 'react';
import { Eye, EyeOff, Loader2, CheckCircle2, XCircle, FlaskConical } from 'lucide-react';
import { testModel, type TestModelResult } from '../../lib/tauri';
import type { AppSettings } from '../../lib/tauri';

interface Props {
  /** 表单值（来自 Settings 父组件，受控）。 */
  form: AppSettings;
  /** 改任意字段时由父组件接管 —— Settings 持有最终态。 */
  onChange: (patch: Partial<AppSettings>) => void;
}

/**
 * AI 服务配置表单。
 *
 * 设计要点：
 * - 受控组件：状态由 Settings 父组件持有，本组件只渲染 + 通知。
 *   Settings 才能在最底部的全局「保存 / 保存并关闭」里一次性写入所有字段。
 * - 每个模型字段右侧有独立的「测试」按钮 —— 测出来才知道 baseUrl/key/model
 *   组合能不能跑通，节省"保存→重启→发现 401"的折返。
 * - 测试入参从当前表单值直传，未保存也能测。
 */
export default function ApiKeyForm({ form, onChange }: Props) {
  const [showKey, setShowKey] = useState(false);

  return (
    <div className="space-y-4">
      <div>
        <label className="mb-1 block text-sm font-medium">API Base URL</label>
        <input
          type="url"
          value={form.baseUrl}
          onChange={(e) => onChange({ baseUrl: e.target.value })}
          className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
        />
      </div>

      <ModelField
        label="轻量模型"
        hint="用于翻译、摘要等简单任务，不开思维链"
        placeholder="claude-haiku-4-5-..."
        value={form.lightweightModel}
        model={form.lightweightModel}
        baseUrl={form.baseUrl}
        apiKey={form.apiKey}
        onChange={(v) => onChange({ lightweightModel: v })}
      />

      <ModelField
        label="高性能模型"
        hint="用于讲解、深度分析等复杂任务"
        placeholder="claude-sonnet-..."
        value={form.performanceModel}
        model={form.performanceModel}
        baseUrl={form.baseUrl}
        apiKey={form.apiKey}
        onChange={(v) => onChange({ performanceModel: v })}
      />

      <div>
        <label className="mb-1 block text-sm font-medium">API Key</label>
        <div className="relative">
          <input
            type={showKey ? 'text' : 'password'}
            value={form.apiKey}
            onChange={(e) => onChange({ apiKey: e.target.value })}
            placeholder="sk-..."
            autoComplete="off"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 pr-10 text-sm"
          />
          <button
            type="button"
            onClick={() => setShowKey((v) => !v)}
            aria-label={showKey ? '隐藏' : '显示'}
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-text-muted hover:bg-surface-2"
          >
            {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        </div>
        <p className="mt-1 text-xs text-text-muted">
          仅保存在本机配置文件，不会上传。
        </p>
      </div>
    </div>
  );
}

/**
 * 单个模型输入行 + 「测试」按钮。
 *
 * 测试状态独立于父组件 —— 不同模型的测试结果在视觉上分开展示，
 * 单独重测某个时不影响另一个。
 */
function ModelField({
  label,
  hint,
  placeholder,
  value,
  model,
  baseUrl,
  apiKey,
  onChange,
}: {
  label: string;
  hint: string;
  placeholder: string;
  value: string;
  /** 实际发送给后端的 model ID（与 value 相等但作为单独 prop 表达"测试用"的语义）。 */
  model: string;
  baseUrl: string;
  apiKey: string;
  onChange: (v: string) => void;
}) {
  // 'idle' | 'testing' | TestModelResult。把"加载中"与"结果"放进同一个 union：
  // 'testing' 显式区分"还没回来"和"刚回来"，UI 用旋转图标 / 钩号 / 红叉表达。
  const [state, setState] = useState<'idle' | 'testing' | TestModelResult>('idle');

  async function runTest() {
    // 前端就先把"非空"卡掉 —— 节省一次无谓的后端请求往返；后端还会再校验，
    // 双层兜底避免用户输入半截状态就被误判为可用。
    if (!model.trim() || !baseUrl.trim() || !apiKey.trim()) {
      setState({
        model,
        ok: false,
        message: '请先填写 Base URL、API Key 和模型 ID',
      });
      return;
    }
    setState('testing');
    try {
      const r = await testModel({ baseUrl, apiKey, model });
      setState(r);
    } catch (e) {
      // 后端抛错（IPC 层面）只在网络 / 命令未注册时发生 —— 普通 401/404 是 ok:false。
      setState({ model, ok: false, message: String(e) });
    }
  }

  const result = state !== 'testing' && state !== 'idle' ? state : null;

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-sm font-medium">
        <label className="flex-1">
          <span>{label}</span>
          <span className="ml-2 text-xs font-normal text-text-muted">{hint}</span>
        </label>
        <button
          type="button"
          onClick={runTest}
          disabled={state === 'testing'}
          className="inline-flex shrink-0 items-center gap-1 rounded border border-border px-2 py-0.5 text-xs text-text-muted hover:bg-surface-2 disabled:opacity-50"
        >
          {state === 'testing' ? (
            <Loader2 size={12} className="animate-spin" />
          ) : (
            <FlaskConical size={12} />
          )}
          测试
        </button>
      </div>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
      />
      {state === 'testing' && (
        <p className="mt-1 flex items-center gap-1 text-xs text-text-muted">
          <Loader2 size={12} className="animate-spin" />
          正在测试…
        </p>
      )}
      {result && (
        <p
          className={`mt-1 flex items-center gap-1 text-xs ${
            result.ok ? 'text-green-600' : 'text-red-600'
          }`}
          role="status"
        >
          {result.ok ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
          {result.message}
        </p>
      )}
    </div>
  );
}
