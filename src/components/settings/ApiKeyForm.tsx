import { useEffect, useState } from 'react';
import { Eye, EyeOff, Save } from 'lucide-react';
import { getSettings, saveSettings, DEFAULTS, type AppSettings } from '../../lib/tauri';

export default function ApiKeyForm() {
  const [form, setForm] = useState<AppSettings>(DEFAULTS);
  const [showKey, setShowKey] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'saved' | 'error'>('idle');

  useEffect(() => {
    setStatus('loading');
    getSettings()
      .then((s) => { setForm(s); setStatus('idle'); })
      .catch(() => setStatus('error'));
  }, []);

  async function onSave(e: React.FormEvent) {
    e.preventDefault();
    try {
      await saveSettings(form);
      setStatus('saved');
      setTimeout(() => setStatus('idle'), 1500);
    } catch {
      setStatus('error');
    }
  }

  return (
    <form onSubmit={onSave} className="space-y-4">
      <div>
        <label className="mb-1 block text-sm font-medium">API Base URL</label>
        <input
          type="url"
          value={form.baseUrl}
          onChange={(e) => setForm({ ...form, baseUrl: e.target.value })}
          className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
        />
      </div>

      <div>
        <label className="mb-1 flex items-baseline justify-between text-sm font-medium">
          <span>轻量模型</span>
          <span className="text-xs font-normal text-text-muted">
            用于翻译、摘要等简单任务，不开思维链
          </span>
        </label>
        <input
          type="text"
          value={form.lightweightModel}
          onChange={(e) => setForm({ ...form, lightweightModel: e.target.value })}
          placeholder="claude-haiku-4-5-..."
          className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
        />
      </div>

      <div>
        <label className="mb-1 flex items-baseline justify-between text-sm font-medium">
          <span>高性能模型</span>
          <span className="text-xs font-normal text-text-muted">
            用于讲解、深度分析等复杂任务
          </span>
        </label>
        <input
          type="text"
          value={form.performanceModel}
          onChange={(e) => setForm({ ...form, performanceModel: e.target.value })}
          placeholder="claude-sonnet-..."
          className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
        />
      </div>

      <div>
        <label className="mb-1 block text-sm font-medium">API Key</label>
        <div className="relative">
          <input
            type={showKey ? 'text' : 'password'}
            value={form.apiKey}
            onChange={(e) => setForm({ ...form, apiKey: e.target.value })}
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

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={status === 'loading'}
          className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
        >
          <Save size={16} />
          保存
        </button>
        {status === 'saved' && <span className="text-sm text-green-600">已保存</span>}
        {status === 'error' && <span className="text-sm text-red-600">保存失败</span>}
      </div>
    </form>
  );
}
