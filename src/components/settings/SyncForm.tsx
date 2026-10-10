import { useEffect, useState } from 'react';
import { Save } from 'lucide-react';
import { getSettings, saveSettings } from '../../lib/tauri';
import { setSyncBaseUrl } from '../../lib/sync';
import { useLibraryStore } from '../../stores/library';

export default function SyncForm() {
  const [url, setUrl] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'saved' | 'error'>('idle');
  const syncNow = useLibraryStore((s) => s.syncNow);

  useEffect(() => {
    getSettings()
      .then((s) => { setUrl(s.syncBaseUrl); setStatus('idle'); })
      .catch(() => setStatus('error'));
  }, []);

  async function onSave(e: React.FormEvent) {
    e.preventDefault();
    try {
      await saveSettings({ syncBaseUrl: url });
      // 后端也要拿到它 —— read_resource 的自动下载读的是 app_config
      await setSyncBaseUrl(url);
      await syncNow();
      setStatus('saved');
      setTimeout(() => setStatus('idle'), 1500);
    } catch {
      setStatus('error');
    }
  }

  return (
    <form onSubmit={onSave} className="space-y-3">
      <div>
        <label className="mb-1 block text-sm font-medium">知识库同步地址</label>
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://your-host/kb"
          className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm"
        />
        <p className="mt-1 text-xs text-text-muted">
          指向静态托管的发布根目录（与 <code>manifest.json</code> 同级，
          内容是 <code>&lt;分类&gt;/&lt;文件&gt;</code>，不再带 <code>knowledge/</code>）。
          留空则只用本机已缓存的内容（开发模式下仍读取仓库里的 resources/knowledge）。
        </p>
      </div>
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={status === 'loading'}
          className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
        >
          <Save size={16} />
          保存并同步
        </button>
        {status === 'saved' && <span className="text-sm text-green-600">已保存并同步</span>}
        {status === 'error' && <span className="text-sm text-red-600">同步失败</span>}
      </div>
    </form>
  );
}
