import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, Save, X } from 'lucide-react';
import ApiKeyForm from '../components/settings/ApiKeyForm';
import SyncForm from '../components/settings/SyncForm';
import { getSettings, saveSettings, type AppSettings, DEFAULTS } from '../lib/tauri';
import { setSyncBaseUrl } from '../lib/sync';
import { useLibraryStore } from '../stores/library';

/**
 * 设置页。
 *
 * - 表单状态由本页持有（受控），传给两个子表单 —— 这样最底部只有一个
 *   "保存 / 保存并关闭"按钮同时写完 AI 配置 + 同步地址，不会出现每个 form
 *   都带个保存按钮的散乱状态。
 * - 「保存并关闭」：写入后 navigate 到默认分类页（"关闭"在此场景等价于
 *   离开设置页回到主页 —— 本应用没有 modal/弹窗形式的设置）。
 * - 拉取清单/开始同步的进度由 SyncForm 自己订阅 useLibraryStore，
 *   与 topbar 的 SyncStatusBar 共用同一份状态；不在本组件重复保存同步 URL。
 */
export default function Settings() {
  const navigate = useNavigate();
  const syncNow = useLibraryStore((s) => s.syncNow);
  const refreshSyncStatus = useLibraryStore((s) => s.refreshSyncStatus);

  const [form, setForm] = useState<AppSettings>(DEFAULTS);
  const [loadStatus, setLoadStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  useEffect(() => {
    setLoadStatus('loading');
    getSettings()
      .then((s) => {
        setForm(s);
        setLoadStatus('ready');
      })
      .catch(() => setLoadStatus('error'));
  }, []);

  /**
   * 把当前表单写入两个后端存储（plugin-store 给前端读、app_config 给后端命令读）。
   * 同步状态条（topbar SyncStatusBar）依赖 sync_status 命令，
   * 所以保存后顺手 refreshSyncStatus，让"已配置远端地址"立刻反映在状态条上。
   */
  async function persistAll(): Promise<void> {
    await saveSettings(form);
    await setSyncBaseUrl(form.syncBaseUrl);
    // 状态条隐含的信息是"是否配了远端 → 是否要展示下载条"；
    // 用户改了 URL 立即在状态条看到生效是合理的预期。
    await refreshSyncStatus();
    // 如果用户已经配置过远端地址，又改了一次（URL 改了），不应自动拉远端——
    // 用户没要求"保存即同步"。这里故意只刷状态不拉，遵循需求里的两段式
    // 交互："拉取清单"按钮才是触发同步的入口。
    void syncNow; // 显式标记"暂不调用"，防止后续无脑加回来。
  }

  async function onSave() {
    setSaveStatus('saving');
    try {
      await persistAll();
      setSaveStatus('saved');
      setTimeout(() => setSaveStatus('idle'), 1500);
    } catch (e) {
      console.error('[settings] save failed', e);
      setSaveStatus('error');
    }
  }

  async function onSaveAndClose() {
    setSaveStatus('saving');
    try {
      await persistAll();
      // "关闭"在路由层面没有专门的 close 动作 —— 把用户带回默认分类页即可。
      // 用 replace 而不是 push，避免用户点后退再回到已经保存过的设置页。
      navigate(-1);
    } catch (e) {
      console.error('[settings] save failed', e);
      setSaveStatus('error');
    }
  }

  if (loadStatus === 'loading') {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <div className="flex items-center gap-2 text-sm text-text-muted">
          <Loader2 size={16} className="animate-spin" />
          正在加载设置…
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <header>
        <h2 className="text-xl font-semibold">设置</h2>
        <p className="mt-1 text-sm text-text-muted">
          配置 AI 云端 API。Key 仅存本地，不上传任何服务器。
        </p>
      </header>

      <section>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-text-muted">
          AI 服务
        </h3>
        <ApiKeyForm
          form={form}
          onChange={(patch) => setForm((f) => ({ ...f, ...patch }))}
        />
      </section>

      <section>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-text-muted">
          知识库同步
        </h3>
        <SyncForm
          url={form.syncBaseUrl}
          onUrlChange={(v) => setForm((f) => ({ ...f, syncBaseUrl: v }))}
        />
      </section>

      <footer className="flex items-center gap-3 border-t border-border pt-4">
        <button
          type="button"
          onClick={onSave}
          disabled={saveStatus === 'saving'}
          className="inline-flex items-center gap-1.5 rounded-md border border-border px-4 py-2 text-sm font-medium hover:bg-surface-2 disabled:opacity-50"
        >
          {saveStatus === 'saving' ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
          保存
        </button>
        <button
          type="button"
          onClick={onSaveAndClose}
          disabled={saveStatus === 'saving'}
          className="inline-flex items-center gap-1.5 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
        >
          {saveStatus === 'saving' ? <Loader2 size={16} className="animate-spin" /> : <X size={16} />}
          保存并关闭
        </button>
        {saveStatus === 'saved' && (
          <span className="text-sm text-green-600">已保存</span>
        )}
        {saveStatus === 'error' && (
          <span className="text-sm text-red-600">保存失败</span>
        )}
        {loadStatus === 'error' && (
          <span className="text-sm text-red-600">加载设置失败</span>
        )}
      </footer>
    </div>
  );
}
