import ApiKeyForm from '../components/settings/ApiKeyForm';

export default function Settings() {
  return (
    <div className="mx-auto max-w-2xl p-6">
      <h2 className="text-xl font-semibold">设置</h2>
      <p className="mt-1 text-sm text-text-muted">
        配置 AI 云端 API。Key 仅存本地，不上传任何服务器。
      </p>

      <section className="mt-6">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-text-muted">
          AI 服务
        </h3>
        <ApiKeyForm />
      </section>
    </div>
  );
}
