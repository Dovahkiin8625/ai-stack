interface Props { html: string }

export default function MarkdownReader({ html }: Props) {
  return (
    <div className="prose prose-sm max-w-none p-6 dark:prose-invert">
      {/* comrak 输出受信任；Phase 2 不引入 DOMPurify */}
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}