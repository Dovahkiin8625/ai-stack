import { useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { FileText, BookOpen, FileType, Presentation, AlertTriangle, CloudDownload } from 'lucide-react';
import type { ComponentType } from 'react';
import type {
  Resource,
  ResourceType,
  ParsedSection,
  ParsedGroup,
  IndexEntry,
} from '../../types';
import { useLibraryStore } from '../../stores/library';

const RESOURCE_ICONS: Record<ResourceType, ComponentType<{ size?: number; className?: string }>> = {
  markdown: FileText,
  pdf: BookOpen,
  docx: FileType,
  pptx: Presentation,
};

/**
 * 单个文档的类型/数量 tag —— 复用 Sidebar 已有的视觉语言（图标 + 文字）。
 * Resource 只有 pageCount / wordCount 两个统计字段；pptx 把"张数"塞进 pageCount。
 */
function formatMeta(r: Resource): string {
  switch (r.type) {
    case 'markdown':
      return r.wordCount != null ? `MD · ${r.wordCount} 字` : 'MD';
    case 'pdf':
      return r.pageCount != null ? `PDF · ${r.pageCount} 页` : 'PDF';
    case 'docx':
      return r.wordCount != null ? `DOCX · ${r.wordCount} 字` : 'DOCX';
    case 'pptx':
      return r.pageCount != null ? `PPTX · ${r.pageCount} 张` : 'PPTX';
  }
}

/**
 * 用 `_index.md` 的 entry 解析结果跟 DB 里的 Resource 对齐。
 * rel_path 是 key：相同 relPath 表示同一篇。
 * - 命中的 resource：用于渲染（title 优先用 index entry 的，覆盖文件名版本以贴合中文标题）
 * - index 里提到但 DB 里没有的 entry：仍展示，点击会落到 broken 路径（罕见；不静默吞掉）
 */
function entryToResource(
  entry: IndexEntry,
  resourceByRel: Map<string, Resource>,
): { entry: IndexEntry; resource: Resource | null } {
  return { entry, resource: resourceByRel.get(entry.relPath) ?? null };
}

export interface ArticleIndexSection {
  /** 该 subcategory 在 sidebar 显示的标题（fallback —— H1 没有时用这个） */
  title: string;
  /** 完整路径（= 父 cat.path / 子 slug），用于构造导航 URL 与触发懒加载 */
  categoryPath: string;
  /** undefined = 仍在加载；空数组 = 已加载但无文章 */
  articles?: Resource[];
}

interface Props {
  /**
   * single：只渲染 sections[0]，无 H1/## 重复包装 —— 用于 "/library/:cat/:sub"。
   * grouped：每个 section 一个完整页面块（H1 + preamble + sections）—— 用于 "/library/:cat"。
   */
  mode: 'single' | 'grouped';
  sections: ArticleIndexSection[];
}

/**
 * 中间区域在「未选中具体文章」时的替代视图。
 * 渲染严格依据该 subcategory 下 `_index.md` 的结构：
 * - H1（页面标题）
 * - preamble（H1~## 之间的前言，含 blockquote）
 * - 每个 ## section 按顺序渲染；含 ### 子分组时分组，否则单层列表
 * - 末尾追加 "未在索引中" 组（DB 里有但 _index.md 没列出的文件）
 *
 * 点击列表项导航到 /library/:cat/:sub/:articleId（和 Sidebar 行为一致）。
 */
export default function ArticleIndexView({ mode, sections }: Props) {
  return (
    <div className="h-full overflow-auto p-6">
      <div
        className={
          mode === 'grouped' ? 'mx-auto max-w-3xl space-y-10' : 'mx-auto max-w-3xl'
        }
      >
        {sections.map((s) => (
          <SectionBlock key={s.categoryPath} section={s} />
        ))}
      </div>
    </div>
  );
}

function SectionBlock({
  section,
}: {
  section: ArticleIndexSection;
}) {
  const articles = useLibraryStore(
    (s) => s.articlesByPath[section.categoryPath],
  );
  const index = useLibraryStore(
    (s) => s.indexByPath[section.categoryPath],
  );
  const loadArticles = useLibraryStore((s) => s.loadArticles);
  const loadIndex = useLibraryStore((s) => s.loadIndex);

  // articlesByPath 走缓存（DB 来源，依赖 scan 重建）：只在未缓存时拉
  useEffect(() => {
    if (articles === undefined) {
      void loadArticles(section.categoryPath);
    }
  }, [articles, section.categoryPath, loadArticles]);

  // index 不缓存（_index.md 随时可编辑）：每次进入该 subcategory 都重新读
  useEffect(() => {
    void loadIndex(section.categoryPath);
  }, [section.categoryPath, loadIndex]);

  // relPath → Resource 映射，方便 entry 渲染时 O(1) 查资源
  const resourceByRel = useMemo(() => {
    const m = new Map<string, Resource>();
    if (articles) {
      for (const r of articles) m.set(r.relPath, r);
    }
    return m;
  }, [articles]);

  // DB 里有但 _index.md 没列出的文件
  const unlisted = useMemo(() => {
    if (!articles || !index) return [];
    const listed = new Set<string>();
    for (const sec of index.sections) {
      for (const g of sec.groups) {
        for (const e of g.entries) listed.add(e.relPath);
      }
    }
    return articles.filter((r) => !listed.has(r.relPath));
  }, [articles, index]);

  // index 还在加载（undefined 表示还没触发 IPC；首次 mount 时短暂处于此态）
  if (index === undefined) {
    return (
      <div>
        <h1 className="mb-3 text-xl font-semibold tracking-tight text-text">
          {section.title}
        </h1>
        <p className="text-sm text-text-muted">加载中…</p>
      </div>
    );
  }

  // index === null：该子分类没有 _index.md —— 把所有已加载文章当作"未在索引中"
  // 这样作者一眼看到"我漏写了 _index.md"，而不是空白页
  if (index === null) {
    return (
      <article>
        <h1 className="mb-3 text-xl font-semibold tracking-tight text-text">
          {section.title}
        </h1>
        {articles && articles.length > 0 ? (
          <UnlistedGroup
            heading="未在索引中（该目录缺少 _index.md）"
            articles={articles}
            categoryPath={section.categoryPath}
          />
        ) : articles === undefined ? (
          <p className="text-sm text-text-muted">加载中…</p>
        ) : (
          <p className="text-sm text-text-muted">该目录暂无文章</p>
        )}
      </article>
    );
  }

  return (
    <article>
      {/* H1 —— index.title 优先；没 H1 时 fallback 到 subcategory title */}
      <h1 className="mb-3 text-xl font-semibold tracking-tight text-text">
        {index.title ?? section.title}
      </h1>

      {/* preamble —— H1~## 之间的 markdown */}
      {index.preamble && (
        <MarkdownBlock
          markdown={index.preamble}
          className="mb-6 text-sm leading-relaxed text-text-muted prose prose-sm max-w-none prose-blockquote:border-l-2 prose-blockquote:border-border prose-blockquote:pl-3 prose-blockquote:italic"
        />
      )}

      {/* ## sections */}
      {index.sections.map((sec, secIdx) => (
        <SectionRenderer
          key={`${sec.heading}-${secIdx}`}
          section={sec}
          resourceByRel={resourceByRel}
        />
      ))}

      {/* "未在索引中" —— DB 里有但 _index.md 没列出的文件 */}
      {unlisted.length > 0 && (
        <UnlistedGroup
          heading="未在索引中"
          articles={unlisted}
          categoryPath={section.categoryPath}
        />
      )}

      {/* index 完整但 articles 还没加载 → 等 articles */}
      {articles === undefined && (
        <p className="mt-4 text-xs text-text-muted">文章列表加载中…</p>
      )}
    </article>
  );
}

function SectionRenderer({
  section,
  resourceByRel,
}: {
  section: ParsedSection;
  resourceByRel: Map<string, Resource>;
}) {
  return (
    <section className="mb-6">
      <h2 className="mb-3 text-base font-semibold text-text">{section.heading}</h2>

      {section.rawMarkdown.trim() && (
        <MarkdownBlock
          markdown={section.rawMarkdown}
          className="mb-3 text-sm leading-relaxed text-text-muted prose prose-sm max-w-none"
        />
      )}

      {section.groups.map((group, idx) => (
        <GroupRenderer
          key={`${group.subheading ?? '_'}-${idx}`}
          group={group}
          resourceByRel={resourceByRel}
        />
      ))}
    </section>
  );
}

function GroupRenderer({
  group,
  resourceByRel,
}: {
  group: ParsedGroup;
  resourceByRel: Map<string, Resource>;
}) {
  // 空 group（无 entries 又无 rawMarkdown）不渲染
  if (group.entries.length === 0 && !group.rawMarkdown.trim()) return null;

  return (
    <div className="mb-3">
      {group.subheading && (
        <h3 className="mb-2 text-sm font-medium text-text-muted">{group.subheading}</h3>
      )}
      {group.rawMarkdown.trim() && (
        <MarkdownBlock
          markdown={group.rawMarkdown}
          className="mb-2 text-sm leading-relaxed text-text-muted prose prose-sm max-w-none"
        />
      )}
      {group.entries.length > 0 && (
        <ul className="divide-y divide-border overflow-hidden rounded-md border border-border bg-surface">
          {group.entries.map((entry) => {
            const { resource } = entryToResource(entry, resourceByRel);
            return (
              <li key={entry.relPath}>
                {resource ? (
                  <EntryLink entry={entry} resource={resource} />
                ) : (
                  <MissingEntry entry={entry} />
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function EntryLink({
  entry,
  resource,
}: {
  entry: IndexEntry;
  resource: Resource;
}) {
  const slashIdx = resource.categoryPath.indexOf('/');
  const parent = slashIdx > 0 ? resource.categoryPath.slice(0, slashIdx) : resource.categoryPath;
  const subSlug = slashIdx > 0 ? resource.categoryPath.slice(slashIdx + 1) : '';
  const linkBase = subSlug ? `/library/${parent}/${subSlug}` : `/library/${parent}`;
  const Icon = RESOURCE_ICONS[resource.type];

  return (
    <Link
      to={`${linkBase}/${resource.id}`}
      className="block px-4 py-2.5 transition-colors hover:bg-surface-2"
    >
      <div className="flex items-center gap-3 text-sm">
        {Icon && <Icon size={16} className="shrink-0 text-text-muted" />}
        <span className="flex-1 truncate text-text">{entry.title || resource.title}</span>
        {/* 未缓存角标：affordance 而非 alarm —— 后端 read_resource 会自动下载，
            点开照样能读；这里只是提示"首次打开时会现拉一次"。 */}
        {!resource.present && (
          <CloudDownload size={12} className="shrink-0 text-text-muted" aria-label="未缓存" />
        )}
        <span className="shrink-0 text-xs text-text-muted">{formatMeta(resource)}</span>
      </div>
      {entry.description && (
        <p className="mt-1 line-clamp-2 pl-7 pr-2 text-xs leading-relaxed text-text-muted">
          {entry.description}
        </p>
      )}
    </Link>
  );
}

function MissingEntry({ entry }: { entry: IndexEntry }) {
  // _index.md 列了但 DB 里没有 —— 可能是文件被删 / 路径写错。
  // 仍渲染一行提示，不要静默吞掉。
  return (
    <div className="flex items-start gap-3 px-4 py-2.5 text-sm">
      <AlertTriangle size={16} className="mt-0.5 shrink-0 text-amber-500" />
      <div className="min-w-0 flex-1">
        <div className="truncate text-text-muted">{entry.title}</div>
        <div className="mt-0.5 text-xs text-text-muted">
          路径 <code className="rounded bg-surface-2 px-1">{entry.relPath}</code> 在磁盘上不存在
        </div>
      </div>
    </div>
  );
}

function UnlistedGroup({
  heading,
  articles,
  categoryPath,
}: {
  heading: string;
  articles: Resource[];
  categoryPath: string;
}) {
  const slashIdx = categoryPath.indexOf('/');
  const parent = slashIdx > 0 ? categoryPath.slice(0, slashIdx) : categoryPath;
  const subSlug = slashIdx > 0 ? categoryPath.slice(slashIdx + 1) : '';
  const linkBase = subSlug ? `/library/${parent}/${subSlug}` : `/library/${parent}`;

  return (
    <section className="mb-6">
      <h2 className="mb-3 flex items-center gap-2 text-base font-semibold text-text">
        {heading}
        <span className="rounded-full bg-surface-2 px-2 py-0.5 text-xs font-normal text-text-muted">
          {articles.length}
        </span>
      </h2>
      <ul className="divide-y divide-border overflow-hidden rounded-md border border-dashed border-border bg-surface">
        {articles.map((r) => {
          const Icon = RESOURCE_ICONS[r.type];
          return (
            <li key={r.id}>
              <Link
                to={`${linkBase}/${r.id}`}
                className="block px-4 py-2.5 transition-colors hover:bg-surface-2"
              >
                <div className="flex items-center gap-3 text-sm">
                  {Icon && <Icon size={16} className="shrink-0 text-text-muted" />}
                  <span className="flex-1 truncate text-text">{r.title}</span>
                  {/* 未缓存角标：与 EntryLink 保持一致 —— 后端 auto-download on read。 */}
                  {!r.present && (
                    <CloudDownload size={12} className="shrink-0 text-text-muted" aria-label="未缓存" />
                  )}
                  <span className="shrink-0 text-xs text-text-muted">{formatMeta(r)}</span>
                </div>
              </Link>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

/**
 * 渲染原始 markdown 的小组件。
 * - 用 react-markdown 处理（标准库，安全转义）
 * - 文章正文走 MarkdownReader（自带 KaTeX 等）；这里只渲染 _index.md 的说明性小段
 */
function MarkdownBlock({
  markdown,
  className,
}: {
  markdown: string;
  className?: string;
}) {
  return (
    <div className={className}>
      <ReactMarkdown
        components={{
          // 标题降级（_index.md 的 preamble/section 内容里出现的多余 H2/H3 不要喧宾夺主）
          h1: ({ node, ...props }) => <p {...props} />,
          a: ({ node, ...props }) => (
            <a
              {...props}
              target="_blank"
              rel="noreferrer"
              className="text-accent underline-offset-2 hover:underline"
            />
          ),
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
