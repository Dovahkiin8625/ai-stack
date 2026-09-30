import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AlertCircle } from 'lucide-react';
import { CATEGORIES } from '../data/categories';
import * as api from '../lib/library-api';
import { useLibraryStore } from '../stores/library';
import { useNotesStore } from '../stores/notes';
import type { Resource } from '../types';
import ResourceList from '../components/library/ResourceList';
import ReaderToolbar from '../components/library/ReaderToolbar';
import MarkdownReader, {
  type MarkdownReaderHandle,
} from '../components/library/reader/MarkdownReader';
import PdfReader from '../components/library/reader/PdfReader';
import DocxReader from '../components/library/reader/DocxReader';
import PptxReader from '../components/library/reader/PptxReader';
import SubCategoryTabs from '../components/library/SubCategoryTabs';
import NotesPanel, {
  type NotesPanelHandle,
} from '../components/library/NotesPanel';

export default function CategoryPage() {
  const { category, subPath, article } = useParams<{
    category?: string;
    subPath?: string;
    article?: string;
  }>();
  const navigate = useNavigate();

  const topCategory = useMemo(
    () => CATEGORIES.find((c) => c.path === category) ?? null,
    [category],
  );
  const subCategories = topCategory?.children ?? [];

  const [resources, setResources] = useState<Resource[]>([]);
  const [loadingResources, setLoadingResources] = useState(false);

  const currentSub = useMemo(() => {
    if (subPath) {
      const full = `${category}/${subPath}`;
      return subCategories.find((s) => s.path === full) ?? null;
    }
    return subCategories[0] ?? null;
  }, [category, subPath, subCategories]);

  const currentFullPath = currentSub?.path ?? null;

  useEffect(() => {
    if (!currentFullPath) {
      setResources([]);
      return;
    }
    let cancelled = false;
    setLoadingResources(true);
    api
      .listResources(currentFullPath)
      .then((rs) => {
        if (!cancelled) setResources(rs);
      })
      .catch((e) => {
        if (!cancelled) console.error('listResources failed', e);
        if (!cancelled) setResources([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingResources(false);
      });
    return () => {
      cancelled = true;
    };
  }, [currentFullPath]);

  const selectedResource = useMemo(() => {
    if (!article) return null;
    const articleId = Number(article);
    if (Number.isFinite(articleId)) {
      return resources.find((r) => r.id === articleId) ?? null;
    }
    const expected = `${article}.md`;
    return resources.find((r) => r.relPath === expected || r.relPath === article) ?? null;
  }, [article, resources]);

  const selectResource = useLibraryStore((s) => s.selectResource);
  const resourceContent = useLibraryStore((s) => s.resourceContent);
  const error = useLibraryStore((s) => s.error);

  useEffect(() => {
    if (selectedResource) {
      void selectResource(selectedResource.id);
    } else {
      void selectResource(null);
    }
  }, [selectedResource, selectResource]);

  // 切换资源时加载对应笔记
  const loadNotes = useNotesStore((s) => s.load);
  const clearNotes = useNotesStore((s) => s.clear);
  const notes = useNotesStore((s) => s.notes);
  const createNote = useNotesStore((s) => s.create);

  useEffect(() => {
    if (selectedResource) {
      void loadNotes(selectedResource.id);
    } else {
      clearNotes();
    }
  }, [selectedResource, loadNotes, clearNotes]);

  // 双向定位的 ref 桥
  const readerRef = useRef<MarkdownReaderHandle>(null);
  const notesPanelRef = useRef<NotesPanelHandle>(null);

  const handleSelectSub = (path: string) => {
    if (!category) return;
    const sub = path.startsWith(`${category}/`) ? path.slice(category.length + 1) : path;
    navigate(`/library/${category}/${sub}`);
  };

  const handleSelectArticle = (id: number) => {
    if (!category || !currentSub) return;
    const r = resources.find((x) => x.id === id);
    if (!r) return;
    navigate(`/library/${category}/${currentSub.path.slice(category.length + 1)}/${r.id}`);
  };

  if (!topCategory) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-text-muted">
        未找到分类 “{category}”
      </div>
    );
  }

  if (subCategories.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-text-muted">
        该分类暂无子分类
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Left column: sub-category tabs + resource list */}
      <div className="flex w-[360px] shrink-0 flex-col border-r border-border bg-surface">
        <SubCategoryTabs
          subCategories={subCategories}
          selectedPath={currentSub?.path ?? null}
          onSelect={handleSelectSub}
        />
        <div className="flex-1 overflow-y-auto p-2">
          {loadingResources ? (
            <div className="p-3 text-sm text-text-muted">加载中...</div>
          ) : (
            <ResourceList
              resources={resources}
              selectedId={selectedResource?.id ?? null}
              onSelect={handleSelectArticle}
            />
          )}
        </div>
      </div>

      {/* Middle: reader */}
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        {selectedResource ? (
          <>
            <ReaderToolbar
              title={selectedResource.title}
              type={selectedResource.type}
              pageCount={selectedResource.pageCount ?? undefined}
              wordCount={selectedResource.wordCount ?? undefined}
              onBack={() => {
                if (category && currentSub) {
                  const sub = currentSub.path.slice(category.length + 1);
                  navigate(`/library/${category}/${sub}`);
                }
              }}
            />
            <div className="flex flex-1 overflow-hidden">
              <div className="flex-1 min-h-0 overflow-hidden">
                {resourceContent ? (
                  <>
                    {resourceContent.type === 'markdown' && (
                      <MarkdownReader
                        ref={readerRef}
                        html={resourceContent.html}
                        notes={notes}
                        onMarkClick={(id) => notesPanelRef.current?.focusNote(id)}
                        onAddNoteAtSelection={async ({ anchorText, anchorOccurrence }) => {
                          const note = await createNote({
                            content: anchorText, // 默认内容 = 选中的文字，方便用户继续编辑
                            anchorText,
                            anchorOccurrence,
                          });
                          // create 后 store 已更新；下一次渲染会自动高亮新 mark
                          void note;
                        }}
                      />
                    )}
                    {resourceContent.type === 'pdf' && (
                      <PdfReader
                        pages={resourceContent.pages}
                        pageCount={resourceContent.pageCount}
                      />
                    )}
                    {resourceContent.type === 'docx' && (
                      <DocxReader
                        blocks={resourceContent.blocks}
                        wordCount={resourceContent.wordCount}
                      />
                    )}
                    {resourceContent.type === 'pptx' && (
                      <PptxReader
                        slides={resourceContent.slides}
                        slideCount={resourceContent.slideCount}
                      />
                    )}
                  </>
                ) : (
                  <div className="m-4 flex items-start gap-2 rounded-md border border-red-300 bg-red-50 p-4 text-sm text-red-700">
                    <AlertCircle size={16} className="mt-0.5 shrink-0" />
                    <div>
                      <div className="font-medium">无法读取：{selectedResource.title}</div>
                      <div className="mt-1 text-xs text-red-600">
                        {error ?? '请尝试刷新索引或更换文件'}
                      </div>
                    </div>
                  </div>
                )}
              </div>
              {selectedResource && (
                <NotesPanel
                  ref={notesPanelRef}
                  onAnchorClick={(id) => readerRef.current?.scrollToAnchor(id)}
                />
              )}
            </div>
          </>
        ) : (
          <div className="flex h-full items-center justify-center p-6 text-sm text-text-muted">
            选择左侧文章开始阅读
          </div>
        )}
      </div>
    </div>
  );
}
