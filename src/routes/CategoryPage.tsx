import { useEffect, useMemo, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { AlertCircle } from 'lucide-react';
import { type UnlistenFn } from '@tauri-apps/api/event';
import * as api from '../lib/library-api';
import { useLibraryStore } from '../stores/library';
import { useNotesStore } from '../stores/notes';
import { getSettings } from '../lib/tauri';
import type { Resource } from '../types';
import MarkdownReader, {
  type MarkdownReaderHandle,
} from '../components/library/reader/MarkdownReader';
import PdfReader, {
  type PdfReaderHandle,
} from '../components/library/reader/PdfReader';
import DocxReader, {
  type DocxReaderHandle,
} from '../components/library/reader/DocxReader';
import PptxReader from '../components/library/reader/PptxReader';
import NotesPanel, {
  type NotesPanelHandle,
} from '../components/library/NotesPanel';
import ArticleIndexView, {
  type ArticleIndexSection,
} from '../components/library/ArticleIndexView';
import { subscribeAiStream } from './subscribe-ai-stream';

export default function CategoryPage() {
  const { category, subPath, article } = useParams<{
    category?: string;
    subPath?: string;
    article?: string;
  }>();

  // 分类从 DB 派生（scanner 从 resources/knowledge/ 实际目录 + _index.md H1 派生）
  const categories = useLibraryStore((s) => s.categories);

  const topCategory = useMemo(
    () => categories.find((c) => c.parentPath === null && c.path === category) ?? null,
    [categories, category],
  );
  const subCategories = useMemo(
    () => (topCategory ? categories.filter((c) => c.parentPath === topCategory.path) : []),
    [categories, topCategory],
  );

  const currentSub = useMemo(() => {
    if (subPath) {
      const full = `${category}/${subPath}`;
      return subCategories.find((s) => s.path === full) ?? null;
    }
    return subCategories[0] ?? null;
  }, [category, subPath, subCategories]);

  const currentFullPath = currentSub?.path ?? null;

  // 资源列表由 Sidebar 加载并缓存，这里仅按当前 subcategory 取一份用于派生 selectedResource
  const articlesByPath = useLibraryStore((s) => s.articlesByPath);

  const selectedResource = useMemo<Resource | null>(() => {
    if (!article) return null;
    const articleId = Number(article);
    // 尝试在当前 subcategory 缓存中找
    if (currentFullPath) {
      const list = articlesByPath[currentFullPath] ?? [];
      if (Number.isFinite(articleId)) {
        return list.find((r) => r.id === articleId) ?? null;
      }
      const expected = `${article}.md`;
      return list.find((r) => r.relPath === expected || r.relPath === article) ?? null;
    }
    return null;
  }, [article, currentFullPath, articlesByPath]);

  const selectResource = useLibraryStore((s) => s.selectResource);
  const resourceContent = useLibraryStore((s) => s.resourceContent);
  const loadingResource = useLibraryStore((s) => s.loadingResource);
  const error = useLibraryStore((s) => s.error);

  useEffect(() => {
    if (selectedResource) {
      // dirty guard：当前 MarkdownReader 还在编辑未保存 → 切换文章前确认一次
      // 阻止 selectResource 把 textarea 内容顶没。拒绝则什么都不做（URL 已经变了，
      // 用户从 sidebar 点错的文章，看不到内容就明白了）。
      if (readerRef.current?.hasUnsavedChanges()) {
        const ok = window.confirm(
          '当前文章有未保存的修改，切换后将丢弃。是否继续？',
        );
        if (!ok) return;
      }
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
  const setPanelOpen = useNotesStore((s) => s.setPanelOpen);
  const panelOpen = useNotesStore((s) => s.panelOpen);

  useEffect(() => {
    if (selectedResource) {
      void loadNotes(selectedResource.id);
    } else {
      clearNotes();
    }
  }, [selectedResource, loadNotes, clearNotes]);

  // 双向定位的 ref 桥
  const readerRef = useRef<MarkdownReaderHandle>(null);
  const pdfReaderRef = useRef<PdfReaderHandle>(null);
  const docxReaderRef = useRef<DocxReaderHandle>(null);
  const notesPanelRef = useRef<NotesPanelHandle>(null);

  // 流式 AI 讲解 / AI 问答的监听器清理函数。每次 onAiAnnotate / onAiAsk 创建一组监听，
  // done/error 时由监听器自己清理；同时组件卸载时兜底再清一遍，避免监听器泄漏。
  // **AI 讲解和 AI 问答共用同一个 ref** —— 用户不会同时跑两条流，但若有重叠，
  // 后启动的流会清掉前一组（同 AI 讲解 → AI 讲解的旧行为）。
  const aiUnlistenRef = useRef<UnlistenFn | null>(null);
  useEffect(() => {
    return () => {
      if (aiUnlistenRef.current) {
        void aiUnlistenRef.current();
        aiUnlistenRef.current = null;
      }
    };
  }, []);

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

  // 未选中具体文章时，中间区域展示该目录下所有文档的 index：
  // - 有 subPath → 单组视图（该子分类的文章）
  // - 只有 category → 分组视图（按子分类分组，每组是该子分类的文章）
  const indexSections: ArticleIndexSection[] =
    subPath && currentFullPath
      ? [
          {
            title: currentSub?.title ?? subPath,
            categoryPath: currentFullPath,
            articles: articlesByPath[currentFullPath],
          },
        ]
      : subCategories.map((sub) => ({
          title: sub.title,
          categoryPath: sub.path,
          articles: articlesByPath[sub.path],
        }));

  return (
    <div className="flex h-full flex-col">
      {selectedResource ? (
        <>
          <div className={`flex-1 overflow-hidden transition-[padding] duration-200 ${
            panelOpen ? 'md:pr-[336px]' : ''
          }`}>
            {resourceContent ? (
              <div className="h-full w-full transition-all duration-200">
                {resourceContent.type === 'markdown' && (
                  <MarkdownReader
                    ref={readerRef}
                    html={resourceContent.html}
                    markdown={resourceContent.markdown}
                    notes={notes}
                    onSaveMarkdown={async (next) => {
                      if (!selectedResource) return;
                      // write_resource 后端返回新 ResourceContent（已重新 comrak），
                      // 直接塞进 store —— 不重走 selectResource，避免 editing textarea 被清。
                      const fresh = await api.writeResource(
                        selectedResource.id,
                        next,
                      );
                      useLibraryStore.getState().updateResourceContent(fresh);
                    }}
                    onMarkClick={(id) => {
                      // 用户点击正文 mark 时主动展开抽屉，否则抽屉关闭时完全看不见笔记
                      setPanelOpen(true);
                      notesPanelRef.current?.focusNote(id);
                    }}
                    onAddNoteAtSelection={async ({ anchorText, anchorOccurrence }) => {
                      const note = await createNote({
                        content: anchorText, // 默认内容 = 选中的文字，方便用户继续编辑
                        anchorText,
                        anchorOccurrence,
                      });
                      // 添加笔记后展开抽屉，让用户看到刚创建的讲解
                      setPanelOpen(true);
                      void note;
                    }}
                    onAiAnnotate={async (input) => {
                      const settings = await getSettings();
                      // **先注册监听器**，再调 startAiAnnotate。
                      // 后端在 start_ai_annotate 内部已经 tokio::spawn 流式任务，
                      // 任务极快失败（如 401/404）时 error 事件会早 emit；
                      // 若监听器在 startAiAnnotate 之后再注册，error 事件就被错过，
                      // 笔记卡片空白。监听器自己用 activeNoteId 锁定目标笔记，
                      // 配合 store 的 pendingByNoteId 队列覆盖"事件先于 addNote"竞态。
                      if (aiUnlistenRef.current) {
                        void aiUnlistenRef.current();
                        aiUnlistenRef.current = null;
                      }
                      const cleanup = await subscribeAiStream(
                        'ai-annotate',
                        'AI annotate error',
                      );
                      aiUnlistenRef.current = cleanup as unknown as UnlistenFn;
                      const note = await api.startAiAnnotate({
                        resourceId: selectedResource.id,
                        selectedText: input.selectedText,
                        contextBefore: input.contextBefore,
                        contextAfter: input.contextAfter,
                        sectionTitle: input.sectionTitle,
                        baseUrl: settings.baseUrl,
                        performanceModel: settings.performanceModel,
                        apiKey: settings.apiKey,
                      });
                      // 把占位笔记放进 store —— 这会触发 pendingByNoteId 队列 drain，
                      // 把 listener 在 addNote 之前收到的事件一并 apply。
                      useNotesStore.getState().addNote(note);
                      // 展开抽屉 + 滚动定位到新讲解
                      setPanelOpen(true);
                      notesPanelRef.current?.focusNote(note.id);
                    }}
                    onAiAsk={async (input) => {
                      const settings = await getSettings();
                      // 同 AI 讲解：先注册监听器，再 invoke。
                      if (aiUnlistenRef.current) {
                        void aiUnlistenRef.current();
                        aiUnlistenRef.current = null;
                      }
                      const cleanup = await subscribeAiStream('ai-qa', 'AI ask error');
                      aiUnlistenRef.current = cleanup as unknown as UnlistenFn;
                      const note = await api.startAiAsk({
                        resourceId: selectedResource.id,
                        selectedText: input.selectedText,
                        contextBefore: input.contextBefore,
                        contextAfter: input.contextAfter,
                        sectionTitle: input.sectionTitle,
                        question: input.question,
                        baseUrl: settings.baseUrl,
                        performanceModel: settings.performanceModel,
                        apiKey: settings.apiKey,
                      });
                      useNotesStore.getState().addNote(note);
                      setPanelOpen(true);
                      notesPanelRef.current?.focusNote(note.id);
                    }}
                  />
                )}
                {resourceContent.type === 'pdf' && (
                  <PdfReader
                    ref={pdfReaderRef}
                    resourceId={selectedResource.id}
                    pageCount={resourceContent.pageCount}
                    notes={notes}
                    onMarkClick={(id) => {
                      setPanelOpen(true);
                      notesPanelRef.current?.focusNote(id);
                    }}
                    onAddNoteAtSelection={async ({ anchorText, anchorOccurrence, pageIdx }) => {
                      const note = await createNote({
                        content: anchorText,
                        anchorText,
                        anchorOccurrence,
                        pageIdx,
                      });
                      setPanelOpen(true);
                      void note;
                    }}
                    onAiAnnotate={async (input) => {
                      const settings = await getSettings();
                      if (aiUnlistenRef.current) {
                        void aiUnlistenRef.current();
                        aiUnlistenRef.current = null;
                      }
                      const cleanup = await subscribeAiStream(
                        'ai-annotate',
                        'AI annotate error',
                      );
                      aiUnlistenRef.current = cleanup as unknown as UnlistenFn;
                      const note = await api.startAiAnnotate({
                        resourceId: selectedResource.id,
                        selectedText: input.selectedText,
                        contextBefore: input.contextBefore,
                        contextAfter: input.contextAfter,
                        sectionTitle: input.sectionTitle,
                        pageIdx: input.pageIdx,
                        baseUrl: settings.baseUrl,
                        performanceModel: settings.performanceModel,
                        apiKey: settings.apiKey,
                      });
                      useNotesStore.getState().addNote(note);
                      setPanelOpen(true);
                      notesPanelRef.current?.focusNote(note.id);
                    }}
                    onAiAsk={async (input) => {
                      const settings = await getSettings();
                      if (aiUnlistenRef.current) {
                        void aiUnlistenRef.current();
                        aiUnlistenRef.current = null;
                      }
                      const cleanup = await subscribeAiStream('ai-qa', 'AI ask error');
                      aiUnlistenRef.current = cleanup as unknown as UnlistenFn;
                      const note = await api.startAiAsk({
                        resourceId: selectedResource.id,
                        selectedText: input.selectedText,
                        contextBefore: input.contextBefore,
                        contextAfter: input.contextAfter,
                        sectionTitle: input.sectionTitle,
                        question: input.question,
                        pageIdx: input.pageIdx,
                        baseUrl: settings.baseUrl,
                        performanceModel: settings.performanceModel,
                        apiKey: settings.apiKey,
                      });
                      useNotesStore.getState().addNote(note);
                      setPanelOpen(true);
                      notesPanelRef.current?.focusNote(note.id);
                    }}
                  />
                )}
                {resourceContent.type === 'docx' && (
                  <DocxReader
                    ref={docxReaderRef}
                    resourceId={selectedResource.id}
                    wordCount={resourceContent.wordCount}
                    notes={notes}
                    onMarkClick={(id) => {
                      setPanelOpen(true);
                      notesPanelRef.current?.focusNote(id);
                    }}
                    onAddNoteAtSelection={async ({ anchorText, anchorOccurrence }) => {
                      const note = await createNote({
                        content: anchorText,
                        anchorText,
                        anchorOccurrence,
                      });
                      setPanelOpen(true);
                      void note;
                    }}
                    onAiAnnotate={async (input) => {
                      const settings = await getSettings();
                      if (aiUnlistenRef.current) {
                        void aiUnlistenRef.current();
                        aiUnlistenRef.current = null;
                      }
                      const cleanup = await subscribeAiStream(
                        'ai-annotate',
                        'AI annotate error',
                      );
                      aiUnlistenRef.current = cleanup as unknown as UnlistenFn;
                      const note = await api.startAiAnnotate({
                        resourceId: selectedResource.id,
                        selectedText: input.selectedText,
                        contextBefore: input.contextBefore,
                        contextAfter: input.contextAfter,
                        sectionTitle: input.sectionTitle,
                        baseUrl: settings.baseUrl,
                        performanceModel: settings.performanceModel,
                        apiKey: settings.apiKey,
                      });
                      useNotesStore.getState().addNote(note);
                      setPanelOpen(true);
                      notesPanelRef.current?.focusNote(note.id);
                    }}
                    onAiAsk={async (input) => {
                      const settings = await getSettings();
                      if (aiUnlistenRef.current) {
                        void aiUnlistenRef.current();
                        aiUnlistenRef.current = null;
                      }
                      const cleanup = await subscribeAiStream('ai-qa', 'AI ask error');
                      aiUnlistenRef.current = cleanup as unknown as UnlistenFn;
                      const note = await api.startAiAsk({
                        resourceId: selectedResource.id,
                        selectedText: input.selectedText,
                        contextBefore: input.contextBefore,
                        contextAfter: input.contextAfter,
                        sectionTitle: input.sectionTitle,
                        question: input.question,
                        baseUrl: settings.baseUrl,
                        performanceModel: settings.performanceModel,
                        apiKey: settings.apiKey,
                      });
                      useNotesStore.getState().addNote(note);
                      setPanelOpen(true);
                      notesPanelRef.current?.focusNote(note.id);
                    }}
                  />
                )}
                {resourceContent.type === 'pptx' && (
                  <PptxReader
                    resourceId={selectedResource.id}
                    slideCount={resourceContent.slideCount}
                  />
                )}
              </div>
            ) : loadingResource ? (
              <div className="flex h-full items-center justify-center gap-2 text-sm text-text-muted">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-border border-t-primary" />
                正在加载 {selectedResource.title}…
              </div>
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
          <NotesPanel
            ref={notesPanelRef}
            onAnchorClick={(id) => {
              // 当前格式对应的 reader 才有 scrollToAnchor。markdown / pdf / docx 已支持；
              // pptx 暂未支持笔记功能，notesPanel 里的笔记其实是 PDF 视角（schema 一致），
              // 但 pptx 内的笔记没有对应 reader → 静默不跳转。
              if (resourceContent?.type === 'markdown') {
                readerRef.current?.scrollToAnchor(id);
              } else if (resourceContent?.type === 'pdf') {
                pdfReaderRef.current?.scrollToAnchor(id);
              } else if (resourceContent?.type === 'docx') {
                docxReaderRef.current?.scrollToAnchor(id);
              }
            }}
          />
        </>
      ) : (
        <ArticleIndexView
          mode={subPath ? 'single' : 'grouped'}
          sections={indexSections}
        />
      )}
    </div>
  );
}