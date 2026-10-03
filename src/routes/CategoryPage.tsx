import { useEffect, useMemo, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { AlertCircle } from 'lucide-react';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import * as api from '../lib/library-api';
import { CATEGORIES } from '../data/categories';
import { useLibraryStore } from '../stores/library';
import { useNotesStore } from '../stores/notes';
import { getSettings } from '../lib/tauri';
import type { Resource } from '../types';
import MarkdownReader, {
  type MarkdownReaderHandle,
} from '../components/library/reader/MarkdownReader';
import PdfReader from '../components/library/reader/PdfReader';
import DocxReader from '../components/library/reader/DocxReader';
import PptxReader from '../components/library/reader/PptxReader';
import NotesPanel, {
  type NotesPanelHandle,
} from '../components/library/NotesPanel';

export default function CategoryPage() {
  const { category, subPath, article } = useParams<{
    category?: string;
    subPath?: string;
    article?: string;
  }>();

  const topCategory = useMemo(
    () => CATEGORIES.find((c) => c.path === category) ?? null,
    [category],
  );
  const subCategories = topCategory?.children ?? [];

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
  const notesPanelRef = useRef<NotesPanelHandle>(null);

  // 流式 AI 讲解的监听器清理函数。每次 onAiAnnotate 创建一组监听，
  // done/error 时由监听器自己清理；同时组件卸载时兜底再清一遍，避免监听器泄漏。
  const aiUnlistenRef = useRef<UnlistenFn | null>(null);
  useEffect(() => {
    return () => {
      if (aiUnlistenRef.current) {
        void aiUnlistenRef.current();
        aiUnlistenRef.current = null;
      }
    };
  }, []);

  /**
   * 注册流式 AI 讲解的监听器：
   * - ai-annotate-chunk：拼到 note.content
   * - ai-annotate-done / ai-annotate-error：清理监听器
   *
   * 不再要求调用方传入 noteId：listener 一律注册，根据第一个事件动态识别
   * "当前讲解"。这样监听器可以在 startAiAnnotate 之前就完成 setup，
   * 不会错过后端极快失败时早 emit 的 error 事件。
   *
   * 事件落到 store 时由 store 的 applyOrQueue 兜底 —— 笔记还没入 store
   * 就先排队等 addNote 时 drain，彻底消除"event 先于 addNote" 的竞态。
   */
  const subscribeAiAnnotation = async () => {
    // 先清理上一组（如果存在）—— 同一时刻只跟一个 note 互动
    if (aiUnlistenRef.current) {
      void aiUnlistenRef.current();
      aiUnlistenRef.current = null;
    }

    const unlistens: UnlistenFn[] = [];

    // 同一时刻只有一个 AI 讲解在跑，用第一个事件锁定 noteId，
    // 后续事件按 noteId 过滤，防止旧 listener 串扰到下一轮。
    let activeNoteId: number | null = null;

    const uChunk = await listen<{ noteId: number; text: string }>(
      'ai-annotate-chunk',
      (e) => {
        if (activeNoteId == null) activeNoteId = e.payload.noteId;
        if (e.payload.noteId !== activeNoteId) return;
        useNotesStore.getState().appendToNote(activeNoteId, e.payload.text);
      },
    );
    unlistens.push(uChunk);

    const cleanup = async () => {
      for (const u of unlistens) await u();
      activeNoteId = null;
    };

    const uDone = await listen<number>('ai-annotate-done', (e) => {
      if (activeNoteId == null) activeNoteId = e.payload;
      if (e.payload !== activeNoteId) return;
      void cleanup();
      if (aiUnlistenRef.current === uDone) aiUnlistenRef.current = null;
    });
    unlistens.push(uDone);

    const uError = await listen<{ noteId: number; message: string; finalContent: string }>(
      'ai-annotate-error',
      (e) => {
        if (activeNoteId == null) activeNoteId = e.payload.noteId;
        if (e.payload.noteId !== activeNoteId) return;
        console.error('AI annotate error:', e.payload.message);
        // 同步本地 store：后端已经把错误描述写进 DB，但前端占位卡片仍是空字符串。
        // 把 final_content 套回去，让用户在面板里立刻看到失败原因（而不是一个空卡片）。
        useNotesStore.getState().setNoteContent(activeNoteId, e.payload.finalContent);
        void cleanup();
        if (aiUnlistenRef.current === uError) aiUnlistenRef.current = null;
      },
    );
    unlistens.push(uError);

    // 监听器 cleanup 整体暴露给 useEffect 卸载时兜底调用
    aiUnlistenRef.current = cleanup as unknown as UnlistenFn;
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
    <div className="flex h-full flex-col">
      {selectedResource ? (
        <>
          <div className={`flex-1 overflow-hidden transition-[padding] duration-200 ${
            panelOpen ? 'pr-[336px]' : ''
          }`}>
            {resourceContent ? (
              <div className="h-full w-full transition-all duration-200">
                {resourceContent.type === 'markdown' && (
                  <MarkdownReader
                    ref={readerRef}
                    html={resourceContent.html}
                    notes={notes}
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
                      await subscribeAiAnnotation();
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
                  />
                )}
                {resourceContent.type === 'pdf' && (
                  <PdfReader
                    resourceId={selectedResource.id}
                    pageCount={resourceContent.pageCount}
                  />
                )}
                {resourceContent.type === 'docx' && (
                  <DocxReader
                    resourceId={selectedResource.id}
                    wordCount={resourceContent.wordCount}
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
            onAnchorClick={(id) => readerRef.current?.scrollToAnchor(id)}
          />
        </>
      ) : (
        <div className="flex h-full items-center justify-center p-6 text-sm text-text-muted">
          选择左侧文章开始阅读
        </div>
      )}
    </div>
  );
}