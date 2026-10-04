import { invoke } from '@tauri-apps/api/core';
import type {
  Category,
  NewNoteInput,
  Note,
  NoteUpdateInput,
  Resource,
  ResourceContent,
  ScanSummary,
  SubcategoryIndex,
} from '../types';

export async function scanLibrary(force = false): Promise<ScanSummary> {
  return invoke<ScanSummary>('scan_library', { force });
}

export async function listCategories(): Promise<Category[]> {
  return invoke<Category[]>('list_categories');
}

export async function listResources(categoryPath: string): Promise<Resource[]> {
  return invoke<Resource[]>('list_resources', { categoryPath });
}

export async function readResource(id: number): Promise<ResourceContent> {
  return invoke<ResourceContent>('read_resource', { id });
}

/**
 * 读取并解析子分类目录下的 `_index.md`，得到结构化的条目列表
 * （含每篇文章的简介）。编辑 `_index.md` 后无需重启，调用一次即可。
 */
export async function readSubcategoryIndex(
  categoryPath: string,
): Promise<SubcategoryIndex> {
  return invoke<SubcategoryIndex>('read_subcategory_index', { categoryPath });
}

/**
 * 返回资源的原始字节，供前端 reader 自己解析：
 * - PDF：pdf.js `getDocument({data})` 直接渲染 + 自带 text layer
 * - DOCX：mammoth.js `convertToHtml({arrayBuffer})` → semantic HTML
 * - PPTX：pptxviewjs `loadFromArrayBuffer(...)` → canvas 翻页
 *
 * Tauri 2 不同 IPC 路径返回不同 JS 类型，归一化由调用方负责（见 PdfReader.tsx）。
 * 只支持 pdf / docx / pptx；其它 reader 类型不通过此命令传输字节流。
 */
export async function readResourceBytes(id: number): Promise<Uint8Array | ArrayBuffer | number[]> {
  return invoke('read_resource_bytes', { id });
}

/**
 * PDF 单页栅格化 —— 已废弃。
 *
 * 老实现：后端用 pdfium 栅格化单页 → PNG bytes → 前端 canvas + ImageBitmap 显示。
 * 缺点：纯图像层，无法选中文本。
 *
 * 新实现：前端 pdf.js 直接渲染 PDF（拿到原始字节 → `getDocument({data})`），自带 text layer。
 * 见 `readResourceBytes` 与 `PdfReader.tsx`。
 *
 * 保留函数签名只是为了让其他模块引用不报 TS 错；调用时会抛后端 "command not found"。
 */
export async function renderPdfPage(
  _resourceId: number,
  _pageIndex: number,
  _targetWidthPx: number,
): Promise<never> {
  throw new Error(
    'renderPdfPage 已废弃：PDF 改用前端 pdf.js 渲染，请调用 readResourceBytes 并交给 PdfReader',
  );
}

export async function listNotes(resourceId: number): Promise<Note[]> {
  return invoke<Note[]>('list_notes', { resourceId });
}

export async function createNote(input: NewNoteInput): Promise<Note> {
  return invoke<Note>('create_note', { payload: input });
}

export async function updateNote(input: NoteUpdateInput): Promise<Note> {
  return invoke<Note>('update_note', { payload: input });
}

export async function deleteNote(id: number): Promise<void> {
  return invoke<void>('delete_note', { id });
}

export interface TranslateInput {
  text: string;
  baseUrl: string;
  lightweightModel: string;
  apiKey: string;
}

export async function translateText(input: TranslateInput): Promise<string> {
  return invoke<string>('translate_text', { payload: input });
}

export interface AiAnnotateInput {
  resourceId: number;
  selectedText: string;
  contextBefore: string;
  contextAfter: string;
  sectionTitle: string;
  baseUrl: string;
  performanceModel: string;
  apiKey: string;
}

/**
 * 流式 AI 讲解：
 * - 立即返回占位笔记（content=""，source="ai"），前端把它放进笔记列表、打开抽屉
 * - 后端在 tokio 后台任务里拉 Anthropic SSE，逐 chunk emit "ai-annotate-chunk" 事件
 * - 全部结束 emit "ai-annotate-done"；失败 emit "ai-annotate-error" 并把错误文本写进 note.content
 * 整个过程无 loading 气泡，用户在抽屉里看到文字边生成边流入
 */
export async function startAiAnnotate(input: AiAnnotateInput): Promise<Note> {
  return invoke<Note>('start_ai_annotate', { payload: input });
}