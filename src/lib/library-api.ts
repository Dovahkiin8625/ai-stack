import { invoke } from '@tauri-apps/api/core';
import type {
  Category,
  NewNoteInput,
  Note,
  NoteUpdateInput,
  Resource,
  ResourceContent,
  ScanSummary,
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