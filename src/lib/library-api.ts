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