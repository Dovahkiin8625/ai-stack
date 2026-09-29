import { invoke } from '@tauri-apps/api/core';
import type {
  Category,
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