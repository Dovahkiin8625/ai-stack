import { load, type Store } from '@tauri-apps/plugin-store';

/**
 * 应用全局 store：阶段 1 只存 API Key；
 * 后续阶段扩展 baseUrl、模型、温度等。
 */
const STORE_FILE = 'settings.json';
const KEY_API_KEY = 'apiKey';
const KEY_BASE_URL = 'baseUrl';
const KEY_MODEL = 'model';

export interface AppSettings {
  apiKey: string;
  baseUrl: string;
  model: string;
}

const DEFAULTS: AppSettings = {
  apiKey: '',
  baseUrl: 'https://api.anthropic.com',
  model: 'claude-sonnet-4-5',
};

let storePromise: Promise<Store> | null = null;

function getStore(): Promise<Store> {
  if (!storePromise) {
    storePromise = load(STORE_FILE, { autoSave: true });
  }
  return storePromise;
}

export async function getSettings(): Promise<AppSettings> {
  // Review Focus #1：首次启动无设置文件时优雅降级到默认值
  try {
    const store = await getStore();
    const apiKey = (await store.get<string>(KEY_API_KEY)) ?? DEFAULTS.apiKey;
    const baseUrl = (await store.get<string>(KEY_BASE_URL)) ?? DEFAULTS.baseUrl;
    const model = (await store.get<string>(KEY_MODEL)) ?? DEFAULTS.model;
    return { apiKey, baseUrl, model };
  } catch {
    return DEFAULTS;
  }
}

export async function saveSettings(s: Partial<AppSettings>): Promise<void> {
  const store = await getStore();
  if (s.apiKey !== undefined) await store.set(KEY_API_KEY, s.apiKey);
  if (s.baseUrl !== undefined) await store.set(KEY_BASE_URL, s.baseUrl);
  if (s.model !== undefined) await store.set(KEY_MODEL, s.model);
  await store.save();
}
