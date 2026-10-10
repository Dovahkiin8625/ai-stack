import { load, type Store } from '@tauri-apps/plugin-store';

/**
 * 应用全局 store：阶段 1 只存 API Key；
 * 后续阶段扩展 baseUrl、模型、温度等。
 */
const STORE_FILE = 'settings.json';
const KEY_API_KEY = 'apiKey';
const KEY_BASE_URL = 'baseUrl';
const KEY_MODEL_LIGHTWEIGHT = 'lightweightModel';
const KEY_MODEL_PERFORMANCE = 'performanceModel';
/** 旧版单一 model 字段，仅在 getSettings 时做一次性迁移读取。 */
const KEY_MODEL_LEGACY = 'model';
const KEY_SYNC_BASE_URL = 'syncBaseUrl';

export interface AppSettings {
  apiKey: string;
  baseUrl: string;
  /** 轻量模型：翻译、摘要等简单任务。请求时不带 thinking 字段。 */
  lightweightModel: string;
  /** 高性能模型：讲解、深度分析等复杂任务。可启用 thinking。 */
  performanceModel: string;
  /** 知识库远端同步静态托管根目录（manifest.json 与各分类目录平铺的发布树）。空字符串 = 未配置，纯本地模式。 */
  syncBaseUrl: string;
}

const DEFAULTS: AppSettings = {
  apiKey: '',
  baseUrl: 'https://api.anthropic.com',
  // Haiku 4.5：响应快、价格低，适合单词/句子级翻译
  lightweightModel: 'claude-haiku-4-5-20251001',
  // Sonnet 4.5：能力强、可挂 thinking，适合讲解/分析类任务
  performanceModel: 'claude-sonnet-4-5',
  syncBaseUrl: '',
};

export { DEFAULTS };

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
    let lightweightModel =
      (await store.get<string>(KEY_MODEL_LIGHTWEIGHT)) ?? '';
    let performanceModel =
      (await store.get<string>(KEY_MODEL_PERFORMANCE)) ?? '';
    // 一次性迁移：旧版只存了单个 `model` 字段时，两个层级都先用它
    // （用户下次保存后会拆开）。避免升级后变成空白。
    if (!lightweightModel && !performanceModel) {
      const legacy = (await store.get<string>(KEY_MODEL_LEGACY)) ?? '';
      if (legacy) {
        lightweightModel = legacy;
        performanceModel = legacy;
      }
    }
    return {
      apiKey,
      baseUrl,
      lightweightModel: lightweightModel || DEFAULTS.lightweightModel,
      performanceModel: performanceModel || DEFAULTS.performanceModel,
      syncBaseUrl: (await store.get<string>(KEY_SYNC_BASE_URL)) ?? DEFAULTS.syncBaseUrl,
    };
  } catch {
    return DEFAULTS;
  }
}

export async function saveSettings(s: Partial<AppSettings>): Promise<void> {
  const store = await getStore();
  if (s.apiKey !== undefined) await store.set(KEY_API_KEY, s.apiKey);
  if (s.baseUrl !== undefined) await store.set(KEY_BASE_URL, s.baseUrl);
  if (s.lightweightModel !== undefined)
    await store.set(KEY_MODEL_LIGHTWEIGHT, s.lightweightModel);
  if (s.performanceModel !== undefined)
    await store.set(KEY_MODEL_PERFORMANCE, s.performanceModel);
  if (s.syncBaseUrl !== undefined)
    await store.set(KEY_SYNC_BASE_URL, s.syncBaseUrl);
  await store.save();
}
