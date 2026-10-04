import { describe, it, expect } from 'vitest';
import { CATEGORY_ICONS, iconForCategoryPath, type LucideIconName } from '../data/categories';

const KNOWN_TOP_PATHS = [
  '01-foundations',
  '02-deep-learning',
  '03-large-language-models',
  '04-computer-vision',
  '05-nlp',
  '06-speech-audio',
  '07-ai-engineering',
  '08-ai-agents',
  '09-ai-safety',
  '10-applications',
  '11-tools-ecosystem',
  '12-industry-trends',
];

describe('CATEGORY_ICONS', () => {
  it('所有已知一级分类 path 都有图标登记', () => {
    for (const path of KNOWN_TOP_PATHS) {
      expect(CATEGORY_ICONS[path], `missing icon for ${path}`).toBeTruthy();
    }
  });

  it('图标名是合法的 lucide-react 组件名集合之一', () => {
    const validNames: LucideIconName[] = [
      'BookOpen', 'Layers', 'MessageSquareText', 'Eye', 'Languages', 'Mic',
      'Wrench', 'Bot', 'Shield', 'Sparkles', 'Package', 'TrendingUp',
    ];
    for (const [path, name] of Object.entries(CATEGORY_ICONS)) {
      expect(validNames, `${path}: '${name}' 不是已知 lucide 图标名`).toContain(name);
    }
  });
});

describe('iconForCategoryPath', () => {
  it('一级分类 path 直接命中', () => {
    expect(iconForCategoryPath('02-deep-learning')).toBe('Layers');
  });

  it('子分类 path 命中其一级（最长前缀匹配）', () => {
    expect(iconForCategoryPath('02-deep-learning/transformers')).toBe('Layers');
    expect(iconForCategoryPath('04-computer-vision/3d-vision')).toBe('Eye');
  });

  it('未知 path 返回 undefined（UI 应优雅降级）', () => {
    expect(iconForCategoryPath('99-unknown')).toBeUndefined();
    expect(iconForCategoryPath('')).toBeUndefined();
  });
});
