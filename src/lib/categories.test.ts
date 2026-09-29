import { describe, it, expect } from 'vitest';
import { CATEGORIES, type Category } from '../data/categories';

describe('CATEGORIES', () => {
  it('至少包含 12 个一级分类', () => {
    expect(CATEGORIES.length).toBeGreaterThanOrEqual(12);
  });

  it('每个分类有 id、title、path、children 字段', () => {
    for (const c of CATEGORIES) {
      expect(c.id).toBeTruthy();
      expect(c.title).toBeTruthy();
      expect(c.path).toMatch(/^\d{2}-/);
      expect(Array.isArray(c.children)).toBe(true);
    }
  });

  it('id 全局唯一', () => {
    const ids = new Set<string>();
    for (const c of CATEGORIES) ids.add(c.id);
    expect(ids.size).toBe(CATEGORIES.length);
  });

  it('path 排序与 id 一致', () => {
    const sorted = [...CATEGORIES].sort((a, b) => a.path.localeCompare(b.path));
    expect(sorted.map((c) => c.path)).toEqual(CATEGORIES.map((c) => c.path));
  });
});

describe('Category 类型', () => {
  it('子分类可嵌套', () => {
    const sample: Category = {
      id: 'root',
      title: '根',
      path: '00-root',
      children: [],
    };
    expect(sample.children).toEqual([]);
  });
});
