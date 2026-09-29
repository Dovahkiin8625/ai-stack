import { describe, it, expect } from 'vitest';
import { CATEGORIES, type Category } from '../data/categories';

function* walk(cats: Category[]): Generator<Category> {
  for (const c of cats) {
    yield c;
    if (c.children.length > 0) yield* walk(c.children);
  }
}

describe('CATEGORIES', () => {
  it('至少包含 12 个一级分类', () => {
    expect(CATEGORIES.length).toBeGreaterThanOrEqual(12);
  });

  it('每个一级分类有 id、title、path、children 字段', () => {
    for (const c of CATEGORIES) {
      expect(c.id).toBeTruthy();
      expect(c.title).toBeTruthy();
      expect(c.path).toMatch(/^\d{2}-/);
      expect(Array.isArray(c.children)).toBe(true);
    }
  });

  it('一级分类 id 全局唯一', () => {
    const ids = new Set<string>();
    for (const c of CATEGORIES) ids.add(c.id);
    expect(ids.size).toBe(CATEGORIES.length);
  });

  it('path 排序与 id 一致', () => {
    const sorted = [...CATEGORIES].sort((a, b) => a.path.localeCompare(b.path));
    expect(sorted.map((c) => c.path)).toEqual(CATEGORIES.map((c) => c.path));
  });
});

describe('CATEGORIES 全树', () => {
  const all = [...walk(CATEGORIES)];

  it('包含子分类（不全为空）', () => {
    const withChildren = all.filter((c) => c.children.length > 0);
    expect(withChildren.length).toBeGreaterThanOrEqual(12);
  });

  it('全树 id 全局唯一（含子分类）', () => {
    const ids = all.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('全树每个节点字段齐全', () => {
    for (const c of all) {
      expect(c.id, `id missing in ${c.path}`).toBeTruthy();
      expect(c.title, `title missing in ${c.path}`).toBeTruthy();
      expect(c.path, `path missing for ${c.id}`).toBeTruthy();
      expect(c.path, `bad path format: ${c.path}`).toMatch(/^[a-z0-9-]+(\/[a-z0-9-]+)*$/);
      expect(Array.isArray(c.children)).toBe(true);
    }
  });

  it('一级 path 是二级 path 的前缀', () => {
    for (const top of CATEGORIES) {
      for (const child of top.children) {
        expect(child.path.startsWith(top.path + '/')).toBe(true);
      }
    }
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
