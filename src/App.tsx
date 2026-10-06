import { useEffect, useMemo } from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import CategoryPage from './routes/CategoryPage';
import Settings from './routes/Settings';
import { useLibraryStore } from './stores/library';

/**
 * 把 unknown 路由重定向到"第一个一级分类"。
 * 一级分类从 scanner 扫到的 `categories`（parentPath === null）派生，按
 * sortOrder 升序取第一项 —— 改名/删除/添加都不会再硬编码路径。
 */
function useDefaultCategoryPath(): string {
  const categories = useLibraryStore((s) => s.categories);
  return useMemo(() => {
    const top = categories
      .filter((c) => c.parentPath === null)
      .sort((a, b) => a.sortOrder - b.sortOrder)[0];
    return top ? `/library/${top.path}` : '/library';
  }, [categories]);
}

export default function App() {
  const scan = useLibraryStore((s) => s.scan);
  const loadCachedCategories = useLibraryStore((s) => s.loadCachedCategories);
  const defaultCategoryPath = useDefaultCategoryPath();

  useEffect(() => {
    // 启动序列（顺序 await，避免两者并发时 cache 覆盖刚跑完的 scan 结果）：
    // 1. 立刻从 DB 读分类缓存 —— 后续启动此时侧栏即可渲染（不再阻塞扫描）
    // 2. 后台跑一次扫描，更新新增/删除/变动的资源
    // 首次启动（DB 空）loadCachedCategories 是 no-op，scan() 跑完后列表照常出现。
    void (async () => {
      await loadCachedCategories();
      await scan(false);
    })();
  }, [scan, loadCachedCategories]);

  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to={defaultCategoryPath} replace />} />
          <Route path="/library" element={<Navigate to={defaultCategoryPath} replace />} />
          <Route path="/library/:category" element={<CategoryPage />} />
          <Route path="/library/:category/:subPath" element={<CategoryPage />} />
          <Route path="/library/:category/:subPath/:article" element={<CategoryPage />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to={defaultCategoryPath} replace />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
