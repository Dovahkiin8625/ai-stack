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
    // 2. syncNow：拉远端 manifest 并落库；落库成功时内部 scan(false) + refreshStatus
    //    都做完了，返回 true
    // 3. 仅当 syncNow 返回 false（后端 skipped 或抛错）时才自己再 scan(false) 一次
    //    —— 避免"syncNow 内 + 启动序列"走两遍（一旦 walk 380 个文件，那个耗时很显著）
    //
    // 注：success 路径不需要单独 refreshSyncStatus —— syncNow 内部已经做了一次，
    // 再来一次就是重复 IPC。skipped/throw 路径下 stay-with-stale-status 是有意的：
    // skipped 时远端没配，状态条本来就隐藏；throw 时上一次成功拉到的 status 仍可显示。
    //
    // 首次启动（DB 空）loadCachedCategories 是 no-op，syncNow skipped 或抛错，
    // scan() 兜底跑完后列表照常出现。
    void (async () => {
      await loadCachedCategories();
      const synced = await useLibraryStore.getState().syncNow();
      if (!synced) await scan(false);
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
