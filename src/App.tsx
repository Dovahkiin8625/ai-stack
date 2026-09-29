import { useEffect } from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import { listen } from '@tauri-apps/api/event';
import Layout from './components/layout/Layout';
import CategoryPage from './routes/CategoryPage';
import Notes from './routes/Notes';
import Dashboard from './routes/Dashboard';
import Settings from './routes/Settings';
import { useThemeStore } from './stores/theme';

export default function App() {
  const toggle = useThemeStore((s) => s.toggle);

  useEffect(() => {
    const unlisten = listen<string>('menu', (e) => {
      if (e.payload === 'toggle_theme') toggle();
      // 其他菜单项在后续阶段实现
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [toggle]);

  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/library/01-foundations" replace />} />
          <Route path="/library" element={<Navigate to="/library/01-foundations" replace />} />
          <Route path="/library/:category" element={<CategoryPage />} />
          <Route path="/library/:category/:subPath" element={<CategoryPage />} />
          <Route path="/library/:category/:subPath/:article" element={<CategoryPage />} />
          <Route path="/notes" element={<Notes />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/library/01-foundations" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}