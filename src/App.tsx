import { useEffect } from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import { listen } from '@tauri-apps/api/event';
import Layout from './components/layout/Layout';
import Library from './routes/Library';
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
    return () => { unlisten.then((fn) => fn()); };
  }, [toggle]);

  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/library" replace />} />
          <Route path="/library" element={<Library />} />
          <Route path="/notes" element={<Notes />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/library" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
