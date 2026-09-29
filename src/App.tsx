import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Library from './routes/Library';
import Notes from './routes/Notes';
import Dashboard from './routes/Dashboard';
import Settings from './routes/Settings';

export default function App() {
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
