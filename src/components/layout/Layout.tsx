import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Topbar from './Topbar';
import SyncStatusBar from '../sync/SyncStatusBar';

export default function Layout() {
  return (
    <div className="flex h-full min-w-[900px] bg-bg text-text">
      <Sidebar />
      <div className="relative flex min-w-0 flex-1 flex-col">
        <Topbar />
        <SyncStatusBar />
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
