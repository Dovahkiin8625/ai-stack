import { Outlet } from 'react-router-dom';

export default function Layout() {
  return (
    <div className="flex h-full flex-col">
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
