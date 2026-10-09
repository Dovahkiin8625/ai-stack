import { useEffect, useState } from 'react';

/** 匹配 Tailwind 的 md 断点（768px）。SSR / 测试环境下默认 false = 桌面布局。
 *
 * 用 typeof 守卫 window.matchMedia —— jsdom 默认不实现，组件首渲就直接抛错。
 * 没有该 API 时按"无信号"处理 = 默认桌面（false），与 SSR 一致；
 * 下次有人加一个会渲染 Topbar/Sidebar 的组件测试时，这条守卫就是真正的护栏。
 */
export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const mq = window.matchMedia('(max-width: 767px)');
    const onChange = () => setIsMobile(mq.matches);
    onChange();
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return isMobile;
}