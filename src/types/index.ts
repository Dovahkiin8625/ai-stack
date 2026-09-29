export type RoutePath = '/library' | '/notes' | '/dashboard' | '/settings';

export interface NavItem {
  path: RoutePath;
  label: string;
  icon: string;
}
