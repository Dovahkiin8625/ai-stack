import { useMemo } from 'react';
import { ChevronRight, ChevronDown, Folder } from 'lucide-react';
import { useState } from 'react';
import type { Category } from '../../types';

interface Props {
  categories: Category[];
  selectedPath: string | null;
  onSelect: (path: string) => void;
}

interface Node {
  cat: Category;
  children: Node[];
}

function buildTree(cats: Category[]): Node[] {
  const byParent = new Map<string | null, Category[]>();
  for (const c of cats) {
    const k = c.parentPath;
    if (!byParent.has(k)) byParent.set(k, []);
    byParent.get(k)!.push(c);
  }
  const make = (parent: string | null): Node[] =>
    (byParent.get(parent) ?? [])
      .sort((a, b) => a.sortOrder - b.sortOrder || a.path.localeCompare(b.path))
      .map((c) => ({ cat: c, children: make(c.path) }));
  return make(null);
}

function NodeRow({
  node,
  depth,
  selectedPath,
  onSelect,
}: {
  node: Node;
  depth: number;
  selectedPath: string | null;
  onSelect: (path: string) => void;
}) {
  const [open, setOpen] = useState(depth === 0);
  const hasChildren = node.children.length > 0;
  const active = selectedPath === node.cat.path;
  return (
    <div>
      <button
        onClick={() => {
          if (hasChildren) setOpen((v) => !v);
          onSelect(node.cat.path);
        }}
        className={`flex w-full items-center gap-1 rounded px-2 py-1 text-left text-sm ${
          active ? 'bg-accent/10 text-accent' : 'hover:bg-surface-2'
        }`}
        style={{ paddingLeft: 8 + depth * 12 }}
      >
        {hasChildren ? (
          open ? <ChevronDown size={14} /> : <ChevronRight size={14} />
        ) : (
          <span className="inline-block w-[14px]" />
        )}
        <Folder size={14} />
        <span className="truncate">{node.cat.title}</span>
      </button>
      {open && hasChildren && (
        <div>
          {node.children.map((c) => (
            <NodeRow
              key={c.cat.path}
              node={c}
              depth={depth + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function Tree({ categories, selectedPath, onSelect }: Props) {
  const tree = useMemo(() => buildTree(categories), [categories]);
  if (tree.length === 0) {
    return <div className="p-3 text-sm text-text-muted">暂无分类</div>;
  }
  return (
    <div className="space-y-0.5">
      {tree.map((n) => (
        <NodeRow
          key={n.cat.path}
          node={n}
          depth={0}
          selectedPath={selectedPath}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}