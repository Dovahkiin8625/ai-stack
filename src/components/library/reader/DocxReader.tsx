interface Block {
  kind: 'heading' | 'paragraph' | 'list' | 'table';
  level?: 1 | 2 | 3;
  text?: string;
  ordered?: boolean;
  items?: string[];
  rows?: string[][];
}

interface Props { blocks: Block[]; wordCount: number }

export default function DocxReader({ blocks }: Props) {
  return (
    <div className="prose prose-sm max-w-none p-6 dark:prose-invert">
      {blocks.map((b, i) => {
        if (b.kind === 'heading') {
          const Tag = (`h${b.level ?? 1}` as 'h1' | 'h2' | 'h3');
          return <Tag key={i}>{b.text}</Tag>;
        }
        if (b.kind === 'paragraph') return <p key={i}>{b.text}</p>;
        if (b.kind === 'list') {
          const Tag = b.ordered ? 'ol' : 'ul';
          return (
            <Tag key={i}>
              {(b.items ?? []).map((it, j) => <li key={j}>{it}</li>)}
            </Tag>
          );
        }
        if (b.kind === 'table') {
          const rows = b.rows ?? [];
          return (
            <table key={i} className="border-collapse border border-border">
              <tbody>
                {rows.map((r, ri) => (
                  <tr key={ri}>
                    {r.map((c, ci) => <td key={ci} className="border border-border px-2 py-1">{c}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          );
        }
        return null;
      })}
    </div>
  );
}