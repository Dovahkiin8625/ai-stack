// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor } from '@testing-library/react';

vi.mock('mammoth', () => ({
  default: {
    convertToHtml: async () => ({
      value: '<h1>第一章</h1><p>这是 DOCX 的内容，包含选区文字。</p>',
      messages: [],
    }),
    images: { dataUri: vi.fn() },
  },
}));

vi.mock('../../../lib/library-api', () => ({
  readResourceBytes: async () => new Uint8Array(),
}));

import DocxReader from './DocxReader';

beforeEach(() => {
  document.body.innerHTML = '';
});

describe('DocxReader (DOCX 阅读器 — 笔记集成)', () => {
  it('renders mammoth HTML into the host', async () => {
    const { container } = render(<DocxReader resourceId={1} wordCount={100} />);
    await waitFor(() => {
      expect(container.querySelector('.prose h1')?.textContent).toBe('第一章');
    });
  });

  it('opens SelectionMenu on right-click when text is selected', async () => {
    const { container, getByText } = render(<DocxReader resourceId={1} wordCount={100} />);
    await waitFor(() => {
      expect(container.querySelector('.prose p')).not.toBeNull();
    });
    const p = container.querySelector('.prose p')!;
    const range = document.createRange();
    range.selectNodeContents(p);
    const sel = window.getSelection();
    sel?.removeAllRanges();
    sel?.addRange(range);
    fireEvent.contextMenu(container.querySelector('.prose')!, {
      clientX: 50,
      clientY: 80,
      bubbles: true,
    });
    await waitFor(() => {
      expect(getByText('添加批注')).toBeTruthy();
    });
  });

  it('inserts <sup.note-marker> for notes with anchorText', async () => {
    const { container } = render(
      <DocxReader
        resourceId={1}
        wordCount={100}
        notes={[
          {
            id: 7,
            resourceId: 1,
            content: 'note',
            anchorText: '选区文字',
            anchorOccurrence: 0,
            source: 'user',
            createdAt: '2026-01-01T00:00:00Z',
            updatedAt: '2026-01-01T00:00:00Z',
          },
        ]}
      />,
    );
    await waitFor(() => {
      const marker = container.querySelector('sup.note-marker');
      expect(marker).not.toBeNull();
      expect(marker!.getAttribute('data-note-id')).toBe('7');
    });
  });

  it('forwards onAddNoteAtSelection with anchor info when 添加批注 clicked', async () => {
    const onAddNoteAtSelection = vi.fn();
    const { container, getByText } = render(
      <DocxReader
        resourceId={1}
        wordCount={100}
        onAddNoteAtSelection={onAddNoteAtSelection}
      />,
    );
    await waitFor(() => {
      expect(container.querySelector('.prose p')).not.toBeNull();
    });
    const p = container.querySelector('.prose p')!;
    const range = document.createRange();
    range.selectNodeContents(p);
    const sel = window.getSelection();
    sel?.removeAllRanges();
    sel?.addRange(range);
    fireEvent.contextMenu(container.querySelector('.prose')!, {
      clientX: 50,
      clientY: 80,
      bubbles: true,
    });
    const addBtn = await waitFor(() => getByText('添加批注'));
    fireEvent.click(addBtn);
    expect(onAddNoteAtSelection).toHaveBeenCalled();
    const arg = onAddNoteAtSelection.mock.calls[0][0];
    expect(arg.anchorText).toContain('选区文字');
  });
});