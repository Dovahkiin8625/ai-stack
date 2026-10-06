/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach } from 'vitest';
import {
  computeAnchor,
  extractContext,
  insertNoteMarkers,
  unwrapAnchors,
  type AnchorConfig,
} from './anchor';
import type { Note } from '../types';

/** 创建一个挂载根，把给定的 HTML 字符串放进去。返回 root + 一个 helper 让测试
 *  构造指向具体文本节点的 Range。 */
function mountRoot(html: string): { root: HTMLElement; rangeAt: (selector: string, text: string, occurrenceInText?: number) => Range; rangeForTextNode: (text: string) => Range | null } {
  const root = document.createElement('div');
  root.innerHTML = html;
  document.body.appendChild(root);

  const rangeAt = (selector: string, text: string, occurrenceInText = 0): Range => {
    const els = root.querySelectorAll(selector);
    let found = 0;
    for (const el of Array.from(els)) {
      if ((el.textContent ?? '').includes(text)) {
        if (found === occurrenceInText) {
          const r = document.createRange();
          r.selectNodeContents(el);
          return r;
        }
        found += 1;
      }
    }
    throw new Error(`rangeAt: no match for "${text}" in "${selector}"`);
  };

  const rangeForTextNode = (text: string): Range | null => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let n: Node | null;
    while ((n = walker.nextNode())) {
      if ((n.textContent ?? '').includes(text)) {
        const r = document.createRange();
        r.setStart(n, 0);
        r.setEnd(n, n.textContent!.length);
        return r;
      }
    }
    return null;
  };

  return { root, rangeAt, rangeForTextNode };
}

const MARKDOWN_CONFIG: AnchorConfig = {
  // 每个 atom 的 matcher 表示"该 element 就是 atom 根"：
  // - .katex：公式整体一个单元
  // - <code>：行内代码与代码块共用一个 atom 类型（marker 插到 <pre> 后，若无 pre 则插到 <code> 后）
  //   与 MarkdownReader 原行为一致 —— anchorText 都走 whitespace-collapsed textContent
  // occurrenceSkipSelector：plain text 计数时跳过这些容器，避免把代码块里的 textContent
  // 误算进 occurrence
  atoms: [
    {
      matcher: (el) => el.matches('.katex'),
      textFrom: (el) => (el.querySelector('.katex-html')?.textContent ?? '').trim(),
    },
    {
      matcher: (el) => el.matches('code'),
      textFrom: (el) => (el.textContent ?? '').replace(/\s+/g, ' ').trim(),
      insertTarget: (el) => el.closest('pre') ?? el,
    },
  ],
  occurrenceSkipSelector: '.katex-mathml, pre, code',
};

function makeNote(overrides: Partial<Note> = {}): Note {
  return {
    id: 1,
    resourceId: 1,
    content: '',
    anchorText: null,
    anchorOccurrence: 0,
    source: 'user',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

beforeEach(() => {
  document.body.innerHTML = '';
});

describe('computeAnchor — plain text selection', () => {
  it('uses trimmed selection text as anchor and counts occurrences before start', () => {
    const { root } = mountRoot(
      `<p>hello world hello <span>again</span> hello</p>`,
    );
    const span = root.querySelector('span')!;
    const text = span.firstChild!;
    const range = document.createRange();
    range.setStart(text, 0);
    range.setEnd(text, text.textContent!.length);
    const result = computeAnchor(root, range);
    expect(result.anchorText).toBe('again');
    expect(result.anchorOccurrence).toBe(0);
  });

  it('counts occurrences of identical text nodes before the selection start', () => {
    const { root } = mountRoot(
      `<p>foo bar foo <span>baz</span> foo qux foo</p>`,
    );
    const span = root.querySelector('span')!;
    const text = span.firstChild!;
    const range = document.createRange();
    range.setStart(text, 0);
    range.setEnd(text, text.textContent!.length);
    const result = computeAnchor(root, range);
    expect(result.anchorText).toBe('baz');
    expect(result.anchorOccurrence).toBe(0);
  });

  it('counts prior occurrence when same text appears multiple times before selection', () => {
    const { root } = mountRoot(
      `<p>repeat repeat repeat <span>target</span></p>`,
    );
    const span = root.querySelector('span')!;
    const text = span.firstChild!;
    const range = document.createRange();
    range.setStart(text, 0);
    range.setEnd(text, text.textContent!.length);
    const result = computeAnchor(root, range);
    expect(result.anchorText).toBe('target');
    expect(result.anchorOccurrence).toBe(0);
  });
});

describe('computeAnchor — katex atom', () => {
  it('anchors on formula text when selection is inside a .katex', () => {
    const { root, rangeAt } = mountRoot(
      `<p>before <span class="katex"><span class="katex-mathml">x^2</span><span class="katex-html">x²</span></span> after</p>`,
    );
    const range = rangeAt('.katex-html', 'x²');
    const result = computeAnchor(root, range, MARKDOWN_CONFIG);
    expect(result.anchorText).toBe('x²');
  });

  it('counts prior katex with same formula text as anchor occurrence', () => {
    const { root, rangeAt } = mountRoot(`
      <p>a <span class="katex"><span class="katex-mathml">x</span><span class="katex-html">E=mc²</span></span>
         b
        <span class="katex"><span class="katex-mathml">x</span><span class="katex-html">E=mc²</span></span>
         c
        <span class="katex"><span class="katex-mathml">x</span><span class="katex-html">E=mc²</span></span></p>
      `);
    const range = rangeAt('.katex-html', 'E=mc²', 2); // 第 3 个
    const result = computeAnchor(root, range, MARKDOWN_CONFIG);
    expect(result.anchorText).toBe('E=mc²');
    expect(result.anchorOccurrence).toBe(2);
  });
});

describe('computeAnchor — code atom', () => {
  it('anchors on code text when selection starts inside <code>', () => {
    const { root, rangeAt } = mountRoot(
      `<p>inline <code>fn main()</code> rest</p>`,
    );
    const range = rangeAt('code', 'fn main()');
    const result = computeAnchor(root, range, MARKDOWN_CONFIG);
    expect(result.anchorText).toBe('fn main()');
  });

  it('counts prior code elements with same text as anchor occurrence', () => {
    const { root, rangeAt } = mountRoot(`
      <p><code>foo()</code></p>
      <p><code>foo()</code></p>
      <p><code>foo()</code></p>
      `);
    const range = rangeAt('code', 'foo()', 2);
    const result = computeAnchor(root, range, MARKDOWN_CONFIG);
    expect(result.anchorText).toBe('foo()');
    expect(result.anchorOccurrence).toBe(2);
  });
});

describe('extractContext', () => {
  it('returns preceding h1/h2/h3 as sectionTitle', () => {
    const { root, rangeForTextNode } = mountRoot(
      `<h2>反向传播</h2><p>链式法则用于求导...</p>`,
    );
    const range = rangeForTextNode('链式法则')!;
    const ctx = extractContext(root, range);
    expect(ctx.sectionTitle).toBe('反向传播');
  });

  it('returns empty sectionTitle when no heading precedes', () => {
    const { root, rangeForTextNode } = mountRoot(`<p>plain text</p>`);
    const range = rangeForTextNode('plain text')!;
    const ctx = extractContext(root, range);
    expect(ctx.sectionTitle).toBe('');
  });

  it('returns up to windowChars chars of context before selection', () => {
    const before = 'a'.repeat(800);
    const after = 'b'.repeat(800);
    const { root, rangeForTextNode } = mountRoot(
      `<p>${before}<span>|SELECTED|</span>${after}</p>`,
    );
    const range = rangeForTextNode('|SELECTED|')!;
    const ctx = extractContext(root, range, { windowChars: 600 });
    expect(ctx.contextBefore.length).toBeLessThanOrEqual(600);
    expect(ctx.contextAfter.length).toBeLessThanOrEqual(600);
    // 末尾必须是 'a'（截尾），开头必须是 'b'（截首）
    expect(ctx.contextBefore.endsWith('a')).toBe(true);
    expect(ctx.contextAfter.startsWith('b')).toBe(true);
  });

  it('uses whole text when context is shorter than windowChars', () => {
    const { root, rangeForTextNode } = mountRoot(
      `<p><span>short</span><span>|SELECTED|</span></p>`,
    );
    const range = rangeForTextNode('|SELECTED|')!;
    const ctx = extractContext(root, range, { windowChars: 600 });
    expect(ctx.contextBefore).toContain('short');
  });
});

describe('insertNoteMarkers + unwrapAnchors', () => {
  it('inserts a <sup.note-marker> after the matched text in plain text', () => {
    const { root } = mountRoot(`<p>hello world</p>`);
    const note = makeNote({ id: 42, anchorText: 'world', anchorOccurrence: 0 });
    insertNoteMarkers(root, [note]);
    const marker = root.querySelector<HTMLElement>('sup.note-marker');
    expect(marker).not.toBeNull();
    expect(marker!.dataset.noteId).toBe('42');
    expect(marker!.textContent).toBe('[1]');
    // 文本拆成 before/after，marker 在中间
    expect(root.querySelector('p')!.textContent).toBe('hello world[1]');
  });

  it('numbers markers 1..N by notes array position', () => {
    const { root } = mountRoot(`<p>alpha beta gamma</p>`);
    insertNoteMarkers(root, [
      makeNote({ id: 1, anchorText: 'alpha', anchorOccurrence: 0 }),
      makeNote({ id: 2, anchorText: 'beta', anchorOccurrence: 0 }),
      makeNote({ id: 3, anchorText: 'gamma', anchorOccurrence: 0 }),
    ]);
    const markers = root.querySelectorAll<HTMLElement>('sup.note-marker');
    expect(markers.length).toBe(3);
    expect(markers[0].textContent).toBe('[1]');
    expect(markers[1].textContent).toBe('[2]');
    expect(markers[2].textContent).toBe('[3]');
  });

  it('skips markers for notes without anchorText', () => {
    const { root } = mountRoot(`<p>hello</p>`);
    insertNoteMarkers(root, [
      makeNote({ id: 1, anchorText: null }),
      makeNote({ id: 2, anchorText: 'hello', anchorOccurrence: 0 }),
    ]);
    // Only the anchored note gets a marker
    expect(root.querySelectorAll('sup.note-marker').length).toBe(1);
  });

  it('matches the Nth occurrence (anchorOccurrence)', () => {
    const { root } = mountRoot(`<p>foo foo foo</p>`);
    insertNoteMarkers(root, [
      makeNote({ id: 1, anchorText: 'foo', anchorOccurrence: 2 }),
    ]);
    const markers = root.querySelectorAll<HTMLElement>('sup.note-marker');
    expect(markers.length).toBe(1);
    // 第 3 个 foo 后插 marker
    expect(root.textContent).toBe('foo foo foo[1]');
  });

  it('unwrapAnchors removes all <sup.note-marker> and normalizes text nodes', () => {
    const { root } = mountRoot(`<p>hello world</p>`);
    insertNoteMarkers(root, [
      makeNote({ id: 1, anchorText: 'hello', anchorOccurrence: 0 }),
      makeNote({ id: 2, anchorText: 'world', anchorOccurrence: 0 }),
    ]);
    expect(root.querySelectorAll('sup.note-marker').length).toBe(2);
    unwrapAnchors(root);
    expect(root.querySelectorAll('sup.note-marker').length).toBe(0);
    expect(root.textContent).toBe('hello world');
  });

  it('inserts after .katex element (does not break katex DOM)', () => {
    const { root } = mountRoot(
      `<p>before <span class="katex"><span class="katex-html">x²</span></span> after</p>`,
    );
    insertNoteMarkers(root, [
      makeNote({ id: 1, anchorText: 'x²', anchorOccurrence: 0 }),
    ], MARKDOWN_CONFIG);
    const marker = root.querySelector<HTMLElement>('sup.note-marker');
    expect(marker).not.toBeNull();
    // Marker must be a sibling AFTER .katex, not inside it
    const katex = root.querySelector('.katex')!;
    expect(katex.nextElementSibling).toBe(marker);
    // .katex-html is preserved
    expect(katex.querySelector('.katex-html')).not.toBeNull();
  });

  // 回归：纯文本 + MARKDOWN_CONFIG（旧 collectAnchorUnits 同时收集 atoms 和 text，
  // 新版在 atoms 分支提前 return 漏掉 text → 普通笔记角标全消失）。
  it('matches plain text in document order even when MARKDOWN_CONFIG is provided', () => {
    const { root } = mountRoot(
      `<p>hello world <span class="katex"><span class="katex-html">x²</span></span> more text</p>`,
    );
    insertNoteMarkers(root, [
      makeNote({ id: 1, anchorText: 'hello world', anchorOccurrence: 0 }),
      makeNote({ id: 2, anchorText: 'more text', anchorOccurrence: 0 }),
    ], MARKDOWN_CONFIG);
    const markers = root.querySelectorAll<HTMLElement>('sup.note-marker');
    expect(markers.length).toBe(2);
    expect(markers[0].dataset.noteId).toBe('1');
    expect(markers[1].dataset.noteId).toBe('2');
  });

  // 纯文本里 anchorOccurrence 计数也要和 atoms 配置兼容（katex 算一个 atom，
  // plain text 算 text，二者按 document 顺序排成统一队列）。
  it('counts occurrences across text and atoms with MARKDOWN_CONFIG', () => {
    const { root } = mountRoot(
      `<p>foo <span class="katex"><span class="katex-html">bar</span></span> foo</p>`,
    );
    insertNoteMarkers(root, [
      // bar 在 katex atom 里，foo 出现两次（首尾） → 选第二个 foo
      makeNote({ id: 1, anchorText: 'foo', anchorOccurrence: 1 }),
    ], MARKDOWN_CONFIG);
    const markers = root.querySelectorAll<HTMLElement>('sup.note-marker');
    expect(markers.length).toBe(1);
    // 第二个 foo 后插角标
    expect(root.textContent).toBe('foo bar foo[1]');
  });

  // 回归：选区跨越多个 textNode（PDF textLayer 把每个 text item 拆一个 span；
  // DOCX mammoth 输出常带 <strong>/<em> 切断文本节点）。needle "hello world"
  // 分布在两个 textNode 里，旧版按 textNode 逐个 indexOf 永远 -1 → 角标消失。
  it('matches a needle that spans multiple adjacent text nodes (PDF/DOCX selection)', () => {
    const { root } = mountRoot(
      `<p>before <span>hello</span> <strong>world</strong> after</p>`,
    );
    insertNoteMarkers(root, [
      makeNote({ id: 1, anchorText: 'hello world', anchorOccurrence: 0 }),
    ]);
    const marker = root.querySelector<HTMLElement>('sup.note-marker');
    expect(marker).not.toBeNull();
    expect(marker!.dataset.noteId).toBe('1');
    // 跨节点匹配 → "world" 末尾后插 marker，<strong> 内部不会被破坏
    const strong = root.querySelector('strong')!;
    expect(strong.nextElementSibling).toBe(marker);
    expect(strong.querySelector('strong') ?? null).toBeNull();
  });

  // PDF 实际场景：每个 text item 是独立的绝对定位 span。needle 完整跨 3 个 span。
  it('matches a needle that spans 3+ adjacent text nodes (PDF textLayer layout)', () => {
    const { root } = mountRoot(
      `<div><span>hello </span><span>foo </span><span>world</span><span> tail</span></div>`,
    );
    insertNoteMarkers(root, [
      makeNote({ id: 1, anchorText: 'hello foo world', anchorOccurrence: 0 }),
    ]);
    const marker = root.querySelector<HTMLElement>('sup.note-marker');
    expect(marker).not.toBeNull();
    // marker 落在 "world" span 之后
    expect(marker!.previousElementSibling?.textContent).toBe('world');
  });

  // PDF 真实场景回归：textLayer 的每个 text item span 是 absolute 定位。
  // marker 必须停在 span **内部** current 之后 —— 否则会落到 textLayer (0,0) 看不见。
  // jsdom 不会算 computedStyle，我们用 inline style 触发判定。
  it('stays inside an absolutely-positioned span (PDF textLayer real layout)', () => {
    const root = document.createElement('div');
    root.innerHTML =
      `<span style="position: absolute; left: 100px; top: 20px;">hello </span>` +
      `<span style="position: absolute; left: 200px; top: 20px;">world</span>`;
    document.body.appendChild(root);
    insertNoteMarkers(root, [
      makeNote({ id: 1, anchorText: 'hello world', anchorOccurrence: 0 }),
    ]);
    const marker = root.querySelector<HTMLElement>('sup.note-marker');
    expect(marker).not.toBeNull();
    // marker 必须落在第二个 span 内部（继承它的 absolute 定位），不是容器的 (0,0)
    const lastSpan = root.querySelectorAll('span')[1]!;
    expect(lastSpan.contains(marker)).toBe(true);
    // 同时是 span 最后一个 child
    expect(lastSpan.lastChild).toBe(marker);
  });
});