/**
 * Reader-agnostic anchor logic for notes:
 * - computeAnchor: 从选区算出 {anchorText, anchorOccurrence}
 * - extractContext: 取出选区两侧的"局部上下文"（章节标题 + 前后窗口）
 * - insertNoteMarkers: 在 DOM 里插入笔记角标
 * - unwrapAnchors: 移除先前插入的角标（保证 idempotent）
 *
 * 设计取舍：anchor 是"原文里出现一次"的定位标记。
 * - Markdown: 用 atoms[] 配置识别 .katex / <code> 原子单元
 * - PDF / DOCX: 不传 atoms，走 plain text 路径
 *
 * 关键概念 —— "anchor unit"：DOM 中一个可选中的最小匹配单元。
 *   - text: 文本节点；插入 marker = 拆 textNode 拼回 marker
 *   - atom: 整个元素作为不可拆的单元；插入 marker = 插到元素之后
 *     （KaTeX 公式、代码块、行内代码属于这一类，避免破坏内部 DOM）
 */

import type { Note } from '../types';

export interface AtomSelector {
  /** 选区起点的父级/根 element 满足条件即视为"起点在 atom 内"。
   *  返回 true 时 anchor 走 atom 文本，occurrence 统计"同 matcher 同文本"的 atom 数。 */
  matcher: (el: Element) => boolean;
  /** 提取 atom 的可见文本，作为 anchorText + occurrence key。 */
  textFrom: (el: Element) => string;
  /** marker 要"插到其后"的元素；默认 = matcher 命中的 element。
   *  例如 <pre> 内的 <code> 时，marker 应插到 <pre> 之后（保持 hljs / pre 整体性）。 */
  insertTarget?: (el: Element) => Element;
}

export interface AnchorConfig {
  /** 原子单元列表（按 DOM 顺序处理）；为空 = 全文 plain text。 */
  atoms?: AtomSelector[];
  /** occurrenceOfSelection 走 plain text 时要排除的 element 选择器
   *  （如 '.katex-mathml, pre, code'），避免代码块/隐藏副本干扰。 */
  occurrenceSkipSelector?: string;
}

export interface AnchorResult {
  anchorText: string;
  anchorOccurrence: number;
}

export interface ContextOptions {
  /** 上下文窗口字符数。默认 600。 */
  windowChars?: number;
}

export interface ContextResult {
  contextBefore: string;
  contextAfter: string;
  sectionTitle: string;
}

const DEFAULT_WINDOW_CHARS = 600;
const DEFAULT_HEADING_SELECTOR = 'h1, h2, h3';

/**
 * 从浏览器选区算出锚点文本与 0-based 出现位置：
 * - 选区起点若落在任一 atom 内 → anchor = atom 文本，occurrence = 前面同文本的 atom 数
 * - 否则 anchor = trim 后的选区文本，occurrence = 前面相同文本出现次数（跳过 occurrenceSkipSelector）
 */
export function computeAnchor(
  rootEl: HTMLElement,
  range: Range,
  config?: AnchorConfig,
): AnchorResult {
  const startContainer = range.startContainer;
  const startEl =
    startContainer.nodeType === Node.ELEMENT_NODE
      ? (startContainer as Element)
      : startContainer.parentElement;

  // 优先级按 atoms 数组顺序：前面的 atom 优先匹配
  if (startEl && config?.atoms) {
    for (const atom of config.atoms) {
      // closest 接受 CSS 选择器不接受 predicate，自己向上找匹配的 element
      const anchorEl = findAtomEnclosing(startEl, atom);
      if (!anchorEl) continue;
      const text = atom.textFrom(anchorEl);
      const occurrence = countPriorAtoms(rootEl, anchorEl, atom, text);
      return { anchorText: text, anchorOccurrence: occurrence };
    }
  }

  // 普通文本路径：anchor = 选区文本，occurrence 用 TreeWalker 计数
  const needle = range.toString().trim();
  if (!needle) return { anchorText: '', anchorOccurrence: 0 };
  const occurrence = occurrenceOfSelection(rootEl, needle, range, config?.occurrenceSkipSelector);
  return { anchorText: needle, anchorOccurrence: occurrence };
}

/**
 * 给出选区的"局部上下文"：最近 h1/h2/h3 标题 + 两侧最多 windowChars 字符窗口。
 * 设计取舍：仅送章节标题 + 局部窗口，不送整篇 → prompt 通常 < 2k tokens，
 * 又能让它知道用户选的内容属于哪个章节。
 */
export function extractContext(
  rootEl: HTMLElement,
  range: Range,
  opts?: ContextOptions,
): ContextResult {
  const windowChars = opts?.windowChars ?? DEFAULT_WINDOW_CHARS;

  // 最近的章节标题：遍历所有 heading，找最后一个"在 range 起点之前"的
  let sectionTitle = '';
  const startNode = range.startContainer;
  const headings = rootEl.querySelectorAll(DEFAULT_HEADING_SELECTOR);
  for (const h of Array.from(headings)) {
    const pos = startNode.compareDocumentPosition(h);
    if (pos & Node.DOCUMENT_POSITION_PRECEDING) {
      sectionTitle = (h.textContent ?? '').trim();
    } else if (pos & Node.DOCUMENT_POSITION_FOLLOWING) {
      break;
    }
  }

  // 窗口文本：用一个覆盖全文的 range 收缩到选区起点/终点
  const preRange = document.createRange();
  preRange.selectNodeContents(rootEl);
  preRange.setEnd(range.startContainer, range.startOffset);
  const beforeRaw = preRange.toString();

  const postRange = document.createRange();
  postRange.selectNodeContents(rootEl);
  postRange.setStart(range.endContainer, range.endOffset);
  const afterRaw = postRange.toString();

  const norm = (s: string) => s.replace(/\s+/g, ' ').trim();
  const beforeTrim = norm(beforeRaw);
  const afterTrim = norm(afterRaw);
  const contextBefore =
    beforeTrim.length > windowChars
      ? beforeTrim.slice(-windowChars)
      : beforeTrim;
  const contextAfter =
    afterTrim.length > windowChars
      ? afterTrim.slice(0, windowChars)
      : afterTrim;
  return { sectionTitle, contextBefore, contextAfter };
}

/**
 * 把所有带 anchor 的笔记在正文中插入角标。
 * 角标序号：按 notes 数组原序，1, 2, 3...（仅统计带 anchor 的笔记）。
 * 对每个 note，按 anchorOccurrence（0-based）找到第 N+1 次出现：
 *   - text 节点：拆 textNode，在匹配段末尾插入角标
 *   - atom：在 atom 之后插入角标（不破坏原子单元内部 DOM）
 * 长文本优先匹配，避免短 anchor 把长 anchor 截断。
 * 原文不做任何包裹/高亮处理，保持阅读体验。
 *
 * 每次循环重新收集 units —— 前面的 note 修改了 DOM，旧的 unit.node.parentNode
 * 可能已变 null；rebuild 保证后续 note 拿到最新结构。
 */
export function insertNoteMarkers(rootEl: HTMLElement, notes: Note[], config?: AnchorConfig): void {
  // 1) 按 notes 数组原序分配角标序号（与 NotesPanel 展示顺序一致）
  const numberById = new Map<number, number>();
  let anchoredIdx = 0;
  for (const n of notes) {
    if (n.anchorText && n.anchorText.trim().length > 0) {
      anchoredIdx += 1;
      numberById.set(n.id, anchoredIdx);
    }
  }

  // 2) 按 anchor 长度降序匹配；空 anchor 跳过
  const anchors = notes
    .filter((n) => n.anchorText && n.anchorText.trim().length > 0)
    .sort((a, b) => (b.anchorText!.length - a.anchorText!.length));
  if (anchors.length === 0) return;

  for (const note of anchors) {
    const needle = note.anchorText!;
    const targetOccurrence = note.anchorOccurrence;

    // 每次循环重新走一次 DOM（flat-text 视图）：前面的 note 改过 DOM，旧 unit
    // 可能已 detached；rebuild 保证后续 note 拿到最新结构。
    //
    // 设计要点：needle（用户的选区）经常会跨多个 textNode —— PDF textLayer 把
    // 每个 text item 拆一个 span，DOCX mammoth 输出常带 <strong>/<em> 切断文本，
    // markdown 的 **bold** 也拆成 <strong> 包裹。旧版按"每个 textNode 独立
    // indexOf"搜索会全部 -1 → 静默不插角标。新版把 text + atom 拼成一段
    // "logical text"，再 indexOf；找到后按 endOffset 映射回具体 textNode / atom。
    const segments = collectFlatSegments(rootEl, config);
    const fullText = segments.map((s) => s.text).join('');

    // 找第 targetOccurrence 次出现
    let count = 0;
    let from = 0;
    let endOffset = -1;
    while (true) {
      const i = fullText.indexOf(needle, from);
      if (i < 0) break;
      if (count === targetOccurrence) {
        endOffset = i + needle.length;
        break;
      }
      count += 1;
      from = i + needle.length;
    }
    if (endOffset < 0) continue;

    // 定位 endOffset 落在哪个 segment
    const seg = locateSegment(segments, endOffset);
    if (!seg) continue;

    const markerNumber = numberById.get(note.id) ?? note.id;
    const marker = createNoteMarker(note, markerNumber);

    if (seg.kind === 'atom') {
      // 匹配落在 atom 内（.katex / <code>）：marker 插到 insertAfter 之后，
      // 不破坏 atom 内部 DOM（KaTeX 公式 / hljs 高亮）。
      const parent = seg.insertAfter.parentNode;
      if (parent) parent.insertBefore(marker, seg.insertAfter.nextSibling);
    } else {
      // 匹配落在 textNode 内：拆 textNode，在匹配段末尾插入角标
      const textNode = seg.node;
      const textParent = textNode.parentNode;
      if (!textParent) continue;
      const localOffset = endOffset - seg.startOffset;
      const originalText = textNode.textContent ?? '';
      const beforeText = originalText.slice(0, localOffset);
      const afterText = originalText.slice(localOffset);

      // 决定 marker 的插入位置：
      // - offset 落在 textNode 中间 → 必须拆 textNode，marker 留在 textNode 原 parent 里
      // - offset 正好在 textNode 末尾 → 沿"是最后一个 child"链上浮到最近的
      //   容器边界，把 marker 作为其 nextSibling 插入（避免把 marker 塞进
      //   <strong>/<span> 内部；PDF textLayer 的 span、mammoth 输出的 inline
      //   元素都不希望被 marker "穿透"）。
      const isAtEnd = localOffset >= originalText.length;
      const insertion = isAtEnd
        ? findBoundaryInsertion(textNode, rootEl)
        : { parent: textParent, beforeNode: textNode };
      if (!insertion) continue;

      if (!isAtEnd) {
        // offset 在中间：拆 textNode，marker 插在中间
        const beforeNode = document.createTextNode(beforeText);
        const afterNode = document.createTextNode(afterText);
        insertion.parent.insertBefore(beforeNode, textNode);
        insertion.parent.insertBefore(marker, textNode);
        insertion.parent.insertBefore(afterNode, textNode);
        textParent.removeChild(textNode);
      } else {
        // offset 在 textNode 末尾（且已上浮到合适位置）：marker 直接插到 beforeNode 之前
        insertion.parent.insertBefore(marker, insertion.beforeNode);
      }
    }
  }
}

/**
 * 常见的 block-level 标签 —— 选区走到这种容器边界时，marker 应该停在容器**内
 * 部** current 之后，避免穿透 inline 元素（如 <strong>、<em>、PDF span）。
 *
 * 用 tagName 黑名单判断，不依赖 computedStyle（jsdom 没有 layout，也保证跨 reader
 * 行为一致：markdown <p>/<h1>、PDF textLayer 的 textLayer 容器、mammoth 输出
 * 的 <p>/<h1> 都命中 block 名单；KaTeX 公式里的 <span> 不命中，会继续上浮到
 * .katex 之外的容器）。
 */
const BLOCK_LEVEL_TAGS = new Set([
  'P', 'DIV', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
  'UL', 'OL', 'LI', 'BLOCKQUOTE', 'PRE', 'HR',
  'TABLE', 'TR', 'TD', 'TH',
  'HEADER', 'FOOTER', 'NAV', 'SECTION', 'ARTICLE', 'ASIDE', 'MAIN', 'ADDRESS',
  'FIGURE', 'FIGCAPTION', 'DL', 'DT', 'DD', 'FORM', 'BODY', 'HTML',
]);

function isBlockLevel(el: Node): boolean {
  if (el.nodeType !== Node.ELEMENT_NODE) return false;
  return BLOCK_LEVEL_TAGS.has((el as Element).tagName);
}

/**
 * 检查元素是否 absolute / fixed 定位。PDF 的 textLayer 每个 text item span
 * 都是 absolute（用 transform 平移到页面坐标），marker 必须停在 span 内
 * 而不是 textLayer 容器里 —— 否则会落到 (0,0) 看不见（容器其他孩子全部
 * absolute，正常流式位置的 marker 没有"邻居"可以参照）。
 *
 * 用 getComputedStyle 拿 computed value（CSS class / !important 都生效）。
 * jsdom 没有 layout 时 computed value 通常为空串，等价 "static"，所以测试
 * 里仍然按"非 absolute"走 block-level 上浮路径，与 markdown 行为一致。
 */
function isAbsolutelyPositioned(el: Node): boolean {
  if (el.nodeType !== Node.ELEMENT_NODE) return false;
  if (typeof window === 'undefined') return false;
  const cs = window.getComputedStyle(el as Element);
  return cs.position === 'absolute' || cs.position === 'fixed';
}

/**
 * 给定一个文本节点，假设 needle 结束位置正好在该节点末尾，
 * 沿 DOM 向上走到最近的 block-level 祖先，把 marker 插入到该祖先内、current 之后。
 *
 * 举例：
 *   <p>before <strong>world</strong> after</p>
 *   textNode = "world"，isAtEnd=true
 *   → current="world" → 上浮到 <strong>（不是 block）→ 上浮到 <p>（block）
 *     → 停在 <p>，marker 插到 <strong> 之后
 *
 *   <div><p>hello world</p></div>，textNode="hello world"
 *   → parent=<p>（block）→ 立即停，marker 作为 <p> 的末位 child
 *
 *   <p><strong>world</strong></p>，textNode="world"
 *   → <strong> 不 block → 上浮 → <p> block → 停，marker 作为 <p> 的末位 child
 *     （避免 marker 套上 strong 的粗体样式）
 */
function findBoundaryInsertion(textNode: Text, rootEl: HTMLElement): { parent: Node; beforeNode: Node | null } | null {
  let current: Node = textNode;
  let parent: Node | null = current.parentNode;
  while (parent) {
    // PDF textLayer 场景：textNode 的 parent（一个 text item span）是 absolute 定位。
    // 此时 marker 必须停在 span **内部** current 之后 —— 否则它会落在 textLayer
    // 容器的 (0,0)，因为容器所有其他孩子都是 absolute，没有"流式邻居"可参照。
    if (isAbsolutelyPositioned(parent)) {
      return { parent, beforeNode: current.nextSibling };
    }
    if (isBlockLevel(parent)) {
      // 找到 block-level 容器：在它内部、current 之后插入 marker
      return { parent, beforeNode: current.nextSibling };
    }
    // inline / 非 block 容器：继续上浮
    if (parent === rootEl) {
      // 到顶了仍非 block，append 到 rootEl 末尾兜底
      return { parent: rootEl, beforeNode: null };
    }
    current = parent;
    parent = current.parentNode;
  }
  return null;
}

/**
 * 移除先前 insertNoteMarkers 留下的 <sup.note-marker>。
 * 这是 idempotent 的前提：每次重跑前必须先清理，否则会把已有角标
 * 当作"不可见"而把同一段文字的另一次出现误判成目标 → 重复插入。
 */
export function unwrapAnchors(rootEl: HTMLElement): void {
  const markers = rootEl.querySelectorAll('sup.note-marker');
  markers.forEach((m) => {
    m.remove();
  });
  rootEl.normalize();
}

// === Internals ===

type FlatSegment =
  | { kind: 'text'; node: Text; text: string; startOffset: number }
  | { kind: 'atom'; node: Element; text: string; startOffset: number; insertAfter: Element };

/**
 * 把 root 走一遍，得到"按 document 顺序的 flat 文本序列"：
 * - text segment：每个 textNode 一段（含 occurrenceSkipSelector 过滤）
 * - atom segment：每个被 atom matcher 命中的 element 一段；其内部 textNode
 *   不再单独收集（避免 katex-mathml 副本 / hljs token 干扰）
 *
 * segments[i].startOffset 是该段在拼起来后的 fullText 中的起点。
 * 设计目的：让 needle 可以跨多个 textNode / atom 匹配，再按 endOffset
 * 映射回具体 segment —— 解决 PDF textLayer 多 span 与 DOCX 内联元素切断
 * textNode 的问题（旧版每个 textNode 单独 indexOf 永远 -1）。
 */
function collectFlatSegments(rootEl: HTMLElement, config?: AnchorConfig): FlatSegment[] {
  const segments: FlatSegment[] = [];
  const seenAtoms = new WeakSet<Element>();
  const skipSelector = config?.occurrenceSkipSelector;
  let fullTextLen = 0;
  let insideAtom: Element | null = null;

  const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_ALL);
  let node: Node | null = walker.nextNode();
  while (node) {
    if (insideAtom && !insideAtom.contains(node)) {
      insideAtom = null;
    }

    if (node.nodeType === Node.ELEMENT_NODE && !insideAtom) {
      const el = node as Element;
      if (!seenAtoms.has(el) && config?.atoms) {
        for (const atom of config.atoms) {
          if (atom.matcher(el)) {
            seenAtoms.add(el);
            const text = atom.textFrom(el);
            if (text) {
              const insertAfter = atom.insertTarget ? atom.insertTarget(el) : el;
              segments.push({
                kind: 'atom',
                node: el,
                text,
                startOffset: fullTextLen,
                insertAfter,
              });
              fullTextLen += text.length;
              insideAtom = el;
            }
            break;
          }
        }
      }
    } else if (node.nodeType === Node.TEXT_NODE && !insideAtom) {
      const textNode = node as Text;
      const parent = textNode.parentElement;
      if (!parent) {
        node = walker.nextNode();
        continue;
      }
      if (skipSelector && parent.closest(skipSelector)) {
        node = walker.nextNode();
        continue;
      }
      const text = textNode.textContent ?? '';
      if (text) {
        segments.push({
          kind: 'text',
          node: textNode,
          text,
          startOffset: fullTextLen,
        });
        fullTextLen += text.length;
      }
    }

    node = walker.nextNode();
  }
  return segments;
}

/**
 * 给定 flat 序列 + 一个 endOffset（needle 结束位置在 fullText 里的索引），
 * 返回该位置所在的 segment。
 *
 * 必须正向遍历：从第一个 segEnd >= endOffset 的 segment 才是 needle 真正
 * 落点。倒着遍历会因为"最后一个 segment 的 segEnd 总是 >= endOffset"误中
 * —— 旧版（anchor.ts 重构前 MarkdownReader 的本地版本）是倒着找的，在
 * PDF/DOCX 这种 textNode 极碎的场景里，把 marker 错插到了文档末尾的 segment。
 */
function locateSegment(segments: FlatSegment[], endOffset: number): FlatSegment | null {
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const segEnd = seg.startOffset + seg.text.length;
    if (endOffset <= segEnd) return seg;
  }
  return null;
}

function createNoteMarker(note: Note, markerNumber: number): HTMLElement {
  const marker = document.createElement('sup');
  marker.className = 'note-marker';
  marker.dataset.noteId = String(note.id);
  marker.textContent = `[${markerNumber}]`;
  marker.title = '点击查看笔记';
  return marker;
}

/**
 * 把 plain text range 的 occurrence 数出来。
 * 与 collectAnchorUnits 中 "只看 plain text" 的路径对齐 —— 通过 occurrenceSkipSelector 排除
 * atom 单元（katex-mathml / pre / code 等）以免 occurrence 偏高。
 */
function occurrenceOfSelection(
  rootEl: HTMLElement,
  needle: string,
  range: Range,
  skipSelector?: string,
): number {
  if (!needle) return 0;
  const target = range.startContainer;
  let before = '';
  const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = (node as Text).parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      if (skipSelector && parent.closest(skipSelector)) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let cur: Node | null = walker.nextNode();
  while (cur) {
    const textNode = cur as Text;
    if (textNode === target) {
      before += (textNode.textContent ?? '').slice(0, range.startOffset);
      break;
    }
    before += textNode.textContent ?? '';
    cur = walker.nextNode();
  }

  let count = 0;
  let from = 0;
  while (true) {
    const i = before.indexOf(needle, from);
    if (i < 0) break;
    count += 1;
    from = i + needle.length;
  }
  return count;
}

/** 找 closest 的、matcher 命中的 element。 */
function findAtomEnclosing(el: Element, atom: AtomSelector): Element | null {
  let cur: Element | null = el;
  while (cur) {
    if (atom.matcher(cur)) return cur;
    cur = cur.parentElement;
  }
  return null;
}

/** 统计目标 atom 之前的"同 matcher 同文本"的 atom 数。 */
function countPriorAtoms(
  rootEl: HTMLElement,
  targetEl: Element,
  atom: AtomSelector,
  text: string,
): number {
  let prior = 0;
  const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_ELEMENT);
  let el: Element | null;
  while ((el = walker.nextNode() as Element | null)) {
    if (el === targetEl) break;
    if (!atom.matcher(el)) continue;
    if (atom.textFrom(el) === text) prior += 1;
  }
  return prior;
}