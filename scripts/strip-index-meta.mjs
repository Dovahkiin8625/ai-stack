#!/usr/bin/env node
/**
 * 精简 `_index.md`：删除 `## 命名约定` / `## 收录范围` / `## 命名规范`
 * 等面向作者的风格指南 section。这些在动态渲染时只对作者有用，对读者无意义。
 *
 * 用法：node scripts/strip-index-meta.mjs [--apply]
 * 默认 dry-run；加 --apply 才真改。
 */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const ROOT = path.resolve(process.cwd(), 'resources', 'knowledge');
const TARGET_HEADINGS = new Set(['命名约定', '收录范围', '命名规范']);
const APPLY = process.argv.includes('--apply');

async function* walkMarkdown(dir) {
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walkMarkdown(abs);
    else if (entry.isFile() && entry.name === '_index.md') yield abs;
  }
}

/**
 * 删掉指定 `## heading` 段（连同它下面的所有内容，直到下一个 `## ` 或 `### ` 或文件尾）。
 * 注意 `## ` 匹配二级 section；遇到 `### ` 时停止剥离。
 */
function stripSection(content, heading) {
  const lines = content.split('\n');
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const h2 = line.match(/^##\s+(.+?)\s*$/);
    if (h2 && TARGET_HEADINGS.has(h2[1])) {
      // skip until next ## or ###
      i++;
      while (i < lines.length) {
        const next = lines[i];
        if (/^##\s+/.test(next) || /^###\s+/.test(next)) break;
        i++;
      }
      continue;
    }
    out.push(line);
    i++;
  }
  // 收尾：去掉文件尾部因删除产生的多余空行（最多保留一个）
  let end = out.length;
  while (end > 0 && out[end - 1].trim() === '') end--;
  // 保留一个 trailing newline（POSIX 风格）
  return out.slice(0, end).join('\n') + '\n';
}

async function main() {
  let touched = 0;
  let stripped = 0;
  for await (const abs of walkMarkdown(ROOT)) {
    const rel = path.relative(ROOT, abs);
    const original = await fs.readFile(abs, 'utf8');
    const updated = stripSection(original, '');
    if (updated !== original) {
      const removedSections = original.match(/^##\s+(.+?)\s*$/gm)
        ?.filter((line) => {
          const m = line.match(/^##\s+(.+?)\s*$/);
          return m && TARGET_HEADINGS.has(m[1]);
        }) ?? [];
      // 只对含目标 section 的文件写回 —— 否则只是 trailing whitespace 差异
      if (removedSections.length > 0) {
        if (APPLY) {
          await fs.writeFile(abs, updated, 'utf8');
        }
        touched++;
        stripped += removedSections.length;
        console.log(
          `${APPLY ? '[stripped]' : '[dry-run]'} ${rel}: 移除 ${removedSections.join(', ')}`,
        );
      }
    }
  }
  console.log(`\n${APPLY ? '已应用' : 'Dry-run'}：扫描到 ${touched} 个文件含目标 section，共 ${stripped} 段。`);
  if (!APPLY) console.log('加 --apply 真正执行。');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
