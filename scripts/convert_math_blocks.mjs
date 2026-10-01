// Convert ```math ... ``` blocks to $$ ... $$ across all knowledge articles.
//
// 形如：
//   ```math
//   H(P) = -\sum_x P(x) \log P(x)
//   ```
// 替换为：
//   $$
//   H(P) = -\sum_x P(x) \log P(x)
//   $$
//
// 用 Node 跑（项目已经有 Node 工具链，Python 没装）。非贪婪匹配让
// 同一文件多个 math 块互不影响；DOTALL 用 `s` flag 让 `.` 跨换行。
import { readdirSync, statSync, readFileSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = 'C:/project/ai-stack/resources/knowledge';

const PATTERN = /^```math[ \t]*\n([\s\S]+?)^```[ \t]*$/gm;

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) yield* walk(p);
    else if (p.endsWith('.md')) yield p;
  }
}

let totalFiles = 0;
let totalBlocks = 0;
const out = [];
for (const p of walk(ROOT)) {
  const text = readFileSync(p, 'utf8');
  let n = 0;
  const newText = text.replace(PATTERN, (_full, b) => {
    n += 1;
    return '$$\n' + b + '$$';
  });
  if (n > 0) {
    writeFileSync(p, newText, 'utf8');
    totalFiles += 1;
    totalBlocks += n;
    out.push(`  ${relative(join(ROOT, '..'), p)}: ${n}`);
  }
}

console.log(`converted ${totalBlocks} blocks across ${totalFiles} files`);
for (const line of out) console.log(line);