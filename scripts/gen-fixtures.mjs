// 生成 cargo test 用的微型目录树
import { mkdir, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..', 'src-tauri', 'tests', 'fixtures', 'knowledge-tree');

async function main() {
  // 清空
  await import('node:fs/promises').then((m) => m.rm(ROOT, { recursive: true, force: true }));

  // 01-foundations/01-mathematics/_index.md + note.md
  await mkdir(join(ROOT, '01-foundations', '01-mathematics'), { recursive: true });
  await writeFile(
    join(ROOT, '01-foundations', '01-mathematics', '_index.md'),
    '# 数学基础\n\n线性代数、概率统计等。\n',
    'utf8',
  );
  await writeFile(
    join(ROOT, '01-foundations', '01-mathematics', 'linear-algebra-notes.md'),
    '# Linear Algebra Notes\n\nVectors and matrices.\n',
    'utf8',
  );
  await writeFile(
    join(ROOT, '01-foundations', '01-mathematics', 'calculus-notes.md'),
    '# Calculus Notes\n\nDerivatives and integrals.\n',
    'utf8',
  );

  // 02-deep-learning/_index.md (no resource) + sub empty
  await mkdir(join(ROOT, '02-deep-learning', 'transformers'), { recursive: true });
  await writeFile(
    join(ROOT, '02-deep-learning', '_index.md'),
    '# 深度学习\n',
    'utf8',
  );
  // 故意放一个 _index.md 但不放资源，验证 _index.md 不入 resources 表
  await writeFile(
    join(ROOT, '02-deep-learning', 'transformers', '_index.md'),
    '# Transformer\n',
    'utf8',
  );

  console.log(`fixtures written under ${ROOT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });