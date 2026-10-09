import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync, writeFileSync, mkdirSync, cpSync } from 'node:fs';
import { join, relative, sep, dirname } from 'node:path';

const DOC_EXT = new Set(['.md', '.pdf', '.docx', '.pptx']);

/** 递归列出 root 下所有文件（相对路径，统一用 / 分隔）。 */
function walk(root, dir = root, out = []) {
  for (const name of readdirSync(dir)) {
    const abs = join(dir, name);
    if (statSync(abs).isDirectory()) walk(root, abs, out);
    else out.push(relative(root, abs).split(sep).join('/'));
  }
  return out;
}

/**
 * 生成 manifest。_index.md 归入 indexes（scanner 不把它当资源），
 * 未知扩展名直接忽略。seedFiles 非空时只保留这些路径 —— 用于打包 APK 种子库。
 */
export function buildManifest(root, { seedFiles = null } = {}) {
  const allow = seedFiles ? new Set(seedFiles) : null;
  const files = [];
  const indexes = [];
  for (const rel of walk(root)) {
    if (allow && !allow.has(rel)) continue;
    const buf = readFileSync(join(root, rel));
    const entry = {
      path: rel,
      size: buf.length,
      sha256: createHash('sha256').update(buf).digest('hex'),
    };
    if (rel.endsWith('/_index.md') || rel === '_index.md') indexes.push(entry);
    else if (DOC_EXT.has(rel.slice(rel.lastIndexOf('.')).toLowerCase())) files.push(entry);
  }
  files.sort((a, b) => a.path.localeCompare(b.path));
  indexes.sort((a, b) => a.path.localeCompare(b.path));
  return { version: new Date().toISOString(), files, indexes };
}

/** 种子库：只挑每个二级分类下的 _index.md —— 目录骨架可见、正文按需下载。 */
export function seedSelection(root) {
  return walk(root).filter((rel) => rel.endsWith('/_index.md') || rel === '_index.md');
}

function parseArgs(argv) {
  const out = { out: 'resources/dist', seed: false };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--out') out.out = argv[++i];
    else if (argv[i] === '--seed') out.seed = true;
  }
  return out;
}

const invokedDirectly = process.argv[1] && process.argv[1].endsWith('gen-remote-manifest.mjs');
if (invokedDirectly) {
  const args = parseArgs(process.argv.slice(2));
  const root = 'resources/knowledge';
  const manifest = args.seed
    ? buildManifest(root, { seedFiles: seedSelection(root) })
    : buildManifest(root);

  if (args.seed) {
    // 种子库交给 Tauri 打包：_index.md 文件 + seed-manifest.json。
    // _index.md 由 buildManifest 归入 indexes（不在 files），所以从这里拷。
    const seedDir = join('src-tauri', 'seed');
    mkdirSync(seedDir, { recursive: true });
    writeFileSync(join(seedDir, 'seed-manifest.json'), JSON.stringify(manifest, null, 2));
    for (const e of manifest.indexes) {
      const dest = join(seedDir, 'knowledge', e.path);
      mkdirSync(dirname(dest), { recursive: true });
      cpSync(join(root, e.path), dest);
    }
    console.log(`[seed] ${manifest.indexes.length} index files -> ${seedDir}/knowledge`);
  } else {
    mkdirSync(args.out, { recursive: true });
    writeFileSync(join(args.out, 'manifest.json'), JSON.stringify(manifest, null, 2));
    cpSync(root, join(args.out, 'knowledge'), { recursive: true });
    console.log(
      `[manifest] ${manifest.files.length} files, ${manifest.indexes.length} indexes -> ${args.out}`,
    );
    console.log('把这个目录整个传到任意静态托管即可。');
  }
}