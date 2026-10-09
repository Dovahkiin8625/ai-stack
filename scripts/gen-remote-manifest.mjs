import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync, writeFileSync, mkdirSync, cpSync } from 'node:fs';
import { join, relative, sep, dirname } from 'node:path';

const DOC_EXT = new Set(['.md', '.pdf', '.docx', '.pptx']);

/** _index.md 的相对路径判定：根目录或任何子目录下的 _index.md 都算。 */
function isIndexPath(rel) {
  return rel.endsWith('/_index.md') || rel === '_index.md';
}

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
 * 未知扩展名直接忽略。
 *
 * 当 `seedFiles` 非空时（即 APK 种子库打包），`files` 数组必须包含所有
 * `seedFiles` 列出的路径 —— 这是因为 Android 上 `platform::parse_seed_manifest`
 * 只读 `files`。这时 `indexes` 必然为空。
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
    if (allow) {
      // seed 模式：所有显式列出的路径都进 files（包括 _index.md），
      // 由 Android 端 parse_seed_manifest 读取并 dispatch 释放。
      files.push(entry);
    } else if (isIndexPath(rel)) {
      indexes.push(entry);
    } else if (DOC_EXT.has(rel.slice(rel.lastIndexOf('.')).toLowerCase())) {
      files.push(entry);
    }
  }
  files.sort((a, b) => a.path.localeCompare(b.path));
  indexes.sort((a, b) => a.path.localeCompare(b.path));
  return { version: new Date().toISOString(), files, indexes };
}

/** 种子库：只挑每个分类下的 _index.md —— 目录骨架可见、正文按需下载。 */
export function seedSelection(root) {
  return walk(root).filter(isIndexPath);
}

/**
 * 把知识库 root 平铺拷贝到 outDir/knowledge/ 下，并写出 seed-manifest.json。
 * 用于 Tauri 打包：tauri.conf.json 把 src-tauri/seed/knowledge/ 映射为运行时的
 * knowledge/，Android 上 `materialize_seed` 按 entries 读取每个文件。
 *
 * 返回实际拷贝的文件数。
 */
export function materializeSeed(root, outDir) {
  const seedFiles = seedSelection(root);
  const manifest = buildManifest(root, { seedFiles });
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, 'seed-manifest.json'), JSON.stringify(manifest, null, 2));
  let count = 0;
  for (const e of manifest.files) {
    const dest = join(outDir, 'knowledge', e.path);
    mkdirSync(dirname(dest), { recursive: true });
    cpSync(join(root, e.path), dest);
    count++;
  }
  return count;
}

/**
 * 把知识库 root **平铺**拷贝到 outDir 下（无 knowledge/ 包裹层），并写出 manifest.json。
 * 平铺是为了让 Rust 端 `sync::ensure_local_at` 拼出的 `{base}/{category_path}/{rel_path}`
 * 直接命中 outDir 里的文件 —— 这样 sync_base_url 只需要指向 outDir 根就能同时取 manifest 和文件。
 */
export function materializeDist(root, outDir) {
  const manifest = buildManifest(root);
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2));
  cpSync(root, outDir, { recursive: true });
  return { files: manifest.files.length, indexes: manifest.indexes.length };
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
  if (args.seed) {
    const seedDir = join('src-tauri', 'seed');
    const n = materializeSeed(root, seedDir);
    console.log(`[seed] ${n} index files -> ${seedDir}/knowledge`);
  } else {
    const { files, indexes } = materializeDist(root, args.out);
    console.log(`[manifest] ${files} files, ${indexes} indexes -> ${args.out}`);
    console.log('把这个目录整个传到任意静态托管即可。sync_base_url 指向该根目录。');
  }
}
