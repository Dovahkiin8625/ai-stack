import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildManifest, seedSelection, materializeSeed, materializeDist } from './gen-remote-manifest.mjs';

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'kb-'));
  mkdirSync(join(root, '01-基础', '回归'), { recursive: true });
  mkdirSync(join(root, '01-基础', '三方资料'), { recursive: true });
  writeFileSync(join(root, '01-基础', '_index.md'), '# 基础\n');
  writeFileSync(join(root, '01-基础', '回归', '线性.md'), 'linear\n');
  writeFileSync(join(root, '01-基础', '三方资料', 'book.pdf'), 'PDF');
  writeFileSync(join(root, '01-基础', 'notes.txt'), 'ignored');
  return root;
}

test('manifest 只收 md/pdf/docx/pptx，_index.md 进 indexes', () => {
  const { files, indexes } = buildManifest(fixture());
  assert.deepEqual(files.map((f) => f.path).sort(), ['01-基础/三方资料/book.pdf', '01-基础/回归/线性.md']);
  assert.deepEqual(indexes.map((f) => f.path), ['01-基础/_index.md']);
});

test('每条都带 size 与 sha256', () => {
  const { files } = buildManifest(fixture());
  const md = files.find((f) => f.path.endsWith('.md'));
  assert.equal(md.size, Buffer.byteLength('linear\n'));
  assert.match(md.sha256, /^[0-9a-f]{64}$/);
});

test('seed 子集把所有显式列出的路径都放进 files（Rust 端的 parse_seed_manifest 只读 files）', () => {
  const root = fixture();
  const seedFiles = seedSelection(root);
  // seedSelection 默认挑的就是 _index.md —— 关键断言：它们必须出现在 files
  // 而不是 indexes，否则 Android 上 parse_seed_manifest 拿到空 Vec，
  // APK 里打的 177 KB 数据根本没人读，UI 上看到的是空库。
  const { files, indexes } = buildManifest(root, { seedFiles });
  assert.deepEqual(indexes, []);
  assert.equal(files.length, seedFiles.length);
  for (const e of files) {
    assert.ok(seedFiles.includes(e.path), `${e.path} 应来自 seedFiles`);
    assert.match(e.path, /_index\.md$/);
  }
});

test('seed 产物与 Android 端 parse_seed_manifest 的契约一致（producer/consumer seam）', () => {
  // 模拟 CLI 的 --seed：写 seed-manifest.json + 平铺文件到 outDir/knowledge/。
  // Rust 端 parse_seed_manifest 只读 files 来 dispatch 读取 —— 必须能从
  // files 数组还原出实际拷贝出来的全部文件，否则 APK 里的数据没人读。
  const root = fixture();
  const out = mkdtempSync(join(tmpdir(), 'seed-out-'));
  try {
    const written = materializeSeed(root, out);
    const manifest = JSON.parse(readFileSync(join(out, 'seed-manifest.json'), 'utf8'));
    assert.ok(Array.isArray(manifest.files), 'seed-manifest.json 必须有 files 数组');
    assert.equal(manifest.files.length, written);
    for (const entry of manifest.files) {
      const onDisk = join(out, 'knowledge', entry.path);
      const stat = readFileSync(onDisk);
      assert.equal(stat.length, entry.size, `${entry.path} 大小与 manifest 不一致`);
    }
  } finally {
    rmSync(out, { recursive: true, force: true });
  }
});

test('远程发布路径与 sync::ensure_local_at 拼出的 URL 一致（producer/consumer seam）', () => {
  // Rust 端 ensure_local_at 拼的 URL 是 {base}/{category_path}/{rel_path}（无 knowledge/ 前缀），
  // 所以 out/ 下必须直接有 {category}/{rel}，而不是 out/knowledge/{category}/{rel}。
  // 同时 manifest.json 自身必须在 out 根下。这样一个 base_url 既能取 manifest 又能取文件。
  const root = fixture();
  const out = mkdtempSync(join(tmpdir(), 'dist-out-'));
  try {
    materializeDist(root, out);
    const manifest = JSON.parse(readFileSync(join(out, 'manifest.json'), 'utf8'));
    const sample = manifest.files.find((e) => e.path.endsWith('.pdf'));
    const slash = sample.path.lastIndexOf('/');
    const categoryPath = sample.path.slice(0, slash);
    const relPath = sample.path.slice(slash + 1);
    // 这就是 {base}/{category_path}/{rel_path} 解析到文件系统后的路径
    const expected = join(out, categoryPath, relPath);
    const stat = readFileSync(expected);
    assert.equal(stat.length, sample.size, `${expected} 应等于 manifest 中声明的大小`);
  } finally {
    rmSync(out, { recursive: true, force: true });
  }
});
