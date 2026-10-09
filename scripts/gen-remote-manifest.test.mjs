import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildManifest } from './gen-remote-manifest.mjs';

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

test('seed 子集只保留显式列出的路径', () => {
  const { files } = buildManifest(fixture(), { seedFiles: ['01-基础/_index.md'] });
  assert.deepEqual(files, []);
});