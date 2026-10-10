#!/usr/bin/env node
// Build the Windows desktop installer.
// Cross-platform: works on both Windows and Unix. Tauri picks the host target.
//
// Output:
//   src-tauri/target/release/bundle/msi/*.msi
//   src-tauri/target/release/bundle/nsis/*.exe
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Invoke the Tauri CLI via node directly. Spawning `npx` fails on Windows
// because Node refuses to execute `.cmd` shims without `shell: true`.
const cliPath = path.join(__dirname, '..', 'node_modules', '@tauri-apps', 'cli', 'tauri.js');

const args = ['build'];
if (process.argv.includes('--debug')) args.push('--debug');

console.log(`> node ${path.relative(process.cwd(), cliPath)} ${args.join(' ')}`);
const r = spawnSync(process.execPath, [cliPath, ...args], { stdio: 'inherit' });
if (r.error) {
  console.error(`spawn failed: ${r.error.message}`);
  process.exit(1);
}
process.exit(r.status ?? 1);
