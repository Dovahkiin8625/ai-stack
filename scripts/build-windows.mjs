#!/usr/bin/env node
// Build the Windows desktop installer.
// Cross-platform: works on both Windows and Unix. Tauri picks the host target.
//
// Output:
//   src-tauri/target/release/bundle/msi/*.msi
//   src-tauri/target/release/bundle/nsis/*.exe
import { spawnSync } from 'node:child_process';

const args = ['--no-install', 'tauri', 'build'];
if (process.argv.includes('--debug')) args.push('--debug');

console.log(`> npx ${args.join(' ')}`);
const r = spawnSync('npx', args, { stdio: 'inherit' });
process.exit(r.status ?? 1);
