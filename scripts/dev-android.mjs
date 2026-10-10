#!/usr/bin/env node
// Launch the app on a connected Android device or emulator with hot reload
// via the Tauri Android dev driver.
//
// Prerequisite: an Android device with USB debugging enabled (or an emulator),
// visible to `adb devices`. Rust + Android target installed (see README).
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Invoke the Tauri CLI via node directly. Spawning `npx` fails on Windows
// because Node refuses to execute `.cmd` shims without `shell: true`.
const cliPath = path.join(__dirname, '..', 'node_modules', '@tauri-apps', 'cli', 'tauri.js');

const args = ['android', 'dev'];
console.log(`> node ${path.relative(process.cwd(), cliPath)} ${args.join(' ')}`);
const r = spawnSync(process.execPath, [cliPath, ...args], { stdio: 'inherit' });
if (r.error) {
  console.error(`spawn failed: ${r.error.message}`);
  process.exit(1);
}
process.exit(r.status ?? 1);
