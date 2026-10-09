#!/usr/bin/env node
// Launch the app on a connected Android device or emulator with hot reload
// via the Tauri Android dev driver.
//
// Prerequisite: an Android device with USB debugging enabled (or an emulator),
// visible to `adb devices`. Rust + Android target installed (see README).
import { spawnSync } from 'node:child_process';

const args = ['--no-install', 'tauri', 'android', 'dev'];
console.log(`> npx ${args.join(' ')}`);
const r = spawnSync('npx', args, { stdio: 'inherit' });
process.exit(r.status ?? 1);
