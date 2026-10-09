#!/usr/bin/env node
// Build a release APK for arm64-v8a Android and sign it with the Android
// debug keystore so it can be sideloaded to a personal device.
//
// Cross-platform: invokes Gradle via the Tauri CLI on any host that has
// the Android SDK + NDK + Rust Android target installed (see README).
//
// Output (default):
//   src-tauri/gen/android/app/build/outputs/apk/universal/release/
//     app-universal-release.apk            (signed, ready to install)
//
//   └─── app-universal-release-unsigned.apk (the Gradle output before signing)
//
// Usage:
//   node scripts/build-android.mjs            # release arm64, signed
//   node scripts/build-android.mjs --debug    # debug build (no signing)
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import os from 'node:os';

// ---- locate the SDK tools we need -----------------------------------------
const isWin = process.platform === 'win32';
const ext = isWin ? '.bat' : '';
const localAppData = process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local');
const sdkRoot = process.env.ANDROID_HOME || process.env.ANDROID_SDK_ROOT
  || (isWin ? path.join(localAppData, 'Android', 'Sdk') : path.join(os.homedir(), 'Android', 'Sdk'));
const buildTools = path.join(sdkRoot, 'build-tools');
const apksigner = path.join(buildTools, '35.0.0', `apksigner${ext}`);
const debugKeystore = process.env.HOME
  ? path.join(process.env.HOME, '.android', 'debug.keystore')
  : path.join(os.homedir(), '.android', 'debug.keystore');

// ---- run the Tauri build --------------------------------------------------
const debug = process.argv.includes('--debug');
const tauriArgs = ['--no-install', 'tauri', 'android', 'build',
  '--apk',
  '--target', 'aarch64',
];
if (debug) tauriArgs.splice(3, 0, '--debug');

console.log(`> npx ${tauriArgs.join(' ')}`);
const build = spawnSync('npx', tauriArgs, { stdio: 'inherit' });
if (build.status !== 0) process.exit(build.status ?? 1);

// ---- sign the unsigned APK (release builds only) -------------------------
const unsignedPath = 'src-tauri/gen/android/app/build/outputs/apk/universal/release/app-universal-release-unsigned.apk';
const signedPath = 'src-tauri/gen/android/app/build/outputs/apk/universal/release/app-universal-release.apk';

if (debug) {
  console.log('Debug build — skipping signing.');
  process.exit(0);
}

if (!existsSync(unsignedPath)) {
  console.error(`Expected unsigned APK at: ${unsignedPath}`);
  process.exit(1);
}
if (!existsSync(apksigner)) {
  console.error(
    `apksigner not found at ${apksigner}.\n` +
    `Install build-tools 35.0.0 via sdkmanager: sdkmanager "build-tools;35.0.0"`,
  );
  process.exit(1);
}
if (!existsSync(debugKeystore)) {
  console.error(
    `Debug keystore not found at ${debugKeystore}.\n` +
    `It is auto-generated the first time you build a debug APK on this machine. ` +
    `Run \`npm run build:android -- --debug\` once to create it, or run a Gradle build.`,
  );
  process.exit(1);
}

console.log(`\n> apksigner sign ${path.basename(unsignedPath)}`);
const sign = spawnSync(apksigner, [
  'sign',
  '--ks', debugKeystore,
  '--ks-pass', 'pass:android',
  '--key-pass', 'pass:android',
  '--ks-key-alias', 'androiddebugkey',
  '--out', signedPath,
  unsignedPath,
], { stdio: 'inherit' });
if (sign.status !== 0) process.exit(sign.status ?? 1);

console.log(`\nDone. Signed APK: ${signedPath}`);
console.log('Install with: adb install -r "' + signedPath + '"');
