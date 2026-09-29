// scripts/make-icons.mjs
// 用 Node 内置 zlib 生成纯色 PNG 占位图标，并把 PNG 嵌入 .ico 容器。
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';

function makePng(width, height, color = [37, 99, 235]) {
  const sig = Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;   // bit depth
  ihdr[9] = 2;   // color type RGB
  ihdr[10] = 0;  // compression
  ihdr[11] = 0;  // filter
  ihdr[12] = 0;  // interlace

  const rowLen = width * 3 + 1;
  const raw = Buffer.alloc(rowLen * height);
  for (let y = 0; y < height; y++) {
    raw[y * rowLen] = 0;
    for (let x = 0; x < width; x++) {
      const off = y * rowLen + 1 + x * 3;
      raw[off] = color[0];
      raw[off + 1] = color[1];
      raw[off + 2] = color[2];
    }
  }
  const idat = deflateSync(raw);

  function crc32(buf) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < buf.length; i++) {
      c ^= buf[i];
      for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xEDB88320 & -(c & 1));
    }
    return (~c) >>> 0;
  }

  function chunk(type, data) {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length, 0);
    const typeBuf = Buffer.from(type, 'ascii');
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
    return Buffer.concat([len, typeBuf, data, crc]);
  }

  return Buffer.concat([
    sig,
    chunk('IHDR', ihdr),
    chunk('IDAT', idat),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

mkdirSync('src-tauri/icons', { recursive: true });

const sizes = [
  { name: '32x32.png', size: 32 },
  { name: '128x128.png', size: 128 },
  { name: '128x128@2x.png', size: 256 },
  { name: 'icon.png', size: 512 },
];

for (const { name, size } of sizes) {
  writeFileSync(`src-tauri/icons/${name}`, makePng(size, size));
  console.log(`Created src-tauri/icons/${name} (${size}x${size})`);
}

// 把 32x32 PNG 嵌入到 ICO 容器（Vista+ 支持 PNG-in-ICO 格式）
// ICO header: 6 bytes + 16 bytes per entry + PNG payload
function makeIco(pngBuffer, size) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0);                  // reserved
  header.writeUInt16LE(1, 2);                  // type = icon
  header.writeUInt16LE(1, 4);                  // count = 1
  const entry = Buffer.alloc(16);
  entry[0] = size === 256 ? 0 : size;          // width (0 = 256)
  entry[1] = size === 256 ? 0 : size;          // height
  entry[2] = 0;                                // colors in palette
  entry[3] = 0;                                // reserved
  entry.writeUInt16LE(1, 4);                   // planes
  entry.writeUInt16LE(32, 6);                  // bit count
  entry.writeUInt32LE(pngBuffer.length, 8);    // size
  entry.writeUInt32LE(6 + 16, 12);             // offset to PNG data
  return Buffer.concat([header, entry, pngBuffer]);
}

const png32 = readFileSync('src-tauri/icons/32x32.png');
writeFileSync('src-tauri/icons/icon.ico', makeIco(png32, 32));
console.log('Created src-tauri/icons/icon.ico (PNG-in-ICO, 32x32)');

console.log('\n提示: 安装 Rust 后可重新生成更高质量图标:');
console.log('  cd C:\\project\\ai-stack && npx @tauri-apps/cli icon src-tauri/icons/icon.png');
