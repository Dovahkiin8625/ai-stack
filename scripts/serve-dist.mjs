import { createServer } from 'node:http';
import { createReadStream, statSync, existsSync } from 'node:fs';
import { join, normalize, extname } from 'node:path';

const ROOT = process.argv[2] ?? 'resources/dist';
const PORT = Number(process.argv[3] ?? 8787);
const TYPES = {
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.pdf': 'application/pdf',
  '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
};

createServer((req, res) => {
  // 目录穿越防护：规范化后必须仍在 ROOT 内
  const rel = normalize(decodeURIComponent(req.url.split('?')[0])).replace(/^([/\\])+/, '');
  const abs = join(ROOT, rel);
  if (!abs.startsWith(normalize(ROOT)) || !existsSync(abs) || !statSync(abs).isFile()) {
    res.writeHead(404).end('not found');
    return;
  }
  res.writeHead(200, {
    'content-type': TYPES[extname(abs)] ?? 'application/octet-stream',
    'content-length': statSync(abs).size,
    'access-control-allow-origin': '*',
  });
  createReadStream(abs).pipe(res);
}).listen(PORT, () => console.log(`serving ${ROOT} at http://127.0.0.1:${PORT}/`));