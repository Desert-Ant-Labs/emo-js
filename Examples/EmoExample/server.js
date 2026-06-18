// Minimal zero-dependency static server for the Emo Todo example.
//
//   node server.js [port]
//
// Serves the files in ./public. @desert-ant-labs/emo is loaded from npm (via a
// CDN) using the import map in index.html, so there's nothing else to serve.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, extname, join, normalize, sep } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = join(__dirname, "public");
const PORT = Number(process.argv[2]) || 5173;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function safeJoin(base, target) {
  const p = normalize(join(base, target));
  if (p !== base && !p.startsWith(base + sep)) return null; // path traversal guard
  return p;
}

const server = createServer(async (req, res) => {
  try {
    let urlPath = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (urlPath === "/") urlPath = "/index.html";
    const filePath = safeJoin(PUBLIC_DIR, urlPath.slice(1));

    if (!filePath) {
      res.writeHead(403).end("Forbidden");
      return;
    }

    const body = await readFile(filePath);
    res.writeHead(200, { "content-type": MIME[extname(filePath)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" }).end("Not found");
  }
});

server.listen(PORT, () => {
  console.log(`\n  Emo Todo example running at http://localhost:${PORT}\n`);
});
