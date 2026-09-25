// Headless driver: serves this directory, runs main.js to completion in Chrome,
// and writes the results as JSON. This is how every number in results/ was produced.
//
//   npm install
//   node bench-headless.mjs results/run.json
//   node bench-headless.mjs results/run-norobust.json --enable-dawn-features=disable_robustness
//
// Any extra argv entries are passed through to Chrome as flags.
import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const OUT = process.argv[2] ?? 'results/run.json';
const EXTRA_FLAGS = process.argv.slice(3);
const QUERY = process.env.QUERY ?? '';
const HTTP_PORT = Number(process.env.HTTP_PORT ?? 8210);
const CDP_PORT = Number(process.env.CDP_PORT ?? 9410);

const CHROME_CANDIDATES = [
  process.env.CHROME,
  '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  `${process.env.HOME}/.cache/chrome-for-testing/chrome-linux64/chrome`,
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);
const CHROME = CHROME_CANDIDATES.find(p => { try { fs.accessSync(p, fs.constants.X_OK); return true; } catch { return false; } });
if (!CHROME) {
  console.error('No Chrome found. Set CHROME=/path/to/chrome.');
  process.exit(1);
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.wgsl': 'text/plain' };
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]);
  const file = path.join(ROOT, rel === '/' ? 'index.html' : rel);
  if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404); res.end('not found'); return;
  }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(HTTP_PORT, '127.0.0.1', r));

const chrome = spawn(CHROME, [
  '--headless=new',
  `--remote-debugging-port=${CDP_PORT}`,
  `--user-data-dir=${fs.mkdtempSync(path.join(process.env.TMPDIR ?? '/tmp', 'psum-'))}`,
  '--no-first-run', '--no-default-browser-check',
  '--no-sandbox', '--disable-dev-shm-usage',
  // Headless Linux needs the Vulkan backend to reach the discrete GPU.
  '--use-angle=vulkan', '--enable-features=Vulkan',
  '--ignore-gpu-blocklist',
  '--enable-gpu-rasterization',
  '--disable-software-rasterizer',
  '--disable-gpu-driver-bug-workarounds',
  '--enable-unsafe-webgpu',
  '--disable-dawn-features=timestamp_quantization',
  ...EXTRA_FLAGS,
  'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
chrome.stderr.on('data', () => {});   // Chrome is chatty on stderr; the page log is what matters

let browser;
for (let i = 0; i < 120 && !browser; i++) {
  try { browser = await puppeteer.connect({ browserURL: `http://127.0.0.1:${CDP_PORT}`, protocolTimeout: 0 }); }
  catch { await sleep(500); }
}
if (!browser) { console.error('could not connect to Chrome'); process.exit(1); }

const page = await browser.newPage();
page.setDefaultTimeout(0);
page.on('console', m => { const t = m.text(); if (t.startsWith('PROGRESS') || t.startsWith('CANARY') || t.startsWith('RECOVER')) console.log(t); });
page.on('pageerror', e => console.error('[page error]', e.message));

const url = `http://localhost:${HTTP_PORT}/index.html${QUERY ? '?' + QUERY : ''}`;
console.log(`running ${url}`);
const started = Date.now();
await page.goto(url, { waitUntil: 'domcontentloaded' });

// Poll rather than waitForFunction, so a wedged GPU still yields partial results.
while (Date.now() - started < 6 * 3600 * 1000) {
  let done = false;
  try { done = await page.evaluate(() => !!window.__BENCH_DONE__); } catch { /* navigating */ }
  if (done) break;
  await sleep(5000);
}

const results = await page.evaluate(() => window.__BENCH_RESULTS__ ?? null).catch(() => null);
const err = await page.evaluate(() => window.__BENCH_ERROR__ ?? null).catch(() => null);
if (err) console.error('benchmark error:', err);
if (results) {
  fs.mkdirSync(path.dirname(path.resolve(ROOT, OUT)), { recursive: true });
  fs.writeFileSync(path.resolve(ROOT, OUT), JSON.stringify(results, null, 2));
  console.log(`\nwrote ${OUT} after ${((Date.now() - started) / 1000).toFixed(0)}s`);
}

await browser.close();
chrome.kill();
server.close();
process.exit(results ? 0 : 1);
