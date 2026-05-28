#!/usr/bin/env node
import { createServer } from 'node:http';
import { promises as fs } from 'node:fs';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dashboardDir = path.resolve(here, '..');
const distDir = path.resolve(process.env.DASHBOARD_DIST || path.join(dashboardDir, 'dist'));
const stateRoot = process.env.BENCH_STATE_ROOT || '/mnt/100g/agent-bench/state';
const controlDir = process.env.BENCH_CONTROL_DIR || path.join(stateRoot, 'control');
const drainedHostsFile = process.env.BENCH_DRAINED_HOSTS_FILE || path.join(controlDir, 'drained-hosts.txt');
const blockedGpusFile = process.env.BENCH_BLOCKED_GPUS_FILE || path.join(controlDir, 'blocked-gpus.txt');
const allowedHosts = new Set(
  (process.env.BENCH_CONTROL_HOSTS || 'a100,3090,2080ti,h100,h100-2')
    .split(',')
    .map((host) => host.trim())
    .filter(Boolean),
);
const basePath = normalizeBase(process.env.DASHBOARD_BASE_PATH || '/agentic-serve');
const host = process.env.HOST || '127.0.0.1';
const port = Number(process.env.PORT || '4180');

function normalizeBase(value) {
  const trimmed = value.trim();
  if (!trimmed || trimmed === '/') return '';
  return `/${trimmed.replace(/^\/+|\/+$/g, '')}`;
}

function stripBase(pathname) {
  if (!basePath) return pathname;
  if (pathname === basePath) return '/';
  if (pathname.startsWith(`${basePath}/`)) return pathname.slice(basePath.length);
  return null;
}

function sendJson(res, status, payload) {
  const body = `${JSON.stringify(payload, null, 2)}\n`;
  res.writeHead(status, {
    'cache-control': 'no-store',
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
  });
  res.end(body);
}

function sendText(res, status, text) {
  res.writeHead(status, {
    'cache-control': 'no-store',
    'content-type': 'text/plain; charset=utf-8',
  });
  res.end(text);
}

function mimeType(filePath) {
  switch (path.extname(filePath)) {
    case '.css':
      return 'text/css; charset=utf-8';
    case '.html':
      return 'text/html; charset=utf-8';
    case '.js':
      return 'text/javascript; charset=utf-8';
    case '.json':
      return 'application/json; charset=utf-8';
    case '.svg':
      return 'image/svg+xml';
    default:
      return 'application/octet-stream';
  }
}

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 4096) throw new Error('request body too large');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function readDrainedHosts() {
  try {
    const text = await fs.readFile(drainedHostsFile, 'utf8');
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#'))
      .map((line) => line.split(/\s+/)[0])
      .filter((hostName) => allowedHosts.has(hostName));
  } catch (error) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }
}

async function writeDrainedHosts(hosts) {
  const uniqueHosts = [...new Set(hosts)].filter((hostName) => allowedHosts.has(hostName)).sort();
  await fs.mkdir(path.dirname(drainedHostsFile), { recursive: true });
  const tmp = path.join(path.dirname(drainedHostsFile), `.drained-hosts.${process.pid}.${Date.now()}.tmp`);
  const body = [
    '# Hosts in this file are drained: current jobs may finish, but new jobs will not dispatch.',
    '# Managed by the private dashboard.',
    ...uniqueHosts,
    '',
  ].join('\n');
  await fs.writeFile(tmp, body, 'utf8');
  await fs.rename(tmp, drainedHostsFile);
  return uniqueHosts;
}

function parseBlockedGpuLine(line) {
  const parts = line.replace(':', ' ').trim().split(/\s+/);
  if (parts.length < 2) return null;
  const [hostName, gpu] = parts;
  if (!allowedHosts.has(hostName) || !/^\d+$/.test(gpu)) return null;
  return { host: hostName, gpu };
}

function gpuBlockKey(entry) {
  return `${entry.host}:${entry.gpu}`;
}

function sortGpuBlocks(entries) {
  return entries
    .filter((entry) => allowedHosts.has(entry.host) && /^\d+$/.test(entry.gpu))
    .sort((left, right) => {
      const hostCompare = left.host.localeCompare(right.host);
      if (hostCompare !== 0) return hostCompare;
      return Number(left.gpu) - Number(right.gpu);
    });
}

async function readBlockedGpus() {
  try {
    const text = await fs.readFile(blockedGpusFile, 'utf8');
    const entries = [];
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith('#')) continue;
      const parsed = parseBlockedGpuLine(line);
      if (parsed) entries.push(parsed);
    }
    return sortGpuBlocks([...new Map(entries.map((entry) => [gpuBlockKey(entry), entry])).values()]);
  } catch (error) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }
}

async function writeBlockedGpus(entries) {
  const uniqueEntries = sortGpuBlocks([...new Map(entries.map((entry) => [gpuBlockKey(entry), entry])).values()]);
  await fs.mkdir(path.dirname(blockedGpusFile), { recursive: true });
  const tmp = path.join(path.dirname(blockedGpusFile), `.blocked-gpus.${process.pid}.${Date.now()}.tmp`);
  const body = [
    '# Host/GPU pairs in this file are unavailable for new sweep dispatches.',
    '# Format: <host> <gpu-index>. Managed by the private dashboard.',
    ...uniqueEntries.map((entry) => `${entry.host} ${entry.gpu}`),
    '',
  ].join('\n');
  await fs.writeFile(tmp, body, 'utf8');
  await fs.rename(tmp, blockedGpusFile);
  return uniqueEntries;
}

async function handleDrainApi(req, res) {
  if (req.method === 'GET') {
    sendJson(res, 200, {
      drained_hosts: await readDrainedHosts(),
      allowed_hosts: [...allowedHosts].sort(),
      file: drainedHostsFile,
    });
    return;
  }

  if (req.method !== 'POST') {
    sendJson(res, 405, { error: 'method not allowed' });
    return;
  }

  let payload;
  try {
    payload = JSON.parse(await readBody(req));
  } catch (error) {
    sendJson(res, 400, { error: error.message || 'invalid JSON body' });
    return;
  }

  const targetHost = String(payload.host || '').trim();
  if (!allowedHosts.has(targetHost)) {
    sendJson(res, 400, { error: `host is not allowed: ${targetHost}` });
    return;
  }
  if (typeof payload.drained !== 'boolean') {
    sendJson(res, 400, { error: 'drained must be a boolean' });
    return;
  }

  const current = new Set(await readDrainedHosts());
  if (payload.drained) current.add(targetHost);
  else current.delete(targetHost);
  const drainedHosts = await writeDrainedHosts([...current]);
  sendJson(res, 200, {
    ok: true,
    host: targetHost,
    drained: payload.drained,
    drained_hosts: drainedHosts,
    file: drainedHostsFile,
  });
}

async function handleGpuBlockApi(req, res) {
  if (req.method === 'GET') {
    sendJson(res, 200, {
      blocked_gpus: await readBlockedGpus(),
      allowed_hosts: [...allowedHosts].sort(),
      file: blockedGpusFile,
    });
    return;
  }

  if (req.method !== 'POST') {
    sendJson(res, 405, { error: 'method not allowed' });
    return;
  }

  let payload;
  try {
    payload = JSON.parse(await readBody(req));
  } catch (error) {
    sendJson(res, 400, { error: error.message || 'invalid JSON body' });
    return;
  }

  const targetHost = String(payload.host || '').trim();
  const targetGpu = String(payload.gpu ?? '').trim();
  if (!allowedHosts.has(targetHost)) {
    sendJson(res, 400, { error: `host is not allowed: ${targetHost}` });
    return;
  }
  if (!/^\d+$/.test(targetGpu)) {
    sendJson(res, 400, { error: `gpu must be a numeric index: ${targetGpu}` });
    return;
  }
  if (typeof payload.blocked !== 'boolean') {
    sendJson(res, 400, { error: 'blocked must be a boolean' });
    return;
  }

  const current = new Map((await readBlockedGpus()).map((entry) => [gpuBlockKey(entry), entry]));
  const entry = { host: targetHost, gpu: targetGpu };
  if (payload.blocked) current.set(gpuBlockKey(entry), entry);
  else current.delete(gpuBlockKey(entry));
  const blockedGpus = await writeBlockedGpus([...current.values()]);
  sendJson(res, 200, {
    ok: true,
    host: targetHost,
    gpu: targetGpu,
    blocked: payload.blocked,
    blocked_gpus: blockedGpus,
    file: blockedGpusFile,
  });
}

async function serveStatic(reqPath, res) {
  const pathname = reqPath === '/' ? '/index.html' : reqPath;
  const decoded = decodeURIComponent(pathname);
  let target = path.resolve(distDir, `.${decoded}`);
  if (!target.startsWith(`${distDir}${path.sep}`) && target !== distDir) {
    sendText(res, 403, 'forbidden\n');
    return;
  }

  try {
    const stat = await fs.stat(target);
    if (stat.isDirectory()) {
      target = path.join(target, 'index.html');
    }
  } catch (error) {
    if (path.extname(decoded)) {
      sendText(res, 404, 'not found\n');
      return;
    }
    target = path.join(distDir, 'index.html');
  }

  try {
    const stat = await fs.stat(target);
    res.writeHead(200, {
      'cache-control': path.basename(target) === 'index.html' ? 'no-cache' : 'public, max-age=60',
      'content-length': stat.size,
      'content-type': mimeType(target),
    });
    createReadStream(target).pipe(res);
  } catch {
    sendText(res, 404, 'not found\n');
  }
}

const server = createServer((req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host || `${host}:${port}`}`);
  const reqPath = stripBase(url.pathname);
  if (reqPath == null) {
    sendText(res, 404, 'not found\n');
    return;
  }

  if (reqPath === '/api/host-drain') {
    handleDrainApi(req, res).catch((error) => {
      sendJson(res, 500, { error: error.message || 'internal error' });
    });
    return;
  }

  if (reqPath === '/api/gpu-block') {
    handleGpuBlockApi(req, res).catch((error) => {
      sendJson(res, 500, { error: error.message || 'internal error' });
    });
    return;
  }

  serveStatic(reqPath, res).catch((error) => {
    sendJson(res, 500, { error: error.message || 'internal error' });
  });
});

server.listen(port, host, () => {
  console.log(`dashboard listening on http://${host}:${port}${basePath || '/'}`);
  console.log(`host drain file: ${drainedHostsFile}`);
  console.log(`gpu block file: ${blockedGpusFile}`);
});
