const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const userDataDir =
  process.env.PLAYWRIGHT_MCP_USER_DATA_DIR || path.join(process.cwd(), '.playwright-mcp', 'edge-profile');
const outputDir = process.env.PLAYWRIGHT_MCP_OUTPUT_DIR || path.join(process.cwd(), '.playwright-mcp', 'output');
const initPageScript = path.join(process.cwd(), 'scripts', 'mcp-init-page.js');
const ownerFilePath =
  process.env.PLAYWRIGHT_MCP_OWNER_FILE || path.join(process.cwd(), '.playwright-mcp', 'active-owner.txt');
const owner = String(process.env.PLAYWRIGHT_MCP_OWNER || 'unknown').trim().toLowerCase();
const explicitActiveOwner = String(process.env.PLAYWRIGHT_MCP_ACTIVE_OWNER || '').trim().toLowerCase();
const lockFilePath = path.join(userDataDir, '.mcp-owner-lock.json');

const extraArgs = process.argv.slice(2);
let child = null;

function toBool(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on';
}

function normalizeOwner(value) {
  return String(value || '').trim().toLowerCase();
}

function info(message) {
  process.stderr.write(`${message}\n`);
}

function tryReadJson(filePath) {
  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(raw);
  } catch (_) {
    return null;
  }
}

function isProcessAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (_) {
    return false;
  }
}

function getActiveOwner() {
  if (explicitActiveOwner) return explicitActiveOwner;
  try {
    if (!fs.existsSync(ownerFilePath)) return '';
    return normalizeOwner(fs.readFileSync(ownerFilePath, 'utf8'));
  } catch (_) {
    return '';
  }
}

function releaseOwnerLock() {
  const lock = tryReadJson(lockFilePath);
  if (!lock || lock.pid !== process.pid) return;
  try {
    fs.unlinkSync(lockFilePath);
  } catch (_) {
    // best effort
  }
}

function acquireOwnerLock() {
  const existing = tryReadJson(lockFilePath);
  if (existing && isProcessAlive(existing.pid)) {
    return {
      ok: false,
      reason: `Profile is locked by owner='${existing.owner || 'unknown'}' pid=${existing.pid}. Stop that session first.`
    };
  }

  const lockPayload = {
    owner,
    pid: process.pid,
    startedAt: new Date().toISOString(),
    profile: userDataDir,
    workspace: process.cwd()
  };

  try {
    fs.writeFileSync(lockFilePath, JSON.stringify(lockPayload, null, 2), 'utf8');
    return { ok: true, lock: lockPayload };
  } catch (error) {
    return { ok: false, reason: `Failed to write owner lock: ${error.message}` };
  }
}

const persistProfile = toBool(process.env.PLAYWRIGHT_MCP_PERSIST_PROFILE, false);
const saveSession = toBool(process.env.PLAYWRIGHT_MCP_SAVE_SESSION, false);
const saveTrace = toBool(process.env.PLAYWRIGHT_MCP_SAVE_TRACE, false);
const outputMode = process.env.PLAYWRIGHT_MCP_OUTPUT_MODE || 'stdout';
const snapshotMode = process.env.PLAYWRIGHT_MCP_SNAPSHOT_MODE || 'incremental';
const consoleLevel = process.env.PLAYWRIGHT_MCP_CONSOLE_LEVEL || 'error';
const timeoutActionMs = String(process.env.PLAYWRIGHT_MCP_TIMEOUT_ACTION_MS || '12000').trim();
const timeoutNavigationMs = String(process.env.PLAYWRIGHT_MCP_TIMEOUT_NAVIGATION_MS || '60000').trim();
const sharedBrowserContext = toBool(process.env.PLAYWRIGHT_MCP_SHARED_BROWSER_CONTEXT, true);
const allowedHosts = process.env.PLAYWRIGHT_MCP_ALLOWED_HOSTS || '';
const allowedOrigins = process.env.PLAYWRIGHT_MCP_ALLOWED_ORIGINS || '';
const blockedOrigins = process.env.PLAYWRIGHT_MCP_BLOCKED_ORIGINS || '';
const blockServiceWorkers = toBool(process.env.PLAYWRIGHT_MCP_BLOCK_SERVICE_WORKERS, false);

fs.mkdirSync(outputDir, { recursive: true });
if (persistProfile) {
  fs.mkdirSync(userDataDir, { recursive: true });
}
fs.mkdirSync(path.dirname(ownerFilePath), { recursive: true });

const activeOwner = getActiveOwner();
if (activeOwner && owner && owner !== activeOwner) {
  info(`[MCP] Owner blocked. Active owner is '${activeOwner}', current owner is '${owner}'. Exiting.`);
  process.exit(2);
}

const lockState = acquireOwnerLock();
if (!lockState.ok) {
  info(`[MCP] ${lockState.reason}`);
  process.exit(3);
}

const baseArgs = [
  'playwright',
  'run-mcp-server',
  '--browser',
  'msedge',
  '--output-dir',
  outputDir,
  '--output-mode',
  outputMode,
  '--console-level',
  consoleLevel,
  '--snapshot-mode',
  snapshotMode,
  '--timeout-action',
  timeoutActionMs,
  '--timeout-navigation',
  timeoutNavigationMs,
  '--caps',
  'vision,pdf'
];

if (sharedBrowserContext) {
  baseArgs.push('--shared-browser-context');
}

if (persistProfile) {
  baseArgs.push('--user-data-dir', userDataDir);
} else {
  baseArgs.push('--isolated');
}
if (saveSession) baseArgs.push('--save-session');
if (saveTrace) baseArgs.push('--save-trace');
if (allowedHosts.trim()) baseArgs.push('--allowed-hosts', allowedHosts.trim());
if (allowedOrigins.trim()) baseArgs.push('--allowed-origins', allowedOrigins.trim());
if (blockedOrigins.trim()) baseArgs.push('--blocked-origins', blockedOrigins.trim());
if (blockServiceWorkers) baseArgs.push('--block-service-workers');
if (fs.existsSync(initPageScript)) {
  baseArgs.push('--init-page', initPageScript);
}

const isWindows = process.platform === 'win32';
const command = isWindows ? (process.env.ComSpec || 'cmd.exe') : 'npx';
const commandArgs = isWindows
  ? ['/d', '/s', '/c', 'npx', ...baseArgs, ...extraArgs]
  : [...baseArgs, ...extraArgs];

info(
  persistProfile
    ? `[MCP] Starting Playwright MCP server (playwright-edge) with local Edge profile: ${userDataDir}`
    : '[MCP] Starting Playwright MCP server (playwright-edge) in isolated profile mode'
);
info(`[MCP] Profile mode: ${persistProfile ? 'persistent' : 'isolated'}`);
info(`[MCP] Artifact mode: saveSession=${saveSession}, saveTrace=${saveTrace}, outputDir=${outputDir}`);
info(`[MCP] Runtime mode: outputMode=${outputMode}, snapshotMode=${snapshotMode}, consoleLevel=${consoleLevel}`);
info(`[MCP] Timeouts: action=${timeoutActionMs}ms, navigation=${timeoutNavigationMs}ms`);
info(`[MCP] Shared browser context: ${sharedBrowserContext}`);
info(`[MCP] Owner: ${owner} | Active owner: ${activeOwner || '(unset)'}`);
info(`[MCP] Owner lock: ${lockFilePath}`);
if (allowedHosts.trim()) info(`[MCP] Network allow hosts: ${allowedHosts.trim()}`);
if (allowedOrigins.trim()) info(`[MCP] Network allow origins: ${allowedOrigins.trim()}`);
if (blockedOrigins.trim()) info(`[MCP] Network block origins: ${blockedOrigins.trim()}`);
if (blockServiceWorkers) info('[MCP] Network: service workers blocked');
info(`[MCP] Init page script: ${initPageScript}`);
info('[MCP] Press Ctrl+C to stop.');

child = spawn(command, commandArgs, {
  stdio: 'inherit',
  shell: false
});

child.on('error', (error) => {
  releaseOwnerLock();
  console.error(`[MCP] Failed to start server: ${error.message}`);
  process.exit(1);
});

child.on('exit', (code) => {
  releaseOwnerLock();
  process.exit(code === null ? 1 : code);
});

process.on('SIGINT', () => {
  releaseOwnerLock();
});

process.on('SIGTERM', () => {
  releaseOwnerLock();
});

process.on('exit', () => {
  releaseOwnerLock();
});
