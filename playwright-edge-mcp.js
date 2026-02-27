const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const userDataDir = process.env.PLAYWRIGHT_MCP_USER_DATA_DIR || path.join(process.cwd(), '.playwright-mcp', 'edge-profile');
const outputDir = process.env.PLAYWRIGHT_MCP_OUTPUT_DIR || path.join(process.cwd(), '.playwright-mcp', 'output');
const initPageScript = path.join(process.cwd(), 'scripts', 'mcp-init-page.js');

const extraArgs = process.argv.slice(2);

function toBool(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on';
}

const persistProfile = toBool(process.env.PLAYWRIGHT_MCP_PERSIST_PROFILE, false);
const saveSession = toBool(process.env.PLAYWRIGHT_MCP_SAVE_SESSION, false);
const saveTrace = toBool(process.env.PLAYWRIGHT_MCP_SAVE_TRACE, false);
const outputMode = process.env.PLAYWRIGHT_MCP_OUTPUT_MODE || 'stdout';
const snapshotMode = process.env.PLAYWRIGHT_MCP_SNAPSHOT_MODE || 'incremental';
const consoleLevel = process.env.PLAYWRIGHT_MCP_CONSOLE_LEVEL || 'error';
const allowedHosts = process.env.PLAYWRIGHT_MCP_ALLOWED_HOSTS || '';
const allowedOrigins = process.env.PLAYWRIGHT_MCP_ALLOWED_ORIGINS || '';
const blockedOrigins = process.env.PLAYWRIGHT_MCP_BLOCKED_ORIGINS || '';
const blockServiceWorkers = toBool(process.env.PLAYWRIGHT_MCP_BLOCK_SERVICE_WORKERS, false);

fs.mkdirSync(outputDir, { recursive: true });
if (persistProfile) {
  fs.mkdirSync(userDataDir, { recursive: true });
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
  '30000',
  '--timeout-navigation',
  '180000',
  '--shared-browser-context',
  '--caps',
  'vision,pdf'
];

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
const command = isWindows ? 'cmd' : 'npx';
const commandArgs = isWindows
  ? ['/c', 'npx', ...baseArgs, ...extraArgs]
  : [...baseArgs, ...extraArgs];

function info(message) {
  // Keep stdout clean for MCP protocol messages.
  process.stderr.write(`${message}\n`);
}

info(
  persistProfile
    ? `[MCP] Starting Playwright MCP server (playwright-edge) with local Edge profile: ${userDataDir}`
    : '[MCP] Starting Playwright MCP server (playwright-edge) in isolated profile mode'
);
info(`[MCP] Profile mode: ${persistProfile ? 'persistent' : 'isolated'}`);
info(`[MCP] Artifact mode: saveSession=${saveSession}, saveTrace=${saveTrace}, outputDir=${outputDir}`);
info(`[MCP] Runtime mode: outputMode=${outputMode}, snapshotMode=${snapshotMode}, consoleLevel=${consoleLevel}`);
if (allowedHosts.trim()) info(`[MCP] Network allow hosts: ${allowedHosts.trim()}`);
if (allowedOrigins.trim()) info(`[MCP] Network allow origins: ${allowedOrigins.trim()}`);
if (blockedOrigins.trim()) info(`[MCP] Network block origins: ${blockedOrigins.trim()}`);
if (blockServiceWorkers) info('[MCP] Network: service workers blocked');
info(`[MCP] Init page script: ${initPageScript}`);
info('[MCP] Press Ctrl+C to stop.');

const child = spawn(command, commandArgs, {
  stdio: 'inherit',
  shell: false
});

child.on('error', (error) => {
  console.error(`[MCP] Failed to start server: ${error.message}`);
  process.exit(1);
});

child.on('exit', (code) => {
  process.exit(code === null ? 1 : code);
});
