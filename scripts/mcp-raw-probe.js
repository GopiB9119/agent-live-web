const fs = require('fs');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const { buildPlaywrightMcpArgs, normalizeCaps, toBool } = require('../playwright-mcp-launch-config');
const { CLIENT_FORMAT, createMcpMessageParser, writeMcpMessage } = require('../mcp-jsonrpc-transport');

function parseOptionalBool(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  return toBool(value, fallback);
}

function applyLaunchProfile(parsed, profile) {
  const normalized = String(profile || '').trim().toLowerCase();
  if (!normalized || normalized === 'full') {
    parsed.profile = 'full';
    return;
  }
  if (normalized === 'minimal') {
    parsed.profile = 'minimal';
    parsed.caps = '';
    parsed.initPageEnabled = false;
    parsed.sharedBrowserContext = false;
    parsed.persistProfile = false;
    return;
  }
  throw new Error(`Unsupported profile '${profile}'. Use 'full' or 'minimal'.`);
}

function parseArgs(argv) {
  const parsed = {
    target: 'launcher',
    step: 'tools-list',
    profile: 'full',
    browser: String(process.env.PLAYWRIGHT_MCP_BROWSER || 'msedge').trim() || 'msedge',
    headless: toBool(process.env.PLAYWRIGHT_MCP_HEADLESS, true),
    caps: normalizeCaps(process.env.PLAYWRIGHT_MCP_CAPS, 'vision,pdf'),
    initPageEnabled: toBool(process.env.PLAYWRIGHT_MCP_INIT_PAGE_ENABLED, true),
    sharedBrowserContext: toBool(process.env.PLAYWRIGHT_MCP_SHARED_BROWSER_CONTEXT, true),
    persistProfile: toBool(process.env.PLAYWRIGHT_MCP_PERSIST_PROFILE, true),
    initTimeoutMs: Number.parseInt(String(process.env.PLAYWRIGHT_MCP_PROBE_INIT_TIMEOUT_MS || '40000'), 10) || 40000,
    requestTimeoutMs:
      Number.parseInt(String(process.env.PLAYWRIGHT_MCP_PROBE_REQUEST_TIMEOUT_MS || '25000'), 10) || 25000,
    launcherInitTimeoutMs:
      Number.parseInt(String(process.env.PLAYWRIGHT_MCP_PROXY_INIT_TIMEOUT_MS || ''), 10) || null,
    launcherRequestTimeoutMs:
      Number.parseInt(String(process.env.PLAYWRIGHT_MCP_PROXY_REQUEST_TIMEOUT_MS || ''), 10) || null,
    keepRuntime: toBool(process.env.PLAYWRIGHT_MCP_PROBE_KEEP_RUNTIME, false),
    saveTrace: toBool(process.env.PLAYWRIGHT_MCP_SAVE_TRACE, false),
    saveSession: toBool(process.env.PLAYWRIGHT_MCP_SAVE_SESSION, false),
    verbose: toBool(process.env.PLAYWRIGHT_MCP_PROBE_VERBOSE, true),
    runtimeRoot: '',
    launcher: ''
  };

  for (const arg of argv) {
    if (arg === '--help' || arg === '-h') {
      parsed.help = true;
      continue;
    }
    if (arg === '--headless') {
      parsed.headless = true;
      continue;
    }
    if (arg === '--headed') {
      parsed.headless = false;
      continue;
    }
    if (arg === '--keep-runtime') {
      parsed.keepRuntime = true;
      continue;
    }
    if (arg === '--quiet') {
      parsed.verbose = false;
      continue;
    }
    const [flag, value] = arg.split('=', 2);
    if (!value) continue;
    if (flag === '--step') parsed.step = value;
    if (flag === '--target') parsed.target = value;
    if (flag === '--profile') applyLaunchProfile(parsed, value);
    if (flag === '--browser') parsed.browser = value;
    if (flag === '--caps') parsed.caps = normalizeCaps(value, parsed.caps);
    if (flag === '--init-page') parsed.initPageEnabled = parseOptionalBool(value, parsed.initPageEnabled);
    if (flag === '--shared-context') {
      parsed.sharedBrowserContext = parseOptionalBool(value, parsed.sharedBrowserContext);
    }
    if (flag === '--persist-profile') {
      parsed.persistProfile = parseOptionalBool(value, parsed.persistProfile);
    }
    if (flag === '--runtime-root') parsed.runtimeRoot = value;
    if (flag === '--launcher') parsed.launcher = value;
    if (flag === '--init-timeout-ms') parsed.initTimeoutMs = Number.parseInt(value, 10) || parsed.initTimeoutMs;
    if (flag === '--request-timeout-ms') {
      parsed.requestTimeoutMs = Number.parseInt(value, 10) || parsed.requestTimeoutMs;
    }
    if (flag === '--launcher-init-timeout-ms') {
      parsed.launcherInitTimeoutMs = Number.parseInt(value, 10) || parsed.launcherInitTimeoutMs;
    }
    if (flag === '--launcher-request-timeout-ms') {
      parsed.launcherRequestTimeoutMs = Number.parseInt(value, 10) || parsed.launcherRequestTimeoutMs;
    }
  }

  return parsed;
}

function printHelp() {
  process.stdout.write(`Usage: node scripts/mcp-raw-probe.js [options]

Raw stdio probe for playwright-edge-mcp.js.

Options:
  --target=launcher|playwright-direct|playwright-cli
                                 Probe the repo launcher, raw Playwright MCP server,
                                 or Playwright CLI JS entrypoint directly
  --step=initialize|tools-list   Probe depth. Default: tools-list
  --profile=full|minimal         Launch preset. minimal disables caps, init-page,
                                 shared context, and persistent profile
  --browser=<browser>            Browser/channel to target. Default: msedge
  --caps=<csv|none>              Override Playwright MCP caps. Use none to disable
  --init-page=on|off             Toggle init-page script injection
  --shared-context=on|off        Toggle shared browser context
  --persist-profile=on|off       Toggle user-data-dir profile persistence
  --headless                     Force headless browser mode
  --headed                       Force headed browser mode
  --keep-runtime                 Keep isolated runtime directory after success
  --runtime-root=<path>          Reuse a specific runtime directory
  --init-timeout-ms=<ms>         Timeout for initialize request
  --request-timeout-ms=<ms>      Timeout for tools/list request
  --launcher-init-timeout-ms=<ms>
                                 Timeout enforced inside the launcher for initialize
  --launcher-request-timeout-ms=<ms>
                                 Timeout enforced inside the launcher for tools/list
  --launcher=<path>              Override launcher path
  --quiet                        Do not echo launcher stderr while probing
  --help                         Show this message

Examples:
  npm run mcp:probe
  npm run mcp:probe -- --profile=minimal --step=initialize
  npm run mcp:probe -- --step=initialize
  node scripts/mcp-raw-probe.js --target=playwright-direct
  node scripts/mcp-raw-probe.js --target=playwright-cli --profile=minimal
  node scripts/mcp-raw-probe.js --launcher-init-timeout-ms=35000
  node scripts/mcp-raw-probe.js --headed --keep-runtime
  node scripts/mcp-raw-probe.js --target=playwright-direct --caps=none --init-page=off
`);
}

function tail(items, count) {
  return items.slice(Math.max(0, items.length - count));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function runtimeFilesSummary(runtimeRoot, limit = 24) {
  const collected = [];
  if (!fs.existsSync(runtimeRoot)) return collected;

  const stack = [runtimeRoot];
  while (stack.length) {
    const current = stack.pop();
    let entries = [];
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch (_) {
      continue;
    }
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }
      try {
        const stat = fs.statSync(fullPath);
        collected.push({
          path: path.relative(runtimeRoot, fullPath) || path.basename(fullPath),
          size: stat.size,
          mtime: stat.mtime.toISOString()
        });
      } catch (_) {
        // best effort
      }
    }
  }

  return collected
    .sort((a, b) => String(b.mtime).localeCompare(String(a.mtime)))
    .slice(0, limit);
}

function readTextIfExists(filePath) {
  try {
    if (!fs.existsSync(filePath)) return '';
    return fs.readFileSync(filePath, 'utf8').trim();
  } catch (_) {
    return '';
  }
}

async function terminateChildTree(child) {
  if (!child || child.exitCode !== null || child.killed) return;

  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/pid', String(child.pid), '/t', '/f'], { stdio: 'ignore', windowsHide: true });
    return;
  }

  child.kill('SIGTERM');
  await sleep(800);
  if (child.exitCode === null && !child.killed) {
    child.kill('SIGKILL');
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    return;
  }

  const workspaceRoot = path.resolve(__dirname, '..');
  const launcherPath = path.resolve(args.launcher || path.join(workspaceRoot, 'playwright-edge-mcp.js'));
  const playwrightCliPath = path.resolve(workspaceRoot, 'node_modules', 'playwright', 'cli.js');
  if (args.target === 'launcher' && !fs.existsSync(launcherPath)) {
    throw new Error(`Launcher not found: ${launcherPath}`);
  }
  if (args.target === 'playwright-cli' && !fs.existsSync(playwrightCliPath)) {
    throw new Error(`Playwright CLI entrypoint not found: ${playwrightCliPath}`);
  }

  const runtimeParent = path.join(workspaceRoot, '.playwright-mcp', 'probes');
  fs.mkdirSync(runtimeParent, { recursive: true });
  const runtimeRoot = args.runtimeRoot
    ? path.resolve(args.runtimeRoot)
    : fs.mkdtempSync(path.join(runtimeParent, 'raw-probe-'));
  const ownerFile = path.join(runtimeRoot, 'active-owner.txt');
  const userDataDir = path.join(runtimeRoot, 'edge-profile');
  const outputDir = path.join(runtimeRoot, 'output');
  const owner = `raw-probe-${process.pid}`;
  const protocolVersion = String(process.env.MCP_PROTOCOL_VERSION || '2025-11-25').trim();
  const initPageScript = path.join(workspaceRoot, 'scripts', 'mcp-init-page.js');

  fs.mkdirSync(runtimeRoot, { recursive: true });
  const env = { ...process.env };
  let command = 'node';
  let commandArgs = [launcherPath];

  if (args.target === 'launcher') {
    Object.assign(env, {
      PLAYWRIGHT_MCP_OWNER: owner,
      PLAYWRIGHT_MCP_FORCE_OWNER: 'true',
      PLAYWRIGHT_MCP_BROWSER: args.browser,
      PLAYWRIGHT_MCP_HEADLESS: args.headless ? 'true' : 'false',
      PLAYWRIGHT_MCP_CAPS: args.caps || 'none',
      PLAYWRIGHT_MCP_INIT_PAGE_ENABLED: args.initPageEnabled ? 'true' : 'false',
      PLAYWRIGHT_MCP_SHARED_BROWSER_CONTEXT: args.sharedBrowserContext ? 'true' : 'false',
      PLAYWRIGHT_MCP_PERSIST_PROFILE: args.persistProfile ? 'true' : 'false',
      PLAYWRIGHT_MCP_OWNER_FILE: ownerFile,
      PLAYWRIGHT_MCP_USER_DATA_DIR: userDataDir,
      PLAYWRIGHT_MCP_OUTPUT_DIR: outputDir,
      PLAYWRIGHT_MCP_SAVE_TRACE: args.saveTrace ? 'true' : 'false',
      PLAYWRIGHT_MCP_SAVE_SESSION: args.saveSession ? 'true' : 'false',
      PLAYWRIGHT_MCP_PROXY_INIT_TIMEOUT_MS: String(
        args.launcherInitTimeoutMs || Math.max(1000, args.initTimeoutMs - 3000)
      ),
      PLAYWRIGHT_MCP_PROXY_REQUEST_TIMEOUT_MS: String(
        args.launcherRequestTimeoutMs || Math.max(1000, args.requestTimeoutMs - 2000)
      ),
      PLAYWRIGHT_MCP_PROXY_DEBUG: process.env.PLAYWRIGHT_MCP_PROXY_DEBUG || 'true'
    });
  } else if (args.target === 'playwright-direct' || args.target === 'playwright-cli') {
    const playwrightArgs = buildPlaywrightMcpArgs({
      browserChannel: args.browser,
      outputDir,
      outputMode: process.env.PLAYWRIGHT_MCP_OUTPUT_MODE || 'stdout',
      consoleLevel: process.env.PLAYWRIGHT_MCP_CONSOLE_LEVEL || 'error',
      snapshotMode: process.env.PLAYWRIGHT_MCP_SNAPSHOT_MODE || 'incremental',
      timeoutActionMs: process.env.PLAYWRIGHT_MCP_TIMEOUT_ACTION_MS || '18000',
      timeoutNavigationMs: process.env.PLAYWRIGHT_MCP_TIMEOUT_NAVIGATION_MS || '90000',
      caps: args.caps,
      sharedBrowserContext: args.sharedBrowserContext,
      headless: args.headless,
      persistProfile: args.persistProfile,
      userDataDir,
      saveSession: args.saveSession,
      saveTrace: args.saveTrace,
      initPageEnabled: args.initPageEnabled && fs.existsSync(initPageScript),
      initPagePath: initPageScript
    });
    if (args.target === 'playwright-cli') {
      command = process.execPath;
      commandArgs = [playwrightCliPath, ...playwrightArgs.slice(1)];
    } else if (process.platform === 'win32') {
      command = process.env.ComSpec || 'cmd.exe';
      commandArgs = ['/d', '/s', '/c', 'npx', ...playwrightArgs];
    } else {
      command = 'npx';
      commandArgs = playwrightArgs;
    }
  } else {
    throw new Error(`Unsupported target '${args.target}'. Use 'launcher', 'playwright-direct', or 'playwright-cli'.`);
  }

  const child = spawn(command, commandArgs, {
    cwd: workspaceRoot,
    env,
    stdio: ['pipe', 'pipe', 'pipe'],
    shell: false,
    windowsHide: true
  });

  const stderrLines = [];
  const malformedStdout = [];
  const notifications = [];
  const pendingRequests = new Map();
  let nextId = 1;
  let childExit = null;
  let success = false;

  function log(line) {
    process.stdout.write(`${line}\n`);
  }

  function sendMessage(message) {
    writeMcpMessage(child.stdin, message, CLIENT_FORMAT.JSONL);
  }

  function sendNotification(method, params = {}) {
    sendMessage({
      jsonrpc: '2.0',
      method,
      params
    });
  }

  function sendRequest(method, params, timeoutMs) {
    return new Promise((resolve, reject) => {
      const id = `probe-${nextId++}`;
      const startedAt = Date.now();
      const timer = setTimeout(() => {
        pendingRequests.delete(id);
        reject(new Error(`${method} timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      pendingRequests.set(id, {
        method,
        startedAt,
        resolve: (message) => {
          clearTimeout(timer);
          resolve({ message, durationMs: Date.now() - startedAt });
        },
        reject: (error) => {
          clearTimeout(timer);
          reject(error);
        }
      });

      sendMessage({
        jsonrpc: '2.0',
        id,
        method,
        params
      });
    });
  }

  child.stdout.on(
    'data',
    createMcpMessageParser(
      (message) => {
        const id = message && (typeof message.id === 'string' || typeof message.id === 'number') ? String(message.id) : '';
        if (id && pendingRequests.has(id)) {
          const pending = pendingRequests.get(id);
          pendingRequests.delete(id);
          if (message.error) {
            pending.reject(new Error(`${pending.method} failed: ${message.error.message || 'Unknown MCP error'}`));
          } else {
            pending.resolve(message);
          }
          return;
        }
        notifications.push(message);
      },
      (chunk) => {
        malformedStdout.push(String(chunk || '').trim());
      }
    )
  );

  child.stderr.on('data', (chunk) => {
    for (const line of String(chunk).split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      stderrLines.push(trimmed);
      if (args.verbose) {
        process.stderr.write(`${trimmed}\n`);
      }
    }
  });

  child.on('exit', (code, signal) => {
    childExit = { code, signal };
    for (const [id, pending] of pendingRequests.entries()) {
      pending.reject(new Error(`${pending.method} failed because the launcher exited early (code=${code}, signal=${signal || 'none'})`));
      pendingRequests.delete(id);
    }
  });

  log(`[probe] target=${args.target}`);
  log(`[probe] launcher=${launcherPath}`);
  log(`[probe] runtime_root=${runtimeRoot}`);
  log(`[probe] browser_target=${args.browser}`);
  log(`[probe] browser_mode=${args.headless ? 'headless' : 'headed'}`);
  log(`[probe] launch_profile=${args.profile}`);
  log(`[probe] launch_caps=${args.caps || '(disabled)'}`);
  log(`[probe] launch_init_page=${args.initPageEnabled}`);
  log(`[probe] launch_shared_context=${args.sharedBrowserContext}`);
  log(`[probe] launch_persist_profile=${args.persistProfile}`);
  log(`[probe] step=${args.step}`);
  if (args.target === 'launcher') {
    log(`[probe] launcher_init_timeout_ms=${env.PLAYWRIGHT_MCP_PROXY_INIT_TIMEOUT_MS}`);
    log(`[probe] launcher_request_timeout_ms=${env.PLAYWRIGHT_MCP_PROXY_REQUEST_TIMEOUT_MS}`);
  }

  try {
    const initialize = await sendRequest(
      'initialize',
      {
        protocolVersion,
        capabilities: {},
        clientInfo: {
          name: 'agent-live-web-raw-probe',
          version: '1.0.0'
        }
      },
      args.initTimeoutMs
    );
    const initializeResult = initialize.message.result || {};
    log(
      `[probe] initialize_ok duration_ms=${initialize.durationMs} protocol=${initializeResult.protocolVersion || '(missing)'}`
    );

    sendNotification('notifications/initialized', {});
    log('[probe] initialized_notification_sent');

    if (args.step === 'initialize') {
      success = true;
      return;
    }

    const toolsList = await sendRequest('tools/list', {}, args.requestTimeoutMs);
    const toolNames = Array.isArray(toolsList.message.result && toolsList.message.result.tools)
      ? toolsList.message.result.tools.map((tool) => tool.name).filter(Boolean)
      : [];
    log(
      `[probe] tools_list_ok duration_ms=${toolsList.durationMs} tool_count=${toolNames.length} sample=${toolNames
        .slice(0, 8)
        .join(',')}`
    );
    success = true;
  } catch (error) {
    log(`[probe] failure=${error.message}`);
    const ownerText = readTextIfExists(ownerFile);
    const lockText = readTextIfExists(path.join(userDataDir, '.mcp-owner-lock.json'));
    if (ownerText) {
      log(`[probe] active_owner=${ownerText}`);
    }
    if (lockText) {
      log(`[probe] owner_lock=${lockText}`);
    }
    const files = runtimeFilesSummary(runtimeRoot);
    if (files.length) {
      log('[probe] runtime_files=');
      for (const file of files) {
        log(`  - ${file.path} size=${file.size} mtime=${file.mtime}`);
      }
    }
    const stderrTail = tail(stderrLines, 40);
    if (stderrTail.length) {
      log('[probe] stderr_tail=');
      for (const line of stderrTail) {
        log(`  - ${line}`);
      }
    }
    const malformedTail = tail(malformedStdout, 10);
    if (malformedTail.length) {
      log('[probe] malformed_stdout_tail=');
      for (const line of malformedTail) {
        log(`  - ${line}`);
      }
    }
    if (notifications.length) {
      log(`[probe] notifications_seen=${notifications.length}`);
    }
    throw error;
  } finally {
    try {
      if (child.stdin && !child.stdin.destroyed) {
        child.stdin.end();
      }
    } catch (_) {
      // best effort
    }
    await terminateChildTree(child);

    if (childExit) {
      log(`[probe] launcher_exit code=${childExit.code} signal=${childExit.signal || 'none'}`);
    }

    if (success && !args.keepRuntime) {
      fs.rmSync(runtimeRoot, { recursive: true, force: true });
      log('[probe] runtime_removed');
    } else {
      log('[probe] runtime_kept');
    }
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
