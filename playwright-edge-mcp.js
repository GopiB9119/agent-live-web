const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { buildPlaywrightMcpArgs, normalizeCaps, toBool } = require('./playwright-mcp-launch-config');
const { CLIENT_FORMAT, createMcpMessageParser, writeMcpMessage } = require('./mcp-jsonrpc-transport');
const {
  AGENT_PROXY_STATUS_TOOL_NAME,
  PolicyDecision,
  augmentMcpToolDefinitions,
  augmentMcpToolResult,
  buildMcpSafetyToolResult,
  buildProxyStatusToolResult,
  captureMcpPageState,
  evaluateMcpToolCall,
  shouldRetryMcpExecution,
  verifyMcpExecution,
  writeMcpSafetyEvent
} = require('./mcp-safety-adapter');
const { initTracing, runInSpan, recordException, shutdownTracing } = require('./tracing');

const defaultRuntimeRoot = path.join(process.cwd(), '.playwright-mcp');
const userDataDir =
  process.env.PLAYWRIGHT_MCP_USER_DATA_DIR || path.join(defaultRuntimeRoot, 'edge-profile');
const outputDir = process.env.PLAYWRIGHT_MCP_OUTPUT_DIR || path.join(defaultRuntimeRoot, 'output');
const initPageScript = path.join(process.cwd(), 'scripts', 'mcp-init-page.js');
const ownerFilePath =
  process.env.PLAYWRIGHT_MCP_OWNER_FILE || path.join(defaultRuntimeRoot, 'active-owner.txt');
const owner = (() => {
  const envOwner = normalizeOwner(process.env.PLAYWRIGHT_MCP_OWNER || '');
  if (envOwner) return envOwner;
  try {
    if (fs.existsSync(ownerFilePath)) {
      const fileOwner = normalizeOwner(fs.readFileSync(ownerFilePath, 'utf8'));
      if (fileOwner) return fileOwner;
    }
  } catch (_) {
    // best effort
  }
  return 'vscode';
})();
const explicitActiveOwner = String(process.env.PLAYWRIGHT_MCP_ACTIVE_OWNER || '').trim().toLowerCase();
const lockFilePath = path.join(userDataDir, '.mcp-owner-lock.json');

const extraArgs = process.argv.slice(2);
let child = null;
const tracingEnabled = ['1', 'true', 'yes', 'on'].includes(
  String(process.env.EDGE_TRACING_ENABLED || '').trim().toLowerCase()
);

if (tracingEnabled) {
  initTracing('agent-live-web-vscode-mcp').catch(() => {
    // best effort; never block MCP startup
  });
}

function normalizeOwner(value) {
  return String(value || '').trim().toLowerCase();
}

function info(message) {
  process.stderr.write(`${message}\n`);
}

function debug(message) {
  if (proxyDebug) {
    info(`[MCP][proxy] ${message}`);
  }
}

function toIdKey(id) {
  return typeof id === 'string' || typeof id === 'number' ? String(id) : '';
}

function stripConfirmFields(args = {}) {
  const clean = { ...(args || {}) };
  delete clean.confirm;
  delete clean.confirm_token;
  delete clean.confirmToken;
  return clean;
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

const persistProfile = toBool(process.env.PLAYWRIGHT_MCP_PERSIST_PROFILE, true);
const saveSession = toBool(process.env.PLAYWRIGHT_MCP_SAVE_SESSION, false);
const saveTrace = toBool(process.env.PLAYWRIGHT_MCP_SAVE_TRACE, false);
const forceOwner = toBool(process.env.PLAYWRIGHT_MCP_FORCE_OWNER, true);
const browserChannel = String(process.env.PLAYWRIGHT_MCP_BROWSER || 'msedge').trim() || 'msedge';
const outputMode = process.env.PLAYWRIGHT_MCP_OUTPUT_MODE || 'stdout';
const snapshotMode = process.env.PLAYWRIGHT_MCP_SNAPSHOT_MODE || 'incremental';
const consoleLevel = process.env.PLAYWRIGHT_MCP_CONSOLE_LEVEL || 'error';
const timeoutActionMs = String(process.env.PLAYWRIGHT_MCP_TIMEOUT_ACTION_MS || '18000').trim();
const timeoutNavigationMs = String(process.env.PLAYWRIGHT_MCP_TIMEOUT_NAVIGATION_MS || '90000').trim();
const caps = normalizeCaps(process.env.PLAYWRIGHT_MCP_CAPS, 'vision,pdf');
const sharedBrowserContext = toBool(process.env.PLAYWRIGHT_MCP_SHARED_BROWSER_CONTEXT, true);
const headless = toBool(process.env.PLAYWRIGHT_MCP_HEADLESS, false);
const allowedHosts = process.env.PLAYWRIGHT_MCP_ALLOWED_HOSTS || '';
const allowedOrigins = process.env.PLAYWRIGHT_MCP_ALLOWED_ORIGINS || '';
const blockedOrigins = process.env.PLAYWRIGHT_MCP_BLOCKED_ORIGINS || '';
const blockServiceWorkers = toBool(process.env.PLAYWRIGHT_MCP_BLOCK_SERVICE_WORKERS, false);
const cdpEndpoint = String(process.env.PLAYWRIGHT_MCP_CDP_ENDPOINT || '').trim();
const confirmationSecret =
  String(process.env.PLAYWRIGHT_MCP_CONFIRMATION_SECRET || '').trim() || `mcp:${process.pid}:${process.cwd()}`;
const allowBrowserCodeExecution = toBool(process.env.PLAYWRIGHT_MCP_ALLOW_BROWSER_CODE_EXECUTION, false);
const auditEnabled = toBool(process.env.PLAYWRIGHT_MCP_SAFETY_AUDIT_ENABLED, true);
const auditFile = String(process.env.PLAYWRIGHT_MCP_SAFETY_AUDIT_FILE || '').trim();
const proxyRetryWaitSeconds = Math.max(0, Number.parseFloat(String(process.env.PLAYWRIGHT_MCP_PROXY_RETRY_WAIT_SEC || '0.8')) || 0);
const proxyVerifyWithSnapshot = toBool(process.env.PLAYWRIGHT_MCP_VERIFY_SNAPSHOT, false);
const proxyDebug = toBool(process.env.PLAYWRIGHT_MCP_PROXY_DEBUG, false);
const proxyInitTimeoutMs = Math.max(
  0,
  Number.parseInt(String(process.env.PLAYWRIGHT_MCP_PROXY_INIT_TIMEOUT_MS || '60000'), 10) || 60000
);
const proxyRequestTimeoutMs = Math.max(
  0,
  Number.parseInt(String(process.env.PLAYWRIGHT_MCP_PROXY_REQUEST_TIMEOUT_MS || '45000'), 10) || 45000
);
const initPageEnabled = toBool(process.env.PLAYWRIGHT_MCP_INIT_PAGE_ENABLED, true);
let clientMessageFormat = CLIENT_FORMAT.JSONL;

fs.mkdirSync(outputDir, { recursive: true });
if (persistProfile && !cdpEndpoint) {
  fs.mkdirSync(userDataDir, { recursive: true });
}
fs.mkdirSync(path.dirname(ownerFilePath), { recursive: true });
fs.mkdirSync(path.dirname(lockFilePath), { recursive: true });

const activeOwner = getActiveOwner();
if (activeOwner && owner && owner !== activeOwner) {
  if (!forceOwner) {
    info(`[MCP] Owner blocked. Active owner is '${activeOwner}', current owner is '${owner}'. Exiting.`);
    process.exit(2);
  }
  info(`[MCP] Owner override: active owner '${activeOwner}' will be replaced with '${owner}' if lock is available.`);
}

const lockState = acquireOwnerLock();
if (!lockState.ok) {
  info(`[MCP] ${lockState.reason}`);
  process.exit(3);
}

try {
  if (owner) {
    fs.writeFileSync(ownerFilePath, `${owner}\n`, 'utf8');
  }
} catch (_) {
  // best effort
}

const baseArgs = buildPlaywrightMcpArgs({
  browserChannel,
  outputDir,
  outputMode,
  consoleLevel,
  snapshotMode,
  timeoutActionMs,
  timeoutNavigationMs,
  caps,
  sharedBrowserContext,
  headless,
  cdpEndpoint,
  persistProfile,
  userDataDir,
  saveSession,
  saveTrace,
  allowedHosts,
  allowedOrigins,
  blockedOrigins,
  blockServiceWorkers,
  initPageEnabled: initPageEnabled && fs.existsSync(initPageScript),
  initPagePath: initPageScript
});

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
info(`[MCP] Caps: ${caps || '(disabled)'}`);
info(`[MCP] Shared browser context: ${sharedBrowserContext}`);
info(`[MCP] Browser mode: ${headless ? 'headless' : 'headed'}`);
info(`[MCP] Browser target: ${browserChannel}`);
if (cdpEndpoint) info(`[MCP] CDP endpoint: ${cdpEndpoint}`);
info(`[MCP] Owner: ${owner} | Active owner: ${activeOwner || '(unset)'}`);
info(`[MCP] Owner lock: ${lockFilePath}`);
if (allowedHosts.trim()) info(`[MCP] Network allow hosts: ${allowedHosts.trim()}`);
if (allowedOrigins.trim()) info(`[MCP] Network allow origins: ${allowedOrigins.trim()}`);
if (blockedOrigins.trim()) info(`[MCP] Network block origins: ${blockedOrigins.trim()}`);
if (blockServiceWorkers) info('[MCP] Network: service workers blocked');
info(`[MCP] Init page script: ${initPageEnabled && fs.existsSync(initPageScript) ? initPageScript : '(disabled)'}`);
info('[MCP] Press Ctrl+C to stop.');

const pendingRequests = new Map();
const expiredClientRequestIds = new Set();
let proxyRequestCounter = 0;
if (proxyDebug) {
  const heartbeat = setInterval(() => {
    debug(`heartbeat pending=${pendingRequests.size}`);
  }, 5000);
  heartbeat.unref();
}
const proxyOptions = {
  workspaceRoot: process.cwd(),
  confirmationSecret,
  allowBrowserCodeExecution,
  auditEnabled,
  auditFile,
  owner,
  retryWaitSeconds: proxyRetryWaitSeconds,
  verifyWithSnapshot: proxyVerifyWithSnapshot,
  retryableTools: new Set([
    'browser_navigate',
    'browser_click',
    'browser_type',
    'browser_fill_form',
    'browser_select_option',
    'browser_press_key',
    'browser_wait_for'
  ]),
  stateChangeTools: new Set([
    'browser_click',
    'browser_type',
    'browser_fill_form',
    'browser_select_option',
    'browser_press_key'
  ])
};

function resolveProxyAuditFilePath() {
  const override = String(auditFile || '').trim();
  if (override) {
    return path.resolve(override);
  }
  return path.resolve(process.cwd(), '.agent-state', 'safety-events.jsonl');
}

function buildProxyRuntimeStatusPayload() {
  const lock = tryReadJson(lockFilePath);
  const childPid = child && Number.isInteger(child.pid) ? child.pid : null;
  const childRunning = Boolean(childPid && isProcessAlive(childPid));
  const active = getActiveOwner() || owner || 'unknown';
  const profileMode = persistProfile ? 'persistent' : 'isolated';
  const runtimeStatus = childRunning ? 'ready' : 'degraded';
  const summary = childRunning
    ? 'Guarded direct MCP mode is ready; resume state is not applicable in direct MCP mode.'
    : 'Direct MCP proxy is degraded; child process is not running and resume state is not applicable in direct MCP mode.';

  return {
    runtime_status: runtimeStatus,
    startup_trust: 'direct MCP proxy',
    resume_state: 'not applicable in direct MCP mode',
    summary,
    trust_summary: summary,
    ownership: {
      owner,
      active_owner: active,
      force_owner: forceOwner
    },
    owner_lock: {
      path: lockFilePath,
      held_by_proxy: Boolean(lock && lock.pid === process.pid),
      owner: lock && lock.owner ? lock.owner : owner,
      pid: lock && Number.isInteger(lock.pid) ? lock.pid : null
    },
    browser: {
      target: browserChannel,
      headless,
      shared_context: sharedBrowserContext,
      profile_mode: profileMode,
      user_data_dir: persistProfile && !cdpEndpoint ? userDataDir : '',
      output_dir: outputDir
    },
    safety: {
      confirmation_model: 'preview_confirm_block',
      audit_enabled: auditEnabled,
      audit_file: resolveProxyAuditFilePath(),
      verify_with_snapshot: proxyVerifyWithSnapshot,
      retry_wait_sec: proxyRetryWaitSeconds,
      allow_browser_code_execution: allowBrowserCodeExecution
    },
    timeouts: {
      initialize_ms: proxyInitTimeoutMs,
      request_ms: proxyRequestTimeoutMs,
      action_ms: Number.parseInt(timeoutActionMs, 10) || timeoutActionMs,
      navigation_ms: Number.parseInt(timeoutNavigationMs, 10) || timeoutNavigationMs
    },
    child_process: {
      pid: childPid,
      running: childRunning,
      transport: 'stdio'
    }
  };
}

child = spawn(command, commandArgs, {
  stdio: ['pipe', 'pipe', 'pipe'],
  shell: false
});

function sendProxyRequest(method, params = {}, timeoutMs = 4000) {
  if (!child || !child.stdin || child.stdin.destroyed) {
    return Promise.reject(new Error('Child MCP process is not writable.'));
  }

  return new Promise((resolve, reject) => {
    const id = `proxy-${++proxyRequestCounter}`;
    const timer = setTimeout(() => {
      pendingRequests.delete(id);
      reject(new Error(`Timed out waiting for child MCP response to ${method}.`));
    }, timeoutMs);

    pendingRequests.set(id, {
      origin: 'proxy',
      method,
      resolve: (message) => {
        clearTimeout(timer);
        resolve(message);
      },
      reject: (error) => {
        clearTimeout(timer);
        reject(error);
      }
    });

    writeMcpMessage(child.stdin, {
      jsonrpc: '2.0',
      id,
      method,
      params
    }, CLIENT_FORMAT.JSONL);
  });
}

async function callProxyTool(toolName, args = {}) {
  const response = await sendProxyRequest('tools/call', {
    name: toolName,
    arguments: args
  });
  if (response && response.error) {
    return {
      ok: false,
      error: response.error.message || 'Unknown MCP error',
      result: null
    };
  }
  const result = response && response.result && typeof response.result === 'object' ? response.result : {};
  return {
    ok: !Boolean(result.isError),
    error: result.isError ? 'Tool reported an error.' : null,
    result
  };
}

function shouldCaptureBeforeState(toolName) {
  return toolName === 'browser_navigate' || toolName === 'browser_tabs' || proxyOptions.stateChangeTools.has(toolName);
}

function shouldCaptureSnapshot(toolName) {
  return proxyOptions.verifyWithSnapshot && proxyOptions.stateChangeTools.has(toolName);
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clearPendingClientTimeout(pending) {
  if (pending && pending.timeoutHandle) {
    clearTimeout(pending.timeoutHandle);
  }
}

function getClientRequestTimeoutMs(method) {
  return method === 'initialize' ? proxyInitTimeoutMs : proxyRequestTimeoutMs;
}

function startClientRequestTimeout(idKey, messageId, method) {
  const timeoutMs = getClientRequestTimeoutMs(method);
  if (!(Number.isFinite(timeoutMs) && timeoutMs > 0)) {
    return null;
  }

  debug(`schedule client timeout id=${idKey} method=${method} timeout_ms=${timeoutMs}`);
  return setTimeout(() => {
    debug(`client timeout id=${idKey} method=${method} timeout_ms=${timeoutMs}`);
    const pending = pendingRequests.get(idKey);
    if (!pending || pending.origin !== 'client') {
      return;
    }

    clearPendingClientTimeout(pending);
    pendingRequests.delete(idKey);
    expiredClientRequestIds.add(idKey);

    if (pending.toolName) {
      writeMcpSafetyEvent(process.cwd(), {
        event_type: 'execution',
        transport: 'mcp_stdio_proxy',
        tool: pending.toolName,
        decision: pending.safety ? pending.safety.decision : 'allow_with_verification',
        status: 'failed',
        duration_ms: Math.max(0, Date.now() - Number(pending.startedAt || Date.now())),
        arguments_summary:
          pending.safety && pending.safety.previewSummary ? pending.safety.previewSummary : { tool_name: pending.toolName },
        error: `Timed out waiting for child MCP response to ${method}.`
      }, proxyOptions);
    }

      writeMcpMessage(process.stdout, {
        jsonrpc: '2.0',
        id: messageId,
        error: {
          code: -32001,
          message: `Timed out waiting for child MCP response to ${method}.`
        }
      }, clientMessageFormat);
  }, timeoutMs);
}

const clientParser = createMcpMessageParser(async (message, format) => {
  clientMessageFormat = format || clientMessageFormat;
  const idKey = toIdKey(message && message.id);
  try {
    if (idKey && typeof message.method === 'string') {
      debug(`client request id=${idKey} method=${message.method}`);
      pendingRequests.set(idKey, {
        origin: 'client',
        method: message.method,
        startedAt: Date.now(),
        timeoutHandle: startClientRequestTimeout(idKey, message.id, message.method)
      });
    }

    if (message && message.method === 'tools/call' && idKey) {
      const params = message.params && typeof message.params === 'object' ? message.params : {};
      const toolName = String(params.name || '').trim();
      const args = params.arguments && typeof params.arguments === 'object' ? params.arguments : {};

      if (toolName === AGENT_PROXY_STATUS_TOOL_NAME) {
        const payload = buildProxyRuntimeStatusPayload();
        writeMcpSafetyEvent(process.cwd(), {
          event_type: 'execution',
          transport: 'mcp_stdio_proxy',
          tool: toolName,
          decision: PolicyDecision.ALLOW,
          action_class: 'read_only',
          risk_level: 'low',
          status: 'ok',
          duration_ms: 0,
          arguments_summary: { tool_name: toolName },
          result_summary: {
            runtime_status: payload.runtime_status,
            startup_trust: payload.startup_trust,
            resume_state: payload.resume_state
          }
        }, proxyOptions);
        clearPendingClientTimeout(pendingRequests.get(idKey));
        pendingRequests.delete(idKey);
        writeMcpMessage(process.stdout, {
          jsonrpc: '2.0',
          id: message.id,
          result: buildProxyStatusToolResult(payload)
        }, clientMessageFormat);
        return;
      }

      if (toolName.startsWith('browser_')) {
        const safety = evaluateMcpToolCall(toolName, args, proxyOptions);
        writeMcpSafetyEvent(process.cwd(), {
          event_type: 'decision',
          transport: 'mcp_stdio_proxy',
          tool: toolName,
          decision: safety.decision,
          action_class: safety.actionClass,
          risk_level: safety.riskLevel,
          reason_codes: safety.reasonCodes,
          arguments_summary: safety.previewSummary || { tool_name: toolName }
        }, proxyOptions);
        if (
          [
            PolicyDecision.PREVIEW_REQUIRED,
            PolicyDecision.CONFIRM_REQUIRED,
            PolicyDecision.BLOCKED
          ].includes(safety.decision)
        ) {
          clearPendingClientTimeout(pendingRequests.get(idKey));
          pendingRequests.delete(idKey);
          writeMcpMessage(process.stdout, {
            jsonrpc: '2.0',
            id: message.id,
            result: buildMcpSafetyToolResult(toolName, args, safety)
          }, clientMessageFormat);
          return;
        }

        let beforeState = null;
        if (shouldCaptureBeforeState(toolName)) {
          try {
            beforeState = await captureMcpPageState(
              {
                callTool: callProxyTool,
                verifyWithSnapshot: proxyOptions.verifyWithSnapshot,
                stateChangeTools: proxyOptions.stateChangeTools
              },
              { includeSnapshot: shouldCaptureSnapshot(toolName) }
            );
          } catch (_) {
            beforeState = null;
          }
        }

        const existingPending = pendingRequests.get(idKey) || {};
        pendingRequests.set(idKey, {
          origin: 'client',
          method: message.method,
          toolName,
          args: stripConfirmFields(args),
          safety,
          beforeState,
          startedAt: existingPending.startedAt || Date.now(),
          timeoutHandle: existingPending.timeoutHandle || null
        });
        debug(`forward browser tool id=${idKey} tool=${toolName}`);
        writeMcpMessage(child.stdin, {
          ...message,
          params: {
            ...params,
            arguments: stripConfirmFields(args)
          }
        }, CLIENT_FORMAT.JSONL);
        return;
      }
    }

    if (idKey && typeof message.method === 'string') {
      debug(`forward request id=${idKey} method=${message.method}`);
    }
    writeMcpMessage(child.stdin, message, CLIENT_FORMAT.JSONL);
  } catch (error) {
    info(`[MCP] Safety proxy failed for client request: ${error.message}`);
    if (idKey) {
      clearPendingClientTimeout(pendingRequests.get(idKey));
      pendingRequests.delete(idKey);
      writeMcpMessage(process.stdout, {
        jsonrpc: '2.0',
        id: message.id,
        error: {
          code: -32603,
          message: `Safety proxy failed: ${error.message}`
        }
      }, clientMessageFormat);
    }
  }
}, (chunk) => {
  info(`[MCP] Failed to parse client JSON-RPC payload: ${chunk}`);
});

const serverParser = createMcpMessageParser(async (message) => {
  const idKey = toIdKey(message && message.id);
  const pending = idKey ? pendingRequests.get(idKey) : null;

  if (!pending && idKey && expiredClientRequestIds.has(idKey)) {
    debug(`drop late response id=${idKey}`);
    expiredClientRequestIds.delete(idKey);
    return;
  }

  if (pending && pending.origin === 'proxy') {
    debug(`proxy response id=${idKey} method=${pending.method}`);
    clearPendingClientTimeout(pending);
    pendingRequests.delete(idKey);
    if (message && message.error) {
      pending.reject(new Error(message.error.message || 'Unknown MCP error'));
    } else {
      pending.resolve(message);
    }
    return;
  }

  if (pending && pending.method === 'tools/list' && message && message.result && Array.isArray(message.result.tools)) {
    debug(`client response id=${idKey} method=tools/list`);
    clearPendingClientTimeout(pending);
    pendingRequests.delete(idKey);
    writeMcpMessage(process.stdout, {
      ...message,
      result: {
        ...message.result,
        tools: augmentMcpToolDefinitions(message.result.tools)
      }
    }, clientMessageFormat);
    return;
  }

  if (pending && pending.method === 'tools/call' && pending.toolName) {
    debug(`client response id=${idKey} tool=${pending.toolName}`);
    clearPendingClientTimeout(pending);
    pendingRequests.delete(idKey);
    if (message && message.result && typeof message.result === 'object') {
      let finalResult = message.result;
      let attempts = 1;
      let recovered = false;
      let verification = null;
      try {
        verification = await verifyMcpExecution(
          pending.toolName,
          pending.args || {},
          finalResult,
          {
            callTool: callProxyTool,
            fileExists: fs.existsSync,
            verifyWithSnapshot: proxyOptions.verifyWithSnapshot,
            stateChangeTools: proxyOptions.stateChangeTools
          },
          pending.beforeState || null
        );
      } catch (error) {
        verification = {
          ok: false,
          reason: `Proxy verification failed: ${error.message}`,
          details: {}
        };
      }

      if (shouldRetryMcpExecution(pending.toolName, finalResult, verification, proxyOptions)) {
        attempts = 2;
        if (proxyOptions.retryWaitSeconds > 0) {
          await delay(proxyOptions.retryWaitSeconds * 1000);
        }
        let retryBeforeState = null;
        if (shouldCaptureBeforeState(pending.toolName)) {
          try {
            retryBeforeState = await captureMcpPageState(
              {
                callTool: callProxyTool,
                verifyWithSnapshot: proxyOptions.verifyWithSnapshot,
                stateChangeTools: proxyOptions.stateChangeTools
              },
              { includeSnapshot: shouldCaptureSnapshot(pending.toolName) }
            );
          } catch (_) {
            retryBeforeState = null;
          }
        }
        try {
          const retryOutcome = await callProxyTool(pending.toolName, pending.args || {});
          finalResult =
            retryOutcome && retryOutcome.result
              ? retryOutcome.result
              : {
                  isError: true,
                  content: [{ type: 'text', text: retryOutcome.error || 'Proxy retry failed.' }],
                  structuredContent: { error: retryOutcome.error || 'Proxy retry failed.' }
                };
        } catch (error) {
          finalResult = {
            isError: true,
            content: [{ type: 'text', text: `Proxy retry failed: ${error.message}` }],
            structuredContent: { error: `Proxy retry failed: ${error.message}` }
          };
        }

        try {
          verification = await verifyMcpExecution(
            pending.toolName,
            pending.args || {},
            finalResult,
            {
              callTool: callProxyTool,
              fileExists: fs.existsSync,
              verifyWithSnapshot: proxyOptions.verifyWithSnapshot,
              stateChangeTools: proxyOptions.stateChangeTools
            },
            retryBeforeState || null
          );
        } catch (error) {
          verification = {
            ok: false,
            reason: `Proxy retry verification failed: ${error.message}`,
            details: {}
          };
        }
        recovered = !Boolean(finalResult && finalResult.isError) && Boolean(verification && verification.ok);
      }

      const augmented = augmentMcpToolResult(
        pending.toolName,
        pending.args || {},
        finalResult,
        pending.safety,
        verification,
        { attempts, recovered }
      );
      writeMcpSafetyEvent(process.cwd(), {
        event_type: 'execution',
        transport: 'mcp_stdio_proxy',
        tool: pending.toolName,
        decision: pending.safety ? pending.safety.decision : 'allow_with_verification',
        status: augmented.isError ? 'failed' : verification && verification.ok === false ? 'verification_failed' : 'ok',
        attempts,
        recovered,
        duration_ms: Math.max(0, Date.now() - Number(pending.startedAt || Date.now())),
        arguments_summary: pending.safety && pending.safety.previewSummary ? pending.safety.previewSummary : { tool_name: pending.toolName },
        result_summary: augmented.structuredContent && augmented.structuredContent.evidence
          ? augmented.structuredContent.evidence
          : {}
      }, proxyOptions);
      writeMcpMessage(process.stdout, {
        ...message,
        result: augmented
      }, clientMessageFormat);
      return;
    }

    if (message && message.error) {
      writeMcpSafetyEvent(process.cwd(), {
        event_type: 'execution',
        transport: 'mcp_stdio_proxy',
        tool: pending.toolName,
        decision: pending.safety ? pending.safety.decision : 'allow_with_verification',
        status: 'failed',
        duration_ms: Math.max(0, Date.now() - Number(pending.startedAt || Date.now())),
        arguments_summary: pending.safety && pending.safety.previewSummary ? pending.safety.previewSummary : { tool_name: pending.toolName },
        error: message.error.message || 'Unknown MCP error'
      }, proxyOptions);
    }
  }

  if (pending) {
    debug(`client response id=${idKey} method=${pending.method}`);
    clearPendingClientTimeout(pending);
    pendingRequests.delete(idKey);
  }
  writeMcpMessage(process.stdout, message, clientMessageFormat);
}, (chunk) => {
  info(`[MCP] Failed to parse child JSON-RPC payload: ${chunk}`);
});

process.stdin.on('data', clientParser);
process.stdin.on('end', () => {
  if (child && child.stdin && !child.stdin.destroyed) {
    child.stdin.end();
  }
});

if (child.stdout) {
  child.stdout.on('data', serverParser);
}
if (child.stderr) {
  child.stderr.on('data', (chunk) => {
    process.stderr.write(chunk);
  });
}

runInSpan(
  'mcp.server.launch',
  {
    'app.mcp.owner': owner || 'unknown',
    'app.mcp.profile_persistent': persistProfile,
    'app.mcp.shared_context': sharedBrowserContext
  },
  async (span) => {
    span.addEvent('mcp_child_spawned', { pid: child.pid || 0 });
  }
).catch(() => {
  // best effort
});

child.on('error', (error) => {
  runInSpan('mcp.server.error', {}, async (span) => {
    recordException(span, error);
  }).catch(() => {
    // best effort
  });
  releaseOwnerLock();
  console.error(`[MCP] Failed to start server: ${error.message}`);
  process.exit(1);
});

child.on('exit', (code) => {
  releaseOwnerLock();
  if (tracingEnabled) {
    shutdownTracing().catch(() => {
      // best effort
    });
  }
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
  if (tracingEnabled) {
    shutdownTracing().catch(() => {
      // best effort
    });
  }
});
