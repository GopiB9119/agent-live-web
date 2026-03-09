const {
  ActionClass,
  PolicyDecision,
  buildPreviewSummary,
  classifyBrowserAction,
  issueBrowserConfirmToken,
  normalizeActionParams,
  validateBrowserConfirmToken
} = require('./browser-safety');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const READ_ONLY_TOOL_NAMES = new Set([
  'browser_snapshot',
  'browser_console_messages',
  'browser_network_requests'
]);
const AGENT_PROXY_STATUS_TOOL_NAME = 'agent_proxy_status';

const ARTIFACT_TOOL_NAMES = new Set([
  'browser_take_screenshot',
  'browser_pdf_save'
]);
const DEFAULT_RETRYABLE_TOOL_NAMES = new Set([
  'browser_navigate',
  'browser_click',
  'browser_type',
  'browser_fill_form',
  'browser_select_option',
  'browser_press_key',
  'browser_wait_for'
]);
const DEFAULT_STATE_CHANGE_TOOL_NAMES = new Set([
  'browser_click',
  'browser_type',
  'browser_fill_form',
  'browser_select_option',
  'browser_press_key'
]);
const SENSITIVE_KEY_PATTERN = /(pass(word)?|pwd|token|secret|auth|cookie|session|otp|pin|api[_-]?key|bearer|credential|code|client_secret|refresh_token)/i;
const TOKEN_VALUE_PATTERN = /^(Bearer\s+)?[A-Za-z0-9._-]{24,}$/i;
const TAB_LINE_RE = /^\s*-\s*(?<index>\d+):\s*(?<current>\(current\)\s*)?\[(?<title>.*?)\]\((?<url>.*?)\)\s*$/;

function toBool(value, fallback = false) {
  if (value === undefined || value === null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on';
}

function truncate(value, limit = 240) {
  const text = String(value || '');
  return text.length <= limit ? text : `${text.slice(0, limit)}...`;
}

function sanitizeForAudit(value, key = '', seen = new WeakSet(), depth = 0) {
  if (value === null || value === undefined) return value;
  if (depth > 5) return '[TRUNCATED]';

  if (typeof value === 'string') {
    if (SENSITIVE_KEY_PATTERN.test(key)) return '[REDACTED]';
    if (TOKEN_VALUE_PATTERN.test(value)) return '[REDACTED]';
    return truncate(value, 320);
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 20).map((item) => sanitizeForAudit(item, key, seen, depth + 1));
  }
  if (typeof value === 'object') {
    if (seen.has(value)) return '[CIRCULAR]';
    seen.add(value);
    const result = {};
    for (const [childKey, childValue] of Object.entries(value)) {
      result[childKey] = sanitizeForAudit(childValue, childKey, seen, depth + 1);
    }
    return result;
  }
  return truncate(String(value), 320);
}

function resolveAuditFilePath(workspaceRoot, options = {}) {
  const override = String(options.auditFile || process.env.PLAYWRIGHT_MCP_SAFETY_AUDIT_FILE || '').trim();
  if (override) {
    return path.resolve(override);
  }
  return path.resolve(workspaceRoot || process.cwd(), '.agent-state', 'safety-events.jsonl');
}

function writeMcpSafetyEvent(workspaceRoot, event = {}, options = {}) {
  if (!toBool(options.auditEnabled ?? process.env.PLAYWRIGHT_MCP_SAFETY_AUDIT_ENABLED ?? '1', true)) {
    return false;
  }

  const target = resolveAuditFilePath(workspaceRoot, options);
  const payload = sanitizeForAudit({
    timestamp: new Date().toISOString(),
    source: 'mcp_proxy',
    owner: String(options.owner || process.env.PLAYWRIGHT_MCP_OWNER || 'unknown').trim().toLowerCase() || 'unknown',
    ...(event || {})
  });

  try {
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.appendFileSync(target, `${JSON.stringify(payload)}\n`, 'utf8');
    return true;
  } catch (_) {
    return false;
  }
}

function extractUploadPath(args = {}) {
  if (typeof args.filePath === 'string' && args.filePath.trim()) return args.filePath.trim();
  if (typeof args.path === 'string' && args.path.trim()) return args.path.trim();
  if (Array.isArray(args.paths) && args.paths.length) return String(args.paths[0] || '').trim();
  if (Array.isArray(args.files) && args.files.length) return String(args.files[0] || '').trim();
  return '';
}

function summarizeToolTarget(args = {}) {
  return String(args.element || args.selector || args.ref || args.text || '').trim();
}

function normalizeToolAction(toolName, args = {}) {
  const cleanArgs = normalizeActionParams(args);
  switch (toolName) {
    case 'browser_click':
      return { action: 'click', params: { text: summarizeToolTarget(cleanArgs) } };
    case 'browser_type':
      return {
        action: 'type',
        params: {
          text: summarizeToolTarget(cleanArgs),
          value: cleanArgs.text || cleanArgs.value || ''
        }
      };
    case 'browser_fill_form':
      return {
        action: 'edit',
        params: {
          text: summarizeToolTarget(cleanArgs) || JSON.stringify(cleanArgs.fields || cleanArgs.form || {}),
          value: JSON.stringify(cleanArgs.fields || cleanArgs.form || {})
        }
      };
    case 'browser_select_option':
      return {
        action: 'edit',
        params: {
          text: summarizeToolTarget(cleanArgs),
          value: JSON.stringify(cleanArgs.values || cleanArgs.options || cleanArgs.value || '')
        }
      };
    case 'browser_press_key':
      return {
        action: 'type',
        params: {
          text: summarizeToolTarget(cleanArgs) || String(cleanArgs.key || ''),
          value: String(cleanArgs.key || '')
        }
      };
    case 'browser_file_upload':
      return {
        action: 'upload',
        params: {
          filePath: extractUploadPath(cleanArgs),
          text: summarizeToolTarget(cleanArgs)
        }
      };
    case 'browser_take_screenshot':
      return { action: 'screenshot', params: { path: cleanArgs.path || cleanArgs.savePath || '' } };
    case 'browser_pdf_save':
      return { action: 'pdf', params: { path: cleanArgs.path || cleanArgs.savePath || '' } };
    case 'browser_navigate':
      return { action: 'goto', params: { url: cleanArgs.url || '' } };
    case 'browser_wait_for':
      return { action: 'waitFor', params: { text: summarizeToolTarget(cleanArgs) } };
    default:
      return { action: toolName, params: cleanArgs };
  }
}

function withToolName(toolName, previewSummary = {}) {
  return {
    tool_name: toolName,
    ...previewSummary
  };
}

function confirmOrPreview(toolName, cleanArgs, baseEvaluation, options = {}) {
  const confirmRequested = toBool(cleanArgs.confirm, false);
  const confirmToken = String(cleanArgs.confirm_token || cleanArgs.confirmToken || '').trim();
  const strippedArgs = normalizeActionParams(cleanArgs);
  const secret = String(options.confirmationSecret || '');

  if ([PolicyDecision.PREVIEW_REQUIRED, PolicyDecision.CONFIRM_REQUIRED].includes(baseEvaluation.decision)) {
    if (confirmRequested && validateBrowserConfirmToken(secret, toolName, strippedArgs, confirmToken)) {
      return {
        ...baseEvaluation,
        decision: PolicyDecision.ALLOW_WITH_VERIFICATION,
        reasonCodes: [...(baseEvaluation.reasonCodes || []), 'confirmed'],
        confirmToken: ''
      };
    }
    return {
      ...baseEvaluation,
      confirmToken: issueBrowserConfirmToken(secret, toolName, strippedArgs)
    };
  }

  return {
    ...baseEvaluation,
    confirmToken: ''
  };
}

function evaluateBrowserTabsTool(args = {}, options = {}) {
  const cleanArgs = normalizeActionParams(args);
  const tabAction = String(cleanArgs.action || 'list').trim().toLowerCase() || 'list';
  const previewSummary = {
    tool_name: 'browser_tabs',
    tab_action: tabAction
  };

  if (tabAction === 'list') {
    return {
      actionClass: ActionClass.READ_ONLY,
      riskLevel: 'low',
      decision: PolicyDecision.ALLOW,
      reasonCodes: ['read-only'],
      requiresVerification: false,
      previewSummary
    };
  }

  if (tabAction === 'select') {
    return {
      actionClass: ActionClass.READ_ONLY,
      riskLevel: 'low',
      decision: PolicyDecision.ALLOW,
      reasonCodes: ['tab-select'],
      requiresVerification: false,
      previewSummary
    };
  }

  if (tabAction === 'new') {
    return {
      actionClass: ActionClass.SCOPED_REVERSIBLE_WRITE,
      riskLevel: 'medium',
      decision: PolicyDecision.ALLOW_WITH_VERIFICATION,
      reasonCodes: ['tab-open'],
      requiresVerification: true,
      previewSummary
    };
  }

  if (tabAction === 'close') {
    return confirmOrPreview(
      'browser_tabs',
      cleanArgs,
      {
        actionClass: ActionClass.SCOPED_REVERSIBLE_WRITE,
        riskLevel: 'high',
        decision: PolicyDecision.PREVIEW_REQUIRED,
        reasonCodes: ['tab-close'],
        requiresVerification: true,
        previewSummary: {
          ...previewSummary,
          index: cleanArgs.index
        }
      },
      options
    );
  }

  return {
    actionClass: ActionClass.SCOPED_REVERSIBLE_WRITE,
    riskLevel: 'medium',
    decision: PolicyDecision.ALLOW_WITH_VERIFICATION,
    reasonCodes: ['tab-maintenance'],
    requiresVerification: true,
    previewSummary
  };
}

function evaluateMcpToolCall(toolName, args = {}, options = {}) {
  const cleanName = String(toolName || '').trim();
  const cleanArgs = args && typeof args === 'object' ? { ...args } : {};

  if (READ_ONLY_TOOL_NAMES.has(cleanName)) {
    return {
      actionClass: ActionClass.READ_ONLY,
      riskLevel: 'low',
      decision: PolicyDecision.ALLOW,
      reasonCodes: ['read-only'],
      requiresVerification: false,
      previewSummary: withToolName(cleanName, {})
    };
  }

  if (cleanName === 'browser_tabs') {
    return evaluateBrowserTabsTool(cleanArgs, options);
  }

  if (ARTIFACT_TOOL_NAMES.has(cleanName)) {
    const normalized = normalizeToolAction(cleanName, cleanArgs);
    return {
      actionClass: ActionClass.SCOPED_REVERSIBLE_WRITE,
      riskLevel: 'medium',
      decision: PolicyDecision.ALLOW_WITH_VERIFICATION,
      reasonCodes: ['bounded-local-artifact'],
      requiresVerification: true,
      previewSummary: withToolName(cleanName, buildPreviewSummary(normalized.action, normalized.params, options))
    };
  }

  if (cleanName === 'browser_evaluate' || cleanName === 'browser_run_code') {
    const base = {
      actionClass: ActionClass.DESTRUCTIVE,
      riskLevel: 'critical',
      decision: toBool(options.allowBrowserCodeExecution, false)
        ? PolicyDecision.CONFIRM_REQUIRED
        : PolicyDecision.BLOCKED,
      reasonCodes: [
        toBool(options.allowBrowserCodeExecution, false)
          ? 'browser-code-execution'
          : 'browser-code-execution-disabled'
      ],
      requiresVerification: true,
      previewSummary: withToolName(cleanName, {})
    };
    return confirmOrPreview(cleanName, cleanArgs, base, options);
  }

  if (cleanName === 'browser_close' || cleanName === 'browser_install') {
    return {
      actionClass: ActionClass.SCOPED_REVERSIBLE_WRITE,
      riskLevel: 'medium',
      decision: PolicyDecision.ALLOW_WITH_VERIFICATION,
      reasonCodes: ['browser-session-maintenance'],
      requiresVerification: true,
      previewSummary: withToolName(cleanName, {})
    };
  }

  const normalized = normalizeToolAction(cleanName, cleanArgs);
  const baseEvaluation = classifyBrowserAction(normalized.action, normalized.params, options);
  return confirmOrPreview(
    cleanName,
    cleanArgs,
    {
      ...baseEvaluation,
      previewSummary: withToolName(cleanName, baseEvaluation.previewSummary || {})
    },
    options
  );
}

function ensureConfirmSchemaFields(inputSchema) {
  const schema = inputSchema && typeof inputSchema === 'object' ? { ...inputSchema } : { type: 'object' };
  schema.type = schema.type || 'object';
  schema.properties = { ...(schema.properties || {}) };
  if (!Object.prototype.hasOwnProperty.call(schema.properties, 'confirm')) {
    schema.properties.confirm = {
      type: 'boolean',
      description: 'Set true with confirm_token to continue a previously previewed or gated browser action.'
    };
  }
  if (!Object.prototype.hasOwnProperty.call(schema.properties, 'confirm_token')) {
    schema.properties.confirm_token = {
      type: 'string',
      description: 'Token returned by a previous preview_required or confirm_required browser tool result.'
    };
  }
  return schema;
}

function buildAgentProxyStatusToolDefinition() {
  return {
    name: AGENT_PROXY_STATUS_TOOL_NAME,
    description:
      'Return readiness and safety status for the local agent-live-web MCP proxy, including owner lock, verification, audit, and direct MCP trust state.',
    inputSchema: {
      type: 'object',
      properties: {},
      additionalProperties: false
    }
  };
}

function augmentMcpToolDefinitions(tools = []) {
  const augmented = tools.map((tool) => {
    const copy = { ...(tool || {}) };
    if (!String(copy.name || '').startsWith('browser_')) {
      return copy;
    }
    copy.inputSchema = ensureConfirmSchemaFields(copy.inputSchema);
    return copy;
  });

  if (!augmented.some((tool) => String(tool && tool.name ? tool.name : '').trim() === AGENT_PROXY_STATUS_TOOL_NAME)) {
    augmented.push(buildAgentProxyStatusToolDefinition());
  }

  return augmented;
}

function buildMcpSafetyToolResult(toolName, args, evaluation) {
  const payload = {
    status: evaluation.decision,
    tool: toolName,
    action_class: evaluation.actionClass,
    risk_level: evaluation.riskLevel,
    reason_codes: Array.isArray(evaluation.reasonCodes) ? evaluation.reasonCodes : [],
    requires_verification: Boolean(evaluation.requiresVerification)
  };

  if (evaluation.previewSummary) {
    payload.preview = {
      status: 'ok',
      preview: evaluation.previewSummary
    };
  }

  if (evaluation.confirmToken) {
    payload.confirm_token = evaluation.confirmToken;
  }

  return {
    content: [
      {
        type: 'text',
        text: JSON.stringify(payload)
      }
    ],
    structuredContent: payload,
    isError: evaluation.decision === PolicyDecision.BLOCKED
  };
}

function buildProxyStatusToolResult(status = {}) {
  const payload = {
    ...sanitizeForAudit(status),
    status: 'ok',
    tool: AGENT_PROXY_STATUS_TOOL_NAME,
    mode: 'direct_mcp_proxy'
  };
  const summary = String(
    payload.summary ||
      payload.trust_summary ||
      payload.readiness_summary ||
      'Direct MCP proxy status reported.'
  ).trim();

  return {
    content: [
      {
        type: 'text',
        text: `[agent-live-web] ${summary}`
      }
    ],
    structuredContent: payload,
    isError: false
  };
}

function collectResultText(result = {}) {
  const parts = [];
  for (const item of Array.isArray(result.content) ? result.content : []) {
    if (item && typeof item.text === 'string') {
      parts.push(item.text);
    }
  }
  if (!parts.length && result.structuredContent && typeof result.structuredContent === 'object') {
    parts.push(JSON.stringify(result.structuredContent));
  }
  return truncate(parts.join('\n'), 400);
}

function parseMcpTabsText(text = '') {
  const tabs = [];
  for (const line of String(text || '').split(/\r?\n/)) {
    const match = TAB_LINE_RE.exec(line.trim());
    if (!match || !match.groups) continue;
    tabs.push({
      index: Number.parseInt(match.groups.index, 10),
      current: Boolean(match.groups.current),
      title: match.groups.title || '',
      url: match.groups.url || ''
    });
  }
  tabs.sort((a, b) => a.index - b.index);
  return tabs;
}

function hostFromUrl(url) {
  try {
    return (new URL(String(url || '')).hostname || '').toLowerCase();
  } catch (_) {
    return '';
  }
}

function hostsMatch(expectedUrl, actualUrl) {
  const expectedHost = hostFromUrl(expectedUrl);
  const actualHost = hostFromUrl(actualUrl);
  if (!expectedHost) return Boolean(actualHost);
  if (expectedHost === actualHost) return true;
  return actualHost.endsWith(`.${expectedHost}`);
}

function extractArtifactPath(toolName, args = {}, result = {}) {
  const structured = result.structuredContent && typeof result.structuredContent === 'object' ? result.structuredContent : {};
  if (typeof structured.path === 'string' && structured.path.trim()) {
    return structured.path.trim();
  }
  if (typeof structured.output_path === 'string' && structured.output_path.trim()) {
    return structured.output_path.trim();
  }
  const reportedText = collectResultText(result);
  const reportedArtifactMatch = /\]\(([^)\r\n]+)\)/.exec(reportedText);
  if (reportedArtifactMatch && reportedArtifactMatch[1] && reportedArtifactMatch[1].trim()) {
    return reportedArtifactMatch[1].trim();
  }
  if (toolName === 'browser_take_screenshot' || toolName === 'browser_pdf_save') {
    return String(args.path || args.savePath || '').trim();
  }
  return '';
}

function normalizeToolNameSet(input, fallbackSet) {
  if (input instanceof Set) return input;
  if (Array.isArray(input)) return new Set(input.map((value) => String(value || '').trim()).filter(Boolean));
  return new Set(fallbackSet);
}

function hashSnapshotText(value) {
  return crypto.createHash('sha1').update(String(value || ''), 'utf8').digest('hex');
}

async function captureMcpPageState(helpers = {}, options = {}) {
  const callTool = typeof helpers.callTool === 'function' ? helpers.callTool : null;
  if (!callTool) {
    return {
      tabs_ok: false,
      tabs_count: 0,
      url: '',
      title: '',
      index: null,
      snapshot_hash: null
    };
  }

  const tabsResult = await callTool('browser_tabs', { action: 'list' });
  const tabsText = collectResultText(tabsResult.result || tabsResult);
  const tabs = parseMcpTabsText(tabsText);
  const current = tabs.find((tab) => tab.current) || null;
  const state = {
    tabs_ok: Boolean(tabsResult.ok),
    tabs_count: tabs.length,
    url: current ? current.url : '',
    title: current ? current.title : '',
    index: current ? current.index : null,
    snapshot_hash: null
  };

  if (options.includeSnapshot && Boolean(tabsResult.ok)) {
    const snapshotResult = await callTool('browser_snapshot', {});
    if (snapshotResult && snapshotResult.ok) {
      state.snapshot_hash = hashSnapshotText(collectResultText(snapshotResult.result || snapshotResult));
    }
  }

  return state;
}

async function verifyMcpExecution(toolName, args = {}, result = {}, helpers = {}, beforeState = null) {
  const callTool = typeof helpers.callTool === 'function' ? helpers.callTool : null;
  const fileExists = typeof helpers.fileExists === 'function' ? helpers.fileExists : fs.existsSync;
  const stateChangeTools = normalizeToolNameSet(helpers.stateChangeTools, DEFAULT_STATE_CHANGE_TOOL_NAMES);
  const verifyWithSnapshot = Boolean(helpers.verifyWithSnapshot);

  if (result && result.isError) {
    return {
      ok: false,
      reason: 'Child MCP server reported an error.',
      details: {}
    };
  }

  if (toolName === 'browser_navigate' && callTool) {
    const after = await captureMcpPageState(helpers, { includeSnapshot: false });
    const actualUrl = after.url || '';
    const expectedUrl = String(args.url || '').trim();
    const ok = Boolean(actualUrl && actualUrl !== 'about:blank' && hostsMatch(expectedUrl, actualUrl));
    return {
      ok,
      reason: ok
        ? `Navigation verified on '${actualUrl}'.`
        : `Expected host from '${expectedUrl}', current url is '${actualUrl || '(unknown)'}'.`,
      details: {
        before: beforeState || {},
        after
      }
    };
  }

  if (toolName === 'browser_tabs' && callTool) {
    const tabsResult = await callTool('browser_tabs', { action: 'list' });
    const tabs = parseMcpTabsText(collectResultText(tabsResult.result || tabsResult));
    const current = tabs.find((tab) => tab.current) || null;
    const tabAction = String(args.action || 'list').trim().toLowerCase() || 'list';
    if (tabAction === 'select' && args.index !== undefined) {
      const ok = current && Number(current.index) === Number(args.index);
      return {
        ok,
        reason: `Tab select target=${args.index}, current=${current ? current.index : 'none'}.`,
        details: { before: beforeState || {}, current, tabs }
      };
    }
    if (tabAction === 'close' && args.index !== undefined) {
      const closed = !tabs.some((tab) => Number(tab.index) === Number(args.index));
      return {
        ok: closed,
        reason: closed ? `Tab ${args.index} is no longer present.` : `Tab ${args.index} is still present after close.`,
        details: { before: beforeState || {}, current, tabs }
      };
    }
    if (tabAction === 'new') {
      const ok = tabs.length >= 1;
      return {
        ok,
        reason: ok ? `Tab count is now ${tabs.length}.` : 'No tabs were reported after opening a new tab.',
        details: { before: beforeState || {}, current, tabs }
      };
    }
    return {
      ok: true,
      reason: 'Tab state verified.',
      details: { before: beforeState || {}, current, tabs }
    };
  }

  if ((toolName === 'browser_take_screenshot' || toolName === 'browser_pdf_save')) {
    const artifactPath = extractArtifactPath(toolName, args, result);
    const ok = Boolean(artifactPath && fileExists(artifactPath));
    return {
      ok,
      reason: ok
        ? `Artifact verified at '${artifactPath}'.`
        : `Artifact path '${artifactPath || '(unknown)'}' could not be verified on disk.`,
      details: {
        output_path: artifactPath
      }
    };
  }

  if (stateChangeTools.has(toolName) && callTool) {
    const after = await captureMcpPageState(helpers, { includeSnapshot: verifyWithSnapshot });
    let changed = false;
    if (beforeState && beforeState.url !== after.url) {
      changed = true;
    }
    if (
      beforeState &&
      beforeState.snapshot_hash &&
      after.snapshot_hash &&
      beforeState.snapshot_hash !== after.snapshot_hash
    ) {
      changed = true;
    }
    const stillAlive = Boolean(after.url && after.url !== 'about:blank');
    return {
      ok: changed || stillAlive,
      reason: changed ? 'Page state changed after action.' : 'Page remained stable but active tab is valid.',
      details: {
        before: beforeState || {},
        after
      }
    };
  }

  if (toolName === 'browser_wait_for') {
    return {
      ok: true,
      reason: 'Wait condition satisfied by tool.',
      details: {
        before: beforeState || {}
      }
    };
  }

  if (callTool) {
    const after = await captureMcpPageState(helpers, { includeSnapshot: false });
    const ok = Boolean(after.url || toolName === 'browser_close' || toolName === 'browser_install');
    return {
      ok,
      reason: ok ? 'Browser context is reachable.' : 'Browser context could not be verified.',
      details: {
        before: beforeState || {},
        after
      }
    };
  }

  return {
    ok: true,
    reason: 'No additional proxy verification was available.',
    details: {}
  };
}

function shouldRetryMcpExecution(toolName, result = {}, verification = null, options = {}) {
  const retryableTools = normalizeToolNameSet(options.retryableTools, DEFAULT_RETRYABLE_TOOL_NAMES);
  if (!retryableTools.has(String(toolName || '').trim())) {
    return false;
  }
  if (result && result.isError) {
    return true;
  }
  return Boolean(verification && verification.ok === false);
}

function buildMcpEvidenceSummary(toolName, args = {}, result = {}, verification = null) {
  const reportedText = collectResultText(result);
  const isError = Boolean(result && result.isError);
  const target = summarizeToolTarget(args);
  const artifactPath = extractArtifactPath(toolName, args, result);
  let summary = isError ? `${toolName} reported an error.` : `${toolName} completed according to the child MCP server.`;

  if (toolName === 'browser_navigate' && args.url) {
    summary = isError
      ? `Navigation to '${args.url}' was reported as failed.`
      : `Navigation to '${args.url}' was reported as completed.`;
  } else if (toolName === 'browser_click' && target) {
    summary = isError
      ? `Click on '${target}' was reported as failed.`
      : `Click on '${target}' was reported as completed.`;
  } else if (toolName === 'browser_file_upload') {
    const uploadPath = extractUploadPath(args);
    summary = isError
      ? `Upload of '${uploadPath || '(unspecified file)'}' was reported as failed.`
      : `Upload of '${uploadPath || '(unspecified file)'}' was reported as completed.`;
  } else if ((toolName === 'browser_take_screenshot' || toolName === 'browser_pdf_save') && artifactPath) {
    summary = isError
      ? `Artifact write to '${artifactPath}' was reported as failed.`
      : `Artifact write to '${artifactPath}' was reported as completed.`;
  }

  const evidence = {
    source: 'mcp_proxy',
    status: isError ? 'failed' : verification && verification.ok === false ? 'verification_failed' : verification && verification.ok ? 'verified' : 'reported_ok',
    summary,
    note: 'Evidence is child-reported by the Playwright MCP server; no additional browser snapshot was taken by the proxy.'
  };
  if (verification && typeof verification === 'object') {
    evidence.verification = sanitizeForAudit(verification);
  }
  if (target) evidence.target = truncate(target, 180);
  if (artifactPath) evidence.output_path = artifactPath;
  if (reportedText) evidence.reported_text = reportedText;
  return evidence;
}

function augmentMcpToolResult(toolName, args = {}, result = {}, evaluation = null, verification = null, execution = null) {
  const baseResult = result && typeof result === 'object' ? { ...result } : {};
  const structuredSource =
    baseResult.structuredContent && typeof baseResult.structuredContent === 'object' && !Array.isArray(baseResult.structuredContent)
      ? { ...baseResult.structuredContent }
      : {};
  const evidence = buildMcpEvidenceSummary(toolName, args, baseResult, verification);
  const safety = evaluation
    ? {
        tool: toolName,
        action_class: evaluation.actionClass,
        risk_level: evaluation.riskLevel,
        decision: evaluation.decision,
        reason_codes: Array.isArray(evaluation.reasonCodes) ? evaluation.reasonCodes : [],
        requires_verification: Boolean(evaluation.requiresVerification)
      }
    : {
        tool: toolName,
        action_class: ActionClass.SCOPED_REVERSIBLE_WRITE,
        risk_level: 'medium',
        decision: PolicyDecision.ALLOW_WITH_VERIFICATION,
        reason_codes: ['proxy-executed'],
        requires_verification: true
      };

  baseResult.structuredContent = {
    ...structuredSource,
    safety,
    evidence,
    ...(verification ? { verification } : {}),
    ...(execution ? { execution } : {})
  };

  const content = Array.isArray(baseResult.content) ? [...baseResult.content] : [];
  content.push({
    type: 'text',
    text: `[agent-live-web] ${evidence.summary}`
  });
  baseResult.content = content;
  return baseResult;
}

module.exports = {
  AGENT_PROXY_STATUS_TOOL_NAME,
  PolicyDecision,
  augmentMcpToolDefinitions,
  augmentMcpToolResult,
  buildMcpSafetyToolResult,
  buildProxyStatusToolResult,
  buildMcpEvidenceSummary,
  captureMcpPageState,
  hostsMatch,
  evaluateMcpToolCall,
  parseMcpTabsText,
  shouldRetryMcpExecution,
  verifyMcpExecution,
  writeMcpSafetyEvent
};
