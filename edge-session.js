const { chromium } = require('playwright');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { runInSpan, getActiveTraceMeta } = require('./tracing');

function toInt(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toBool(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on';
}

function escapeRegex(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function escapeAttrValue(value) {
  return String(value).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

function hostFromUrl(value) {
  try {
    return (new URL(String(value || '')).hostname || '').toLowerCase();
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

function safeJson(value) {
  try {
    return JSON.stringify(value);
  } catch (_) {
    return '';
  }
}

function capitalize(value) {
  const text = String(value || '');
  if (!text) return '';
  return text.charAt(0).toUpperCase() + text.slice(1);
}

const SENSITIVE_KEY_PATTERN = /(pass(word)?|pwd|token|secret|auth|cookie|session|otp|pin|api[_-]?key|bearer|credential|code)/i;
const TOKEN_VALUE_PATTERN = /^(Bearer\s+)?[A-Za-z0-9._-]{24,}$/i;
const URL_CREDENTIAL_PATTERN = /(https?:\/\/)([^/@\s]+(?::[^/@\s]+)?@)/gi;
const SENSITIVE_QUERY_PARAM_PATTERN = /([?&](?:api[_-]?key|token|secret|password|passwd|pwd|cookie|authorization|auth|session|bearer|credential|otp|pin|client_secret|access[_-]?token|refresh[_-]?token|code)=)([^&#\s]+)/gi;
const SENSITIVE_TEXT_PATTERNS = [
  {
    type: 'bearer',
    pattern: /\bBearer\s+[A-Za-z0-9._-]{12,}\b/gi
  },
  {
    type: 'key-value',
    pattern: /\b(api[_-]?key|token|secret|password|passwd|pwd|cookie|authorization|client_secret|access[_-]?token|refresh[_-]?token)\s*[:=]\s*([^\s,;]+)/gi
  },
  {
    type: 'generic',
    pattern: /\bghp_[A-Za-z0-9]{20,}\b/g
  },
  {
    type: 'generic',
    pattern: /\bAKIA[0-9A-Z]{16}\b/g
  },
  {
    type: 'generic',
    pattern: /\bAIza[0-9A-Za-z\-_]{20,}\b/g
  }
];
const ACTIONS_WITH_SNAPSHOT_VERIFICATION = new Set(['search', 'click', 'clickXPath', 'clickByText', 'hover', 'doubleClick', 'rightClick']);
const FILE_OUTPUT_ACTIONS = new Set(['download', 'screenshot', 'stopTrace']);

function sanitizeUrlForLog(value) {
  return String(value || '')
    .replace(URL_CREDENTIAL_PATTERN, '$1[REDACTED]@')
    .replace(SENSITIVE_QUERY_PARAM_PATTERN, (_, prefix) => `${prefix}[REDACTED]`);
}

function redactSensitiveText(value) {
  let text = sanitizeUrlForLog(value);
  for (const entry of SENSITIVE_TEXT_PATTERNS) {
    if (entry.type === 'key-value') {
      text = text.replace(entry.pattern, (_, key) => `${key}=[REDACTED]`);
    } else if (entry.type === 'bearer') {
      text = text.replace(entry.pattern, 'Bearer [REDACTED]');
    } else {
      text = text.replace(entry.pattern, '[REDACTED]');
    }
  }
  return text;
}

function sanitizeForLog(value, key = '', seen = new WeakSet(), depth = 0) {
  if (value === null || value === undefined) return value;
  if (depth > 6) return '[TRUNCATED]';

  if (typeof value === 'string') {
    if (SENSITIVE_KEY_PATTERN.test(key)) return '[REDACTED]';
    const sanitized = /url|uri|link|href/i.test(key) ? sanitizeUrlForLog(value) : redactSensitiveText(value);
    if (TOKEN_VALUE_PATTERN.test(sanitized)) return '[REDACTED]';
    return sanitized;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 25).map((item) => sanitizeForLog(item, key, seen, depth + 1));
  }
  if (typeof value === 'object') {
    if (seen.has(value)) return '[CIRCULAR]';
    seen.add(value);
    const result = {};
    for (const [childKey, childValue] of Object.entries(value)) {
      if (SENSITIVE_KEY_PATTERN.test(childKey)) {
        result[childKey] = '[REDACTED]';
      } else {
        result[childKey] = sanitizeForLog(childValue, childKey, seen, depth + 1);
      }
    }
    return result;
  }
  return String(value);
}

class EdgeSession {
  constructor(options = {}) {
    this.context = null;
    this.page = null;
    this.boundPages = new WeakSet();
    this.headless = Boolean(options.headless);
    this.userDataDir = options.userDataDir || process.env.EDGE_USER_DATA_DIR || path.join(process.cwd(), '.playwright-edge-profile');
    this.localOperatorMode = toBool(
      options.localOperatorMode !== undefined ? options.localOperatorMode : process.env.EDGE_LOCAL_OPERATOR_MODE,
      true
    );
    this.workspaceRoot = path.resolve(options.workspaceRoot || process.env.EDGE_WORKSPACE_ROOT || process.cwd());
    this.restrictWriteToWorkspace = toBool(
      options.restrictWriteToWorkspace !== undefined
        ? options.restrictWriteToWorkspace
        : process.env.EDGE_RESTRICT_WRITE_TO_WORKSPACE,
      true
    );
    this.allowDomHtmlAdd = toBool(
      options.allowDomHtmlAdd !== undefined ? options.allowDomHtmlAdd : process.env.EDGE_ALLOW_DOM_HTML_ADD,
      false
    );
    this.allowDomDelete = toBool(
      options.allowDomDelete !== undefined ? options.allowDomDelete : process.env.EDGE_ALLOW_DOM_DELETE,
      false
    );
    this.domFallbackOnFailure = toBool(
      options.domFallbackOnFailure !== undefined
        ? options.domFallbackOnFailure
        : process.env.EDGE_DOM_FALLBACK_ON_FAILURE,
      true
    );
    this.frameAwareLocators = toBool(
      options.frameAwareLocators !== undefined
        ? options.frameAwareLocators
        : process.env.EDGE_FRAME_AWARE_LOCATORS,
      true
    );
    this.logFile = options.logFile || path.join(process.cwd(), 'logs', 'edge-agent.log');
    this.writeLogFile = toBool(
      options.writeLogFile !== undefined ? options.writeLogFile : process.env.EDGE_WRITE_LOG_FILE,
      false
    );
    this.logToConsole = toBool(
      options.logToConsole !== undefined ? options.logToConsole : process.env.EDGE_LOG_TO_CONSOLE,
      false
    );
    this.actionTimeout = toInt(options.actionTimeout || process.env.EDGE_ACTION_TIMEOUT_MS, 30000);
    this.navigationTimeout = toInt(options.navigationTimeout || process.env.EDGE_NAV_TIMEOUT_MS, 180000);
    this.retryCount = toInt(options.retryCount || process.env.EDGE_ACTION_RETRIES, 2);
    this.logActionPayloads = toBool(
      options.logActionPayloads !== undefined ? options.logActionPayloads : process.env.EDGE_LOG_ACTION_PAYLOADS,
      false
    );
    this.traceActive = false;
  }

  async open() {
    return runInSpan(
      'edge.session.open',
      {
        'app.browser.channel': 'msedge',
        'app.edge.headless': this.headless,
        'app.edge.retry_count': this.retryCount
      },
      async () => {
        await fs.promises.mkdir(this.userDataDir, { recursive: true });
        if (this.writeLogFile) {
          await fs.promises.mkdir(path.dirname(this.logFile), { recursive: true });
        }

        this.context = await chromium.launchPersistentContext(this.userDataDir, {
          channel: 'msedge',
          headless: this.headless,
          viewport: null,
          acceptDownloads: true,
          args: [
            '--no-default-browser-check',
            '--disable-features=Translate'
          ]
        });
        this.context.setDefaultTimeout(this.actionTimeout);
        this.context.setDefaultNavigationTimeout(this.navigationTimeout);

        this.context.on('page', (page) => this.bindPage(page));
        const existingPages = this.context.pages();
        this.page = existingPages.length ? existingPages[0] : await this.context.newPage();
        this.page.setDefaultTimeout(this.actionTimeout);
        this.page.setDefaultNavigationTimeout(this.navigationTimeout);
        this.bindPage(this.page);

        this.log('INFO', 'Edge session opened', {
          userDataDir: this.userDataDir,
          headless: this.headless,
          actionTimeout: this.actionTimeout,
          navigationTimeout: this.navigationTimeout,
          retryCount: this.retryCount
        });
        return this.page;
      }
    );
  }

  async close() {
    return runInSpan('edge.session.close', {}, async () => {
      if (!this.context) return;
      await this.context.close();
      this.context = null;
      this.page = null;
      this.log('INFO', 'Edge session closed');
    });
  }

  async newSession() {
    await this.close();
    await this.open();
  }

  async runAction(action, params = {}) {
    const startedAt = Date.now();
    const safeParams = this.logActionPayloads ? this.sanitizeActionParams(action, params) : undefined;
    this.log('INFO', 'Action started', safeParams ? { action, params: safeParams } : { action });

    if (!this.page) {
      const response = this.sanitizeActionResponse(this.formatActionResponse({
        action,
        args: this.sanitizeActionParams(action, params),
        attempts: 0,
        recovered: false,
        verification: {
          ok: false,
          reason: 'Edge session is not open.',
          details: { before: null, after: null }
        },
        result: null,
        error: 'Edge session is not open',
        durationMs: Date.now() - startedAt
      }));
      this.log('ERROR', 'Action failed', { action, durationMs: response.durationMs, error: response.error });
      return response;
    }

    const includeSnapshot = ACTIONS_WITH_SNAPSHOT_VERIFICATION.has(action);
    const beforeState = await this.captureActionState(includeSnapshot);

    try {
      const execution = await runInSpan(
        'edge.action.execute',
        {
          'app.action.name': action,
          'app.action.retry_count': this.retryCount,
          'app.action.dom_fallback_enabled': this.domFallbackOnFailure
        },
        async () => this.executeActionWithRetry(action, params)
      );

      const afterState = await this.captureActionState(includeSnapshot);
      const verification = await this.verifyAction(action, params, beforeState, afterState, execution.result);
      const response = this.sanitizeActionResponse(this.formatActionResponse({
        action,
        args: this.sanitizeActionParams(action, params),
        attempts: execution.attempts,
        recovered: execution.recovered,
        verification,
        result: execution.result,
        error: null,
        durationMs: Date.now() - startedAt
      }));

      const safeResult = this.logActionPayloads ? response : undefined;
      this.log(
        'INFO',
        'Action completed',
        safeResult !== undefined
          ? { action, durationMs: response.durationMs, result: safeResult }
          : { action, durationMs: response.durationMs, status: response.status }
      );
      return response;
    } catch (error) {
      const afterState = await this.captureActionState(includeSnapshot);
      const verification = await this.verifyAction(action, params, beforeState, afterState, null, error);
      const response = this.sanitizeActionResponse(this.formatActionResponse({
        action,
        args: this.sanitizeActionParams(action, params),
        attempts: Number.isFinite(error.attemptCount) ? error.attemptCount : 1,
        recovered: false,
        verification,
        result: null,
        error: error.message,
        durationMs: Date.now() - startedAt
      }));
      this.log('ERROR', 'Action failed', { action, durationMs: response.durationMs, error: response.error });
      return response;
    }
  }

  async act(action, params = {}) {
    const response = await this.runAction(action, params);
    if (response.status !== 'ok') {
      const error = new Error(response.error || response.verification.reason || `Action failed: ${action}`);
      error.actionResponse = response;
      throw error;
    }
    return response.result;
  }

  sanitizeActionParams(action, params) {
    const safe = sanitizeForLog(params);
    if ((action === 'type' || action === 'edit') && safe && Object.prototype.hasOwnProperty.call(safe, 'value')) {
      safe.value = '[REDACTED_INPUT]';
    }
    return safe;
  }

  sanitizeActionResponse(response) {
    const safe = sanitizeForLog(response);
    if (response && Object.prototype.hasOwnProperty.call(response, 'result')) {
      safe.result = sanitizeForLog(response.result);
    }
    if (response && Object.prototype.hasOwnProperty.call(response, 'verification')) {
      safe.verification = sanitizeForLog(response.verification);
    }
    const action = safe && typeof safe.action === 'string' ? safe.action : '';
    if (action === 'type' || action === 'edit') {
      const redactValueFields = (target) => {
        if (!target || typeof target !== 'object') return;
        for (const field of ['beforeValue', 'afterValue', 'requestedValue']) {
          if (Object.prototype.hasOwnProperty.call(target, field)) {
            target[field] = '[REDACTED_INPUT]';
          }
        }
      };
      redactValueFields(safe.result);
      redactValueFields(safe.verification && safe.verification.details);
      if (safe.verification && safe.verification.ok === false) {
        safe.verification.reason = action === 'edit'
          ? 'Field value verification failed after fill.'
          : 'Typed value verification failed.';
      }
    }
    safe.summary = this.buildGroundedActionSummary({
      action: safe && typeof safe.action === 'string' ? safe.action : action,
      verification: safe && typeof safe.verification === 'object' ? safe.verification : null,
      error: safe && typeof safe.error === 'string' ? safe.error : '',
      result: safe && typeof safe.result === 'object' ? safe.result : null,
      status: safe && typeof safe.status === 'string' ? safe.status : ''
    });
    return safe;
  }

  buildGroundedActionSummary({ action, verification, error, result, status }) {
    const actionLabel = typeof action === 'string' && action ? action : 'action';
    const verificationReason = verification && typeof verification.reason === 'string'
      ? verification.reason.trim()
      : '';
    const failureError = typeof error === 'string' ? error.trim() : '';
    const resultMessage = result && typeof result.message === 'string' ? result.message.trim() : '';

    if (status === 'ok') {
      if (verificationReason) {
        return `${capitalize(actionLabel)} succeeded: ${verificationReason}`;
      }
      if (resultMessage) {
        return `${capitalize(actionLabel)} succeeded.`;
      }
      return `${capitalize(actionLabel)} succeeded.`;
    }

    if (failureError) {
      return `${capitalize(actionLabel)} failed: ${failureError}`;
    }
    if (verificationReason) {
      return `${capitalize(actionLabel)} failed: ${verificationReason}`;
    }
    return `${capitalize(actionLabel)} failed.`;
  }

  async executeActionWithRetry(action, params) {
    let lastError;
    for (let attempt = 1; attempt <= this.retryCount + 1; attempt += 1) {
      try {
        if (attempt > 1) {
          this.log('WARN', 'Retrying action', { action, attempt });
          await this.page.waitForTimeout(600);
        }
        const result = await this.executeAction(action, params);
        return { result, attempts: attempt, recovered: attempt > 1 };
      } catch (error) {
        lastError = error;
      }
    }
    if (lastError) {
      lastError.attemptCount = this.retryCount + 1;
    }
    throw lastError;
  }

  async executeAction(action, params = {}) {
    switch (action) {
      case 'goto':
        if (!params.url) throw new Error('Missing url');
        return this.handleGoto(params);
      case 'search':
        return this.handleSearch(params);
      case 'click':
      case 'clickXPath':
      case 'clickByText':
        return this.handleClick(action, params);
      case 'edit':
        return this.handleFill(params);
      case 'type':
        return this.handleType(params);
      case 'delete':
        if (!this.allowDomDelete) {
          throw new Error('delete action is disabled by security policy (EDGE_ALLOW_DOM_DELETE=false)');
        }
        return this.handleDelete(params);
      case 'add':
        if (!this.allowDomHtmlAdd) {
          throw new Error('add action is disabled by security policy (EDGE_ALLOW_DOM_HTML_ADD=false)');
        }
        return this.handleAdd(params);
      case 'exists':
        return this.handleExists(params);
      case 'getText':
        return this.handleGetText(params);
      case 'waitFor':
        return this.handleWaitFor(params);
      case 'wait':
        return this.handleWait(params);
      case 'download':
        return this.handleDownload(params);
      case 'upload':
        return this.handleUpload(params);
      case 'scroll':
        return this.handleScroll(params);
      case 'screenshot':
        return this.handleScreenshot(params);
      case 'startTrace':
        return this.handleStartTrace();
      case 'stopTrace':
        return this.handleStopTrace(params);
      case 'back':
        return this.handleBack();
      case 'forward':
        return this.handleForward();
      case 'refresh':
        return this.handleRefresh();
      case 'press':
        return this.handlePress(params);
      case 'hover':
        return this.handleHover(params);
      case 'select':
        return this.handleSelect(params);
      case 'focus':
        return this.handleFocus(params);
      case 'clear':
        return this.handleClear(params);
      case 'doubleClick':
        return this.handleDoubleClick(params);
      case 'rightClick':
        return this.handleRightClick(params);
      default:
        throw new Error(`Unsupported action: ${action}`);
    }
  }

  async captureActionState(includeSnapshot = false) {
    const state = {
      url: '',
      title: '',
      snapshotHash: null,
      scrollX: 0,
      scrollY: 0
    };
    if (!this.page) {
      return state;
    }

    try {
      state.url = this.page.url() || '';
    } catch (_) {
      state.url = '';
    }

    try {
      state.title = await this.page.title();
    } catch (_) {
      state.title = '';
    }

    if (includeSnapshot) {
      try {
        const content = await this.page.content();
        state.snapshotHash = crypto.createHash('sha1').update(content, 'utf8').digest('hex');
      } catch (_) {
        state.snapshotHash = null;
      }
    }

    try {
      const scrollState = await this.page.evaluate(() => ({ x: window.scrollX || 0, y: window.scrollY || 0 }));
      state.scrollX = Number.isFinite(scrollState && scrollState.x) ? scrollState.x : 0;
      state.scrollY = Number.isFinite(scrollState && scrollState.y) ? scrollState.y : 0;
    } catch (_) {
      state.scrollX = 0;
      state.scrollY = 0;
    }

    return state;
  }

  async verifyAction(action, params, beforeState, afterState, result, error = null) {
    if (error) {
      return {
        ok: false,
        reason: error.message || String(error),
        details: {
          before: beforeState,
          after: afterState
        }
      };
    }

    if (action === 'goto') {
      const actualUrl = afterState.url || '';
      const expectedUrl = String(params.url || '').trim();
      const ok = Boolean(actualUrl && actualUrl !== 'about:blank' && hostsMatch(expectedUrl, actualUrl));
      return {
        ok,
        reason: ok
          ? `Navigation verified on '${actualUrl}'.`
          : `Expected host from '${expectedUrl}', current url is '${actualUrl}'.`,
        details: {
          before: beforeState,
          after: afterState
        }
      };
    }

    if (action === 'search') {
      const changed = beforeState.url !== afterState.url || beforeState.snapshotHash !== afterState.snapshotHash;
      const ok = Boolean(result && result.submitted && (result.urlChanged || changed));
      return {
        ok,
        reason: ok ? 'Search submission completed with visible browser state.' : 'Search submission did not produce a verifiable URL or page-state change.',
        details: {
          before: beforeState,
          after: afterState
        }
      };
    }

    if (action === 'click' || action === 'clickXPath' || action === 'clickByText') {
      const pageChanged = beforeState.url !== afterState.url || beforeState.snapshotHash !== afterState.snapshotHash;
      const targetBefore = result && result.target ? result.target.before : null;
      const targetAfter = result && result.target ? result.target.after : null;
      const targetChanged = Boolean(targetBefore || targetAfter) && safeJson(targetBefore) !== safeJson(targetAfter);
      const ok = pageChanged || targetChanged;
      let reason = 'Click executed but no page or target state change was verified.';
      if (pageChanged) {
        reason = 'Page state changed after click.';
      } else if (targetChanged) {
        reason = 'Target state changed after click.';
      }
      return {
        ok,
        reason,
        details: {
          before: beforeState,
          after: afterState,
          target: result && result.target ? result.target : null
        }
      };
    }

    if (action === 'edit' || action === 'type') {
      const beforeValue = result && typeof result.beforeValue === 'string' ? result.beforeValue : '';
      const afterValue = result && typeof result.afterValue === 'string' ? result.afterValue : null;
      const requestedValue = String(params && Object.prototype.hasOwnProperty.call(params, 'value') ? params.value : '');
      const ok = action === 'edit'
        ? typeof afterValue === 'string' && afterValue === requestedValue
        : typeof afterValue === 'string' && afterValue !== beforeValue && afterValue.includes(requestedValue);
      const reason = action === 'edit'
        ? ok
          ? 'Field value matches requested fill value.'
          : `Field value verification failed. Expected '${requestedValue}', got '${afterValue}'.`
        : ok
          ? 'Typed value was observed in the target field.'
          : `Typed value verification failed. Before='${beforeValue}' After='${afterValue}' Requested fragment='${requestedValue}'.`;
      return {
        ok,
        reason,
        details: {
          before: beforeState,
          after: afterState,
          beforeValue,
          afterValue,
          requestedValue,
          mode: result && result.mode ? result.mode : action
        }
      };
    }

    if (action === 'delete') {
      const beforeCount = Number.isFinite(result && result.beforeCount) ? result.beforeCount : 0;
      const afterCount = Number.isFinite(result && result.afterCount) ? result.afterCount : 0;
      const ok = Boolean(result && result.removed && afterCount < beforeCount);
      return {
        ok,
        reason: ok
          ? `Delete reduced match count from ${beforeCount} to ${afterCount}.`
          : `Delete verification failed. Match count before=${beforeCount}, after=${afterCount}.`,
        details: {
          before: beforeState,
          after: afterState,
          beforeCount,
          afterCount,
          mode: result && result.mode ? result.mode : null
        }
      };
    }

    if (action === 'add') {
      const beforeChildCount = Number.isFinite(result && result.beforeChildCount) ? result.beforeChildCount : 0;
      const afterChildCount = Number.isFinite(result && result.afterChildCount) ? result.afterChildCount : 0;
      const ok = Boolean(result && result.ok && afterChildCount > beforeChildCount);
      return {
        ok,
        reason: ok
          ? `Add increased parent child count from ${beforeChildCount} to ${afterChildCount}.`
          : `Add verification failed. Parent child count before=${beforeChildCount}, after=${afterChildCount}.`,
        details: {
          before: beforeState,
          after: afterState,
          beforeChildCount,
          afterChildCount,
          insertedTag: result && result.insertedTag ? result.insertedTag : null,
          parentSelector: result && result.parentSelector ? result.parentSelector : null
        }
      };
    }

    if (action === 'scroll') {
      const beforeY = Number.isFinite(result && result.beforeY) ? result.beforeY : beforeState.scrollY;
      const afterY = Number.isFinite(result && result.afterY) ? result.afterY : afterState.scrollY;
      const deltaY = Number.isFinite(result && result.deltaY) ? result.deltaY : afterY - beforeY;
      const ok = deltaY !== 0;
      return {
        ok,
        reason: ok
          ? `Scroll moved viewport from ${beforeY} to ${afterY}.`
          : `Scroll verification failed. Viewport Y remained at ${afterY}.`,
        details: {
          before: beforeState,
          after: afterState,
          beforeY,
          afterY,
          deltaY,
          direction: result && result.direction ? result.direction : null
        }
      };
    }

    if (action === 'exists') {
      return {
        ok: typeof (result && result.exists) === 'boolean',
        reason: `Existence check returned ${Boolean(result && result.exists)}.`,
        details: {
          before: beforeState,
          after: afterState
        }
      };
    }

    if (action === 'getText') {
      return {
        ok: typeof (result && result.text) === 'string',
        reason: 'Text extraction returned a string result.',
        details: {
          before: beforeState,
          after: afterState
        }
      };
    }

    if (action === 'waitFor') {
      return {
        ok: Boolean(result && result.ok),
        reason: 'Wait condition satisfied by locator.',
        details: {
          before: beforeState,
          after: afterState
        }
      };
    }

    if (action === 'wait') {
      return {
        ok: Number.isFinite(result && result.waitedMs),
        reason: `Waited ${result && result.waitedMs ? result.waitedMs : 0}ms.`,
        details: {
          before: beforeState,
          after: afterState
        }
      };
    }

    if (FILE_OUTPUT_ACTIONS.has(action)) {
      const outputPath = result && result.path ? String(result.path) : '';
      let size = null;
      let ok = false;
      if (outputPath) {
        try {
          const stats = await fs.promises.stat(outputPath);
          size = stats.size;
          ok = stats.isFile() && stats.size > 0;
        } catch (_) {
          ok = false;
        }
      }
      return {
        ok,
        reason: ok ? `Output file verified at '${outputPath}'.` : `Expected output file was not verified at '${outputPath}'.`,
        details: {
          before: beforeState,
          after: afterState,
          path: outputPath,
          size
        }
      };
    }

    if (action === 'upload') {
      const filePath = result && result.filePath ? String(result.filePath) : '';
      const selectedFiles = Array.isArray(result && result.selectedFiles) ? result.selectedFiles : [];
      let exists = false;
      if (filePath) {
        try {
          const stats = await fs.promises.stat(filePath);
          exists = stats.isFile();
        } catch (_) {
          exists = false;
        }
      }
      const expectedName = filePath ? path.basename(filePath) : '';
      const selectionVerified = Boolean(expectedName && selectedFiles.includes(expectedName));
      return {
        ok: exists && selectionVerified,
        reason: exists && selectionVerified
          ? `Upload file '${expectedName}' is present in the file input.`
          : `Upload verification failed. SourceExists=${exists} SelectedFiles=${JSON.stringify(selectedFiles)}.`,
        details: {
          before: beforeState,
          after: afterState,
          filePath,
          selectedFiles,
          expectedName
        }
      };
    }

    if (action === 'startTrace') {
      const ok = Boolean(result && (result.status === 'started' || result.status === 'already-running'));
      return {
        ok,
        reason: ok ? `Trace state is '${result.status}'.` : 'Trace start could not be verified.',
        details: {
          before: beforeState,
          after: afterState
        }
      };
    }

    if (action === 'back' || action === 'forward') {
      const urlChanged = beforeState.url !== afterState.url;
      const ok = urlChanged || Boolean(afterState.url && afterState.url !== 'about:blank');
      return {
        ok,
        reason: ok
          ? `${capitalize(action)} navigation completed. URL: '${afterState.url}'.`
          : `${capitalize(action)} had no observable effect. URL stayed at '${afterState.url}'.`,
        details: { before: beforeState, after: afterState }
      };
    }

    if (action === 'refresh') {
      const ok = Boolean(afterState.url && afterState.url !== 'about:blank');
      return {
        ok,
        reason: ok
          ? `Page refreshed. URL: '${afterState.url}'.`
          : 'Refresh could not be verified. Page may be on about:blank.',
        details: { before: beforeState, after: afterState }
      };
    }

    if (action === 'press') {
      const key = result && result.key ? result.key : '';
      const ok = Boolean(result && result.pressed);
      return {
        ok,
        reason: ok ? `Key '${key}' pressed.` : `Key press could not be verified.`,
        details: { before: beforeState, after: afterState, key }
      };
    }

    if (action === 'hover') {
      const ok = Boolean(result && result.hovered);
      return {
        ok,
        reason: ok ? `Hover completed via ${result.strategy || 'locator'}.` : 'Hover target could not be verified.',
        details: { before: beforeState, after: afterState }
      };
    }

    if (action === 'select') {
      const selectedLabel = result && result.selectedLabel ? result.selectedLabel : '';
      const afterValue = result && typeof result.afterValue === 'string' ? result.afterValue : null;
      const ok = typeof afterValue === 'string' && afterValue !== '';
      return {
        ok,
        reason: ok
          ? `Selected '${selectedLabel}' (value='${afterValue}').`
          : `Select verification failed. Label='${selectedLabel}', afterValue='${afterValue}'.`,
        details: { before: beforeState, after: afterState, selectedLabel, afterValue }
      };
    }

    if (action === 'focus') {
      const ok = Boolean(result && result.focused);
      return {
        ok,
        reason: ok ? `Focus applied via ${result.strategy || 'locator'}.` : 'Focus target could not be verified.',
        details: { before: beforeState, after: afterState }
      };
    }

    if (action === 'clear') {
      const ok = Boolean(result && result.cleared);
      return {
        ok,
        reason: ok
          ? `Input cleared (was '${result.beforeValue || ''}').`
          : `Clear verification failed. afterValue='${result && result.afterValue}'.`,
        details: { before: beforeState, after: afterState }
      };
    }

    if (action === 'doubleClick') {
      const pageChanged = beforeState.url !== afterState.url || beforeState.snapshotHash !== afterState.snapshotHash;
      const ok = Boolean(result && result.doubleClicked) || pageChanged;
      return {
        ok,
        reason: ok ? `Double-click executed via ${result && result.strategy || 'locator'}.` : 'Double-click could not be verified.',
        details: { before: beforeState, after: afterState }
      };
    }

    if (action === 'rightClick') {
      const pageChanged = beforeState.url !== afterState.url || beforeState.snapshotHash !== afterState.snapshotHash;
      const ok = Boolean(result && result.rightClicked) || pageChanged;
      return {
        ok,
        reason: ok ? `Right-click executed via ${result && result.strategy || 'locator'}.` : 'Right-click could not be verified.',
        details: { before: beforeState, after: afterState }
      };
    }

    const changed = beforeState.url !== afterState.url || beforeState.snapshotHash !== afterState.snapshotHash;
    const ok = changed || Boolean(afterState.url && afterState.url !== 'about:blank');
    return {
      ok,
      reason: changed ? 'Page state changed after action.' : 'Page remained stable but browser context is reachable.',
      details: {
        before: beforeState,
        after: afterState
      }
    };
  }

  formatActionResponse({ action, args, attempts, recovered, verification, result, error, durationMs }) {
    const status = !error && verification && verification.ok ? 'ok' : 'failed';
    return {
      status,
      action,
      tool: action,
      args,
      attempts,
      recovered,
      verification,
      summary: this.buildGroundedActionSummary({ action, verification, error, result, status }),
      result,
      error,
      durationMs
    };
  }

  bindPage(page) {
    if (this.boundPages.has(page)) return;
    this.boundPages.add(page);

    page.on('pageerror', (error) => this.log('ERROR', 'Page error', { message: error.message }));
    page.on('requestfailed', (request) => {
      this.log('WARN', 'Request failed', {
        method: request.method(),
        url: request.url(),
        error: request.failure() ? request.failure().errorText : 'unknown'
      });
    });
    page.on('console', (message) => {
      if (message.type() === 'error') {
        this.log('WARN', 'Browser console error', { text: message.text() });
      }
    });
  }

  async handleGoto(params) {
    const url = /^https?:\/\//i.test(params.url) ? params.url : `https://${params.url}`;
    await this.page.goto(url, { waitUntil: 'domcontentloaded', timeout: this.navigationTimeout });
    try {
      await this.page.waitForLoadState('networkidle', { timeout: 8000 });
    } catch (_) {
      // Many websites keep long-polling requests; domcontentloaded is enough as baseline.
    }
    return { url: this.page.url(), title: await this.page.title() };
  }

  async handleSearch(params) {
    if (!params.query) throw new Error('search requires query');
    const query = String(params.query);
    const beforeUrl = this.page.url();
    const roots = this.getSearchRoots();
    const searchCandidates = roots.flatMap((root) => [
      root.getByRole('searchbox').first(),
      root.locator('input[type="search"]').first(),
      root.getByPlaceholder(/search/i).first(),
      root.getByLabel(/search/i).first(),
      root.getByRole('textbox', { name: /search/i }).first(),
      root
        .locator(
          'input[aria-label*="search" i], textarea[aria-label*="search" i], [contenteditable="true"][aria-label*="search" i]'
        )
        .first()
    ]);
    const searchInput = await this.firstVisibleLocator(searchCandidates);
    if (searchInput) {
      await this.writeValueToLocator(searchInput, query, 'fill');
      await searchInput.press('Enter');
      const urlChanged = await this.waitForUrlChange(beforeUrl, 5000);
      return { query, submitted: true, urlChanged };
    }

    if (this.domFallbackOnFailure) {
      const domResult = await this.writeWithDomFallback({ text: 'search' }, query, 'fill');
      if (domResult.ok) {
        await this.page.keyboard.press('Enter').catch(() => {});
        let urlChanged = await this.waitForUrlChange(beforeUrl, 3500);
        if (!urlChanged) {
          await this.page.evaluate(() => {
            const active = document.activeElement;
            if (!active) return false;
            const isInput = ['INPUT', 'TEXTAREA'].includes(active.tagName) || active.isContentEditable;
            if (!isInput) return false;
            active.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
            active.dispatchEvent(new KeyboardEvent('keyup', { key: 'Enter', code: 'Enter', bubbles: true }));
            if (active.form) {
              if (typeof active.form.requestSubmit === 'function') {
                active.form.requestSubmit();
              } else {
                active.form.submit();
              }
            }
            return true;
          }).catch(() => {});
          urlChanged = await this.waitForUrlChange(beforeUrl, 3500);
        }
        return { query, submitted: true, fallback: 'dom', strategy: domResult.strategy, urlChanged };
      }
    }

    throw new Error('No visible search field found on this page.');
  }

  async handleClick(action, params) {
    const compatibleParams = { ...params };
    if (action === 'clickXPath' && !compatibleParams.xpath && compatibleParams.selector) {
      compatibleParams.xpath = compatibleParams.selector;
    }
    if (action === 'clickByText' && !compatibleParams.text && compatibleParams.selector) {
      compatibleParams.text = compatibleParams.selector;
    }
    try {
      const resolved = await this.resolveStateChangingLocator(compatibleParams);
      const { locator, strategy } = resolved;
      const beforeTarget = await this.describeLocator(locator).catch(() => null);
      await locator.scrollIntoViewIfNeeded();
      await locator.click({ timeout: this.actionTimeout });
      const afterTarget = await this.describeLocator(locator).catch(() => null);
      return { ok: true, strategy, target: { before: beforeTarget, after: afterTarget, strategy } };
    } catch (error) {
      if (!this.domFallbackOnFailure) throw error;
      const domResult = await this.clickWithDomFallback(compatibleParams);
      if (domResult.ok) {
        return {
          ok: true,
          fallback: 'dom',
          strategy: domResult.strategy,
          target: {
            before: domResult.beforeTarget || null,
            after: domResult.afterTarget || null
          }
        };
      }
      throw error;
    }
  }

  async handleFill(params) {
    if (params.value === undefined) throw new Error('Missing value');
    try {
      const resolved = await this.resolveStateChangingInputLocator(params);
      const { locator, strategy } = resolved;
      const writeResult = await this.writeValueToLocator(locator, String(params.value), 'fill');
      return { ok: true, ...writeResult, mode: 'fill', selectorStrategy: strategy };
    } catch (error) {
      if (!this.domFallbackOnFailure) throw error;
      const domResult = await this.writeWithDomFallback(params, String(params.value), 'fill');
      if (domResult.ok) {
        return {
          ok: true,
          fallback: 'dom',
          strategy: domResult.strategy,
          beforeValue: typeof domResult.beforeValue === 'string' ? domResult.beforeValue : '',
          afterValue: typeof domResult.afterValue === 'string' ? domResult.afterValue : '',
          mode: 'fill'
        };
      }
      throw error;
    }
  }

  async handleType(params) {
    if (params.value === undefined) throw new Error('Missing value');
    try {
      const resolved = await this.resolveStateChangingInputLocator(params);
      const { locator, strategy } = resolved;
      const writeResult = await this.writeValueToLocator(locator, String(params.value), 'type');
      return { ok: true, ...writeResult, mode: 'type', selectorStrategy: strategy };
    } catch (error) {
      if (!this.domFallbackOnFailure) throw error;
      const domResult = await this.writeWithDomFallback(params, String(params.value), 'type');
      if (domResult.ok) {
        return {
          ok: true,
          fallback: 'dom',
          strategy: domResult.strategy,
          beforeValue: typeof domResult.beforeValue === 'string' ? domResult.beforeValue : '',
          afterValue: typeof domResult.afterValue === 'string' ? domResult.afterValue : '',
          mode: 'type'
        };
      }
      throw error;
    }
  }

  async handleDelete(params) {
    const beforeCount = await this.countMatches(params);
    if (params.selector) {
      await this.page.evaluate((selector) => {
        const element = document.querySelector(selector);
        if (element) element.remove();
      }, params.selector);
      const afterCount = await this.countMatches(params);
      return { removed: afterCount < beforeCount, mode: 'css', beforeCount, afterCount };
    }
    if (params.xpath) {
      await this.page.evaluate((xpath) => {
        const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
        if (result.singleNodeValue) result.singleNodeValue.remove();
      }, params.xpath);
      const afterCount = await this.countMatches(params);
      return { removed: afterCount < beforeCount, mode: 'xpath', beforeCount, afterCount };
    }
    throw new Error('Delete requires CSS selector or XPath');
  }

  async handleAdd(params) {
    if (!params.parentSelector || !params.html) throw new Error('Add requires parentSelector and html');
    const result = await this.page.evaluate(({ parentSelector, html }) => {
      const parent = document.querySelector(parentSelector);
      if (!parent) throw new Error(`Parent selector not found: ${parentSelector}`);
      const beforeChildCount = parent.childElementCount;
      const wrapper = document.createElement('div');
      wrapper.innerHTML = html;
      if (!wrapper.firstElementChild) throw new Error('Invalid html payload');
      const inserted = wrapper.firstElementChild;
      const insertedTag = inserted.tagName ? inserted.tagName.toLowerCase() : '';
      parent.appendChild(inserted);
      return {
        parentSelector,
        beforeChildCount,
        afterChildCount: parent.childElementCount,
        insertedTag
      };
    }, params);
    return {
      ok: result.afterChildCount > result.beforeChildCount,
      parentSelector: result.parentSelector,
      beforeChildCount: result.beforeChildCount,
      afterChildCount: result.afterChildCount,
      insertedTag: result.insertedTag
    };
  }

  async handleExists(params) {
    try {
      const count = await this.countMatches(params);
      return { exists: count > 0 };
    } catch (_) {
      if (!this.domFallbackOnFailure) {
        return { exists: false };
      }
      const domResult = await this.existsWithDomFallback(params);
      return { exists: Boolean(domResult.exists), fallback: 'dom' };
    }
  }

  async handleGetText(params) {
    try {
      const locator = await this.getLocator(params);
      const text = await locator.textContent();
      return { text: text ? text.trim() : '' };
    } catch (error) {
      if (!this.domFallbackOnFailure) throw error;
      const domResult = await this.getTextWithDomFallback(params);
      if (domResult.ok) {
        return { text: domResult.text, fallback: 'dom' };
      }
      throw error;
    }
  }

  async handleWaitFor(params) {
    const locator = await this.getLocator(params);
    await locator.waitFor({
      state: 'visible',
      timeout: params.timeout ? Number(params.timeout) : 60000
    });
    return { ok: true };
  }

  async handleWait(params) {
    const ms = Number(params.ms);
    if (!Number.isFinite(ms) || ms < 0) throw new Error('wait requires a positive ms value');
    await this.page.waitForTimeout(ms);
    return { waitedMs: ms };
  }

  async handleDownload(params) {
    const resolved = await this.resolveStateChangingLocator(params);
    const { locator, strategy } = resolved;
    const [download] = await Promise.all([
      this.page.waitForEvent('download'),
      locator.click()
    ]);

    const defaultName = download.suggestedFilename();
    const savePath = this.resolveOutputPath(params.savePath, path.join('downloads', defaultName));
    await fs.promises.mkdir(path.dirname(savePath), { recursive: true });
    await download.saveAs(savePath);
    const stats = await fs.promises.stat(savePath);
    return { path: savePath, size: stats.size, selectorStrategy: strategy };
  }

  async handleUpload(params) {
    if (!params.filePath) throw new Error('upload requires filePath');
    const resolved = await this.resolveStateChangingInputLocator(params);
    const { locator, strategy } = resolved;
    const resolvedPath = path.resolve(params.filePath);
    await locator.setInputFiles(resolvedPath);
    const selectedFiles = await locator.evaluate((element) => Array.from(element.files || []).map((file) => file.name));
    return { ok: true, filePath: resolvedPath, selectedFiles, selectedCount: selectedFiles.length, selectorStrategy: strategy };
  }

  async handleScroll(params) {
    const direction = params.direction === 'up' ? 'up' : 'down';
    const amount = Number(params.amount) || 600;
    const beforeY = await this.page.evaluate(() => window.scrollY || 0);
    const afterY = await this.page.evaluate(({ scrollDirection, pixels }) => {
      window.scrollBy(0, scrollDirection === 'down' ? pixels : -pixels);
      return window.scrollY || 0;
    }, { scrollDirection: direction, pixels: amount });
    return { direction, amount, beforeY, afterY, deltaY: afterY - beforeY };
  }

  async handleScreenshot(params) {
    const outputPath = this.resolveOutputPath(params.path, `screenshot-${Date.now()}.png`);
    await fs.promises.mkdir(path.dirname(outputPath), { recursive: true });
    await this.page.screenshot({ path: outputPath, fullPage: true });
    const stats = await fs.promises.stat(outputPath);
    return { path: outputPath, size: stats.size };
  }

  async handleStartTrace() {
    if (!this.context) throw new Error('Browser context is not ready');
    if (this.traceActive) return { status: 'already-running' };
    await this.context.tracing.start({ screenshots: true, snapshots: true, sources: true });
    this.traceActive = true;
    return { status: 'started' };
  }

  async handleStopTrace(params = {}) {
    if (!this.context) throw new Error('Browser context is not ready');
    if (!this.traceActive) return { status: 'not-running' };
    const tracePath = this.resolveOutputPath(params.path, path.join('traces', `trace-${Date.now()}.zip`));
    await fs.promises.mkdir(path.dirname(tracePath), { recursive: true });
    await this.context.tracing.stop({ path: tracePath });
    this.traceActive = false;
    const stats = await fs.promises.stat(tracePath);
    return { status: 'stopped', path: tracePath, size: stats.size };
  }

  async handleBack() {
    await this.page.goBack({ waitUntil: 'domcontentloaded' });
    return { url: this.page.url(), title: await this.page.title() };
  }

  async handleForward() {
    await this.page.goForward({ waitUntil: 'domcontentloaded' });
    return { url: this.page.url(), title: await this.page.title() };
  }

  async handleRefresh() {
    await this.page.reload({ waitUntil: 'domcontentloaded' });
    return { url: this.page.url(), title: await this.page.title() };
  }

  async handlePress(params = {}) {
    const key = String(params.key || '').trim();
    if (!key) throw new Error('Missing key parameter');
    await this.page.keyboard.press(key);
    return { key, pressed: true };
  }

  async handleHover(params = {}) {
    const locatorCandidates = this.buildLocatorCandidateDescriptors(params);
    for (const candidate of locatorCandidates) {
      const count = await candidate.locator.count();
      if (count === 1) {
        await candidate.locator.first().hover();
        return { hovered: true, strategy: candidate.strategy };
      }
    }
    throw new Error(`Hover target not found for ${JSON.stringify(params)}`);
  }

  async handleSelect(params = {}) {
    const selectLocator = await this.getLocator(params);
    const count = await selectLocator.count();
    if (count !== 1) {
      throw new Error(`Select target matched ${count} elements (expected 1)`);
    }
    const value = String(params.value || params.option || '').trim();
    if (!value) throw new Error('Missing value or option parameter for select');
    const beforeValue = await selectLocator.first().inputValue().catch(() => '');
    await selectLocator.first().selectOption({ label: value });
    const afterValue = await selectLocator.first().inputValue().catch(() => '');
    return { beforeValue, afterValue, selectedLabel: value };
  }

  async handleFocus(params = {}) {
    const locatorCandidates = this.buildLocatorCandidateDescriptors(params);
    for (const candidate of locatorCandidates) {
      const count = await candidate.locator.count();
      if (count === 1) {
        await candidate.locator.first().focus();
        return { focused: true, strategy: candidate.strategy };
      }
    }
    throw new Error(`Focus target not found for ${JSON.stringify(params)}`);
  }

  async handleClear(params = {}) {
    const locator = await this.getLocator(params);
    const count = await locator.count();
    if (count !== 1) {
      throw new Error(`Clear target matched ${count} elements (expected 1)`);
    }
    const beforeValue = await locator.first().inputValue().catch(() => '');
    await locator.first().fill('');
    const afterValue = await locator.first().inputValue().catch(() => '');
    return { beforeValue, afterValue, cleared: afterValue === '' };
  }

  async handleDoubleClick(params = {}) {
    const locatorCandidates = this.buildLocatorCandidateDescriptors(params);
    for (const candidate of locatorCandidates) {
      const count = await candidate.locator.count();
      if (count === 1) {
        await candidate.locator.first().dblclick();
        return { doubleClicked: true, strategy: candidate.strategy };
      }
    }
    throw new Error(`Double-click target not found for ${JSON.stringify(params)}`);
  }

  async handleRightClick(params = {}) {
    const locatorCandidates = this.buildLocatorCandidateDescriptors(params);
    for (const candidate of locatorCandidates) {
      const count = await candidate.locator.count();
      if (count === 1) {
        await candidate.locator.first().click({ button: 'right' });
        return { rightClicked: true, strategy: candidate.strategy };
      }
    }
    throw new Error(`Right-click target not found for ${JSON.stringify(params)}`);
  }

  async waitForUrlChange(previousUrl, timeoutMs = 5000) {
    try {
      await this.page.waitForURL((url) => String(url) !== previousUrl, { timeout: timeoutMs });
      return true;
    } catch (_) {
      return this.page.url() !== previousUrl;
    }
  }

  resolveOutputPath(requestedPath, defaultRelativePath) {
    const candidate = path.resolve(requestedPath || defaultRelativePath);
    if (!this.restrictWriteToWorkspace) {
      return candidate;
    }

    const relative = path.relative(this.workspaceRoot, candidate);
    const isInsideWorkspace = Boolean(relative) && !relative.startsWith('..') && !path.isAbsolute(relative);
    const isWorkspaceRoot = relative === '';
    if (isInsideWorkspace || isWorkspaceRoot) {
      return candidate;
    }

    throw new Error(`Output path is blocked by security policy. Use a path under workspace: ${this.workspaceRoot}`);
  }

  getSearchRoots() {
    if (!this.frameAwareLocators || !this.page) {
      return [this.page];
    }
    const roots = [this.page];
    for (const frame of this.page.frames()) {
      if (frame === this.page.mainFrame()) continue;
      roots.push(frame);
    }
    return roots;
  }

  async clickWithDomFallback(params) {
    return this.page.evaluate((input) => {
      const isVisible = (el) => {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (!style || style.visibility === 'hidden' || style.display === 'none') return false;
        return Boolean(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
      };

      const normalize = (value) => String(value || '').trim().toLowerCase();
      const describeElement = (el) => {
        if (!el) return null;
        return {
          tag: (el.tagName || '').toLowerCase(),
          text: String(el.innerText || el.textContent || '').trim().slice(0, 120),
          value: typeof el.value === 'string' ? el.value : '',
          checked: typeof el.checked === 'boolean' ? el.checked : null,
          ariaExpanded: el.getAttribute('aria-expanded'),
          focused: document.activeElement === el
        };
      };
      const clickElement = (el) => {
        if (!el) return false;
        el.scrollIntoView({ block: 'center', inline: 'center' });
        el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
        el.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
        el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        return true;
      };

      let element = null;
      let strategy = '';

      if (input.selector) {
        try {
          element = document.querySelector(input.selector);
          strategy = 'selector';
        } catch (_) {
          element = null;
        }
      } else if (input.xpath) {
        try {
          const result = document.evaluate(input.xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
          element = result.singleNodeValue;
          strategy = 'xpath';
        } catch (_) {
          element = null;
        }
      } else if (input.text) {
        const wanted = normalize(input.text);
        const candidates = Array.from(
          document.querySelectorAll('button,a,[role="button"],[role="link"],[role="option"],[role="menuitem"],label,span,div')
        );
        element = candidates.find((candidate) => isVisible(candidate) && normalize(candidate.innerText).includes(wanted)) || null;
        strategy = 'text-scan';
      }

      if (!element || !isVisible(element)) {
        return { ok: false };
      }
      const beforeTarget = describeElement(element);
      const ok = clickElement(element);
      const afterTarget = describeElement(element);
      return { ok, strategy, beforeTarget, afterTarget };
    }, params);
  }

  async writeWithDomFallback(params, value, mode) {
    return this.page.evaluate(({ target, inputValue, inputMode }) => {
      const normalize = (v) => String(v || '').trim().toLowerCase();
      const isVisible = (el) => {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (!style || style.visibility === 'hidden' || style.display === 'none') return false;
        return Boolean(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
      };

      const readValue = (el) => {
        if (!el) return '';
        const tag = (el.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') {
          return String(el.value || '');
        }
        if (el.isContentEditable) {
          return String(el.textContent || '');
        }
        return typeof el.textContent === 'string' ? String(el.textContent) : '';
      };

      const writeToElement = (el) => {
        if (!el) return false;
        const tag = (el.tagName || '').toLowerCase();
        const isEditable = Boolean(el.isContentEditable);

        el.scrollIntoView({ block: 'center', inline: 'center' });
        el.focus();

        if (tag === 'input' || tag === 'textarea' || tag === 'select') {
          const current = String(el.value || '');
          el.value = inputMode === 'type' ? `${current}${inputValue}` : inputValue;
          el.dispatchEvent(new Event('input', { bubbles: true }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
          return true;
        }

        if (isEditable) {
          const current = String(el.textContent || '');
          el.textContent = inputMode === 'type' ? `${current}${inputValue}` : inputValue;
          el.dispatchEvent(new Event('input', { bubbles: true }));
          return true;
        }

        return false;
      };

      const byLabelOrHint = (hint) => {
        const wanted = normalize(hint);
        if (!wanted) return null;

        const directCandidates = Array.from(document.querySelectorAll('input,textarea,[contenteditable="true"],[role="textbox"]'));
        const direct = directCandidates.find((el) => {
          const label = normalize(el.getAttribute('aria-label'));
          const placeholder = normalize(el.getAttribute('placeholder'));
          const name = normalize(el.getAttribute('name'));
          return isVisible(el) && (label.includes(wanted) || placeholder.includes(wanted) || name.includes(wanted));
        });
        if (direct) return direct;

        const labels = Array.from(document.querySelectorAll('label'));
        for (const label of labels) {
          if (!isVisible(label)) continue;
          if (!normalize(label.innerText).includes(wanted)) continue;
          const controlId = label.getAttribute('for');
          if (controlId) {
            const bound = document.getElementById(controlId);
            if (bound && isVisible(bound)) return bound;
          }
          const nested = label.querySelector('input,textarea,[contenteditable="true"],[role="textbox"]');
          if (nested && isVisible(nested)) return nested;
        }

        return null;
      };

      let element = null;
      let strategy = '';

      if (target.selector) {
        try {
          element = document.querySelector(target.selector);
          strategy = 'selector';
        } catch (_) {
          element = null;
        }
      } else if (target.xpath) {
        try {
          const result = document.evaluate(target.xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
          element = result.singleNodeValue;
          strategy = 'xpath';
        } catch (_) {
          element = null;
        }
      } else if (target.text) {
        element = byLabelOrHint(target.text);
        strategy = 'label-hint';
      }

      if (!element || !isVisible(element)) {
        return { ok: false };
      }
      const beforeValue = readValue(element);
      const ok = writeToElement(element);
      const afterValue = readValue(element);
      return { ok, strategy, beforeValue, afterValue };
    }, { target: params, inputValue: value, inputMode: mode });
  }

  async existsWithDomFallback(params) {
    return this.page.evaluate((input) => {
      const normalize = (value) => String(value || '').trim().toLowerCase();
      const bySelector = () => {
        try {
          return Boolean(input.selector && document.querySelector(input.selector));
        } catch (_) {
          return false;
        }
      };
      const byXPath = () => {
        if (!input.xpath) return false;
        try {
          const result = document.evaluate(input.xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
          return Boolean(result.singleNodeValue);
        } catch (_) {
          return false;
        }
      };
      const byText = () => {
        if (!input.text) return false;
        const wanted = normalize(input.text);
        const all = Array.from(document.querySelectorAll('body *'));
        return all.some((el) => normalize(el.innerText).includes(wanted));
      };
      return { exists: bySelector() || byXPath() || byText() };
    }, params);
  }

  async getTextWithDomFallback(params) {
    return this.page.evaluate((input) => {
      const normalize = (value) => String(value || '').trim().toLowerCase();
      let element = null;

      if (input.selector) {
        try {
          element = document.querySelector(input.selector);
        } catch (_) {
          element = null;
        }
      } else if (input.xpath) {
        try {
          const result = document.evaluate(input.xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
          element = result.singleNodeValue;
        } catch (_) {
          element = null;
        }
      } else if (input.text) {
        const wanted = normalize(input.text);
        const all = Array.from(document.querySelectorAll('body *'));
        element = all.find((el) => normalize(el.innerText).includes(wanted)) || null;
      }

      if (!element) {
        return { ok: false, text: '' };
      }
      const text = (element.textContent || element.value || '').trim();
      return { ok: true, text };
    }, params);
  }

  async countMatches(params) {
    const roots = this.getSearchRoots();
    if (params.selector) {
      let total = 0;
      for (const root of roots) {
        try {
          total += await root.locator(params.selector).count();
        } catch (_) {
          // Ignore detached/inaccessible frames.
        }
      }
      return total;
    }
    if (params.xpath) {
      let total = 0;
      for (const root of roots) {
        try {
          total += await root.locator(`xpath=${params.xpath}`).count();
        } catch (_) {
          // Ignore detached/inaccessible frames.
        }
      }
      return total;
    }
    if (params.text) {
      const candidates = roots.flatMap((root) => this.buildTextCandidateLocators(params.text, root));
      let total = 0;
      for (const candidate of candidates) {
        try {
          total += await candidate.count();
        } catch (_) {
          // Ignore detached/inaccessible frames.
        }
      }
      return total;
    }
    return 0;
  }

  buildTextCandidateLocators(text, root = this.page) {
    const normalized = String(text).trim();
    const exactRegex = new RegExp(`^\\s*${escapeRegex(normalized)}\\s*$`, 'i');
    const hasWhitespace = /\s/.test(normalized);
    const candidates = [
      root.getByRole('button', { name: exactRegex }),
      root.getByRole('link', { name: exactRegex }),
      root.getByRole('tab', { name: exactRegex }),
      root.getByRole('menuitem', { name: exactRegex }),
      root.getByRole('option', { name: exactRegex }),
      root.getByLabel(normalized, { exact: false }),
      root.getByPlaceholder(normalized, { exact: false }),
      root.getByText(normalized, { exact: false })
    ];
    if (!hasWhitespace) {
      candidates.push(root.locator(`[aria-label="${escapeAttrValue(normalized)}" i]`));
      candidates.push(root.locator(`[title="${escapeAttrValue(normalized)}" i]`));
    }
    return candidates.map((candidate) => candidate.first());
  }

  buildTextCandidateDescriptors(text, root = this.page) {
    const normalized = String(text).trim();
    const exactRegex = new RegExp(`^\\s*${escapeRegex(normalized)}\\s*$`, 'i');
    const hasWhitespace = /\s/.test(normalized);
    const descriptors = [
      { strategy: 'role-button-exact', locator: root.getByRole('button', { name: exactRegex }) },
      { strategy: 'role-link-exact', locator: root.getByRole('link', { name: exactRegex }) },
      { strategy: 'role-tab-exact', locator: root.getByRole('tab', { name: exactRegex }) },
      { strategy: 'role-menuitem-exact', locator: root.getByRole('menuitem', { name: exactRegex }) },
      { strategy: 'role-option-exact', locator: root.getByRole('option', { name: exactRegex }) },
      { strategy: 'label-match', locator: root.getByLabel(normalized, { exact: false }) },
      { strategy: 'placeholder-match', locator: root.getByPlaceholder(normalized, { exact: false }) },
      { strategy: 'text-match', locator: root.getByText(normalized, { exact: false }) }
    ];
    if (!hasWhitespace) {
      descriptors.push({ strategy: 'aria-label-attr', locator: root.locator(`[aria-label="${escapeAttrValue(normalized)}" i]`) });
      descriptors.push({ strategy: 'title-attr', locator: root.locator(`[title="${escapeAttrValue(normalized)}" i]`) });
    }
    return descriptors;
  }

  buildInputCandidateLocators(text, root = this.page) {
    const target = String(text).trim();
    const targetRegex = new RegExp(escapeRegex(target), 'i');
    return [
      root.getByLabel(target, { exact: false }).first(),
      root.getByPlaceholder(target, { exact: false }).first(),
      root.getByRole('textbox', { name: targetRegex }).first(),
      root
        .locator(
          `input[aria-label*="${escapeAttrValue(target)}" i], textarea[aria-label*="${escapeAttrValue(target)}" i], [contenteditable="true"][aria-label*="${escapeAttrValue(target)}" i]`
        )
        .first()
    ];
  }

  buildInputCandidateDescriptors(text, root = this.page) {
    const target = String(text).trim();
    const targetRegex = new RegExp(escapeRegex(target), 'i');
    return [
      { strategy: 'label-input', locator: root.getByLabel(target, { exact: false }) },
      { strategy: 'placeholder-input', locator: root.getByPlaceholder(target, { exact: false }) },
      { strategy: 'role-textbox-name', locator: root.getByRole('textbox', { name: targetRegex }) },
      {
        strategy: 'aria-label-input',
        locator: root.locator(
          `input[aria-label*="${escapeAttrValue(target)}" i], textarea[aria-label*="${escapeAttrValue(target)}" i], [contenteditable="true"][aria-label*="${escapeAttrValue(target)}" i]`
        )
      }
    ];
  }

  buildLocatorCandidates(params) {
    const roots = this.getSearchRoots();
    if (params.selector) {
      return roots.map((root) => root.locator(params.selector).first());
    }
    if (params.xpath) {
      return roots.map((root) => root.locator(`xpath=${params.xpath}`).first());
    }
    if (params.text) {
      return roots.flatMap((root) => this.buildTextCandidateLocators(params.text, root));
    }
    return [];
  }

  buildLocatorCandidateDescriptors(params) {
    const roots = this.getSearchRoots();
    if (params.selector) {
      return roots.map((root) => ({ strategy: 'selector', locator: root.locator(params.selector) }));
    }
    if (params.xpath) {
      return roots.map((root) => ({ strategy: 'xpath', locator: root.locator(`xpath=${params.xpath}`) }));
    }
    if (params.text) {
      return roots.flatMap((root) => this.buildTextCandidateDescriptors(params.text, root));
    }
    return [];
  }

  buildInputCandidateDescriptorsForParams(params) {
    const roots = this.getSearchRoots();
    if (params.selector) {
      return roots.map((root) => ({ strategy: 'selector', locator: root.locator(params.selector) }));
    }
    if (params.xpath) {
      return roots.map((root) => ({ strategy: 'xpath', locator: root.locator(`xpath=${params.xpath}`) }));
    }
    if (params.text) {
      return roots.flatMap((root) => this.buildInputCandidateDescriptors(params.text, root));
    }
    return [];
  }

  describeTarget(params, kind = 'element') {
    if (params.selector) return `${kind} selector "${params.selector}"`;
    if (params.xpath) return `${kind} xpath "${params.xpath}"`;
    if (params.text) return `${kind} text "${params.text}"`;
    return kind;
  }

  async resolveCandidateDescriptor(descriptors, missingMessage, ambiguityLabel) {
    for (const descriptor of descriptors) {
      let count = 0;
      try {
        count = await descriptor.locator.count();
      } catch (_) {
        count = 0;
      }

      if (!count) {
        continue;
      }

      if (count > 1) {
        throw new Error(`Ambiguous ${ambiguityLabel} via ${descriptor.strategy}: matched ${count} elements. Refine the target.`);
      }

      const locator = descriptor.locator.first();
      try {
        await locator.waitFor({ state: 'visible', timeout: 1500 });
      } catch (_) {
        // A single attached match is still valid for state-changing actions; the click/fill call can raise if unusable.
      }
      return { locator, strategy: descriptor.strategy, matchCount: count };
    }

    throw new Error(missingMessage);
  }

  async resolveStateChangingLocator(params) {
    return this.resolveCandidateDescriptor(
      this.buildLocatorCandidateDescriptors(params),
      `No element found for ${this.describeTarget(params, 'element')}.`,
      this.describeTarget(params, 'element')
    );
  }

  async resolveStateChangingInputLocator(params) {
    return this.resolveCandidateDescriptor(
      this.buildInputCandidateDescriptorsForParams(params),
      `Input target not found for ${this.describeTarget(params, 'input')}. Use CSS/XPath or a clear field label.`,
      this.describeTarget(params, 'input')
    );
  }

  async firstVisibleLocator(candidates, perCandidateTimeout = 1500) {
    for (const candidate of candidates) {
      try {
        await candidate.waitFor({ state: 'visible', timeout: perCandidateTimeout });
        return candidate;
      } catch (_) {
        // Keep checking next candidate.
      }
    }
    return null;
  }

  async firstAttachedLocator(candidates) {
    for (const candidate of candidates) {
      try {
        if (await candidate.count()) return candidate;
      } catch (_) {
        // Ignore detached/inaccessible frames.
      }
    }
    return null;
  }

  async getLocator(params) {
    const candidates = this.buildLocatorCandidates(params);
    if (candidates.length) {
      const visible = await this.firstVisibleLocator(candidates);
      if (visible) return visible;
      const attached = await this.firstAttachedLocator(candidates);
      if (attached) return attached;
      if (params.text) throw new Error(`No element found for text "${params.text}".`);
      if (params.selector) throw new Error(`No element found for selector "${params.selector}".`);
      if (params.xpath) throw new Error(`No element found for xpath "${params.xpath}".`);
    }
    throw new Error('Missing selector target. Use selector, xpath, or text.');
  }

  async getInputLocator(params) {
    const roots = this.getSearchRoots();
    if (params.selector) {
      const candidates = roots.map((root) => root.locator(params.selector).first());
      const visible = await this.firstVisibleLocator(candidates);
      if (visible) return visible;
      const attached = await this.firstAttachedLocator(candidates);
      if (attached) return attached;
    }
    if (params.xpath) {
      const candidates = roots.map((root) => root.locator(`xpath=${params.xpath}`).first());
      const visible = await this.firstVisibleLocator(candidates);
      if (visible) return visible;
      const attached = await this.firstAttachedLocator(candidates);
      if (attached) return attached;
    }
    if (params.text) {
      const candidates = roots.flatMap((root) => this.buildInputCandidateLocators(params.text, root));
      const visible = await this.firstVisibleLocator(candidates);
      if (visible) return visible;
      const attached = await this.firstAttachedLocator(candidates);
      if (attached) return attached;
    }
    throw new Error('Input target not found. Use CSS/XPath or a clear field label.');
  }

  async writeValueToLocator(locator, value, mode = 'fill') {
    const beforeValue = await this.readLocatorValue(locator);
    await locator.scrollIntoViewIfNeeded();
    await locator.waitFor({ state: 'visible', timeout: this.actionTimeout });
    const editableInfo = await locator.evaluate((element) => {
      const tag = element.tagName.toLowerCase();
      return {
        tag,
        contentEditable: element.isContentEditable,
        inputType: element.getAttribute('type') || ''
      };
    });

    if (editableInfo.contentEditable) {
      await locator.click({ timeout: this.actionTimeout });
      if (mode === 'fill') {
        await this.page.keyboard.press('Control+A');
        await this.page.keyboard.press('Backspace');
      }
      await this.page.keyboard.type(value);
      const afterValue = await this.readLocatorValue(locator);
      return { beforeValue, afterValue, target: editableInfo };
    }

    if (editableInfo.tag === 'input' || editableInfo.tag === 'textarea') {
      if (mode === 'fill') {
        await locator.fill(value);
      } else {
        await locator.type(value);
      }
      const afterValue = await this.readLocatorValue(locator);
      return { beforeValue, afterValue, target: editableInfo };
    }

    await locator.click({ timeout: this.actionTimeout });
    await this.page.keyboard.type(value);
    const afterValue = await this.readLocatorValue(locator);
    return { beforeValue, afterValue, target: editableInfo };
  }

  async readLocatorValue(locator) {
    return locator.evaluate((element) => {
      const tag = element.tagName.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') {
        return String(element.value || '');
      }
      if (element.isContentEditable) {
        return String(element.textContent || '');
      }
      if (typeof element.textContent === 'string') {
        return String(element.textContent);
      }
      return '';
    });
  }

  async describeLocator(locator) {
    return locator.evaluate((element) => ({
      tag: element.tagName ? element.tagName.toLowerCase() : '',
      text: String(element.innerText || element.textContent || '').trim().slice(0, 120),
      value: typeof element.value === 'string' ? element.value : '',
      checked: typeof element.checked === 'boolean' ? element.checked : null,
      disabled: Boolean(element.disabled || element.getAttribute('aria-disabled') === 'true'),
      ariaExpanded: element.getAttribute('aria-expanded'),
      focused: document.activeElement === element
    }));
  }

  log(level, message, metadata) {
    const timestamp = new Date().toISOString();
    const traceMeta = getActiveTraceMeta();
    const payload = sanitizeForLog({
      ...(metadata || {}),
      ...(traceMeta ? { trace: traceMeta } : {})
    });
    const suffix = Object.keys(payload).length ? ` ${JSON.stringify(payload)}` : '';
    const line = `[${timestamp}] [${level}] ${message}${suffix}`;
    if (this.logToConsole) {
      console.log(line);
    }
    if (this.writeLogFile) {
      try {
        fs.appendFileSync(this.logFile, `${line}\n`, 'utf8');
      } catch (_) {
        // Keep automation running if logging to file fails.
      }
    }
  }
}

module.exports = { EdgeSession };
