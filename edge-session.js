const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

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

const SENSITIVE_KEY_PATTERN = /(pass(word)?|pwd|token|secret|auth|cookie|session|otp|pin|api[_-]?key|bearer|credential|code)/i;
const TOKEN_VALUE_PATTERN = /^(Bearer\s+)?[A-Za-z0-9._-]{24,}$/i;

function sanitizeUrlForLog(value) {
  try {
    const parsed = new URL(value);
    for (const [key] of parsed.searchParams) {
      if (SENSITIVE_KEY_PATTERN.test(key)) {
        parsed.searchParams.set(key, '[REDACTED]');
      }
    }
    return parsed.toString();
  } catch (_) {
    return value;
  }
}

function sanitizeForLog(value, key = '', seen = new WeakSet(), depth = 0) {
  if (value === null || value === undefined) return value;
  if (depth > 6) return '[TRUNCATED]';

  if (typeof value === 'string') {
    if (SENSITIVE_KEY_PATTERN.test(key)) return '[REDACTED]';
    if (/url|uri|link/i.test(key)) return sanitizeUrlForLog(value);
    if (TOKEN_VALUE_PATTERN.test(value)) return '[REDACTED]';
    return value;
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
    this.logFile = options.logFile || path.join(process.cwd(), 'logs', 'edge-agent.log');
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
    await fs.promises.mkdir(this.userDataDir, { recursive: true });
    await fs.promises.mkdir(path.dirname(this.logFile), { recursive: true });

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

  async close() {
    if (!this.context) return;
    await this.context.close();
    this.context = null;
    this.page = null;
    this.log('INFO', 'Edge session closed');
  }

  async newSession() {
    await this.close();
    await this.open();
  }

  async act(action, params = {}) {
    if (!this.page) throw new Error('Edge session is not open');

    const startedAt = Date.now();
    const safeParams = this.logActionPayloads ? this.sanitizeActionParams(action, params) : undefined;
    this.log('INFO', 'Action started', safeParams ? { action, params: safeParams } : { action });

    try {
      const result = await this.withActionRetry(action, async () => {
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
            return this.handleDelete(params);
          case 'add':
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
          default:
            throw new Error(`Unsupported action: ${action}`);
        }
      });

      const safeResult = this.logActionPayloads ? sanitizeForLog(result) : undefined;
      this.log(
        'INFO',
        'Action completed',
        safeResult !== undefined
          ? { action, durationMs: Date.now() - startedAt, result: safeResult }
          : { action, durationMs: Date.now() - startedAt }
      );
      return result;
    } catch (error) {
      this.log('ERROR', 'Action failed', { action, durationMs: Date.now() - startedAt, error: error.message });
      throw error;
    }
  }

  sanitizeActionParams(action, params) {
    const safe = sanitizeForLog(params);
    if ((action === 'type' || action === 'edit') && safe && Object.prototype.hasOwnProperty.call(safe, 'value')) {
      safe.value = '[REDACTED_INPUT]';
    }
    return safe;
  }

  async withActionRetry(action, operation) {
    let lastError;
    for (let attempt = 0; attempt <= this.retryCount; attempt += 1) {
      try {
        if (attempt > 0) {
          this.log('WARN', 'Retrying action', { action, attempt });
          await this.page.waitForTimeout(600);
        }
        return await operation();
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError;
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
    const searchCandidates = [
      this.page.getByRole('searchbox'),
      this.page.locator('input[type="search"]'),
      this.page.getByPlaceholder(/search/i),
      this.page.getByLabel(/search/i),
      this.page.getByRole('textbox', { name: /search/i }),
      this.page.locator('input[aria-label*="search" i], textarea[aria-label*="search" i], [contenteditable="true"][aria-label*="search" i]')
    ];
    const searchInput = await this.firstVisibleLocator(searchCandidates);
    if (!searchInput) throw new Error('No visible search field found on this page.');
    await this.writeValueToLocator(searchInput, query, 'fill');
    await searchInput.press('Enter');
    return { query, submitted: true };
  }

  async handleClick(action, params) {
    const compatibleParams = { ...params };
    if (action === 'clickXPath' && !compatibleParams.xpath && compatibleParams.selector) {
      compatibleParams.xpath = compatibleParams.selector;
    }
    if (action === 'clickByText' && !compatibleParams.text && compatibleParams.selector) {
      compatibleParams.text = compatibleParams.selector;
    }
    const locator = await this.getLocator(compatibleParams);
    await locator.scrollIntoViewIfNeeded();
    await locator.click({ timeout: this.actionTimeout });
    return { ok: true };
  }

  async handleFill(params) {
    if (params.value === undefined) throw new Error('Missing value');
    const locator = await this.getInputLocator(params);
    await this.writeValueToLocator(locator, String(params.value), 'fill');
    return { ok: true };
  }

  async handleType(params) {
    if (params.value === undefined) throw new Error('Missing value');
    const locator = await this.getInputLocator(params);
    await this.writeValueToLocator(locator, String(params.value), 'type');
    return { ok: true };
  }

  async handleDelete(params) {
    if (params.selector) {
      await this.page.evaluate((selector) => {
        const element = document.querySelector(selector);
        if (element) element.remove();
      }, params.selector);
      return { removed: true, mode: 'css' };
    }
    if (params.xpath) {
      await this.page.evaluate((xpath) => {
        const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
        if (result.singleNodeValue) result.singleNodeValue.remove();
      }, params.xpath);
      return { removed: true, mode: 'xpath' };
    }
    throw new Error('Delete requires CSS selector or XPath');
  }

  async handleAdd(params) {
    if (!params.parentSelector || !params.html) throw new Error('Add requires parentSelector and html');
    await this.page.evaluate(({ parentSelector, html }) => {
      const parent = document.querySelector(parentSelector);
      if (!parent) throw new Error(`Parent selector not found: ${parentSelector}`);
      const wrapper = document.createElement('div');
      wrapper.innerHTML = html;
      if (!wrapper.firstElementChild) throw new Error('Invalid html payload');
      parent.appendChild(wrapper.firstElementChild);
    }, params);
    return { ok: true };
  }

  async handleExists(params) {
    const count = await this.countMatches(params);
    return { exists: count > 0 };
  }

  async handleGetText(params) {
    const locator = await this.getLocator(params);
    const text = await locator.textContent();
    return { text: text ? text.trim() : '' };
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
    const locator = await this.getLocator(params);
    const [download] = await Promise.all([
      this.page.waitForEvent('download'),
      locator.click()
    ]);

    const defaultName = download.suggestedFilename();
    const savePath = path.resolve(params.savePath || path.join('downloads', defaultName));
    await fs.promises.mkdir(path.dirname(savePath), { recursive: true });
    await download.saveAs(savePath);
    return { path: savePath };
  }

  async handleUpload(params) {
    if (!params.filePath) throw new Error('upload requires filePath');
    const locator = await this.getInputLocator(params);
    await locator.setInputFiles(path.resolve(params.filePath));
    return { ok: true, filePath: path.resolve(params.filePath) };
  }

  async handleScroll(params) {
    const direction = params.direction === 'up' ? 'up' : 'down';
    const amount = Number(params.amount) || 600;
    await this.page.evaluate(({ scrollDirection, pixels }) => {
      window.scrollBy(0, scrollDirection === 'down' ? pixels : -pixels);
    }, { scrollDirection: direction, pixels: amount });
    return { direction, amount };
  }

  async handleScreenshot(params) {
    const outputPath = path.resolve(params.path || `screenshot-${Date.now()}.png`);
    await fs.promises.mkdir(path.dirname(outputPath), { recursive: true });
    await this.page.screenshot({ path: outputPath, fullPage: true });
    return { path: outputPath };
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
    const tracePath = path.resolve(params.path || path.join('traces', `trace-${Date.now()}.zip`));
    await fs.promises.mkdir(path.dirname(tracePath), { recursive: true });
    await this.context.tracing.stop({ path: tracePath });
    this.traceActive = false;
    return { status: 'stopped', path: tracePath };
  }

  async countMatches(params) {
    if (params.selector) return this.page.locator(params.selector).count();
    if (params.xpath) return this.page.locator(`xpath=${params.xpath}`).count();
    if (params.text) {
      const candidates = this.buildTextCandidateLocators(params.text);
      let total = 0;
      for (const candidate of candidates) {
        total += await candidate.count();
      }
      return total;
    }
    return 0;
  }

  buildTextCandidateLocators(text) {
    const normalized = String(text).trim();
    const exactRegex = new RegExp(`^\\s*${escapeRegex(normalized)}\\s*$`, 'i');
    const hasWhitespace = /\s/.test(normalized);
    const candidates = [
      this.page.getByRole('button', { name: exactRegex }),
      this.page.getByRole('link', { name: exactRegex }),
      this.page.getByRole('tab', { name: exactRegex }),
      this.page.getByRole('menuitem', { name: exactRegex }),
      this.page.getByRole('option', { name: exactRegex }),
      this.page.getByLabel(normalized, { exact: false }),
      this.page.getByPlaceholder(normalized, { exact: false }),
      this.page.getByText(normalized, { exact: false })
    ];
    if (!hasWhitespace) {
      candidates.push(this.page.locator(`[aria-label="${escapeAttrValue(normalized)}" i]`));
      candidates.push(this.page.locator(`[title="${escapeAttrValue(normalized)}" i]`));
    }
    return candidates.map((candidate) => candidate.first());
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
      if (await candidate.count()) return candidate;
    }
    return null;
  }

  async getLocator(params) {
    if (params.selector) return this.page.locator(params.selector).first();
    if (params.xpath) return this.page.locator(`xpath=${params.xpath}`).first();
    if (params.text) {
      const candidates = this.buildTextCandidateLocators(params.text);
      const visible = await this.firstVisibleLocator(candidates);
      if (visible) return visible;
      const attached = await this.firstAttachedLocator(candidates);
      if (attached) return attached;
      throw new Error(`No element found for text "${params.text}".`);
    }
    throw new Error('Missing selector target. Use selector, xpath, or text.');
  }

  async getInputLocator(params) {
    if (params.selector) return this.page.locator(params.selector).first();
    if (params.xpath) return this.page.locator(`xpath=${params.xpath}`).first();
    if (params.text) {
      const target = String(params.text).trim();
      const targetRegex = new RegExp(escapeRegex(target), 'i');
      const candidates = [
        this.page.getByLabel(target, { exact: false }).first(),
        this.page.getByPlaceholder(target, { exact: false }).first(),
        this.page.getByRole('textbox', { name: targetRegex }).first(),
        this.page.locator(`input[aria-label*="${escapeAttrValue(target)}" i], textarea[aria-label*="${escapeAttrValue(target)}" i], [contenteditable="true"][aria-label*="${escapeAttrValue(target)}" i]`).first()
      ];
      const visible = await this.firstVisibleLocator(candidates);
      if (visible) return visible;
      const attached = await this.firstAttachedLocator(candidates);
      if (attached) return attached;
    }
    throw new Error('Input target not found. Use CSS/XPath or a clear field label.');
  }

  async writeValueToLocator(locator, value, mode = 'fill') {
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
      return;
    }

    if (editableInfo.tag === 'input' || editableInfo.tag === 'textarea') {
      if (mode === 'fill') {
        await locator.fill(value);
      } else {
        await locator.type(value);
      }
      return;
    }

    await locator.click({ timeout: this.actionTimeout });
    await this.page.keyboard.type(value);
  }

  log(level, message, metadata) {
    const timestamp = new Date().toISOString();
    const suffix = metadata ? ` ${JSON.stringify(metadata)}` : '';
    const line = `[${timestamp}] [${level}] ${message}${suffix}`;
    console.log(line);
    try {
      fs.appendFileSync(this.logFile, `${line}\n`, 'utf8');
    } catch (_) {
      // Keep automation running if logging to file fails.
    }
  }
}

module.exports = { EdgeSession };
