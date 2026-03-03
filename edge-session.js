const { chromium } = require('playwright');
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
      this.localOperatorMode
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

  async act(action, params = {}) {
    if (!this.page) throw new Error('Edge session is not open');

    const startedAt = Date.now();
    const safeParams = this.logActionPayloads ? this.sanitizeActionParams(action, params) : undefined;
    this.log('INFO', 'Action started', safeParams ? { action, params: safeParams } : { action });

    try {
      const result = await runInSpan(
        'edge.action.execute',
        {
          'app.action.name': action,
          'app.action.retry_count': this.retryCount,
          'app.action.dom_fallback_enabled': this.domFallbackOnFailure
        },
        async () =>
          this.withActionRetry(action, async () => {
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
              default:
                throw new Error(`Unsupported action: ${action}`);
            }
          })
      );

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
      const locator = await this.getLocator(compatibleParams);
      await locator.scrollIntoViewIfNeeded();
      await locator.click({ timeout: this.actionTimeout });
      return { ok: true };
    } catch (error) {
      if (!this.domFallbackOnFailure) throw error;
      const domResult = await this.clickWithDomFallback(compatibleParams);
      if (domResult.ok) {
        return { ok: true, fallback: 'dom', strategy: domResult.strategy };
      }
      throw error;
    }
  }

  async handleFill(params) {
    if (params.value === undefined) throw new Error('Missing value');
    try {
      const locator = await this.getInputLocator(params);
      await this.writeValueToLocator(locator, String(params.value), 'fill');
      return { ok: true };
    } catch (error) {
      if (!this.domFallbackOnFailure) throw error;
      const domResult = await this.writeWithDomFallback(params, String(params.value), 'fill');
      if (domResult.ok) {
        return { ok: true, fallback: 'dom', strategy: domResult.strategy };
      }
      throw error;
    }
  }

  async handleType(params) {
    if (params.value === undefined) throw new Error('Missing value');
    try {
      const locator = await this.getInputLocator(params);
      await this.writeValueToLocator(locator, String(params.value), 'type');
      return { ok: true };
    } catch (error) {
      if (!this.domFallbackOnFailure) throw error;
      const domResult = await this.writeWithDomFallback(params, String(params.value), 'type');
      if (domResult.ok) {
        return { ok: true, fallback: 'dom', strategy: domResult.strategy };
      }
      throw error;
    }
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
    const locator = await this.getLocator(params);
    const [download] = await Promise.all([
      this.page.waitForEvent('download'),
      locator.click()
    ]);

    const defaultName = download.suggestedFilename();
    const savePath = this.resolveOutputPath(params.savePath, path.join('downloads', defaultName));
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
    const outputPath = this.resolveOutputPath(params.path, `screenshot-${Date.now()}.png`);
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
    const tracePath = this.resolveOutputPath(params.path, path.join('traces', `trace-${Date.now()}.zip`));
    await fs.promises.mkdir(path.dirname(tracePath), { recursive: true });
    await this.context.tracing.stop({ path: tracePath });
    this.traceActive = false;
    return { status: 'stopped', path: tracePath };
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
      return { ok: clickElement(element), strategy };
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
      return { ok: writeToElement(element), strategy };
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
    const traceMeta = getActiveTraceMeta();
    const payload = {
      ...(metadata || {}),
      ...(traceMeta ? { trace: traceMeta } : {})
    };
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
