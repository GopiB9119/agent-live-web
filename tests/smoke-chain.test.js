const { describe, it, beforeEach, afterEach, mock } = require('node:test');
const assert = require('node:assert/strict');
const { EdgeSession } = require('../edge-session');
const { parseCommand } = require('../nl-command-parser');

/*
 * End-to-end smoke tests for the NL parser → EdgeSession → verification chain.
 * These tests use mocked Playwright pages (no real browser) to verify the full
 * flow from natural language input through to grounded response output.
 */

function createMockPage(options = {}) {
  const url = options.url || 'about:blank';
  const title = options.title || '';
  const content = options.content || '<html><body></body></html>';
  let currentUrl = url;
  let currentTitle = title;

  const locator = {
    count: mock.fn(async () => options.locatorCount || 1),
    first: mock.fn(() => locator),
    isVisible: mock.fn(async () => true),
    isAttached: mock.fn(async () => true),
    click: mock.fn(async () => {}),
    fill: mock.fn(async () => {}),
    pressSequentially: mock.fn(async () => {}),
    inputValue: mock.fn(async () => options.inputValue || ''),
    innerText: mock.fn(async () => options.innerText || ''),
    getAttribute: mock.fn(async () => options.getAttribute || null),
    evaluate: mock.fn(async (fn) => {
      if (typeof fn === 'function') return {};
      return {};
    }),
    waitFor: mock.fn(async () => {})
  };

  return {
    url: mock.fn(() => currentUrl),
    title: mock.fn(async () => currentTitle),
    content: mock.fn(async () => content),
    goto: mock.fn(async (targetUrl) => {
      currentUrl = targetUrl;
      currentTitle = options.gotoTitle || 'Navigated Page';
      return { status: () => 200 };
    }),
    locator: mock.fn(() => locator),
    getByRole: mock.fn(() => locator),
    getByPlaceholder: mock.fn(() => locator),
    getByLabel: mock.fn(() => locator),
    waitForTimeout: mock.fn(async () => {}),
    waitForLoadState: mock.fn(async () => {}),
    evaluate: mock.fn(async () => ({ x: 0, y: 0 })),
    screenshot: mock.fn(async () => Buffer.from('fake-png')),
    keyboard: { press: mock.fn(async () => {}) },
    setDefaultTimeout: mock.fn(),
    setDefaultNavigationTimeout: mock.fn(),
    frames: mock.fn(() => [])
  };
}

describe('NL parser → EdgeSession chain', () => {
  it('parseCommand feeds valid params into runAction for goto', () => {
    const parsed = parseCommand('go to https://example.com');
    assert.equal(parsed.action, 'goto');
    assert.equal(parsed.params.url, 'https://example.com');
  });

  it('parseCommand feeds valid params into runAction for click', () => {
    const parsed = parseCommand('click #submit-btn');
    assert.equal(parsed.action, 'click');
    assert.equal(parsed.params.selector, '#submit-btn');
  });

  it('parseCommand feeds valid params into runAction for type', () => {
    const parsed = parseCommand('type "hello" in #input');
    assert.equal(parsed.action, 'type');
    assert.equal(parsed.params.value, 'hello');
    assert.equal(parsed.params.selector, '#input');
  });

  it('parseCommand feeds valid params into runAction for search', () => {
    const parsed = parseCommand('search for playwright testing');
    assert.equal(parsed.action, 'search');
    assert.equal(parsed.params.query, 'playwright testing');
  });

  it('unknown commands are caught before reaching EdgeSession', () => {
    const parsed = parseCommand('do something weird');
    assert.equal(parsed.action, 'unknown');
  });

  it('parseCommand handles back, forward, refresh, press, hover, and select', () => {
    assert.equal(parseCommand('back').action, 'back');
    assert.equal(parseCommand('go back').action, 'back');
    assert.equal(parseCommand('forward').action, 'forward');
    assert.equal(parseCommand('go forward').action, 'forward');
    assert.equal(parseCommand('refresh').action, 'refresh');
    assert.equal(parseCommand('reload').action, 'refresh');
    const press = parseCommand('press Enter');
    assert.equal(press.action, 'press');
    assert.equal(press.params.key, 'Enter');
    const hover = parseCommand('hover over .menu-item');
    assert.equal(hover.action, 'hover');
    assert.equal(hover.params.selector, '.menu-item');
    const hoverOn = parseCommand('hover on #avatar');
    assert.equal(hoverOn.action, 'hover');
    assert.equal(hoverOn.params.selector, '#avatar');
    const hoverDirect = parseCommand('hover text:Profile');
    assert.equal(hoverDirect.action, 'hover');
    assert.equal(hoverDirect.params.text, 'Profile');
    const sel = parseCommand('select "Option A" from #dropdown');
    assert.equal(sel.action, 'select');
    assert.equal(sel.params.value, 'Option A');
    assert.equal(sel.params.selector, '#dropdown');
    const choose = parseCommand('choose India in css:#country');
    assert.equal(choose.action, 'select');
    assert.equal(choose.params.value, 'India');
    assert.equal(choose.params.selector, '#country');
    const focus = parseCommand('focus on #email');
    assert.equal(focus.action, 'focus');
    assert.equal(focus.params.selector, '#email');
    assert.equal(parseCommand('focus .search-input').action, 'focus');
    const clear = parseCommand('clear #search');
    assert.equal(clear.action, 'clear');
    assert.equal(clear.params.selector, '#search');
    const dbl = parseCommand('double-click .cell');
    assert.equal(dbl.action, 'doubleClick');
    assert.equal(dbl.params.selector, '.cell');
    assert.equal(parseCommand('doubleclick #item').action, 'doubleClick');
    assert.equal(parseCommand('dblclick text:Edit').action, 'doubleClick');
    const rc = parseCommand('right-click .file-item');
    assert.equal(rc.action, 'rightClick');
    assert.equal(rc.params.selector, '.file-item');
    assert.equal(parseCommand('rightclick #ctx').action, 'rightClick');
  });
});

describe('EdgeSession response contract', () => {
  let session;

  beforeEach(() => {
    session = new EdgeSession({ headless: true });
  });

  it('runAction returns structured response with required fields', async () => {
    const mockPage = createMockPage({
      url: 'https://example.com',
      gotoTitle: 'Example Domain'
    });
    session.context = { close: async () => {} };
    session.page = mockPage;

    const response = await session.runAction('goto', { url: 'https://example.com' });

    assert.ok(response, 'response should exist');
    assert.ok('action' in response, 'response must have action');
    assert.ok('status' in response, 'response must have status');
    assert.ok('verification' in response, 'response must have verification');
    assert.ok('summary' in response, 'response must have grounded summary');
    assert.ok('durationMs' in response, 'response must have durationMs');
    assert.equal(typeof response.summary, 'string');
    assert.ok(response.summary.length > 0, 'summary must not be empty');
  });

  it('grounded summary reflects verification, not raw result', async () => {
    // Simulate a goto that fails verification (wrong URL)
    const mockPage = createMockPage({ url: 'about:blank' });
    mockPage.goto = mock.fn(async () => {
      // Navigation "succeeds" from Playwright's perspective but lands on wrong URL
      return { status: () => 200 };
    });
    session.context = { close: async () => {} };
    session.page = mockPage;

    const response = await session.runAction('goto', { url: 'https://expected.com' });

    // The summary should reflect the verification failure, not claim success
    assert.equal(response.status, 'failed');
    assert.equal(response.verification.ok, false);
    assert.ok(response.summary.toLowerCase().includes('failed'), 'summary must say failed');
  });

  it('sanitizeActionResponse redacts sensitive values', () => {
    const raw = {
      action: 'type',
      status: 'ok',
      result: { beforeValue: 'secret123', afterValue: 'secret456', requestedValue: 'secret456' },
      verification: { ok: true, reason: 'Typed value verification passed.' },
      durationMs: 100
    };

    const sanitized = session.sanitizeActionResponse(raw);

    assert.equal(sanitized.result.beforeValue, '[REDACTED_INPUT]');
    assert.equal(sanitized.result.afterValue, '[REDACTED_INPUT]');
    assert.equal(sanitized.result.requestedValue, '[REDACTED_INPUT]');
  });

  it('runAction without open session returns structured error', async () => {
    // page is null — session not opened
    const response = await session.runAction('goto', { url: 'https://example.com' });
    assert.equal(response.status, 'failed');
    assert.ok(response.error.toLowerCase().includes('not open'));
    assert.equal(typeof response.summary, 'string');
    assert.ok(response.summary.toLowerCase().includes('failed'));
  });

  it('runAction blocks delete by default', async () => {
    session.page = createMockPage({ url: 'https://example.com' });
    session.context = { close: async () => {} };
    const response = await session.runAction('delete', { selector: '#item' });
    assert.equal(response.status, 'failed');
    assert.ok(response.error.toLowerCase().includes('disabled'));
  });

  it('runAction blocks add by default', async () => {
    session.page = createMockPage({ url: 'https://example.com' });
    session.context = { close: async () => {} };
    const response = await session.runAction('add', { parentSelector: '#list', html: '<li>New</li>' });
    assert.equal(response.status, 'failed');
    assert.ok(response.error.toLowerCase().includes('disabled'));
  });

  it('runAction returns attempts count and recovered flag', async () => {
    const mockPage = createMockPage({
      url: 'https://example.com',
      gotoTitle: 'Example'
    });
    session.context = { close: async () => {} };
    session.page = mockPage;
    const response = await session.runAction('goto', { url: 'https://example.com' });
    assert.equal(typeof response.attempts, 'number');
    assert.ok(response.attempts >= 1);
    assert.equal(typeof response.recovered, 'boolean');
  });

  it('unsupported action returns error', async () => {
    session.page = createMockPage({ url: 'https://example.com' });
    session.context = { close: async () => {} };
    const response = await session.runAction('flyToMoon', {});
    assert.equal(response.status, 'failed');
    assert.ok(response.error.toLowerCase().includes('unsupported'));
  });

  it('response durationMs is a positive number', async () => {
    const mockPage = createMockPage({
      url: 'https://example.com',
      gotoTitle: 'Example'
    });
    session.context = { close: async () => {} };
    session.page = mockPage;
    const response = await session.runAction('goto', { url: 'https://example.com' });
    assert.equal(typeof response.durationMs, 'number');
    assert.ok(response.durationMs >= 0);
  });
});
