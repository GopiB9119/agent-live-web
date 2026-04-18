const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { EdgeSession } = require('../edge-session');

test('runAction returns a structured success response', async () => {
  const session = new EdgeSession({ retryCount: 0 });
  session.page = {};
  session.log = () => {};

  let captureCalls = 0;
  session.captureActionState = async () => {
    captureCalls += 1;
    if (captureCalls === 1) {
      return { url: 'https://example.com/form', title: 'Before', snapshotHash: 'before' };
    }
    return { url: 'https://example.com/form', title: 'After', snapshotHash: 'after' };
  };

  session.executeActionWithRetry = async () => ({
    result: { ok: true, mode: 'locator' },
    attempts: 2,
    recovered: true
  });

  session.verifyAction = async (action, params, beforeState, afterState, result) => ({
    ok: true,
    reason: 'verified',
    details: { before: beforeState, after: afterState, result }
  });

  const response = await session.runAction('click', { text: 'Submit' });

  assert.equal(response.status, 'ok');
  assert.equal(response.action, 'click');
  assert.equal(response.tool, 'click');
  assert.equal(response.attempts, 2);
  assert.equal(response.recovered, true);
  assert.equal(response.verification.reason, 'verified');
  assert.equal(response.summary, 'Click succeeded: verified');
  assert.deepEqual(response.result, { ok: true, mode: 'locator' });
  assert.equal(response.error, null);
  assert.equal(response.args.text, 'Submit');
});

test('runAction returns a structured failure response and redacts typed values', async () => {
  const session = new EdgeSession({ retryCount: 1 });
  session.page = {};
  session.log = () => {};

  session.captureActionState = async () => ({
    url: 'https://example.com/login',
    title: 'Login',
    snapshotHash: 'same'
  });

  session.executeActionWithRetry = async () => {
    const error = new Error('No element found for selector "#password".');
    error.attemptCount = 2;
    throw error;
  };

  const response = await session.runAction('type', { selector: '#password', value: 'super-secret' });

  assert.equal(response.status, 'failed');
  assert.equal(response.action, 'type');
  assert.equal(response.attempts, 2);
  assert.equal(response.recovered, false);
  assert.equal(response.error, 'No element found for selector "#password".');
  assert.equal(response.verification.ok, false);
  assert.equal(response.summary, 'Type failed: No element found for selector "#password".');
  assert.equal(response.args.selector, '#password');
  assert.equal(response.args.value, '[REDACTED_INPUT]');
  assert.equal(response.result, null);
});

test('runAction redacts typed verification details before returning the response', async () => {
  const session = new EdgeSession({ retryCount: 0 });
  session.page = {};
  session.log = () => {};

  let captureCalls = 0;
  session.captureActionState = async () => {
    captureCalls += 1;
    return captureCalls === 1
      ? { url: 'https://example.com/login', title: 'Before', snapshotHash: 'a' }
      : { url: 'https://example.com/login', title: 'After', snapshotHash: 'b' };
  };
  session.executeActionWithRetry = async () => ({
    result: { beforeValue: '', afterValue: 'super-secret', mode: 'type' },
    attempts: 1,
    recovered: false
  });
  session.verifyAction = async () => ({
    ok: false,
    reason: "Typed value verification failed. Before='' After='super-secret' Requested fragment='super-secret'.",
    details: {
      beforeValue: '',
      afterValue: 'super-secret',
      requestedValue: 'super-secret'
    }
  });

  const response = await session.runAction('type', { selector: '#password', value: 'super-secret' });

  assert.equal(response.status, 'failed');
  assert.equal(response.result.beforeValue, '[REDACTED_INPUT]');
  assert.equal(response.result.afterValue, '[REDACTED_INPUT]');
  assert.equal(response.verification.details.afterValue, '[REDACTED_INPUT]');
  assert.equal(response.verification.details.requestedValue, '[REDACTED_INPUT]');
  assert.equal(response.verification.reason, 'Typed value verification failed.');
  assert.equal(response.summary, 'Type failed: Typed value verification failed.');
});

test('formatActionResponse summary stays grounded on failure even if result text sounds successful', async () => {
  const session = new EdgeSession();

  const response = session.formatActionResponse({
    action: 'click',
    args: { text: 'Save' },
    attempts: 1,
    recovered: false,
    verification: {
      ok: false,
      reason: 'Click executed but no page or target state change was verified.',
      details: {}
    },
    result: { message: 'Saved successfully.' },
    error: null,
    durationMs: 42
  });

  assert.equal(response.status, 'failed');
  assert.equal(response.summary, 'Click failed: Click executed but no page or target state change was verified.');
});

test('act keeps legacy raw-result behavior on success and throws on failure', async () => {
  const successSession = new EdgeSession();
  successSession.runAction = async () => ({
    status: 'ok',
    result: { text: 'hello' },
    verification: { ok: true, reason: 'verified', details: {} }
  });

  const rawResult = await successSession.act('getText', { text: 'Welcome' });
  assert.deepEqual(rawResult, { text: 'hello' });

  const failureSession = new EdgeSession();
  failureSession.runAction = async () => ({
    status: 'failed',
    result: null,
    error: 'Missing selector target.',
    verification: { ok: false, reason: 'Missing selector target.', details: {} }
  });

  await assert.rejects(
    () => failureSession.act('click', { text: 'Missing' }),
    (error) => {
      assert.equal(error.message, 'Missing selector target.');
      assert.equal(error.actionResponse.status, 'failed');
      return true;
    }
  );
});

test('verifyAction confirms fill results by exact final value', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'edit',
    { value: 'John Doe' },
    { url: 'https://example.com/form', title: 'Before', snapshotHash: 'a' },
    { url: 'https://example.com/form', title: 'After', snapshotHash: 'b' },
    { beforeValue: '', afterValue: 'John Doe', mode: 'fill' }
  );

  assert.equal(verification.ok, true);
  assert.equal(verification.details.afterValue, 'John Doe');
});

test('verifyAction confirms type results when typed fragment appears in changed value', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'type',
    { value: 'world' },
    { url: 'https://example.com/form', title: 'Before', snapshotHash: 'a' },
    { url: 'https://example.com/form', title: 'After', snapshotHash: 'b' },
    { beforeValue: 'hello ', afterValue: 'hello world', mode: 'type' }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /Typed value was observed/);
});

test('verifyAction rejects clicks with no page or target change', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'click',
    { text: 'Submit' },
    { url: 'https://example.com/form', title: 'Before', snapshotHash: 'same' },
    { url: 'https://example.com/form', title: 'After', snapshotHash: 'same' },
    {
      target: {
        before: { tag: 'button', text: 'Submit', focused: false },
        after: { tag: 'button', text: 'Submit', focused: false }
      }
    }
  );

  assert.equal(verification.ok, false);
  assert.match(verification.reason, /no page or target state change was verified/i);
});

test('verifyAction confirms upload results by source file and selected file name', async () => {
  const session = new EdgeSession();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'edge-session-upload-'));
  const tempFile = path.join(tempDir, 'report.csv');

  try {
    fs.writeFileSync(tempFile, 'id,name\n1,example\n', 'utf8');

    const verification = await session.verifyAction(
      'upload',
      { filePath: tempFile },
      { url: 'https://example.com/upload', title: 'Before', snapshotHash: 'a' },
      { url: 'https://example.com/upload', title: 'After', snapshotHash: 'b' },
      { filePath: tempFile, selectedFiles: ['report.csv'], selectedCount: 1 }
    );

    assert.equal(verification.ok, true);
    assert.deepEqual(verification.details.selectedFiles, ['report.csv']);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('verifyAction confirms delete only when the target count decreases', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'delete',
    { selector: '.toast' },
    { url: 'https://example.com/app', title: 'Before', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/app', title: 'After', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { removed: true, mode: 'css', beforeCount: 1, afterCount: 0 }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /reduced match count from 1 to 0/i);
});

test('verifyAction confirms add only when the parent child count increases', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'add',
    { parentSelector: '#app' },
    { url: 'https://example.com/app', title: 'Before', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/app', title: 'After', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { ok: true, parentSelector: '#app', beforeChildCount: 2, afterChildCount: 3, insertedTag: 'div' }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /increased parent child count from 2 to 3/i);
});

test('verifyAction confirms scroll only when viewport position changes', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'scroll',
    { direction: 'down', amount: 600 },
    { url: 'https://example.com/app', title: 'Before', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/app', title: 'After', snapshotHash: null, scrollX: 0, scrollY: 600 },
    { direction: 'down', amount: 600, beforeY: 0, afterY: 600, deltaY: 600 }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /moved viewport from 0 to 600/i);
});

test('resolveStateChangingLocator rejects ambiguous matches instead of silently picking the first', async () => {
  const session = new EdgeSession();
  session.buildLocatorCandidateDescriptors = () => [
    {
      strategy: 'text-match',
      locator: {
        count: async () => 2,
        first() {
          return this;
        }
      }
    }
  ];

  await assert.rejects(
    () => session.resolveStateChangingLocator({ text: 'Submit' }),
    /Ambiguous element text "Submit" via text-match: matched 2 elements/
  );
});

test('handleFill exposes the selector strategy used for a state-changing input action', async () => {
  const session = new EdgeSession();
  const fakeLocator = {
    evaluate: async () => '',
    scrollIntoViewIfNeeded: async () => {},
    waitFor: async () => {},
    fill: async () => {}
  };

  session.resolveStateChangingInputLocator = async () => ({
    locator: fakeLocator,
    strategy: 'label-input',
    matchCount: 1
  });
  session.writeValueToLocator = async () => ({ beforeValue: '', afterValue: 'Alice' });

  const result = await session.handleFill({ text: 'Name', value: 'Alice' });
  assert.equal(result.selectorStrategy, 'label-input');
  assert.equal(result.afterValue, 'Alice');
});

test('handleSelect reads before/after value and selects by label', async () => {
  const session = new EdgeSession();
  let selected = false;
  const fakeLocator = {
    count: async () => 1,
    first: () => ({
      inputValue: async () => selected ? 'IN' : '',
      selectOption: async (opts) => {
        assert.equal(opts.label, 'India');
        selected = true;
      }
    })
  };
  session.getLocator = async () => fakeLocator;

  const result = await session.handleSelect({ selector: '#country', value: 'India' });

  assert.equal(result.beforeValue, '');
  assert.equal(result.afterValue, 'IN');
  assert.equal(result.selectedLabel, 'India');
  assert.equal(selected, true);
});

test('runAction blocks delete by default until explicit opt-in is provided', async () => {
  const session = new EdgeSession({ retryCount: 0 });
  session.page = { waitForTimeout: async () => {} };
  session.log = () => {};

  const response = await session.runAction('delete', { selector: '#danger-zone' });

  assert.equal(response.status, 'failed');
  assert.match(response.error, /EDGE_ALLOW_DOM_DELETE=false/);
  assert.equal(response.verification.ok, false);
});

test('runAction blocks add by default until explicit opt-in is provided', async () => {
  const session = new EdgeSession({ retryCount: 0 });
  session.page = { waitForTimeout: async () => {} };
  session.log = () => {};

  const response = await session.runAction('add', { parentSelector: '#app', html: '<div>Injected</div>' });

  assert.equal(response.status, 'failed');
  assert.match(response.error, /EDGE_ALLOW_DOM_HTML_ADD=false/);
  assert.equal(response.verification.ok, false);
});

test('verifyAction confirms back navigation when URL changes', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'back',
    {},
    { url: 'https://example.com/page2', title: 'Page 2', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/page1', title: 'Page 1', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/page1', title: 'Page 1' }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /back navigation completed/i);
});

test('verifyAction confirms forward navigation when URL changes', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'forward',
    {},
    { url: 'https://example.com/page1', title: 'Page 1', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/page2', title: 'Page 2', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/page2', title: 'Page 2' }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /forward navigation completed/i);
});

test('verifyAction confirms refresh when page has a valid URL', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'refresh',
    {},
    { url: 'https://example.com', title: 'Example', snapshotHash: 'abc', scrollX: 0, scrollY: 0 },
    { url: 'https://example.com', title: 'Example', snapshotHash: 'def', scrollX: 0, scrollY: 0 },
    { url: 'https://example.com', title: 'Example' }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /page refreshed/i);
});

test('verifyAction confirms key press when result reports pressed', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'press',
    { key: 'Enter' },
    { url: 'https://example.com', title: 'Form', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com', title: 'Form', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { key: 'Enter', pressed: true }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /key 'Enter' pressed/i);
});

test('verifyAction confirms hover when result reports hovered', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'hover',
    { selector: '.menu-item' },
    { url: 'https://example.com', title: 'Page', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com', title: 'Page', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { hovered: true, strategy: 'css-selector' }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /hover completed/i);
});

test('verifyAction confirms select when dropdown value changes', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'select',
    { selector: '#country', value: 'India' },
    { url: 'https://example.com/form', title: 'Form', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/form', title: 'Form', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { beforeValue: '', afterValue: 'IN', selectedLabel: 'India' }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /selected 'India'/i);
});

test('verifyAction rejects select when afterValue is empty', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'select',
    { selector: '#country', value: 'India' },
    { url: 'https://example.com/form', title: 'Form', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/form', title: 'Form', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { beforeValue: '', afterValue: '', selectedLabel: 'India' }
  );

  assert.equal(verification.ok, false);
  assert.match(verification.reason, /select verification failed/i);
});

test('verifyAction confirms focus when result reports focused', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'focus',
    { selector: '#email' },
    { url: 'https://example.com/form', title: 'Form', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/form', title: 'Form', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { focused: true, strategy: 'css-selector' }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /focus applied/i);
});

test('verifyAction confirms clear when input becomes empty', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'clear',
    { selector: '#search' },
    { url: 'https://example.com', title: 'Page', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { url: 'https://example.com', title: 'Page', snapshotHash: null, scrollX: 0, scrollY: 0 },
    { beforeValue: 'old text', afterValue: '', cleared: true }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /input cleared/i);
});

test('verifyAction confirms doubleClick when result reports success', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'doubleClick',
    { selector: '.cell' },
    { url: 'https://example.com/sheet', title: 'Sheet', snapshotHash: 'abc', scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/sheet', title: 'Sheet', snapshotHash: 'def', scrollX: 0, scrollY: 0 },
    { doubleClicked: true, strategy: 'css-selector' }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /double-click executed/i);
});

test('verifyAction confirms rightClick when result reports success', async () => {
  const session = new EdgeSession();

  const verification = await session.verifyAction(
    'rightClick',
    { selector: '.file-item' },
    { url: 'https://example.com/files', title: 'Files', snapshotHash: 'abc', scrollX: 0, scrollY: 0 },
    { url: 'https://example.com/files', title: 'Files', snapshotHash: 'def', scrollX: 0, scrollY: 0 },
    { rightClicked: true, strategy: 'css-selector' }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /right-click executed/i);
});

test('handleClear reads before value and verifies field is empty after', async () => {
  const session = new EdgeSession();
  let cleared = false;
  const fakeLocator = {
    count: async () => 1,
    first: () => ({
      inputValue: async () => cleared ? '' : 'old text',
      fill: async (val) => {
        assert.equal(val, '');
        cleared = true;
      }
    })
  };
  session.getLocator = async () => fakeLocator;

  const result = await session.handleClear({ selector: '#search' });

  assert.equal(result.beforeValue, 'old text');
  assert.equal(result.afterValue, '');
  assert.equal(result.cleared, true);
});