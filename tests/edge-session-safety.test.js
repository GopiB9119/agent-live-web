const test = require('node:test');
const assert = require('node:assert/strict');

const { classifyBrowserAction } = require('../browser-safety');
const { EdgeSession } = require('../edge-session');

test('dangerous click intent requires confirmation', () => {
  const safety = classifyBrowserAction('click', { text: 'Delete account' }, { workspaceRoot: process.cwd() });
  assert.equal(safety.decision, 'confirm_required');
  assert.equal(safety.actionClass, 'external_side_effect');
});

test('upload requires confirmation', () => {
  const safety = classifyBrowserAction('upload', { filePath: './README.md', selector: 'input[type="file"]' }, { workspaceRoot: process.cwd() });
  assert.equal(safety.decision, 'confirm_required');
  assert.equal(safety.actionClass, 'broad_local_write');
});

test('delete is blocked by default', () => {
  const safety = classifyBrowserAction('delete', { selector: '#danger' }, { workspaceRoot: process.cwd() });
  assert.equal(safety.decision, 'blocked');
  assert.equal(safety.reasonCodes[0], 'destructive-dom-disabled');
});

test('download requires preview and includes workspace output summary', () => {
  const safety = classifyBrowserAction('download', { selector: 'a.download', savePath: './artifacts/report.pdf' }, { workspaceRoot: process.cwd() });
  assert.equal(safety.decision, 'preview_required');
  assert.equal(safety.previewSummary.output.inside_workspace, true);
  assert.match(safety.previewSummary.output.resolved_path, /artifacts[\\/]report\.pdf$/);
});

test('edge session confirmation token unlocks confirmed add action', () => {
  const session = new EdgeSession({
    workspaceRoot: process.cwd(),
    allowDomHtmlAdd: true,
    confirmationSecret: 'test-secret'
  });

  const initial = session.evaluateActionSafety('add', {
    parentSelector: '#root',
    html: '<div>Injected</div>'
  });
  assert.equal(initial.decision, 'confirm_required');
  assert.ok(initial.confirmToken);

  const confirmed = session.evaluateActionSafety('add', {
    parentSelector: '#root',
    html: '<div>Injected</div>',
    confirm: true,
    confirm_token: initial.confirmToken
  });
  assert.equal(confirmed.decision, 'allow_with_verification');
  assert.ok(confirmed.reasonCodes.includes('confirmed'));
});

test('act returns browser safety gate response before requiring an open page', async () => {
  const session = new EdgeSession({
    workspaceRoot: process.cwd(),
    confirmationSecret: 'test-secret'
  });
  const result = await session.act('delete', { selector: '#danger' });
  assert.equal(result.status, 'blocked');
  assert.equal(result.action, 'delete');
  assert.equal(result.action_class, 'destructive');
});
