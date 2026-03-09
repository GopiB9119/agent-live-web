const test = require('node:test');
const assert = require('node:assert/strict');

const { buildPlaywrightMcpArgs, normalizeCaps } = require('../playwright-mcp-launch-config');

test('normalizeCaps disables caps for none', () => {
  assert.equal(normalizeCaps('none', 'vision,pdf'), '');
  assert.equal(normalizeCaps('', 'vision,pdf'), '');
});

test('buildPlaywrightMcpArgs includes full launch options', () => {
  const args = buildPlaywrightMcpArgs({
    browserChannel: 'msedge',
    outputDir: '.playwright-mcp/output',
    outputMode: 'stdout',
    consoleLevel: 'error',
    snapshotMode: 'incremental',
    timeoutActionMs: '18000',
    timeoutNavigationMs: '90000',
    caps: 'vision,pdf',
    sharedBrowserContext: true,
    headless: true,
    persistProfile: true,
    userDataDir: '.playwright-mcp/profile',
    initPageEnabled: true,
    initPagePath: 'scripts/mcp-init-page.js'
  });

  assert.deepEqual(args.slice(0, 4), ['playwright', 'run-mcp-server', '--browser', 'msedge']);
  assert.ok(args.includes('--caps'));
  assert.ok(args.includes('vision,pdf'));
  assert.ok(args.includes('--shared-browser-context'));
  assert.ok(args.includes('--headless'));
  assert.ok(args.includes('--user-data-dir'));
  assert.ok(args.includes('.playwright-mcp/profile'));
  assert.ok(args.includes('--init-page'));
});

test('buildPlaywrightMcpArgs supports minimal isolated launch', () => {
  const args = buildPlaywrightMcpArgs({
    browserChannel: 'chrome',
    outputDir: '.playwright-mcp/output',
    caps: '',
    sharedBrowserContext: false,
    headless: true,
    persistProfile: false,
    userDataDir: '.playwright-mcp/profile',
    initPageEnabled: false,
    initPagePath: 'scripts/mcp-init-page.js'
  });

  assert.ok(!args.includes('--caps'));
  assert.ok(!args.includes('--shared-browser-context'));
  assert.ok(args.includes('--headless'));
  assert.ok(args.includes('--isolated'));
  assert.ok(!args.includes('--user-data-dir'));
  assert.ok(!args.includes('--init-page'));
});
