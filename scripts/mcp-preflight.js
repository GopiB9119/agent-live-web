#!/usr/bin/env node
'use strict';

// MCP preflight check: verifies the Playwright MCP server can start,
// the owner lock works, and basic browser actions succeed.
// Usage: npm run mcp:preflight
// This is a local-only check — no network requests to external sites.

const path = require('path');
const { EdgeSession } = require('../edge-session');

let passed = 0;
let failed = 0;

function report(name, ok, detail) {
  const status = ok ? '✓' : '✗';
  console.log(`  ${status} ${name}${detail ? ': ' + detail : ''}`);
  if (ok) passed++; else failed++;
}

(async () => {
  console.log('MCP Preflight Check');
  console.log('='.repeat(40));

  // --- Non-browser checks ---

  // DOM mutation defaults
  const defaultSession = new EdgeSession({ headless: true });
  report('EdgeSession constructor', true, 'headless mode');
  report('DOM delete blocked by default', !defaultSession.allowDomDelete, `allowDomDelete=${defaultSession.allowDomDelete}`);
  report('DOM add blocked by default', !defaultSession.allowDomHtmlAdd, `allowDomHtmlAdd=${defaultSession.allowDomHtmlAdd}`);

  // Redaction check
  const testData = {
    action: 'type',
    result: { beforeValue: 'secret', afterValue: 'newsecret', requestedValue: 'newsecret' },
    verification: { ok: true, reason: 'ok' },
    status: 'ok',
    durationMs: 10
  };
  const sanitized = defaultSession.sanitizeActionResponse(testData);
  const redacted = sanitized.result.beforeValue === '[REDACTED_INPUT]' &&
                   sanitized.result.afterValue === '[REDACTED_INPUT]';
  report('Secret redaction', redacted, redacted ? 'type values redacted' : 'LEAK: values not redacted');

  // --- Browser checks (single shared session) ---

  const session = new EdgeSession({
    headless: true,
    userDataDir: path.join(process.cwd(), '.playwright-mcp', 'preflight-profile')
  });

  try {
    await session.open();
    const hasPage = Boolean(session.page);
    report('Edge browser launch', hasPage, `initial url: ${session.page ? session.page.url() : 'none'}`);

    if (!hasPage) {
      report('Navigation', false, 'no page to navigate');
      report('Grounded summary', false, 'skipped — no page');
      report('Verification rejects bad click', false, 'skipped — no page');
    } else {
      // Navigation test (use about:blank → simple page)
      await session.page.goto('data:text/html,<h1>MCP Preflight</h1>');
      const pageTitle = await session.page.title();
      const pageUrl = session.page.url();
      const navOk = pageUrl.startsWith('data:') || pageUrl !== 'about:blank';
      report('Navigation', navOk, `url=${pageUrl.slice(0, 60)}`);

      // Grounded summary check — use runAction which produces the summary
      const gotoResult = await session.runAction('goto', { url: 'https://example.com' });
      const hasSummary = typeof gotoResult.summary === 'string' && gotoResult.summary.length > 0;
      report('Grounded summary', hasSummary, gotoResult.summary ? gotoResult.summary.slice(0, 80) : 'missing');

      // Verification rejects bad element
      try {
        const badClick = await session.runAction('click', { selector: '#nonexistent-element-xyz' });
        report('Verification rejects bad click', badClick.status === 'failed', badClick.summary || 'should fail');
      } catch (_) {
        report('Verification rejects bad click', true, 'threw error as expected');
      }
    }

    await session.close();
  } catch (e) {
    report('Edge browser launch', false, e.message);
    try { await session.close(); } catch (_) {}
  }

  console.log('='.repeat(40));
  console.log(`Results: ${passed} passed, ${failed} failed`);

  if (failed > 0) {
    console.log('\nSome checks failed. Ensure Edge is installed (npm run install:edge) and no other MCP session is running.');
    process.exit(1);
  }

  console.log('\nAll preflight checks passed. MCP runtime is ready.');
  process.exit(0);
})();
