const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

const {
  AGENT_PROXY_STATUS_TOOL_NAME,
  PolicyDecision,
  augmentMcpToolDefinitions,
  augmentMcpToolResult,
  buildMcpSafetyToolResult,
  buildProxyStatusToolResult,
  captureMcpPageState,
  buildMcpEvidenceSummary,
  evaluateMcpToolCall,
  parseMcpTabsText,
  shouldRetryMcpExecution,
  verifyMcpExecution,
  writeMcpSafetyEvent
} = require('../mcp-safety-adapter');

test('augmentMcpToolDefinitions adds confirmation fields to browser tools', () => {
  const tools = augmentMcpToolDefinitions([
    {
      name: 'browser_click',
      inputSchema: {
        type: 'object',
        properties: {
          element: { type: 'string' }
        }
      }
    }
  ]);

  assert.equal(typeof tools[0].inputSchema.properties.confirm, 'object');
  assert.equal(typeof tools[0].inputSchema.properties.confirm_token, 'object');
});

test('augmentMcpToolDefinitions appends agent proxy status tool', () => {
  const tools = augmentMcpToolDefinitions([]);
  const statusTool = tools.find((tool) => tool.name === AGENT_PROXY_STATUS_TOOL_NAME);

  assert.ok(statusTool);
  assert.equal(statusTool.inputSchema.type, 'object');
  assert.equal(statusTool.inputSchema.additionalProperties, false);
});

test('evaluateMcpToolCall requires confirmation for dangerous click intent', () => {
  const result = evaluateMcpToolCall(
    'browser_click',
    { element: 'text:Delete account' },
    { workspaceRoot: process.cwd(), confirmationSecret: 'proxy-secret' }
  );

  assert.equal(result.decision, PolicyDecision.CONFIRM_REQUIRED);
  assert.ok(result.confirmToken);
});

test('evaluateMcpToolCall returns preview for closing a tab', () => {
  const result = evaluateMcpToolCall(
    'browser_tabs',
    { action: 'close', index: 2 },
    { workspaceRoot: process.cwd(), confirmationSecret: 'proxy-secret' }
  );

  assert.equal(result.decision, PolicyDecision.PREVIEW_REQUIRED);
  assert.equal(result.previewSummary.tab_action, 'close');
});

test('evaluateMcpToolCall confirms upload after token round-trip', () => {
  const first = evaluateMcpToolCall(
    'browser_file_upload',
    { paths: ['./README.md'], element: 'input[type=file]' },
    { workspaceRoot: process.cwd(), confirmationSecret: 'proxy-secret' }
  );

  const second = evaluateMcpToolCall(
    'browser_file_upload',
    {
      paths: ['./README.md'],
      element: 'input[type=file]',
      confirm: true,
      confirm_token: first.confirmToken
    },
    { workspaceRoot: process.cwd(), confirmationSecret: 'proxy-secret' }
  );

  assert.equal(first.decision, PolicyDecision.CONFIRM_REQUIRED);
  assert.equal(second.decision, PolicyDecision.ALLOW_WITH_VERIFICATION);
});

test('buildMcpSafetyToolResult returns a structured blocked tool result', () => {
  const evaluation = evaluateMcpToolCall(
    'browser_evaluate',
    { expression: 'document.body.remove()' },
    { workspaceRoot: process.cwd(), confirmationSecret: 'proxy-secret', allowBrowserCodeExecution: false }
  );

  const result = buildMcpSafetyToolResult('browser_evaluate', { expression: 'document.body.remove()' }, evaluation);
  assert.equal(result.isError, true);
  assert.equal(result.structuredContent.status, PolicyDecision.BLOCKED);
  assert.equal(result.structuredContent.tool, 'browser_evaluate');
});

test('buildProxyStatusToolResult reports direct MCP trust summary', () => {
  const result = buildProxyStatusToolResult({
    runtime_status: 'ready',
    startup_trust: 'direct MCP proxy',
    resume_state: 'not applicable in direct MCP mode',
    summary: 'Guarded direct MCP mode is ready; resume state is not applicable in direct MCP mode.'
  });

  assert.equal(result.isError, false);
  assert.equal(result.structuredContent.tool, AGENT_PROXY_STATUS_TOOL_NAME);
  assert.equal(result.structuredContent.runtime_status, 'ready');
  assert.equal(result.structuredContent.resume_state, 'not applicable in direct MCP mode');
  assert.match(result.content[0].text, /Guarded direct MCP mode is ready/);
});

test('augmentMcpToolResult adds safety and evidence summaries', () => {
  const evaluation = evaluateMcpToolCall(
    'browser_navigate',
    { url: 'https://example.com' },
    { workspaceRoot: process.cwd(), confirmationSecret: 'proxy-secret' }
  );

  const result = augmentMcpToolResult(
    'browser_navigate',
    { url: 'https://example.com' },
    {
      content: [{ type: 'text', text: 'Navigated to https://example.com' }],
      structuredContent: { url: 'https://example.com' },
      isError: false
    },
    evaluation
  );

  assert.equal(result.structuredContent.safety.tool, 'browser_navigate');
  assert.equal(result.structuredContent.evidence.status, 'reported_ok');
  assert.match(result.content[result.content.length - 1].text, /reported as completed/);
});

test('augmentMcpToolResult carries execution metadata', () => {
  const result = augmentMcpToolResult(
    'browser_click',
    { element: 'text:Save' },
    { content: [{ type: 'text', text: 'Clicked Save' }], structuredContent: {}, isError: false },
    null,
    { ok: true, reason: 'verified', details: {} },
    { attempts: 2, recovered: true }
  );

  assert.equal(result.structuredContent.execution.attempts, 2);
  assert.equal(result.structuredContent.execution.recovered, true);
});

test('buildMcpEvidenceSummary reports artifact path for screenshot tools', () => {
  const evidence = buildMcpEvidenceSummary(
    'browser_take_screenshot',
    { path: './artifacts/page.png' },
    { content: [{ type: 'text', text: 'Saved screenshot' }], isError: false }
  );

  assert.equal(evidence.output_path, './artifacts/page.png');
  assert.equal(evidence.status, 'reported_ok');
});

test('buildMcpEvidenceSummary prefers child-reported artifact path over requested screenshot path', () => {
  const evidence = buildMcpEvidenceSummary(
    'browser_take_screenshot',
    { path: 'C:\\tmp\\requested.png' },
    {
      content: [
        {
          type: 'text',
          text: '### Result\n- [Screenshot of viewport](.playwright-mcp\\\\output\\\\page-123.png)\n'
        }
      ],
      isError: false
    }
  );

  assert.equal(evidence.output_path, '.playwright-mcp\\\\output\\\\page-123.png');
});

test('writeMcpSafetyEvent appends redacted jsonl records', () => {
  const auditDir = path.join(process.cwd(), '.tmp', 'mcp-safety-tests');
  fs.mkdirSync(auditDir, { recursive: true });
  const auditFile = path.join(auditDir, 'safety-events.jsonl');
  try {
    fs.rmSync(auditFile, { force: true });
  } catch (_) {
    // best effort
  }

  const wrote = writeMcpSafetyEvent(
    process.cwd(),
    {
      event_type: 'decision',
      tool: 'browser_click',
      arguments_summary: { element: 'Delete account', bearer_token: 'secret-token-value-1234567890' }
    },
    {
      auditEnabled: true,
      auditFile,
      owner: 'vscode'
    }
  );

  assert.equal(wrote, true);
  const lines = fs.readFileSync(auditFile, 'utf8').trim().split(/\r?\n/);
  assert.equal(lines.length, 1);
  const parsed = JSON.parse(lines[0]);
  assert.equal(parsed.source, 'mcp_proxy');
  assert.equal(parsed.owner, 'vscode');
  assert.equal(parsed.arguments_summary.bearer_token, '[REDACTED]');
});

test('parseMcpTabsText parses current tab state', () => {
  const tabs = parseMcpTabsText('- 0: (current) [Example](https://example.com)\n- 1: [Blank](about:blank)');
  assert.equal(tabs.length, 2);
  assert.equal(tabs[0].current, true);
  assert.equal(tabs[1].url, 'about:blank');
});

test('captureMcpPageState captures snapshot hash when enabled', async () => {
  const state = await captureMcpPageState(
    {
      callTool: async (toolName) => {
        if (toolName === 'browser_tabs') {
          return {
            ok: true,
            result: {
              content: [{ type: 'text', text: '- 0: (current) [Example](https://example.com)' }],
              isError: false
            }
          };
        }
        return {
          ok: true,
          result: {
            content: [{ type: 'text', text: '<html>snapshot</html>' }],
            isError: false
          }
        };
      }
    },
    { includeSnapshot: true }
  );

  assert.equal(state.url, 'https://example.com');
  assert.equal(typeof state.snapshot_hash, 'string');
  assert.ok(state.snapshot_hash.length > 0);
});

test('verifyMcpExecution verifies browser_navigate with live tab state', async () => {
  const verification = await verifyMcpExecution(
    'browser_navigate',
    { url: 'https://example.com' },
    { isError: false },
    {
      callTool: async () => ({
        ok: true,
        result: {
          content: [{ type: 'text', text: '- 0: (current) [Example](https://www.example.com/path)' }],
          isError: false
        }
      })
    }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /Navigation verified/);
});

test('verifyMcpExecution verifies screenshot artifact on disk', async () => {
  const artifactDir = path.join(process.cwd(), '.tmp', 'mcp-safety-tests');
  fs.mkdirSync(artifactDir, { recursive: true });
  const artifactPath = path.join(artifactDir, 'page.png');
  fs.writeFileSync(artifactPath, 'png', 'utf8');

  try {
    const verification = await verifyMcpExecution(
      'browser_take_screenshot',
      { path: artifactPath },
      { isError: false, structuredContent: { path: artifactPath } },
      {}
    );

    assert.equal(verification.ok, true);
    assert.match(verification.reason, /Artifact verified/);
  } finally {
    try {
      fs.rmSync(artifactPath, { force: true });
    } catch (_) {
      // best effort
    }
  }
});

test('verifyMcpExecution uses snapshot hash for state-changing tools', async () => {
  const verification = await verifyMcpExecution(
    'browser_click',
    { element: 'text:Save' },
    { isError: false },
    {
      callTool: async (toolName) => {
        if (toolName === 'browser_tabs') {
          return {
            ok: true,
            result: {
              content: [{ type: 'text', text: '- 0: (current) [Example](https://example.com/settings)' }],
              isError: false
            }
          };
        }
        return {
          ok: true,
          result: {
            content: [{ type: 'text', text: '<html>after</html>' }],
            isError: false
          }
        };
      },
      verifyWithSnapshot: true,
      stateChangeTools: new Set(['browser_click'])
    },
    {
      url: 'https://example.com/settings',
      snapshot_hash: 'before-hash'
    }
  );

  assert.equal(verification.ok, true);
  assert.match(verification.reason, /Page state changed|active tab is valid/);
});

test('shouldRetryMcpExecution retries retryable tool on failed verification', () => {
  const retry = shouldRetryMcpExecution(
    'browser_click',
    { isError: false },
    { ok: false, reason: 'verification failed' },
    {}
  );
  assert.equal(retry, true);
});

test('shouldRetryMcpExecution does not retry non-retryable tool', () => {
  const retry = shouldRetryMcpExecution(
    'browser_tabs',
    { isError: true },
    { ok: false, reason: 'failed' },
    {}
  );
  assert.equal(retry, false);
});
