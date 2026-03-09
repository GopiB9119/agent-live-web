function toBool(value, fallback = false) {
  if (value === undefined || value === null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on';
}

function normalizeCaps(value, fallback = 'vision,pdf') {
  if (value === undefined || value === null) return fallback;
  const normalized = String(value).trim();
  if (!normalized) return '';
  if (normalized.toLowerCase() === 'none') return '';
  return normalized;
}

function buildPlaywrightMcpArgs(options = {}) {
  const args = [
    'playwright',
    'run-mcp-server',
    '--browser',
    String(options.browserChannel || 'msedge').trim() || 'msedge',
    '--output-dir',
    options.outputDir,
    '--output-mode',
    String(options.outputMode || 'stdout').trim() || 'stdout',
    '--console-level',
    String(options.consoleLevel || 'error').trim() || 'error',
    '--snapshot-mode',
    String(options.snapshotMode || 'incremental').trim() || 'incremental',
    '--timeout-action',
    String(options.timeoutActionMs || '18000').trim() || '18000',
    '--timeout-navigation',
    String(options.timeoutNavigationMs || '90000').trim() || '90000'
  ];

  const caps = normalizeCaps(options.caps);
  if (caps) {
    args.push('--caps', caps);
  }

  if (options.sharedBrowserContext) {
    args.push('--shared-browser-context');
  }
  if (options.headless) {
    args.push('--headless');
  }

  const cdpEndpoint = String(options.cdpEndpoint || '').trim();
  if (cdpEndpoint) {
    args.push('--cdp-endpoint', cdpEndpoint);
  } else if (options.persistProfile) {
    args.push('--user-data-dir', options.userDataDir);
  } else {
    args.push('--isolated');
  }

  if (options.saveSession) args.push('--save-session');
  if (options.saveTrace) args.push('--save-trace');

  const allowedHosts = String(options.allowedHosts || '').trim();
  if (allowedHosts) args.push('--allowed-hosts', allowedHosts);

  const allowedOrigins = String(options.allowedOrigins || '').trim();
  if (allowedOrigins) args.push('--allowed-origins', allowedOrigins);

  const blockedOrigins = String(options.blockedOrigins || '').trim();
  if (blockedOrigins) args.push('--blocked-origins', blockedOrigins);

  if (options.blockServiceWorkers) {
    args.push('--block-service-workers');
  }

  const initPagePath = String(options.initPagePath || '').trim();
  if (options.initPageEnabled && initPagePath) {
    args.push('--init-page', initPagePath);
  }

  if (Array.isArray(options.extraArgs) && options.extraArgs.length) {
    args.push(...options.extraArgs);
  }

  return args;
}

module.exports = {
  buildPlaywrightMcpArgs,
  normalizeCaps,
  toBool
};
