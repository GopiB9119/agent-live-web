const http = require('http');
const https = require('https');
const { URL } = require('url');

function toInt(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function requestJson(urlString, timeoutMs) {
  const target = new URL(urlString);
  const transport = target.protocol === 'https:' ? https : http;

  return new Promise((resolve, reject) => {
    const request = transport.request(
      {
        method: 'GET',
        hostname: target.hostname,
        port: target.port || (target.protocol === 'https:' ? 443 : 80),
        path: `${target.pathname}${target.search}`,
        timeout: timeoutMs
      },
      (response) => {
        let raw = '';
        response.setEncoding('utf8');
        response.on('data', (chunk) => {
          raw += chunk;
        });
        response.on('end', () => {
          if (response.statusCode < 200 || response.statusCode >= 300) {
            reject(new Error(`Request failed with status ${response.statusCode}: ${raw.slice(0, 240)}`));
            return;
          }
          try {
            resolve(JSON.parse(raw));
          } catch (error) {
            reject(new Error(`Invalid JSON response: ${error.message}`));
          }
        });
      }
    );

    request.on('timeout', () => {
      request.destroy(new Error(`timeout after ${timeoutMs}ms`));
    });

    request.on('error', (error) => {
      reject(error);
    });

    request.end();
  });
}

function getLogEventNames(span) {
  const logs = Array.isArray(span.logs) ? span.logs : [];
  const events = [];
  for (const log of logs) {
    const fields = Array.isArray(log.fields) ? log.fields : [];
    const eventField = fields.find((field) => field.key === 'event' && typeof field.value === 'string');
    if (eventField) events.push(eventField.value);
  }
  return events;
}

function hasManualEvent(span) {
  const events = getLogEventNames(span);
  return events.includes('ui-refresh-trace') || events.includes('manual_refresh_trace');
}

function getTagValue(span, key) {
  const tags = Array.isArray(span.tags) ? span.tags : [];
  const tag = tags.find((item) => item.key === key);
  return tag ? tag.value : undefined;
}

function microsToIso(value) {
  const millis = Math.floor(Number(value) / 1000);
  return Number.isFinite(millis) ? new Date(millis).toISOString() : 'unknown';
}

function printTrace(trace, span, processMap) {
  const processMeta = processMap && span.processID ? processMap[span.processID] : null;
  const serviceName = processMeta && processMeta.serviceName ? processMeta.serviceName : 'unknown';
  const status = getTagValue(span, 'otel.status_code') || 'UNSET';
  const owner = getTagValue(span, 'app.mcp.owner') || 'unknown';
  const durationMs = (Number(span.duration || 0) / 1000).toFixed(2);
  const startTime = microsToIso(span.startTime);

  console.log(`trace=${trace.traceID}`);
  console.log(`  service=${serviceName}`);
  console.log(`  operation=${span.operationName}`);
  console.log(`  status=${status}`);
  console.log(`  owner=${owner}`);
  console.log(`  duration_ms=${durationMs}`);
  console.log(`  start=${startTime}`);
}

(async () => {
  const service = String(process.env.TRACE_SERVICE || 'agent-live-web-vscode-mcp').trim();
  const operation = String(process.env.TRACE_OPERATION || 'mcp.server.launch').trim();
  const lookback = String(process.env.TRACE_LOOKBACK || '1h').trim();
  const limit = toInt(process.env.TRACE_LIMIT || 50, 50);
  const timeoutMs = toInt(process.env.TRACE_REQUEST_TIMEOUT_MS || 7000, 7000);
  const jaegerBase = String(process.env.JAEGER_BASE_URL || 'http://localhost:16686').trim().replace(/\/+$/, '');

  const query = new URLSearchParams({ service, operation, limit: String(limit), lookback });
  const requestUrl = `${jaegerBase}/api/traces?${query.toString()}`;

  console.log(`[TraceLatest] service=${service} operation=${operation} lookback=${lookback} limit=${limit}`);

  let payload;
  try {
    payload = await requestJson(requestUrl, timeoutMs);
  } catch (error) {
    console.error(`[TraceLatest] query failed: ${error.message}`);
    process.exit(1);
    return;
  }

  const traces = Array.isArray(payload.data) ? payload.data : [];
  const runtimeResults = [];

  for (const traceEntry of traces) {
    const spans = Array.isArray(traceEntry.spans) ? traceEntry.spans : [];
    const processMap = traceEntry.processes || {};

    for (const span of spans) {
      if (span.operationName !== operation) continue;
      if (hasManualEvent(span)) continue;
      runtimeResults.push({ trace: traceEntry, span, processMap });
    }
  }

  runtimeResults.sort((a, b) => Number(b.span.startTime || 0) - Number(a.span.startTime || 0));

  if (!runtimeResults.length) {
    console.log('[TraceLatest] No runtime traces found (manual smoke traces excluded).');
    process.exit(0);
    return;
  }

  console.log(`[TraceLatest] runtime_traces=${runtimeResults.length}`);
  for (const item of runtimeResults.slice(0, limit)) {
    printTrace(item.trace, item.span, item.processMap);
  }
})();
