const http = require('http');
const https = require('https');
const { URL } = require('url');

function toBool(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on';
}

function getTracingEndpoint() {
  return (
    String(process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT || '').trim() ||
    String(process.env.OTEL_EXPORTER_OTLP_ENDPOINT || '').trim() ||
    'http://localhost:4318/v1/traces'
  );
}

async function checkEndpoint(rawEndpoint, timeoutMs) {
  const endpoint = new URL(rawEndpoint);
  const transport = endpoint.protocol === 'https:' ? https : http;
  const options = {
    method: 'HEAD',
    hostname: endpoint.hostname,
    port: endpoint.port || (endpoint.protocol === 'https:' ? 443 : 80),
    path: endpoint.pathname || '/',
    timeout: timeoutMs
  };

  return await new Promise((resolve) => {
    const request = transport.request(options, (response) => {
      response.resume();
      resolve({
        ok: response.statusCode >= 200 && response.statusCode < 500,
        statusCode: response.statusCode,
        statusMessage: response.statusMessage
      });
    });

    request.on('timeout', () => {
      request.destroy(new Error(`timeout after ${timeoutMs}ms`));
    });

    request.on('error', (error) => {
      resolve({ ok: false, error: error.message });
    });

    request.end();
  });
}

async function probePostEndpoint(rawEndpoint, timeoutMs) {
  const endpoint = new URL(rawEndpoint);
  const transport = endpoint.protocol === 'https:' ? https : http;
  const body = '{}';
  const options = {
    method: 'POST',
    hostname: endpoint.hostname,
    port: endpoint.port || (endpoint.protocol === 'https:' ? 443 : 80),
    path: endpoint.pathname || '/',
    timeout: timeoutMs,
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body)
    }
  };

  return await new Promise((resolve) => {
    const request = transport.request(options, (response) => {
      response.resume();
      resolve({
        ok: response.statusCode >= 200 && response.statusCode < 500,
        statusCode: response.statusCode,
        statusMessage: response.statusMessage
      });
    });

    request.on('timeout', () => {
      request.destroy(new Error(`timeout after ${timeoutMs}ms`));
    });

    request.on('error', (error) => {
      resolve({ ok: false, error: error.message });
    });

    request.write(body);
    request.end();
  });
}

(async () => {
  const tracingEnabled = toBool(process.env.EDGE_TRACING_ENABLED, true);
  const endpoint = getTracingEndpoint();
  const timeoutMs = Number(process.env.EDGE_TRACING_HEALTH_TIMEOUT_MS || 4000);

  console.log('[TracingCheck] Starting trace health check...');
  console.log(`[TracingCheck] EDGE_TRACING_ENABLED=${tracingEnabled}`);
  console.log(`[TracingCheck] Endpoint=${endpoint}`);

  if (!tracingEnabled) {
    console.log('[TracingCheck] Tracing is disabled by env. Nothing to validate.');
    process.exit(0);
    return;
  }

  let result;
  let postResult;
  try {
    result = await checkEndpoint(endpoint, Number.isFinite(timeoutMs) ? timeoutMs : 4000);
    postResult = await probePostEndpoint(endpoint, Number.isFinite(timeoutMs) ? timeoutMs : 4000);
  } catch (error) {
    console.error(`[TracingCheck] Failed to check endpoint: ${error.message}`);
    process.exit(1);
    return;
  }

  if (!result.ok) {
    console.error(
      `[TracingCheck] Endpoint not reachable. ${result.error ? `Error=${result.error}` : 'No response metadata.'}`
    );
    process.exit(2);
    return;
  }

  const endpointPath = new URL(endpoint).pathname || '/';

  if (postResult && !postResult.ok) {
    console.error(
      `[TracingCheck] Endpoint path may be invalid. POST probe failed. ${postResult.error ? `Error=${postResult.error}` : 'No response metadata.'}`
    );
    process.exit(3);
    return;
  }

  if (postResult && postResult.statusCode === 404) {
    console.error('[TracingCheck] Endpoint returned 404 on POST.');
    console.error('[TracingCheck] Use OTLP traces path, for example: http://localhost:4318/v1/traces');
    process.exit(4);
    return;
  }

  if (endpointPath === '/' || endpointPath === '') {
    console.error('[TracingCheck] Endpoint points to collector root (`/`), which commonly returns `404 page not found`.');
    console.error('[TracingCheck] Set OTLP traces endpoint to: http://localhost:4318/v1/traces');
    process.exit(5);
    return;
  }

  console.log(
    `[TracingCheck] Endpoint reachable. status=${result.statusCode}${result.statusMessage ? ` ${result.statusMessage}` : ''}`
  );
  if (postResult) {
    console.log(
      `[TracingCheck] POST probe status=${postResult.statusCode}${postResult.statusMessage ? ` ${postResult.statusMessage}` : ''}`
    );
  }
  console.log('[TracingCheck] Trace exporter connectivity looks healthy.');
  process.exit(0);
})();
