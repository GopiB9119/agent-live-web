const { context, trace, SpanStatusCode } = require('@opentelemetry/api');
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { OTLPTraceExporter } = require('@opentelemetry/exporter-trace-otlp-http');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { resourceFromAttributes } = require('@opentelemetry/resources');
const { ATTR_SERVICE_NAME, ATTR_SERVICE_VERSION } = require('@opentelemetry/semantic-conventions');

let sdk = null;
let initialized = false;
let enabled = false;

function toBool(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on';
}

function safeSetAttributes(span, attributes = {}) {
  if (!span || typeof span.setAttribute !== 'function') return;
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === null) continue;
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      span.setAttribute(key, value);
    } else {
      span.setAttribute(key, JSON.stringify(value));
    }
  }
}

function getTracer() {
  return trace.getTracer('agent-live-web');
}

async function initTracing(serviceName = 'agent-live-web') {
  if (initialized) return { enabled };
  initialized = true;

  enabled = toBool(process.env.EDGE_TRACING_ENABLED, false);
  if (!enabled) return { enabled: false };

  const traceEndpoint =
    String(process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT || '').trim() ||
    String(process.env.OTEL_EXPORTER_OTLP_ENDPOINT || '').trim() ||
    'http://localhost:4318/v1/traces';

  try {
    const traceExporter = new OTLPTraceExporter({
      url: traceEndpoint
    });

    sdk = new NodeSDK({
      resource: resourceFromAttributes({
        [ATTR_SERVICE_NAME]: process.env.EDGE_TRACING_SERVICE_NAME || serviceName,
        [ATTR_SERVICE_VERSION]: process.env.npm_package_version || '1.0.0'
      }),
      traceExporter,
      instrumentations: [
        getNodeAutoInstrumentations({
          '@opentelemetry/instrumentation-fs': { enabled: false }
        })
      ]
    });

    await Promise.resolve(sdk.start());
    return { enabled: true, endpoint: traceEndpoint };
  } catch (error) {
    enabled = false;
    return { enabled: false, error: error.message };
  }
}

function recordException(span, error) {
  if (!span || !error) return;
  try {
    span.recordException(error);
    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message || String(error) });
  } catch (_) {
    // best effort
  }
}

async function runInSpan(name, attributes = {}, operation) {
  const tracer = getTracer();
  return await tracer.startActiveSpan(name, async (span) => {
    safeSetAttributes(span, attributes);
    try {
      const result = await operation(span);
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (error) {
      recordException(span, error);
      throw error;
    } finally {
      span.end();
    }
  });
}

function getActiveTraceMeta() {
  const activeSpan = trace.getSpan(context.active());
  if (!activeSpan) return null;
  const spanContext = activeSpan.spanContext();
  if (!spanContext || !spanContext.traceId || !spanContext.spanId) return null;
  return {
    traceId: spanContext.traceId,
    spanId: spanContext.spanId
  };
}

async function shutdownTracing() {
  if (!sdk) return;
  try {
    await Promise.resolve(sdk.shutdown());
  } catch (_) {
    // best effort
  } finally {
    sdk = null;
  }
}

module.exports = {
  initTracing,
  runInSpan,
  recordException,
  getActiveTraceMeta,
  shutdownTracing,
  isTracingEnabled: () => enabled
};
