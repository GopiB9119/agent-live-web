const { spawnSync } = require('child_process');
const path = require('path');

const NETWORK_NAME = 'agent-live-web-tracing-net';
const JAEGER = {
  container: 'agent-live-web-jaeger',
  image: 'jaegertracing/all-in-one:1.62.0'
};
const OTEL = {
  container: 'agent-live-web-otel',
  image: 'otel/opentelemetry-collector:0.111.0'
};

function run(command, args) {
  const result = spawnSync(command, args, {
    stdio: 'pipe',
    encoding: 'utf8'
  });

  return {
    status: result.status === null ? 1 : result.status,
    stdout: result.stdout || '',
    stderr: result.stderr || ''
  };
}

function runOrExit(command, args, failureMessage) {
  const result = run(command, args);
  if (result.status !== 0) {
    const details = [result.stderr.trim(), result.stdout.trim()].filter(Boolean).join('\n');
    console.error(`${failureMessage}${details ? `\n${details}` : ''}`);
    process.exit(result.status || 1);
  }
  return result;
}

function ensureDocker() {
  const result = run('docker', ['info']);
  if (result.status !== 0) {
    const details = [result.stderr.trim(), result.stdout.trim()].filter(Boolean).join('\n');
    console.error(`[TraceStack] Docker daemon is not available.${details ? `\n${details}` : ''}`);
    process.exit(1);
  }
}

function containerExists(container) {
  const result = runOrExit('docker', ['ps', '-a', '--format', '{{.Names}}'], '[TraceStack] Failed to list containers.');
  return result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .includes(container);
}

function ensureNetwork() {
  const inspect = run('docker', ['network', 'inspect', NETWORK_NAME]);
  if (inspect.status === 0) return;
  runOrExit('docker', ['network', 'create', NETWORK_NAME], `[TraceStack] Failed to create network ${NETWORK_NAME}.`);
}

function removeContainerIfPresent(container, label) {
  if (!containerExists(container)) {
    console.log(`[${label}] not running.`);
    return;
  }

  runOrExit('docker', ['rm', '-f', container], `[${label}] failed to stop ${container}.`);
  console.log(`[${label}] stopped: ${container}`);
}

function startJaeger() {
  ensureNetwork();
  if (containerExists(JAEGER.container)) {
    runOrExit('docker', ['rm', '-f', JAEGER.container], `[Jaeger] failed to replace ${JAEGER.container}.`);
  }

  runOrExit(
    'docker',
    [
      'run',
      '-d',
      '--name',
      JAEGER.container,
      '--network',
      NETWORK_NAME,
      '-e',
      'COLLECTOR_OTLP_ENABLED=true',
      '-p',
      '16686:16686',
      JAEGER.image
    ],
    '[Jaeger] failed to start container.'
  );

  console.log('[Jaeger] UI started: http://localhost:16686');
  console.log(`[Jaeger] container: ${JAEGER.container}`);
  console.log(`[Jaeger] network: ${NETWORK_NAME}`);
}

function startCollector() {
  ensureNetwork();
  if (containerExists(OTEL.container)) {
    runOrExit('docker', ['rm', '-f', OTEL.container], `[OTEL] failed to replace ${OTEL.container}.`);
  }

  const configPath = path.resolve(__dirname, 'otel-collector-config.yaml');
  runOrExit(
    'docker',
    [
      'run',
      '-d',
      '--name',
      OTEL.container,
      '--network',
      NETWORK_NAME,
      '-p',
      '4318:4318',
      '-v',
      `${configPath}:/etc/otelcol/config.yaml:ro`,
      OTEL.image
    ],
    '[OTEL] failed to start collector.'
  );

  console.log(`[OTEL] collector started: ${OTEL.container}`);
  console.log('[OTEL] endpoint: http://localhost:4318/v1/traces');
  console.log(`[OTEL] network: ${NETWORK_NAME}`);
}

function usage() {
  console.error('Usage: node scripts/trace-stack-control.js <jaeger|collector|stack> <start|stop>');
  process.exit(1);
}

function main() {
  const target = (process.argv[2] || '').trim();
  const action = (process.argv[3] || '').trim();

  if (!target || !action) {
    usage();
    return;
  }

  ensureDocker();

  if (target === 'jaeger' && action === 'start') {
    startJaeger();
    return;
  }

  if (target === 'jaeger' && action === 'stop') {
    removeContainerIfPresent(JAEGER.container, 'Jaeger');
    return;
  }

  if (target === 'collector' && action === 'start') {
    startCollector();
    return;
  }

  if (target === 'collector' && action === 'stop') {
    removeContainerIfPresent(OTEL.container, 'OTEL');
    return;
  }

  if (target === 'stack' && action === 'start') {
    startJaeger();
    startCollector();
    return;
  }

  if (target === 'stack' && action === 'stop') {
    removeContainerIfPresent(OTEL.container, 'OTEL');
    removeContainerIfPresent(JAEGER.container, 'Jaeger');
    return;
  }

  usage();
}

main();
