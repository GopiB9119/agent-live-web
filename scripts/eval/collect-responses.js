const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const evalDir = path.join(process.cwd(), 'eval');
const queriesPath = path.join(evalDir, 'queries.json');
const responsesPath = path.join(evalDir, 'responses.json');

function runCommand(command, args, extraEnv = {}) {
  return new Promise((resolve) => {
    const startedAt = Date.now();
    const child = spawn(command, args, {
      cwd: process.cwd(),
      env: {
        ...process.env,
        ...extraEnv
      },
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('close', (code) => {
      const finishedAt = Date.now();
      resolve({
        exitCode: Number.isInteger(code) ? code : 1,
        durationMs: finishedAt - startedAt,
        stdout,
        stderr,
        startedAt: new Date(startedAt).toISOString(),
        finishedAt: new Date(finishedAt).toISOString()
      });
    });

    child.on('error', (error) => {
      const finishedAt = Date.now();
      resolve({
        exitCode: 1,
        durationMs: finishedAt - startedAt,
        stdout,
        stderr: `${stderr}\n${error.message}`.trim(),
        startedAt: new Date(startedAt).toISOString(),
        finishedAt: new Date(finishedAt).toISOString()
      });
    });
  });
}

function extractDurations(stdout) {
  const matches = [...stdout.matchAll(/duration_ms=([0-9]+(?:\.[0-9]+)?)/g)];
  return matches.map((match) => Number(match[1])).filter((value) => Number.isFinite(value));
}

function extractRuntimeTraceCount(stdout) {
  const match = stdout.match(/runtime_traces=([0-9]+)/);
  if (!match) return 0;
  const count = Number(match[1]);
  return Number.isFinite(count) ? count : 0;
}

function toExpectedNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

async function main() {
  if (!fs.existsSync(queriesPath)) {
    console.error(`[EvalCollect] Missing dataset: ${queriesPath}`);
    console.error('[EvalCollect] Run: npm run eval:dataset');
    process.exit(1);
  }

  const queries = JSON.parse(fs.readFileSync(queriesPath, 'utf8'));
  if (!Array.isArray(queries) || !queries.length) {
    console.error('[EvalCollect] queries.json is empty or invalid.');
    process.exit(1);
  }

  const responses = [];

  for (const query of queries) {
    const command = String(query.command || '').trim();
    const args = Array.isArray(query.args) ? query.args : [];

    if (!command) {
      responses.push({
        id: query.id || 'unknown',
        name: query.name || 'unknown',
        ok: false,
        error: 'Missing command',
        expectedExitCode: query.expectedExitCode,
        actualExitCode: 1,
        durationMs: 0,
        runtimeTraceCount: 0,
        launchDurationsMs: []
      });
      continue;
    }

    const env = query.env && typeof query.env === 'object' ? query.env : {};
    console.log(`[EvalCollect] Running: ${command} ${args.join(' ')}`);
    const result = await runCommand(command, args, env);

    const expectedExitCode = Number.isInteger(query.expectedExitCode) ? query.expectedExitCode : 0;
    const runtimeTraceCount = extractRuntimeTraceCount(result.stdout);
    const minRuntimeTraces = toExpectedNumber(query.minRuntimeTraces, 0);
    const maxRuntimeTraces = toExpectedNumber(query.maxRuntimeTraces, Number.POSITIVE_INFINITY);
    const launchDurationsMs = extractDurations(result.stdout);

    const exitOk = result.exitCode === expectedExitCode;
    const minTraceOk = runtimeTraceCount >= minRuntimeTraces;
    const maxTraceOk = runtimeTraceCount <= maxRuntimeTraces;
    const ok = exitOk && minTraceOk && maxTraceOk;

    responses.push({
      id: query.id || 'unknown',
      name: query.name || 'unknown',
      category: query.category || 'general',
      expectedExitCode,
      actualExitCode: result.exitCode,
      minRuntimeTraces,
      maxRuntimeTraces: Number.isFinite(maxRuntimeTraces) ? maxRuntimeTraces : null,
      ok,
      checks: {
        exitOk,
        minTraceOk,
        maxTraceOk
      },
      durationMs: result.durationMs,
      runtimeTraceCount,
      launchDurationsMs,
      startedAt: result.startedAt,
      finishedAt: result.finishedAt,
      env,
      stdout: result.stdout,
      stderr: result.stderr
    });
  }

  fs.mkdirSync(evalDir, { recursive: true });
  fs.writeFileSync(responsesPath, JSON.stringify(responses, null, 2), 'utf8');

  console.log(`[EvalCollect] Wrote responses: ${responsesPath}`);
  console.log(`[EvalCollect] Response count: ${responses.length}`);
}

main().catch((error) => {
  console.error(`[EvalCollect] Failed: ${error.message}`);
  process.exit(1);
});
