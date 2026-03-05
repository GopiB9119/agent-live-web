const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const IS_QUIET = process.argv.includes('--quiet') || process.env.HEALTH_SNAPSHOT_QUIET === '1';
const KEEP_LATEST_REPORTS = Number.parseInt(process.env.HEALTH_SNAPSHOT_KEEP_LATEST || '20', 10);

function timestampForFile(date = new Date()) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

function pythonCommandCandidates() {
  const candidates = [];
  if (process.env.AGENT_PYTHON && String(process.env.AGENT_PYTHON).trim()) {
    candidates.push({ cmd: String(process.env.AGENT_PYTHON).trim(), argsPrefix: [] });
  }

  const winVenvPython = path.resolve(process.cwd(), '.venv', 'Scripts', 'python.exe');
  if (fs.existsSync(winVenvPython)) {
    candidates.push({ cmd: winVenvPython, argsPrefix: [] });
  }

  candidates.push({ cmd: 'python', argsPrefix: [] });
  candidates.push({ cmd: 'py', argsPrefix: ['-3'] });
  return candidates;
}

function runCommand(command, args, options = {}) {
  return new Promise((resolve) => {
    const startedAt = new Date();
    const streamOutput = options.streamOutput !== false;
    const child = spawn(command, args, {
      shell: false,
      cwd: process.cwd(),
      env: process.env,
      ...options,
      streamOutput: undefined
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      stdout += text;
      if (streamOutput) process.stdout.write(text);
    });

    child.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      stderr += text;
      if (streamOutput) process.stderr.write(text);
    });

    child.on('error', (error) => {
      resolve({
        ok: false,
        code: 1,
        error: error.message,
        stdout,
        stderr: `${stderr}${stderr ? '\n' : ''}${error.message}`,
        startedAt,
        finishedAt: new Date(),
        command: [command, ...args].join(' ')
      });
    });

    child.on('close', (code) => {
      resolve({
        ok: code === 0,
        code: code || 0,
        stdout,
        stderr,
        startedAt,
        finishedAt: new Date(),
        command: [command, ...args].join(' ')
      });
    });
  });
}

function formatSection(title, result) {
  const status = result.ok ? 'PASS' : 'FAIL';
  const output = [result.stdout, result.stderr].filter(Boolean).join('\n').trim();
  return [
    `## ${title}`,
    `- Status: ${status}`,
    `- Exit code: ${result.code}`,
    `- Command: \`${result.command}\``,
    `- Started: ${result.startedAt.toISOString()}`,
    `- Finished: ${result.finishedAt.toISOString()}`,
    '',
    '```text',
    output || '(no output)',
    '```',
    ''
  ].join('\n');
}

function pruneOldReports(directory, keepLatest) {
  if (!Number.isFinite(keepLatest) || keepLatest < 1) {
    return [];
  }

  const files = fs
    .readdirSync(directory)
    .filter((name) => /^health-snapshot-\d{8}-\d{6}\.md$/.test(name))
    .map((name) => {
      const fullPath = path.join(directory, name);
      const stats = fs.statSync(fullPath);
      return { name, fullPath, mtimeMs: stats.mtimeMs };
    })
    .sort((a, b) => b.mtimeMs - a.mtimeMs);

  const toDelete = files.slice(keepLatest);
  for (const item of toDelete) {
    fs.unlinkSync(item.fullPath);
  }

  return toDelete.map((item) => item.name);
}

async function runPythonTests() {
  const testArgs = ['-m', 'unittest', 'discover', '-s', 'agent/agent/tests', '-p', 'test_*.py', '-v'];
  const candidates = pythonCommandCandidates();

  let last = null;
  for (const candidate of candidates) {
    const result = await runCommand(candidate.cmd, [...candidate.argsPrefix, ...testArgs], {
      streamOutput: !IS_QUIET
    });
    if (result.ok) {
      return result;
    }

    // Keep trying if interpreter path is missing.
    const combined = `${result.stderr}\n${result.stdout}`;
    if (/not recognized|No such file|ENOENT/i.test(combined)) {
      last = result;
      continue;
    }

    return result;
  }

  return (
    last || {
      ok: false,
      code: 1,
      stdout: '',
      stderr: 'Unable to locate a usable Python interpreter.',
      startedAt: new Date(),
      finishedAt: new Date(),
      command: 'python -m unittest discover -s agent/agent/tests -p test_*.py -v'
    }
  );
}

async function runJsCheck() {
  const files = [
    'cli-agent.js',
    'edge-session.js',
    'nl-command-parser.js',
    'playwright-edge-mcp.js',
    'scripts/mcp-init-page.js'
  ];

  const startedAt = new Date();
  const outputs = [];

  for (const file of files) {
    const result = await runCommand(process.execPath, ['--check', file], { streamOutput: !IS_QUIET });
    outputs.push(`$ node --check ${file}`);
    if (result.stdout) outputs.push(result.stdout.trim());
    if (result.stderr) outputs.push(result.stderr.trim());

    if (!result.ok) {
      return {
        ok: false,
        code: result.code,
        stdout: outputs.filter(Boolean).join('\n\n'),
        stderr: result.stderr || '',
        startedAt,
        finishedAt: new Date(),
        command: 'node --check <repo-js-files>'
      };
    }
  }

  return {
    ok: true,
    code: 0,
    stdout: outputs.filter(Boolean).join('\n\n') || 'All JS syntax checks passed.',
    stderr: '',
    startedAt,
    finishedAt: new Date(),
    command: 'node --check <repo-js-files>'
  };
}

(async () => {
  const startedAt = new Date();
  const outDir = path.resolve(process.cwd(), 'logs', 'health');
  fs.mkdirSync(outDir, { recursive: true });

  console.log(`[HealthSnapshot] Running JS syntax checks${IS_QUIET ? ' (quiet mode)' : ''}...`);
  const jsCheck = await runJsCheck();

  console.log(`[HealthSnapshot] Running Python unit tests${IS_QUIET ? ' (quiet mode)' : ''}...`);
  const pyTests = await runPythonTests();

  const finishedAt = new Date();
  const allPass = jsCheck.ok && pyTests.ok;

  const reportLines = [
    '# Health Snapshot',
    '',
    `- Generated: ${finishedAt.toISOString()}`,
    `- Overall: ${allPass ? 'PASS' : 'FAIL'}`,
    `- Duration: ${(finishedAt.getTime() - startedAt.getTime()) / 1000}s`,
    '',
    formatSection('JavaScript Syntax Check', jsCheck),
    formatSection('Python Unit Tests', pyTests)
  ];

  const fileName = `health-snapshot-${timestampForFile(finishedAt)}.md`;
  const reportPath = path.join(outDir, fileName);
  fs.writeFileSync(reportPath, reportLines.join('\n'), 'utf8');

  const removedReports = pruneOldReports(outDir, KEEP_LATEST_REPORTS);

  console.log(`[HealthSnapshot] Report written: ${reportPath}`);
  if (removedReports.length > 0 && !IS_QUIET) {
    console.log(
      `[HealthSnapshot] Pruned ${removedReports.length} old report(s) to keep latest ${KEEP_LATEST_REPORTS}.`
    );
  }
  if (!allPass) {
    process.exit(1);
    return;
  }

  process.exit(0);
})();
