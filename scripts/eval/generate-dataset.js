const fs = require('fs');
const path = require('path');

const evalDir = path.join(process.cwd(), 'eval');
const queriesPath = path.join(evalDir, 'queries.json');
const force = process.argv.includes('--force');

function getArgValue(flag, fallback = '') {
  const arg = process.argv.find((item) => item.startsWith(`${flag}=`));
  if (!arg) return fallback;
  return String(arg.slice(flag.length + 1)).trim();
}

function getProfile() {
  const value = getArgValue('--profile', 'baseline').toLowerCase();
  if (value === 'stress') return 'stress';
  if (value === 'failure') return 'failure';
  return 'baseline';
}

const baselineQueries = [
  {
    id: 'q1_latest_launch_health',
    name: 'Latest launch trace health',
    command: 'npm',
    args: ['run', 'trace:latest'],
    expectedExitCode: 0,
    minRuntimeTraces: 1,
    category: 'reliability'
  },
  {
    id: 'q2_errors_health',
    name: 'Error trace health',
    command: 'npm',
    args: ['run', 'trace:latest:errors'],
    expectedExitCode: 0,
    maxRuntimeTraces: 0,
    category: 'reliability'
  },
  {
    id: 'q3_incident_health',
    name: 'Incident filter health',
    command: 'npm',
    args: ['run', 'trace:incident'],
    expectedExitCode: 0,
    maxRuntimeTraces: 0,
    category: 'performance'
  }
];

const stressExtras = [
  {
    id: 'q4_triage_consistency',
    name: 'Triage consistency check',
    command: 'npm',
    args: ['run', 'trace:triage'],
    expectedExitCode: 0,
    category: 'reliability'
  },
  {
    id: 'q5_latest_strict_window',
    name: 'Latest trace strict window',
    command: 'npm',
    args: ['run', 'trace:latest'],
    env: {
      TRACE_LOOKBACK: '10m'
    },
    expectedExitCode: 0,
    minRuntimeTraces: 1,
    category: 'performance'
  }
];

const failureInjectionQueries = [
  {
    id: 'qf1_missing_service_trace',
    name: 'Failure injection: missing service should violate min traces',
    command: 'npm',
    args: ['run', 'trace:latest'],
    env: {
      TRACE_SERVICE: 'agent-live-web-service-does-not-exist',
      TRACE_LOOKBACK: '5m'
    },
    expectedExitCode: 0,
    minRuntimeTraces: 1,
    category: 'failure-injection'
  }
];

function getQueries(profile) {
  if (profile === 'stress') {
    return [...baselineQueries, ...stressExtras];
  }
  if (profile === 'failure') {
    return [...baselineQueries, ...failureInjectionQueries];
  }
  return baselineQueries;
}

fs.mkdirSync(evalDir, { recursive: true });

const profile = getProfile();
const queries = getQueries(profile);

if (fs.existsSync(queriesPath) && !force) {
  console.log(`[EvalDataset] Keeping existing dataset: ${queriesPath}`);
  console.log('[EvalDataset] Use --force to regenerate.');
  process.exit(0);
}

fs.writeFileSync(queriesPath, JSON.stringify(queries, null, 2), 'utf8');
console.log(`[EvalDataset] Wrote dataset: ${queriesPath}`);
console.log(`[EvalDataset] Profile: ${profile}`);
console.log(`[EvalDataset] Query count: ${queries.length}`);
