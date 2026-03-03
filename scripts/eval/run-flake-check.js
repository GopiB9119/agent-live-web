const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const evalDir = path.join(process.cwd(), 'eval');
const reportPath = path.join(evalDir, 'report.json');
const flakeJsonPath = path.join(evalDir, 'flake-report.json');
const flakeMdPath = path.join(evalDir, 'flake-report.md');

function toNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function runNodeScript(relativePath) {
  const scriptPath = path.join(process.cwd(), relativePath);
  return spawnSync('node', [scriptPath], {
    cwd: process.cwd(),
    env: process.env,
    encoding: 'utf8'
  });
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function toMarkdown(report) {
  return [
    '# Flake Report',
    '',
    `Generated: ${report.generatedAt}`,
    '',
    `- Runs: ${report.runs}`,
    `- Median score: ${report.medianScore}`,
    `- Min score: ${report.minScore}`,
    `- Max score: ${report.maxScore}`,
    `- Failed-check runs: ${report.failedCheckRuns}`,
    `- Score spread: ${report.scoreSpread}`,
    `- Unstable: ${report.unstable ? 'yes' : 'no'}`,
    ''
  ].join('\n');
}

function main() {
  const runs = toNumber(process.env.EVAL_FLAKE_RUNS, 3);
  const allowedSpread = toNumber(process.env.EVAL_FLAKE_ALLOWED_SPREAD, 5);

  if (runs < 2) {
    console.error('[EvalFlake] EVAL_FLAKE_RUNS must be >= 2');
    process.exit(1);
  }

  const scores = [];
  let failedCheckRuns = 0;

  for (let i = 0; i < runs; i += 1) {
    console.log(`[EvalFlake] Iteration ${i + 1}/${runs}`);

    const collect = runNodeScript('scripts/eval/collect-responses.js');
    if (collect.status !== 0) {
      process.stderr.write(collect.stderr || collect.stdout || '');
      process.exit(1);
    }

    const evaluate = runNodeScript('scripts/eval/run-evaluation.js');
    if (evaluate.status !== 0) {
      process.stderr.write(evaluate.stderr || evaluate.stdout || '');
      process.exit(1);
    }

    if (!fs.existsSync(reportPath)) {
      console.error('[EvalFlake] Missing eval/report.json after evaluation run.');
      process.exit(1);
    }

    const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
    const score = Number(report.summary.overallScore || 0);
    scores.push(score);

    if (Number(report.metrics.failedChecks || 0) > 0) {
      failedCheckRuns += 1;
    }
  }

  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const scoreSpread = maxScore - minScore;

  const flakeReport = {
    generatedAt: new Date().toISOString(),
    runs,
    scores,
    medianScore: Math.round(median(scores) * 100) / 100,
    minScore: Math.round(minScore * 100) / 100,
    maxScore: Math.round(maxScore * 100) / 100,
    scoreSpread: Math.round(scoreSpread * 100) / 100,
    failedCheckRuns,
    unstable: failedCheckRuns > 0 || scoreSpread > allowedSpread,
    allowedSpread
  };

  fs.mkdirSync(evalDir, { recursive: true });
  fs.writeFileSync(flakeJsonPath, JSON.stringify(flakeReport, null, 2), 'utf8');
  fs.writeFileSync(flakeMdPath, toMarkdown(flakeReport), 'utf8');

  console.log(`[EvalFlake] Wrote: ${flakeJsonPath}`);
  console.log(`[EvalFlake] Wrote: ${flakeMdPath}`);
  console.log(`[EvalFlake] Unstable: ${flakeReport.unstable ? 'yes' : 'no'}`);

  if (flakeReport.unstable) {
    process.exit(1);
  }
}

main();
