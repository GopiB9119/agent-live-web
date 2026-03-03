const fs = require('fs');
const path = require('path');

const evalDir = path.join(process.cwd(), 'eval');
const reportPath = path.join(evalDir, 'report.json');
const trendPath = path.join(evalDir, 'trend.json');

function toNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function main() {
  if (!fs.existsSync(reportPath)) {
    console.error('[EvalGate] Missing eval/report.json. Run `npm run eval:run` first.');
    process.exit(1);
  }

  const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  const trend = fs.existsSync(trendPath)
    ? JSON.parse(fs.readFileSync(trendPath, 'utf8'))
    : { regressionDetected: false };

  const minScore = toNumber(process.env.EVAL_GATE_MIN_SCORE, 85);
  const allowRegression = String(process.env.EVAL_GATE_ALLOW_REGRESSION || 'false').toLowerCase() === 'true';

  const failures = [];

  if (Number(report.summary.overallScore || 0) < minScore) {
    failures.push(`overallScore ${report.summary.overallScore} < minScore ${minScore}`);
  }

  if (Number(report.metrics.failedChecks || 0) > 0) {
    failures.push(`failedChecks ${report.metrics.failedChecks} > 0`);
  }

  if (Number(report.metrics.errorTraceCount || 0) > 0) {
    failures.push(`errorTraceCount ${report.metrics.errorTraceCount} > 0`);
  }

  if (!allowRegression && trend.regressionDetected) {
    failures.push('regressionDetected=true (score dropped by more than 5 points)');
  }

  if (failures.length) {
    console.error('[EvalGate] FAIL');
    for (const failure of failures) {
      console.error(`- ${failure}`);
    }
    console.error('[EvalGate] See eval/report.md and eval/trend.json for details.');
    process.exit(1);
  }

  console.log('[EvalGate] PASS');
  console.log(`[EvalGate] overallScore=${report.summary.overallScore}`);
  console.log(`[EvalGate] minScore=${minScore}`);
  console.log(`[EvalGate] regressionDetected=${trend.regressionDetected ? 'yes' : 'no'}`);
}

main();
