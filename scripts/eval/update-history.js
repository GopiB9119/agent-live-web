const fs = require('fs');
const path = require('path');

const evalDir = path.join(process.cwd(), 'eval');
const reportPath = path.join(evalDir, 'report.json');
const historyPath = path.join(evalDir, 'history.jsonl');
const trendPath = path.join(evalDir, 'trend.json');

function readHistoryEntries() {
  if (!fs.existsSync(historyPath)) return [];
  const lines = fs
    .readFileSync(historyPath, 'utf8')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

  const entries = [];
  for (const line of lines) {
    try {
      entries.push(JSON.parse(line));
    } catch (_) {
      // ignore invalid line
    }
  }
  return entries;
}

function buildTrend(entries) {
  const recent = entries.slice(-10);
  const previous = entries.length >= 2 ? entries[entries.length - 2] : null;
  const latest = entries.length ? entries[entries.length - 1] : null;

  const averageScore = recent.length
    ? recent.reduce((sum, item) => sum + Number(item.summary.overallScore || 0), 0) / recent.length
    : 0;

  const scoreDelta = latest && previous
    ? Number(latest.summary.overallScore || 0) - Number(previous.summary.overallScore || 0)
    : 0;

  return {
    totalRuns: entries.length,
    recentWindow: recent.length,
    averageScore: Math.round(averageScore * 100) / 100,
    scoreDeltaFromPrevious: Math.round(scoreDelta * 100) / 100,
    regressionDetected: scoreDelta < -5,
    latest: latest || null,
    previous: previous || null
  };
}

function main() {
  if (!fs.existsSync(reportPath)) {
    console.error('[EvalHistory] Missing eval/report.json. Run `npm run eval:run` first.');
    process.exit(1);
  }

  const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  fs.mkdirSync(evalDir, { recursive: true });
  fs.appendFileSync(historyPath, `${JSON.stringify(report)}\n`, 'utf8');

  const entries = readHistoryEntries();
  const trend = buildTrend(entries);
  fs.writeFileSync(trendPath, JSON.stringify(trend, null, 2), 'utf8');

  console.log(`[EvalHistory] Appended run to: ${historyPath}`);
  console.log(`[EvalHistory] Wrote trend: ${trendPath}`);
  console.log(`[EvalHistory] Total runs: ${trend.totalRuns}`);
  console.log(`[EvalHistory] Score delta vs previous: ${trend.scoreDeltaFromPrevious}`);
  console.log(`[EvalHistory] Regression detected: ${trend.regressionDetected ? 'yes' : 'no'}`);
}

main();
