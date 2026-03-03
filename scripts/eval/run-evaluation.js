const fs = require('fs');
const path = require('path');

const evalDir = path.join(process.cwd(), 'eval');
const responsesPath = path.join(evalDir, 'responses.json');
const reportJsonPath = path.join(evalDir, 'report.json');
const reportMdPath = path.join(evalDir, 'report.md');

function round(value) {
  return Math.round(value * 100) / 100;
}

function percentile(values, pct) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil((pct / 100) * sorted.length) - 1);
  return sorted[Math.max(0, index)];
}

function buildRemediationHints(report) {
  const hints = [];

  if (report.metrics.launchTraceCount === 0) {
    hints.push('No launch traces found: run `npm run trace:stack:start`, restart MCP server, then run `npm run trace:latest`.');
  }

  if (report.metrics.errorTraceCount > 0) {
    hints.push('Error traces detected: run `npm run trace:latest:errors` and inspect latest spans in Jaeger (`http://localhost:16686`).');
  }

  if (report.metrics.incidentTraceCount > 0) {
    hints.push('Incident traces detected: run `npm run trace:incident` and investigate slow/error spans first.');
  }

  if (report.metrics.launchP95Ms > 500) {
    hints.push('High launch latency (p95 > 500ms): validate host load and rerun with stress profile to confirm regression.');
  }

  if (report.metrics.failedChecks > 0) {
    hints.push('One or more checks failed: open `eval/report.md`, fix the first failed check, then rerun `npm run eval:all`.');
  }

  if (!hints.length) {
    hints.push('No action needed. Continue monitoring with `npm run trace:triage`.');
  }

  return hints;
}

function computeReport(responses) {
  const total = responses.length;
  const passed = responses.filter((item) => item.ok).length;
  const failedItems = responses.filter((item) => !item.ok);
  const commandSuccessRate = total ? (passed / total) * 100 : 0;

  const latest = responses.find((item) => item.id === 'q1_latest_launch_health');
  const errors = responses.find((item) => item.id === 'q2_errors_health');
  const incident = responses.find((item) => item.id === 'q3_incident_health');

  const launchDurations = latest && Array.isArray(latest.launchDurationsMs) ? latest.launchDurationsMs : [];
  const launchP95Ms = percentile(launchDurations, 95);
  const launchMaxMs = launchDurations.length ? Math.max(...launchDurations) : 0;

  const errorTraceCount = errors ? errors.runtimeTraceCount : 0;
  const incidentTraceCount = incident ? incident.runtimeTraceCount : 0;

  const reliabilityScore = Math.max(0, Math.min(100, commandSuccessRate - (errorTraceCount * 10)));

  const perfPenalty = launchP95Ms > 500 ? 30 : launchP95Ms > 200 ? 10 : 0;
  const performanceScore = Math.max(0, Math.min(100, 100 - perfPenalty - (incidentTraceCount * 10)));

  const overallScore = round((reliabilityScore * 0.6) + (performanceScore * 0.4));

  const report = {
    generatedAt: new Date().toISOString(),
    summary: {
      overallScore,
      reliabilityScore: round(reliabilityScore),
      performanceScore: round(performanceScore),
      commandSuccessRate: round(commandSuccessRate)
    },
    metrics: {
      totalChecks: total,
      passedChecks: passed,
      failedChecks: total - passed,
      errorTraceCount,
      incidentTraceCount,
      launchTraceCount: launchDurations.length,
      launchP95Ms: round(launchP95Ms),
      launchMaxMs: round(launchMaxMs)
    },
    failedItems: failedItems.map((item) => ({
      id: item.id,
      name: item.name,
      expectedExitCode: item.expectedExitCode,
      actualExitCode: item.actualExitCode,
      runtimeTraceCount: item.runtimeTraceCount,
      minRuntimeTraces: item.minRuntimeTraces,
      maxRuntimeTraces: item.maxRuntimeTraces,
      checks: item.checks
    })),
    status: {
      healthy: overallScore >= 80 && errorTraceCount === 0,
      needsAttention: overallScore < 80 || errorTraceCount > 0 || incidentTraceCount > 0 || failedItems.length > 0,
      gateReady: overallScore >= 85 && failedItems.length === 0 && errorTraceCount === 0
    }
  };

  report.remediationHints = buildRemediationHints(report);
  return report;
}

function toMarkdown(report) {
  const { summary, metrics, status, failedItems } = report;
  const failures = failedItems && failedItems.length
    ? failedItems
        .map((item) => `- ${item.id}: exit(${item.actualExitCode}/${item.expectedExitCode}) traces=${item.runtimeTraceCount} min=${item.minRuntimeTraces} max=${item.maxRuntimeTraces ?? '∞'}`)
        .join('\n')
    : '- none';

  return [
    '# Evaluation Report',
    '',
    `Generated: ${report.generatedAt}`,
    '',
    '## Scores',
    `- Overall: ${summary.overallScore}`,
    `- Reliability: ${summary.reliabilityScore}`,
    `- Performance: ${summary.performanceScore}`,
    `- Command Success Rate: ${summary.commandSuccessRate}%`,
    '',
    '## Metrics',
    `- Total checks: ${metrics.totalChecks}`,
    `- Passed checks: ${metrics.passedChecks}`,
    `- Failed checks: ${metrics.failedChecks}`,
    `- Launch trace count: ${metrics.launchTraceCount}`,
    `- Launch p95 (ms): ${metrics.launchP95Ms}`,
    `- Launch max (ms): ${metrics.launchMaxMs}`,
    `- Error trace count: ${metrics.errorTraceCount}`,
    `- Incident trace count: ${metrics.incidentTraceCount}`,
    '',
    '## Failed Checks',
    failures,
    '',
    '## Health',
    `- Healthy: ${status.healthy ? 'yes' : 'no'}`,
    `- Needs attention: ${status.needsAttention ? 'yes' : 'no'}`,
    `- Gate ready: ${status.gateReady ? 'yes' : 'no'}`,
    '',
    '## Remediation Hints',
    ...report.remediationHints.map((hint) => `- ${hint}`),
    ''
  ].join('\n');
}

function main() {
  if (!fs.existsSync(responsesPath)) {
    console.error(`[EvalRun] Missing responses: ${responsesPath}`);
    console.error('[EvalRun] Run: npm run eval:collect');
    process.exit(1);
  }

  const responses = JSON.parse(fs.readFileSync(responsesPath, 'utf8'));
  if (!Array.isArray(responses) || !responses.length) {
    console.error('[EvalRun] responses.json is empty or invalid.');
    process.exit(1);
  }

  const report = computeReport(responses);
  fs.mkdirSync(evalDir, { recursive: true });
  fs.writeFileSync(reportJsonPath, JSON.stringify(report, null, 2), 'utf8');
  fs.writeFileSync(reportMdPath, toMarkdown(report), 'utf8');

  console.log(`[EvalRun] Wrote report: ${reportJsonPath}`);
  console.log(`[EvalRun] Wrote report: ${reportMdPath}`);
  console.log(`[EvalRun] Overall score: ${report.summary.overallScore}`);
  console.log(`[EvalRun] Healthy: ${report.status.healthy ? 'yes' : 'no'}`);
}

main();
