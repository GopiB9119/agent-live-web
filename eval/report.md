# Evaluation Report

Generated: 2026-03-03T13:04:41.005Z

## Scores
- Overall: 40
- Reliability: 0
- Performance: 100
- Command Success Rate: 0%

## Metrics
- Total checks: 5
- Passed checks: 0
- Failed checks: 5
- Launch trace count: 0
- Launch p95 (ms): 0
- Launch max (ms): 0
- Error trace count: 0
- Incident trace count: 0

## Failed Checks
- q1_latest_launch_health: exit(1/0) traces=0 min=1 max=∞
- q2_errors_health: exit(1/0) traces=0 min=0 max=0
- q3_incident_health: exit(1/0) traces=0 min=0 max=0
- q4_triage_consistency: exit(1/0) traces=0 min=0 max=∞
- q5_latest_strict_window: exit(1/0) traces=0 min=1 max=∞

## Health
- Healthy: no
- Needs attention: yes
- Gate ready: no

## Remediation Hints
- No launch traces found: run `npm run trace:stack:start`, restart MCP server, then run `npm run trace:latest`.
- One or more checks failed: open `eval/report.md`, fix the first failed check, then rerun `npm run eval:all`.
