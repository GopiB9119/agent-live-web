const { spawn } = require('child_process');

function runTraceLatest(envOverrides = {}) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, ['scripts/trace-latest.js'], {
      stdio: 'inherit',
      env: {
        ...process.env,
        ...envOverrides
      }
    });

    child.on('close', (code) => {
      resolve(code || 0);
    });

    child.on('error', () => {
      resolve(1);
    });
  });
}

(async () => {
  const checks = [
    {
      label: 'latest',
      env: {}
    },
    {
      label: 'latest:errors',
      env: {
        TRACE_STATUS: 'ERROR'
      }
    },
    {
      label: 'incident',
      env: {
        TRACE_STATUS: 'ERROR',
        TRACE_MIN_DURATION_MS: '500',
        TRACE_LOOKBACK: '24h'
      }
    }
  ];

  for (const check of checks) {
    console.log(`[TraceTriage] Running ${check.label}...`);
    const code = await runTraceLatest(check.env);
    if (code !== 0) {
      console.error(`[TraceTriage] ${check.label} failed with exit code ${code}.`);
      process.exit(code);
      return;
    }
  }

  console.log('[TraceTriage] Completed successfully.');
})();
