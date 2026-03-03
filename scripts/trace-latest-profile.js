const { spawn } = require('child_process');

const profile = String(process.argv[2] || 'latest').trim().toLowerCase();

const profileEnv = {
  latest: {},
  errors: {
    TRACE_STATUS: 'ERROR'
  },
  incident: {
    TRACE_STATUS: 'ERROR',
    TRACE_MIN_DURATION_MS: '500',
    TRACE_LOOKBACK: '24h'
  }
};

if (!profileEnv[profile]) {
  console.error(`[TraceLatestProfile] Unknown profile: ${profile}`);
  console.error('[TraceLatestProfile] Use one of: latest, errors, incident');
  process.exit(1);
}

const child = spawn(process.execPath, ['scripts/trace-latest.js'], {
  stdio: 'inherit',
  env: {
    ...process.env,
    ...profileEnv[profile]
  }
});

child.on('close', (code) => {
  process.exit(code || 0);
});

child.on('error', () => {
  process.exit(1);
});
