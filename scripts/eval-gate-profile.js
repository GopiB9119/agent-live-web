const { spawn } = require('child_process');

const profile = String(process.argv[2] || 'normal').trim().toLowerCase();

const profileEnv = {
  lenient: {
    EVAL_GATE_MIN_SCORE: '75',
    EVAL_GATE_ALLOW_REGRESSION: 'true'
  },
  normal: {
    EVAL_GATE_MIN_SCORE: '85',
    EVAL_GATE_ALLOW_REGRESSION: 'false'
  },
  strict: {
    EVAL_GATE_MIN_SCORE: '92',
    EVAL_GATE_ALLOW_REGRESSION: 'false'
  }
};

if (!profileEnv[profile]) {
  console.error(`[EvalGateProfile] Unknown profile: ${profile}`);
  console.error('[EvalGateProfile] Use one of: lenient, normal, strict');
  process.exit(1);
}

const child = spawn(process.execPath, ['scripts/eval/ci-gate.js'], {
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
