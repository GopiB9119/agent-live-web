const { spawn } = require('child_process');

function runNodeScript(scriptPath) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [scriptPath], {
      stdio: 'inherit',
      env: process.env
    });

    child.on('close', (code) => {
      resolve(code || 0);
    });

    child.on('error', () => {
      resolve(1);
    });
  });
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function runStep(label, scriptPath) {
  console.log(`[TracePlaywrightSmoke] Running ${label}...`);
  const code = await runNodeScript(scriptPath);
  if (code !== 0) {
    console.error(`[TracePlaywrightSmoke] ${label} failed with exit code ${code}.`);
    return code;
  }
  return 0;
}

(async () => {
  let failureCode = 0;

  console.log('[TracePlaywrightSmoke] Running trace-stack:start...');
  const stackStart = await new Promise((resolve) => {
    const child = spawn(process.execPath, ['scripts/trace-stack-control.js', 'stack', 'start'], {
      stdio: 'inherit',
      env: process.env
    });
    child.on('close', (code) => resolve(code || 0));
    child.on('error', () => resolve(1));
  });

  if (stackStart !== 0) {
    process.exit(stackStart);
    return;
  }

  try {
    const maxTraceCheckAttempts = 4;
    let traceCheckPassed = false;
    for (let attempt = 1; attempt <= maxTraceCheckAttempts; attempt += 1) {
      const code = await runStep(`trace:check (attempt ${attempt}/${maxTraceCheckAttempts})`, 'scripts/check-tracing.js');
      if (code === 0) {
        traceCheckPassed = true;
        break;
      }

      if (attempt < maxTraceCheckAttempts) {
        console.log('[TracePlaywrightSmoke] Waiting 3s before retrying trace check...');
        await sleep(3000);
      }
    }

    if (!traceCheckPassed) {
      failureCode = 2;
    }

    if (failureCode === 0) {
      for (const [label, scriptPath] of [
        ['trace:triage (pre)', 'scripts/trace-triage.js'],
        ['playwright-edge-smoke', 'scripts/playwright-edge-smoke.js'],
        ['trace:triage (post)', 'scripts/trace-triage.js']
      ]) {
        const code = await runStep(label, scriptPath);
        if (code !== 0) {
          failureCode = code;
          break;
        }
      }
    }
  } finally {
    const stopCode = await new Promise((resolve) => {
      const child = spawn(process.execPath, ['scripts/trace-stack-control.js', 'stack', 'stop'], {
        stdio: 'inherit',
        env: process.env
      });
      child.on('close', (code) => resolve(code || 0));
      child.on('error', () => resolve(1));
    });

    if (failureCode === 0 && stopCode !== 0) {
      failureCode = stopCode;
    }
  }

  if (failureCode !== 0) {
    process.exit(failureCode);
    return;
  }

  console.log('[TracePlaywrightSmoke] Completed successfully.');
})();
