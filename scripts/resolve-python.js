#!/usr/bin/env node
'use strict';

// Resolves the correct Python executable for this project.
// Prefers the repo-local .venv, falls back to system python.

const { execFileSync, spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const candidates = [
  path.join('.venv', 'Scripts', 'python.exe'), // Windows venv
  path.join('.venv', 'bin', 'python'),          // Unix venv
  'python3',                                     // System python3
  'python'                                       // System python
];

function findPython() {
  for (const candidate of candidates) {
    try {
      if (candidate.includes(path.sep) || candidate.includes('/')) {
        if (!fs.existsSync(candidate)) continue;
      }
      execFileSync(candidate, ['--version'], { stdio: 'pipe' });
      return candidate;
    } catch (_) {
      continue;
    }
  }
  return null;
}

const args = process.argv.slice(2);
const pythonPath = findPython();

if (!pythonPath) {
  process.stderr.write('Error: No Python interpreter found. Create a .venv or install Python 3.11+.\n');
  process.exit(1);
}

if (args.length === 0) {
  process.stdout.write(pythonPath + '\n');
  process.exit(0);
}

try {
  const result = spawnSync(pythonPath, args, { stdio: 'inherit', shell: false });
  if (result.error) {
    throw result.error;
  }
  process.exit(result.status ?? 1);
} catch (error) {
  process.exit(error.status || 1);
}
