#!/usr/bin/env node
'use strict';
const fs = require('fs');

// --- Structure checks ---
const checks = [
  { file: '.github/agents/agent-live-web.agent.md', contains: ['## Scope', '## What the agent must not do', '## Failure impact controls', '## Completion rule'] },
  { file: '.github/prompts/playwright-live-web-task-brief.prompt.md', contains: ['What I Want:', 'What I Do Not Want:', 'Evidence Required:', 'Done When:'] },
  { file: '.github/skills/web-works/SKILL.md', contains: ['## Purpose', '## Do not use this skill for', '## Required task brief', '## Failure and bad-impact control'] },
  { file: '.github/skills/web-works/PROMPTS.md', contains: ['## 0) Structured Task Brief First', '## What This File Is For', '## When To Avoid These Prompts'] },
  { file: '.github/instructions/playwright-edge.instructions.md', contains: ['## What This File Is For', '## What This File Is Not For', '## Bad Impact To Avoid', '## Validation After Edits'] },
  { file: '.github/instructions/live-web-governance.instructions.md', contains: ['# Live Web Governance Structure', 'what it is for', 'validation expectations'] },
  { file: '.github/README.md', contains: ['## Main lanes', '### 1. Live Web Copilot lane', '## Live Web task brief contract'] },
  { file: '.github/copilot-instructions.md', contains: ['## Lane-specific rules', 'Do not duplicate lane-specific rules here'] }
];

let failed = false;
for (const check of checks) {
  if (!fs.existsSync(check.file)) {
    console.error(`[missing-file] ${check.file}`);
    failed = true;
    continue;
  }
  const text = fs.readFileSync(check.file, 'utf8');
  for (const token of check.contains) {
    if (!text.includes(token)) {
      console.error(`[missing-token] ${check.file}: ${token}`);
      failed = true;
    }
  }
}

console.log(failed ? '[structure-check] FAIL' : '[structure-check] PASS');

// --- Frontmatter checks ---
function extractFrontmatter(content) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return null;
  const fields = {};
  for (const line of match[1].split(/\r?\n/)) {
    const kv = line.match(/^(\w[\w-]*):\s*(.*)/);
    if (kv) fields[kv[1]] = kv[2].trim().replace(/^["']|["']$/g, '');
  }
  return fields;
}

const fmChecks = [
  { dir: '.github/agents/', ext: '.agent.md', required: ['name', 'description'] },
  { dir: '.github/prompts/', ext: '.prompt.md', required: ['description'] },
  { dir: '.github/instructions/', ext: '.instructions.md', requiredOneOf: ['description', 'applyTo'] }
];

for (const spec of fmChecks) {
  if (!fs.existsSync(spec.dir)) continue;
  const files = fs.readdirSync(spec.dir).filter(f => f.endsWith(spec.ext));
  for (const file of files) {
    const filePath = spec.dir + file;
    const content = fs.readFileSync(filePath, 'utf8');
    const fm = extractFrontmatter(content);
    if (!fm) {
      console.error(`[frontmatter-missing] ${filePath}: no YAML frontmatter found`);
      failed = true;
      continue;
    }
    if (spec.required) {
      for (const field of spec.required) {
        if (!fm[field]) {
          console.error(`[frontmatter-field] ${filePath}: missing required field '${field}'`);
          failed = true;
        }
      }
    }
    if (spec.requiredOneOf) {
      const hasOne = spec.requiredOneOf.some(f => fm[f]);
      if (!hasOne) {
        console.error(`[frontmatter-field] ${filePath}: needs at least one of: ${spec.requiredOneOf.join(', ')}`);
        failed = true;
      }
    }
    if (fm.description && !fm.description.startsWith('Use when')) {
      console.warn(`[frontmatter-hint] ${filePath}: description should start with "Use when..." for discoverability`);
    }
  }
}

console.log(failed ? '[frontmatter-check] FAIL' : '[frontmatter-check] PASS');

if (failed) {
  process.exit(1);
}
console.log('[validate-governance] ALL PASS');
