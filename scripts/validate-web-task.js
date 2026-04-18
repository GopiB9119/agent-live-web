#!/usr/bin/env node
'use strict';

// Validates a web-task JSON file and reports issues.
// Usage: node scripts/validate-web-task.js <path-to-task.json>

const fs = require('fs');

const REQUIRED_FIELDS = ['version', 'task_id', 'mode', 'start_url', 'objective', 'success_criteria', 'side_effect_policy', 'steps', 'output'];
const VALID_MODES = ['explore', 'extract', 'automate', 'qa'];
const VALID_PROFILES = ['balanced', 'deep', 'turbo'];
const VALID_VERSIONS = ['1.0', '1.1'];
const STEP_ACTIONS = ['navigate', 'click', 'type', 'fill', 'press', 'select', 'hover', 'search', 'wait', 'check', 'extract', 'scroll', 'download', 'upload', 'screenshot', 'snapshot', 'back', 'forward', 'refresh', 'delete', 'focus', 'clear', 'doubleClick', 'rightClick', 'evaluate'];
const TARGET_OPTIONAL_ACTIONS = new Set(['wait', 'screenshot', 'snapshot', 'back', 'forward', 'refresh']);

function normalizeAllowedDomain(domain) {
  if (typeof domain !== 'string') return '';
  const trimmed = domain.trim().toLowerCase();
  if (!trimmed) return '';
  const withoutWildcard = trimmed.startsWith('*.') ? trimmed.slice(2) : trimmed;
  return withoutWildcard.split(':')[0];
}

function validate(taskPath) {
  const errors = [];
  const warnings = [];

  if (!fs.existsSync(taskPath)) {
    errors.push(`File not found: ${taskPath}`);
    return { errors, warnings };
  }

  let data;
  try {
    data = JSON.parse(fs.readFileSync(taskPath, 'utf8'));
  } catch (e) {
    errors.push(`Invalid JSON: ${e.message}`);
    return { errors, warnings };
  }

  // Required fields
  for (const field of REQUIRED_FIELDS) {
    if (!(field in data)) {
      errors.push(`Missing required field: ${field}`);
    }
  }

  // Version
  if (data.version && !VALID_VERSIONS.includes(data.version)) {
    errors.push(`Invalid version "${data.version}". Must be one of: ${VALID_VERSIONS.join(', ')}`);
  }

  // Mode
  if (data.mode && !VALID_MODES.includes(data.mode)) {
    errors.push(`Invalid mode "${data.mode}". Must be one of: ${VALID_MODES.join(', ')}`);
  }

  // Execution profile
  if (data.execution_profile && !VALID_PROFILES.includes(data.execution_profile)) {
    errors.push(`Invalid execution_profile "${data.execution_profile}". Must be one of: ${VALID_PROFILES.join(', ')}`);
  }

  // task_id
  if (data.task_id && typeof data.task_id === 'string' && !data.task_id.trim()) {
    errors.push('task_id must not be empty');
  }

  // start_url
  if (data.start_url) {
    try {
      new URL(data.start_url);
    } catch (_) {
      errors.push(`start_url is not a valid URL: "${data.start_url}"`);
    }
  }

  // objective
  if (data.objective && typeof data.objective === 'string' && data.objective.length < 5) {
    warnings.push('objective is very short — consider being more specific');
  }

  // success_criteria
  if (data.success_criteria) {
    if (!Array.isArray(data.success_criteria)) {
      errors.push('success_criteria must be an array');
    } else if (data.success_criteria.length === 0) {
      errors.push('success_criteria must have at least one entry');
    }
  }

  // steps
  if (data.steps) {
    if (!Array.isArray(data.steps)) {
      errors.push('steps must be an array');
    } else {
      const stepIds = new Set();
      for (let i = 0; i < data.steps.length; i++) {
        const step = data.steps[i];
        const prefix = `steps[${i}]`;

        if (typeof step !== 'object' || step === null) {
          errors.push(`${prefix}: must be an object`);
          continue;
        }

        if (!step.id) {
          errors.push(`${prefix}: missing required field "id"`);
        } else if (stepIds.has(step.id)) {
          errors.push(`${prefix}: duplicate step id "${step.id}"`);
        } else {
          stepIds.add(step.id);
        }

        if (!step.action) {
          errors.push(`${prefix}: missing required field "action"`);
        } else if (!STEP_ACTIONS.includes(step.action)) {
          warnings.push(`${prefix}: action "${step.action}" is not a standard action (${STEP_ACTIONS.join(', ')})`);
        }

        if (!step.target && !TARGET_OPTIONAL_ACTIONS.has(step.action)) {
          warnings.push(`${prefix}: no "target" defined — may need selector or URL`);
        }

        if (!step.verify) {
          warnings.push(`${prefix}: no "verify" block — step completion won't be confirmed`);
        }
      }

      if (data.steps.length === 0) {
        errors.push('steps must have at least one entry');
      }
    }
  }

  // side_effect_policy
  if (data.side_effect_policy) {
    if (typeof data.side_effect_policy !== 'object') {
      errors.push('side_effect_policy must be an object');
    } else {
      if (!Array.isArray(data.side_effect_policy.require_confirmation_for)) {
        warnings.push('side_effect_policy.require_confirmation_for should be an array');
      }
    }
  }

  // output
  if (data.output && typeof data.output !== 'object') {
    errors.push('output must be an object');
  }

  // allowed_domains
  if (data.allowed_domains) {
    if (!Array.isArray(data.allowed_domains)) {
      errors.push('allowed_domains must be an array');
    } else if (data.start_url) {
      try {
        const startHost = new URL(data.start_url).hostname.toLowerCase();
        const domainMatch = data.allowed_domains.some((d) => {
          const normalized = normalizeAllowedDomain(d);
          return normalized && (startHost === normalized || startHost.endsWith(`.${normalized}`));
        });
        if (!domainMatch) {
          warnings.push(`start_url host "${startHost}" is not in allowed_domains — agent may be blocked from navigating`);
        }
      } catch (_) {
        // start_url validation already caught
      }
    }
  }

  return { errors, warnings, data };
}

// CLI
const taskFile = process.argv[2];
if (!taskFile) {
  console.log('Usage: node scripts/validate-web-task.js <path-to-task.json>');
  console.log('');
  console.log('Validates a web-task definition.');
  console.log('');
  console.log('Examples:');
  console.log('  node scripts/validate-web-task.js .github/skills/web-works/examples/extract-github-trending.json');
  console.log('  node scripts/validate-web-task.js my-task.json');
  process.exit(1);
}

const result = validate(taskFile);

if (result.errors.length === 0 && result.warnings.length === 0) {
  console.log(`✓ ${taskFile} is valid (${result.data.steps.length} steps, mode=${result.data.mode}, profile=${result.data.execution_profile || 'balanced'})`);
  process.exit(0);
}

if (result.warnings.length > 0) {
  console.log('Warnings:');
  for (const w of result.warnings) {
    console.log(`  ⚠ ${w}`);
  }
}

if (result.errors.length > 0) {
  console.log('Errors:');
  for (const e of result.errors) {
    console.log(`  ✗ ${e}`);
  }
  process.exit(1);
}

console.log(`✓ ${taskFile} is valid with warnings (${result.data.steps.length} steps, mode=${result.data.mode})`);
process.exit(0);
