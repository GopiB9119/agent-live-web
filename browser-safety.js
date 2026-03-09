const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const ActionClass = Object.freeze({
  READ_ONLY: 'read_only',
  SCOPED_REVERSIBLE_WRITE: 'scoped_reversible_write',
  BROAD_LOCAL_WRITE: 'broad_local_write',
  EXTERNAL_SIDE_EFFECT: 'external_side_effect',
  DESTRUCTIVE: 'destructive'
});

const PolicyDecision = Object.freeze({
  ALLOW: 'allow',
  ALLOW_WITH_VERIFICATION: 'allow_with_verification',
  PREVIEW_REQUIRED: 'preview_required',
  CONFIRM_REQUIRED: 'confirm_required',
  BLOCKED: 'blocked'
});

const DANGEROUS_BROWSER_HINTS = [
  'delete',
  'remove',
  'purchase',
  'checkout',
  'pay',
  'submit',
  'send',
  'confirm order',
  'place order',
  'merge',
  'publish'
];

const READ_ONLY_ACTIONS = new Set(['exists', 'getText', 'wait', 'waitFor']);
const BOUNDED_BROWSER_ACTIONS = new Set([
  'goto',
  'search',
  'click',
  'clickXPath',
  'clickByText',
  'edit',
  'type',
  'scroll',
  'screenshot',
  'startTrace',
  'stopTrace'
]);
const DANGEROUS_INTENT_ACTIONS = new Set(['click', 'clickXPath', 'clickByText', 'edit', 'type']);

function normalizeActionParams(params = {}) {
  const normalized = { ...(params || {}) };
  delete normalized.confirm;
  delete normalized.confirmToken;
  delete normalized.confirm_token;
  return normalized;
}

function truncate(value, limit = 200) {
  const text = String(value || '');
  return text.length <= limit ? text : `${text.slice(0, limit)}...`;
}

function collectArgumentText(action, params) {
  const fields = [
    action,
    params.selector,
    params.xpath,
    params.text,
    params.url,
    params.query,
    params.parentSelector
  ];
  return fields
    .filter((value) => value !== undefined && value !== null && String(value).trim())
    .map((value) => String(value).toLowerCase())
    .join(' ');
}

function isDangerousIntent(action, params) {
  if (action === 'delete' || action === 'add' || action === 'upload') {
    return true;
  }
  if (!DANGEROUS_INTENT_ACTIONS.has(action)) {
    return false;
  }
  const haystack = collectArgumentText(action, params);
  return DANGEROUS_BROWSER_HINTS.some((hint) => haystack.includes(hint));
}

function resolveArtifactPreview(requestedPath, workspaceRoot, defaultRelativePath) {
  const candidate = path.resolve(requestedPath || defaultRelativePath);
  const relative = path.relative(path.resolve(workspaceRoot || process.cwd()), candidate);
  const insideWorkspace = relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
  return {
    requested_path: requestedPath || '',
    resolved_path: candidate,
    default_relative_path: defaultRelativePath,
    inside_workspace: insideWorkspace
  };
}

function buildPreviewSummary(action, rawParams = {}, options = {}) {
  const params = normalizeActionParams(rawParams);
  const preview = { action };

  if (params.selector) preview.selector = truncate(params.selector, 160);
  if (params.xpath) preview.xpath = truncate(params.xpath, 160);
  if (params.text) preview.text = truncate(params.text, 160);
  if (params.url) preview.url = truncate(params.url, 220);
  if (params.query) preview.query = truncate(params.query, 220);

  if (action === 'edit' || action === 'type') {
    preview.value_chars = String(params.value || '').length;
  }

  if (action === 'add') {
    preview.parent_selector = truncate(params.parentSelector || '', 160);
    preview.html_chars = String(params.html || '').length;
  }

  if (action === 'upload') {
    const resolvedPath = path.resolve(String(params.filePath || ''));
    preview.file_path = resolvedPath;
    preview.file_exists = Boolean(resolvedPath && fs.existsSync(resolvedPath));
  }

  if (action === 'download') {
    preview.output = resolveArtifactPreview(
      params.savePath,
      options.workspaceRoot,
      path.join('downloads', '<browser-suggested-filename>')
    );
  }

  if (action === 'screenshot') {
    preview.output = resolveArtifactPreview(params.path, options.workspaceRoot, 'screenshot-<timestamp>.png');
  }

  if (action === 'stopTrace') {
    preview.output = resolveArtifactPreview(params.path, options.workspaceRoot, path.join('traces', 'trace-<timestamp>.zip'));
  }

  return preview;
}

function issueBrowserConfirmToken(secret, action, params = {}) {
  const canonical = JSON.stringify({ action, params: normalizeActionParams(params) });
  return crypto.createHash('sha256').update(String(secret || '')).update(':').update(canonical).digest('hex').slice(0, 24);
}

function validateBrowserConfirmToken(secret, action, params = {}, token = '') {
  const requested = String(token || '').trim();
  if (!requested) return false;
  return requested === issueBrowserConfirmToken(secret, action, params);
}

function classifyBrowserAction(action, rawParams = {}, options = {}) {
  const params = normalizeActionParams(rawParams);
  const previewSummary = buildPreviewSummary(action, params, options);

  if (READ_ONLY_ACTIONS.has(action)) {
    return {
      actionClass: ActionClass.READ_ONLY,
      riskLevel: 'low',
      decision: PolicyDecision.ALLOW,
      reasonCodes: ['read-only'],
      requiresVerification: false,
      previewSummary
    };
  }

  if (action === 'delete') {
    if (!options.allowDestructiveDomActions) {
      return {
        actionClass: ActionClass.DESTRUCTIVE,
        riskLevel: 'critical',
        decision: PolicyDecision.BLOCKED,
        reasonCodes: ['destructive-dom-disabled'],
        requiresVerification: true,
        previewSummary
      };
    }
    return {
      actionClass: ActionClass.DESTRUCTIVE,
      riskLevel: 'critical',
      decision: PolicyDecision.CONFIRM_REQUIRED,
      reasonCodes: ['destructive-dom-delete'],
      requiresVerification: true,
      previewSummary
    };
  }

  if (action === 'add') {
    if (!options.allowDomHtmlAdd) {
      return {
        actionClass: ActionClass.DESTRUCTIVE,
        riskLevel: 'critical',
        decision: PolicyDecision.BLOCKED,
        reasonCodes: ['dom-html-add-disabled'],
        requiresVerification: true,
        previewSummary
      };
    }
    return {
      actionClass: ActionClass.DESTRUCTIVE,
      riskLevel: 'critical',
      decision: PolicyDecision.CONFIRM_REQUIRED,
      reasonCodes: ['dom-html-injection'],
      requiresVerification: true,
      previewSummary
    };
  }

  if (action === 'upload') {
    return {
      actionClass: ActionClass.BROAD_LOCAL_WRITE,
      riskLevel: 'high',
      decision: PolicyDecision.CONFIRM_REQUIRED,
      reasonCodes: ['browser-file-upload'],
      requiresVerification: true,
      previewSummary
    };
  }

  if (action === 'download') {
    return {
      actionClass: ActionClass.BROAD_LOCAL_WRITE,
      riskLevel: 'high',
      decision: PolicyDecision.PREVIEW_REQUIRED,
      reasonCodes: ['browser-download'],
      requiresVerification: true,
      previewSummary
    };
  }

  if (isDangerousIntent(action, params)) {
    return {
      actionClass: ActionClass.EXTERNAL_SIDE_EFFECT,
      riskLevel: 'high',
      decision: PolicyDecision.CONFIRM_REQUIRED,
      reasonCodes: ['browser-dangerous-intent-hint'],
      requiresVerification: true,
      previewSummary
    };
  }

  if (BOUNDED_BROWSER_ACTIONS.has(action)) {
    return {
      actionClass: ActionClass.SCOPED_REVERSIBLE_WRITE,
      riskLevel: 'medium',
      decision: PolicyDecision.ALLOW_WITH_VERIFICATION,
      reasonCodes: ['browser-bounded-action'],
      requiresVerification: true,
      previewSummary
    };
  }

  return {
    actionClass: ActionClass.SCOPED_REVERSIBLE_WRITE,
    riskLevel: 'medium',
    decision: PolicyDecision.ALLOW_WITH_VERIFICATION,
    reasonCodes: ['browser-default-guarded-allow'],
    requiresVerification: true,
    previewSummary
  };
}

module.exports = {
  ActionClass,
  PolicyDecision,
  buildPreviewSummary,
  classifyBrowserAction,
  issueBrowserConfirmToken,
  normalizeActionParams
  ,
  validateBrowserConfirmToken
};
