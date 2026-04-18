// Natural language parser for browser automation commands.

function normalizeWhitespace(value) {
  return value.trim().replace(/\s+/g, ' ');
}

function unquote(value) {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function parseSelectorToken(rawSelector) {
  const selector = unquote(rawSelector);
  const lower = selector.toLowerCase();

  if (lower.startsWith('css:') || lower.startsWith('css=')) {
    return { kind: 'css', value: selector.slice(4).trim() };
  }
  if (lower.startsWith('xpath:') || lower.startsWith('xpath=')) {
    return { kind: 'xpath', value: selector.slice(6).trim() };
  }
  if (lower.startsWith('text:') || lower.startsWith('text=')) {
    return { kind: 'text', value: selector.slice(5).trim() };
  }
  if (selector.startsWith('//') || selector.startsWith('(//')) {
    return { kind: 'xpath', value: selector };
  }
  if (/^[#.[]/.test(selector) || /[>~:]/.test(selector)) {
    return { kind: 'css', value: selector };
  }
  return { kind: 'text', value: selector };
}

function selectorParams(selectorToken) {
  if (selectorToken.kind === 'css') return { selector: selectorToken.value };
  if (selectorToken.kind === 'xpath') return { xpath: selectorToken.value };
  return { text: selectorToken.value };
}

function parseWaitDuration(value, unit) {
  const amount = Number(value);
  if (!Number.isFinite(amount) || amount < 0) return null;
  if (!unit || /^ms|milliseconds?$/i.test(unit)) return amount;
  return amount * 1000;
}

function normalizeUrl(raw) {
  const url = unquote(raw);
  if (/^https?:\/\//i.test(url)) return url;
  return `https://${url}`;
}

/**
 * Parses a natural language command and returns { action, params }.
 */
function parseCommand(input) {
  if (!input || typeof input !== 'string') {
    return { action: 'unknown', params: { command: '' } };
  }

  const command = normalizeWhitespace(input);
  const lowered = command.toLowerCase();

  if (/^(go to|navigate to|open) /i.test(command)) {
    const match = command.match(/^(?:go to|navigate to|open)\s+(.+)$/i);
    return { action: 'goto', params: { url: normalizeUrl(match[1]) } };
  }
  if (/^https?:\/\//i.test(command) || /^[\w.-]+\.[a-z]{2,}/i.test(command)) {
    return { action: 'goto', params: { url: normalizeUrl(command) } };
  }

  {
    const searchMatch = command.match(/^search(?:\s+for)?\s+(.+)$/i);
    if (searchMatch) {
      return { action: 'search', params: { query: unquote(searchMatch[1]) } };
    }
  }

  if (lowered.startsWith('click ')) {
    const match = command.match(/^click\s+(.+)$/i);
    const token = parseSelectorToken(match[1]);
    return { action: 'click', params: selectorParams(token) };
  }

  {
    const typeMatch = command.match(/^type\s+(.+?)\s+(?:in|into)\s+(.+)$/i);
    if (typeMatch) {
      const value = unquote(typeMatch[1]);
      const token = parseSelectorToken(typeMatch[2]);
      return { action: 'type', params: { ...selectorParams(token), value } };
    }
  }

  {
    const editMatch = command.match(/^(?:edit|fill|set|update)\s+(.+?)\s+(?:to|with)\s+(.+)$/i);
    if (editMatch) {
      const token = parseSelectorToken(editMatch[1]);
      const value = unquote(editMatch[2]);
      return { action: 'edit', params: { ...selectorParams(token), value } };
    }
  }

  {
    const deleteMatch = command.match(/^(?:delete|remove)\s+(.+)$/i);
    if (deleteMatch) {
      const token = parseSelectorToken(deleteMatch[1]);
      return { action: 'delete', params: selectorParams(token) };
    }
  }

  {
    const existsMatch = command.match(/^(?:check|verify)\s+(?:if\s+)?(.+?)(?:\s+(?:exists|is visible|is present))?$/i);
    if (existsMatch && existsMatch[1]) {
      const token = parseSelectorToken(existsMatch[1]);
      return { action: 'exists', params: selectorParams(token) };
    }
  }

  {
    const extractMatch = command.match(/^(?:extract|get|read)\s+text(?:\s+from)?\s+(.+)$/i);
    if (extractMatch) {
      const token = parseSelectorToken(extractMatch[1]);
      return { action: 'getText', params: selectorParams(token) };
    }
  }

  {
    const waitForMatch = command.match(/^wait\s+for\s+(.+)$/i);
    if (waitForMatch) {
      const token = parseSelectorToken(waitForMatch[1]);
      return { action: 'waitFor', params: selectorParams(token) };
    }
  }

  {
    const waitMatch = command.match(/^wait\s+(\d+)\s*(ms|milliseconds?|s|sec|secs|seconds)?$/i);
    if (waitMatch) {
      const ms = parseWaitDuration(waitMatch[1], waitMatch[2]);
      if (ms !== null) return { action: 'wait', params: { ms } };
    }
  }

  {
    const uploadMatch = command.match(/^upload\s+(.+?)\s+(?:to|into)\s+(.+)$/i);
    if (uploadMatch) {
      const filePath = unquote(uploadMatch[1]);
      const token = parseSelectorToken(uploadMatch[2]);
      return { action: 'upload', params: { ...selectorParams(token), filePath } };
    }
  }

  {
    const downloadMatch = command.match(/^download\s+(.+?)(?:\s+(?:to|as)\s+(.+))?$/i);
    if (downloadMatch) {
      const token = parseSelectorToken(downloadMatch[1]);
      const savePath = downloadMatch[2] ? unquote(downloadMatch[2]) : undefined;
      return { action: 'download', params: { ...selectorParams(token), savePath } };
    }
  }

  {
    const screenshotMatch = command.match(/^(?:take\s+)?screenshot(?:\s+(?:to|as|at)\s+(.+))?$/i);
    if (screenshotMatch) {
      const path = screenshotMatch[1] ? unquote(screenshotMatch[1]) : undefined;
      return { action: 'screenshot', params: { path } };
    }
  }

  {
    const traceStart = command.match(/^(?:start|begin)\s+trace$/i);
    if (traceStart) {
      return { action: 'startTrace', params: {} };
    }
  }

  {
    const traceStop = command.match(/^(?:stop|end)\s+trace(?:\s+(?:to|as|at)\s+(.+))?$/i);
    if (traceStop) {
      const path = traceStop[1] ? unquote(traceStop[1]) : undefined;
      return { action: 'stopTrace', params: { path } };
    }
  }

  {
    const scrollMatch = command.match(/^scroll\s+(down|up)(?:\s+(\d+))?$/i);
    if (scrollMatch) {
      return {
        action: 'scroll',
        params: {
          direction: scrollMatch[1].toLowerCase(),
          amount: scrollMatch[2] ? Number(scrollMatch[2]) : 600
        }
      };
    }
  }

  // Navigation shortcuts
  if (/^(go\s+)?back$/i.test(command)) {
    return { action: 'back', params: {} };
  }
  if (/^(go\s+)?forward$/i.test(command)) {
    return { action: 'forward', params: {} };
  }
  if (/^(refresh|reload)$/i.test(command)) {
    return { action: 'refresh', params: {} };
  }

  // Press key
  {
    const pressMatch = command.match(/^press\s+(.+)$/i);
    if (pressMatch) {
      return { action: 'press', params: { key: unquote(pressMatch[1]) } };
    }
  }

  // Hover
  {
    const hoverMatch = command.match(/^hover(?:\s+(?:over|on))?\s+(.+)$/i);
    if (hoverMatch) {
      const token = parseSelectorToken(hoverMatch[1]);
      return { action: 'hover', params: selectorParams(token) };
    }
  }

  // Select option from dropdown
  {
    const selectMatch = command.match(/^(?:select|choose)\s+(.+?)\s+(?:from|in)\s+(.+)$/i);
    if (selectMatch) {
      const value = unquote(selectMatch[1]);
      const token = parseSelectorToken(selectMatch[2]);
      return { action: 'select', params: { ...selectorParams(token), value } };
    }
  }

  // Focus element
  {
    const focusMatch = command.match(/^focus(?:\s+on)?\s+(.+)$/i);
    if (focusMatch) {
      const token = parseSelectorToken(focusMatch[1]);
      return { action: 'focus', params: selectorParams(token) };
    }
  }

  // Clear input
  {
    const clearMatch = command.match(/^clear\s+(.+)$/i);
    if (clearMatch) {
      const token = parseSelectorToken(clearMatch[1]);
      return { action: 'clear', params: selectorParams(token) };
    }
  }

  // Double-click
  {
    const dblClickMatch = command.match(/^(?:double[- ]?click|dblclick)\s+(.+)$/i);
    if (dblClickMatch) {
      const token = parseSelectorToken(dblClickMatch[1]);
      return { action: 'doubleClick', params: selectorParams(token) };
    }
  }

  // Right-click
  {
    const rightClickMatch = command.match(/^(?:right[- ]?click|contextmenu)\s+(.+)$/i);
    if (rightClickMatch) {
      const token = parseSelectorToken(rightClickMatch[1]);
      return { action: 'rightClick', params: selectorParams(token) };
    }
  }

  return { action: 'unknown', params: { command } };
}

module.exports = { parseCommand, parseSelectorToken };
