// agent-logic.js
// This file provides a simple agent interface for user-driven web actions using Playwright and Edge.
const { runWithEdge } = require('./playwright-edge-agent');

/**
 * Interprets user instructions and maps them to Playwright actions.
 * @param {string} instruction - The user's command (e.g., 'edit', 'update', 'delete', 'add', 'screenshot', 'getText', 'type').
 * @param {object} params - Parameters for the action (e.g., selector, value, url, path).
 */
async function agentAction(instruction, params) {
  await runWithEdge(async (page) => {
    if (instruction === 'goto' && params.url) {
      await page.goto(params.url);
    } else if (instruction === 'edit' && params.selector && params.value) {
      await page.fill(params.selector, params.value);
    } else if (instruction === 'click' && params.selector) {
      await page.click(params.selector);
    } else if (instruction === 'delete' && params.selector) {
      await page.evaluate((sel) => {
        const el = document.querySelector(sel);
        if (el) el.remove();
      }, params.selector);
    } else if (instruction === 'add' && params.parentSelector && params.html) {
      await page.evaluate(({ parentSelector, html }) => {
        const parent = document.querySelector(parentSelector);
        if (parent) {
          const temp = document.createElement('div');
          temp.innerHTML = html;
          parent.appendChild(temp.firstElementChild);
        }
      }, params);
    } else if (instruction === 'screenshot' && params.path) {
      await page.screenshot({ path: params.path, fullPage: true });
    } else if (instruction === 'getText' && params.selector) {
      const text = await page.textContent(params.selector);
      console.log('Text content:', text);
    } else if (instruction === 'type' && params.selector && params.value) {
      await page.type(params.selector, params.value);
    } else {
      throw new Error('Unsupported instruction or missing parameters');
    }
  });
}

module.exports = { agentAction };
