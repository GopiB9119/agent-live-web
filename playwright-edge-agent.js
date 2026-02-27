const { chromium } = require('playwright');

// Use only Edge browser for all automation
async function runWithEdge(userAction) {
  // Launch Edge (Chromium-based)
  const browser = await chromium.launch({
    channel: 'msedge',
    headless: false // Show browser for live interaction
  });
  const context = await browser.newContext();
  const page = await context.newPage();

  // Interpret and execute user action
  await userAction(page);

  // Optionally, keep browser open or close after action
  // await browser.close();
}

// Example usage: edit a field on a web page
// runWithEdge(async (page) => {
//   await page.goto('https://example.com');
//   await page.fill('#inputField', 'New Value');
// });

module.exports = { runWithEdge };
