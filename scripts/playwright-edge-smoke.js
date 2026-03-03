const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    await page.goto('https://example.com', { waitUntil: 'domcontentloaded' });
    const title = await page.title();
    console.log(`[PlaywrightSmoke] title=${title}`);

    if (title !== 'Example Domain') {
      console.error(`[PlaywrightSmoke] Unexpected title: ${title}`);
      process.exit(1);
      return;
    }
  } finally {
    await browser.close();
  }
})();
