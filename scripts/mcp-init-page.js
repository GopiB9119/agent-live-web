const wiredContexts = new WeakSet();

async function pruneBlankTabs(context) {
  const pages = context.pages().filter((page) => !page.isClosed());
  const nonBlankPages = pages.filter((page) => page.url() !== 'about:blank');
  if (nonBlankPages.length === 0) return;

  for (const page of pages) {
    if (page.url() === 'about:blank') {
      await page.close({ runBeforeUnload: false }).catch(() => {});
    }
  }
}

async function focusLatestRealPage(context) {
  const pages = context.pages().filter((page) => !page.isClosed());
  const nonBlankPages = pages.filter((page) => page.url() !== 'about:blank');
  if (!nonBlankPages.length) return;
  await nonBlankPages[nonBlankPages.length - 1].bringToFront().catch(() => {});
}

module.exports.default = async function initPage({ page }) {
  const context = page.context();

  if (!wiredContexts.has(context)) {
    wiredContexts.add(context);
    context.on('page', async () => {
      await pruneBlankTabs(context);
      await focusLatestRealPage(context);
    });
  }

  await pruneBlankTabs(context);
  await focusLatestRealPage(context);
};
