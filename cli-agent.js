const { EdgeSession } = require('./edge-session');
const { parseCommand } = require('./nl-command-parser');
const readline = require('readline');

const HELP_TEXT = `
Natural language commands:
- open https://example.com
- click css:#submit
- click xpath://button[@type='submit']
- click text:Sign in
- search latest ai news
- type "john@example.com" in css:#email
- edit css:#name to "John Doe"
- check if css:.result exists
- extract text from text:Welcome
- wait for css:.loaded
- wait 2s
- upload ./file.pdf to css:input[type="file"]
- download css:a.download to ./downloads/report.pdf
- scroll down 800
- screenshot to ./screenshots/page.png
- start trace
- stop trace to ./traces/run.zip

Session commands:
- help
- new session
- exit
`.trim();

(async () => {
  const session = new EdgeSession();
  await session.open();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  console.log('Edge browser session started.');
  console.log('Type "help" for commands, "new session" for a fresh browser, or "exit" to quit.');

  async function shutdown() {
    await session.close();
    rl.close();
  }

  async function handleCommand(input) {
    const raw = input.trim();
    const lowered = raw.toLowerCase();

    if (!raw) return;
    if (lowered === 'help') {
      console.log(HELP_TEXT);
      return;
    }
    if (lowered === 'new session') {
      await session.newSession();
      console.log('New Edge browser session started.');
      return;
    }
    if (lowered === 'exit' || lowered === 'quit') {
      await shutdown();
      process.exit(0);
      return;
    }

    const parsed = parseCommand(raw);
    if (parsed.action === 'unknown') {
      console.log('Unknown or unsupported command.');
      console.log('Type "help" to see supported commands.');
      return;
    }

    try {
      const result = await session.act(parsed.action, parsed.params);
      if (result !== undefined) {
        console.log(`Result: ${JSON.stringify(result)}`);
      }
    } catch (error) {
      console.error(`Error: ${error.message}`);
    }
  }

  rl.on('line', handleCommand);
  rl.on('close', async () => {
    await session.close();
  });
  process.on('SIGINT', async () => {
    await shutdown();
    process.exit(0);
  });
})();
