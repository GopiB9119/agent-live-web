const fs = require('fs');
const path = require('path');

const requested = String(process.argv[2] || '').trim().toLowerCase();
const valid = new Set(['vscode', 'python', 'none']);

if (!valid.has(requested)) {
  console.error('Usage: node scripts/set-mcp-owner.js <vscode|python|none>');
  process.exit(1);
}

const ownerFile =
  process.env.PLAYWRIGHT_MCP_OWNER_FILE || path.join(process.cwd(), '.playwright-mcp', 'active-owner.txt');

fs.mkdirSync(path.dirname(ownerFile), { recursive: true });

if (requested === 'none') {
  if (fs.existsSync(ownerFile)) fs.unlinkSync(ownerFile);
  console.log(`MCP active owner cleared (file removed): ${ownerFile}`);
  process.exit(0);
}

fs.writeFileSync(ownerFile, `${requested}\n`, 'utf8');
console.log(`MCP active owner set to '${requested}' in ${ownerFile}`);
