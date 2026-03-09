# VS Code Quickstart

Use this when you want direct VS Code MCP mode. If you are not sure, start with `npm run agent:python` from the repo root instead. Python Agent mode is the primary product path.

If startup fails or owner lock/runtime issues appear, use `docs/TROUBLESHOOTING.md`.

## 1) Install once
```powershell
cd "C:\Users\banot\Agent Works"
npm install
npm run install:edge
```

## 2) Start MCP in VS Code
1. Open this folder in VS Code.
2. In MCP panel, start or restart `playwright-edge`.

Terminal equivalent:
```powershell
cd "C:\Users\banot\Agent Works"
npm run agent:vscode
```

## 3) If owner lock error appears
1. Stop any other MCP/agent process using the same Edge profile.
2. Restart `playwright-edge` in VS Code.
3. If lock still persists, run:
```powershell
cd "C:\Users\banot\Agent Works"
node scripts/set-mcp-owner.js none
```
Then restart `playwright-edge` again.

## 4) Optional terminal mode
```powershell
cd "C:\Users\banot\Agent Works"
npm run agent:vscode
```
