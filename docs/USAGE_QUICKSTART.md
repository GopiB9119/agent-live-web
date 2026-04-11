# Quickstart: Zero to Working

## 1) Prerequisites
- Node.js 18+
- Python 3.11+ (for the Python agent runtime)
- VS Code with GitHub Copilot extension
- Microsoft Edge browser installed

## 2) Install
```powershell
cd "<repo-path>"
npm install
npm run install:edge
```

## 3) Configure environment
```powershell
# Copy the example and fill in your values
cp .env.example .env
```

Edit `.env` and set at minimum:
```
AGENT_PROVIDER=openai
OPENAI_API_KEY=sk-...
AGENT_MODEL=gpt-4o
```

Or for Azure:
```
AGENT_PROVIDER=azure
azure_key=...
azure_endpoint_uri=https://your-resource.openai.azure.com/
azure_deployment_name=gpt-4o
```

## 4) Set up Python agent (optional)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r agent/agent/requirements.txt
```

## 5) Verify everything works
```powershell
npm test                 # JS syntax validation + JS unit tests
npm run test:py          # Python unit tests (requires .venv activated)
npm run test:all         # Both JS and Python in one command
```

## 6) Use: VS Code MCP mode (recommended)
1. Open this folder in VS Code.
2. In the MCP panel, start or restart `playwright-edge`.
3. The agent can now control Edge through Playwright.

## 7) Use: Terminal CLI mode
```powershell
npm run agent:live-web
```
Type natural language commands:
```
open https://github.com
click text:Sign in
type "user@example.com" in css:#login_field
screenshot to ./screenshots/login.png
```
Type `help` for all commands, `exit` to quit.

## 8) Use: Python agent mode
```powershell
.\.venv\Scripts\Activate.ps1
cd agent/agent
python agent.py
```
The Python agent has its own conversation loop with memory, session continuity, and tool routing.

## 9) If owner lock error appears
```powershell
node scripts/set-mcp-owner.js none
```
Then restart `playwright-edge` in VS Code.

## 10) Run a structured web task
Task examples are in `.github/skills/web-works/examples/`. Use the task brief prompt or pass JSON directly to the agent.

## Runtime paths summary
| Path | Best for | Command |
|------|----------|---------|
| VS Code MCP | Interactive work, debugging | Start `playwright-edge` in MCP panel |
| Terminal CLI | Supervised browser control | `npm run agent:live-web` |
| Python agent | CI, batch, scheduling | `python agent/agent/agent.py` |
