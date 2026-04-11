# --- Stage 1: Node.js runtime ---
FROM node:22-slim AS base

WORKDIR /app

# Install system dependencies for Playwright Edge
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Copy package files and install Node dependencies
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Copy Python requirements and install
COPY agent/agent/requirements.txt ./agent/agent/requirements.txt
RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir -r agent/agent/requirements.txt

# Copy application source
COPY index.js edge-session.js nl-command-parser.js \
     playwright-edge-mcp.js cli-agent.js tracing.js ./
COPY scripts/ ./scripts/
COPY agent/ ./agent/
COPY .env.example ./.env.example

# Install Playwright browsers (Edge).
# Set INSTALL_MSEDGE=false to skip (e.g. in CI environments that provide their own browser).
ARG INSTALL_MSEDGE=true
RUN if [ "$INSTALL_MSEDGE" = "true" ]; then npx playwright install --with-deps msedge; else echo "Skipping Edge installation (INSTALL_MSEDGE=false)"; fi

# Set environment defaults
ENV NODE_ENV=production \
    PLAYWRIGHT_MCP_OWNER=docker \
    PLAYWRIGHT_MCP_PERSIST_PROFILE=false \
    EDGE_LOCAL_OPERATOR_MODE=true \
    EDGE_TRACING_ENABLED=false \
    AGENT_MEMORY_AUTO_LOG=false \
    AGENT_RUN_COMMAND_SECURITY_MODE=restricted

EXPOSE 4318 16686

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD node -e "require('./index.js'); console.log('healthy')" || exit 1

# Default: run the MCP server
CMD ["node", "playwright-edge-mcp.js"]
