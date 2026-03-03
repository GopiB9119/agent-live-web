#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="agent-live-web-jaeger"

if ! command -v docker >/dev/null 2>&1; then
  echo "[Jaeger] docker is required but not found on PATH."
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
  echo "[Jaeger] stopped: ${CONTAINER_NAME}"
else
  echo "[Jaeger] not running."
fi
