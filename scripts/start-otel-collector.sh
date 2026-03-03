#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/scripts/otel-collector-config.yaml"
CONTAINER_NAME="agent-live-web-otel"
IMAGE="otel/opentelemetry-collector:0.111.0"
NETWORK_NAME="agent-live-web-tracing-net"

if ! command -v docker >/dev/null 2>&1; then
  echo "[OTEL] docker is required but not found on PATH."
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[OTEL] missing config: $CONFIG_PATH"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
  docker network create "$NETWORK_NAME" >/dev/null
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --network "$NETWORK_NAME" \
  -p 4318:4318 \
  -v "$CONFIG_PATH:/etc/otelcol/config.yaml:ro" \
  "$IMAGE" >/dev/null

echo "[OTEL] collector started: ${CONTAINER_NAME}"
echo "[OTEL] endpoint: http://localhost:4318/v1/traces"
echo "[OTEL] network: ${NETWORK_NAME}"
