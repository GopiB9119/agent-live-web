#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="agent-live-web-jaeger"
IMAGE="jaegertracing/all-in-one:1.62.0"
NETWORK_NAME="agent-live-web-tracing-net"

if ! command -v docker >/dev/null 2>&1; then
  echo "[Jaeger] docker is required but not found on PATH."
  exit 1
fi

if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
  docker network create "$NETWORK_NAME" >/dev/null
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --network "$NETWORK_NAME" \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  "$IMAGE" >/dev/null

echo "[Jaeger] UI started: http://localhost:16686"
echo "[Jaeger] container: ${CONTAINER_NAME}"
echo "[Jaeger] network: ${NETWORK_NAME}"
