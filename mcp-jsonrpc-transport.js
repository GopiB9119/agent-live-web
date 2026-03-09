const CLIENT_FORMAT = {
  CONTENT_LENGTH: 'content-length',
  JSONL: 'jsonl'
};

const CONTENT_LENGTH_PREFIX = Buffer.from('Content-Length:', 'utf8');

function encodeMcpMessage(message, format = CLIENT_FORMAT.JSONL) {
  const body = Buffer.from(JSON.stringify(message), 'utf8');
  if (format === CLIENT_FORMAT.CONTENT_LENGTH) {
    return Buffer.concat([Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, 'utf8'), body]);
  }
  return Buffer.concat([body, Buffer.from('\n', 'utf8')]);
}

function writeMcpMessage(stream, message, format = CLIENT_FORMAT.JSONL) {
  stream.write(encodeMcpMessage(message, format));
}

function createMcpMessageParser(onMessage, onMalformedChunk = () => {}) {
  let buffer = Buffer.alloc(0);

  return (chunk) => {
    buffer = Buffer.concat([buffer, chunk]);

    while (buffer.length) {
      if (buffer[0] === 0x0a || buffer[0] === 0x0d) {
        buffer = buffer.slice(1);
        continue;
      }

      if (looksLikeContentLengthFrame(buffer)) {
        const headerEnd = buffer.indexOf('\r\n\r\n');
        if (headerEnd === -1) return;

        const headerText = buffer.slice(0, headerEnd).toString('utf8');
        let contentLength = null;
        for (const line of headerText.split('\r\n')) {
          const separator = line.indexOf(':');
          if (separator === -1) continue;
          const name = line.slice(0, separator).trim().toLowerCase();
          const value = line.slice(separator + 1).trim();
          if (name === 'content-length') {
            const parsed = Number.parseInt(value, 10);
            if (Number.isFinite(parsed) && parsed >= 0) {
              contentLength = parsed;
            }
          }
        }

        if (!Number.isFinite(contentLength)) {
          onMalformedChunk(headerText);
          buffer = buffer.slice(headerEnd + 4);
          continue;
        }

        const frameEnd = headerEnd + 4 + contentLength;
        if (buffer.length < frameEnd) return;

        const payload = buffer.slice(headerEnd + 4, frameEnd);
        buffer = buffer.slice(frameEnd);
        emitPayload(payload, CLIENT_FORMAT.CONTENT_LENGTH, onMessage, onMalformedChunk);
        continue;
      }

      const newlineIndex = buffer.indexOf('\n');
      if (newlineIndex === -1) return;

      const lineBuffer = buffer.slice(0, newlineIndex);
      buffer = buffer.slice(newlineIndex + 1);

      const line = lineBuffer.toString('utf8').replace(/\r$/, '').trim();
      if (!line) continue;

      emitPayload(Buffer.from(line, 'utf8'), CLIENT_FORMAT.JSONL, onMessage, onMalformedChunk);
    }
  };
}

function looksLikeContentLengthFrame(buffer) {
  const prefixLength = Math.min(buffer.length, CONTENT_LENGTH_PREFIX.length);
  const partialPrefix = buffer.slice(0, prefixLength);
  return CONTENT_LENGTH_PREFIX.slice(0, prefixLength).equals(partialPrefix);
}

function emitPayload(payload, format, onMessage, onMalformedChunk) {
  try {
    const maybePromise = onMessage(JSON.parse(payload.toString('utf8')), format);
    if (maybePromise && typeof maybePromise.then === 'function') {
      maybePromise.catch((error) => {
        onMalformedChunk(`Async JSON-RPC handler failed: ${error.message}`);
      });
    }
  } catch (error) {
    onMalformedChunk(`JSON parse failed: ${error.message}`);
  }
}

module.exports = {
  CLIENT_FORMAT,
  createMcpMessageParser,
  encodeMcpMessage,
  writeMcpMessage
};
