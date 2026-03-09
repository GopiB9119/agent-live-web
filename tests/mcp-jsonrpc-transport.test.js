const test = require('node:test');
const assert = require('node:assert/strict');

const {
  CLIENT_FORMAT,
  createMcpMessageParser,
  encodeMcpMessage
} = require('../mcp-jsonrpc-transport');

test('encodeMcpMessage uses newline-delimited json by default', () => {
  const encoded = encodeMcpMessage({ jsonrpc: '2.0', id: 1, method: 'ping' });
  assert.equal(encoded.toString('utf8'), '{"jsonrpc":"2.0","id":1,"method":"ping"}\n');
});

test('createMcpMessageParser parses jsonl messages', () => {
  const seen = [];
  const parser = createMcpMessageParser((message, format) => {
    seen.push({ message, format });
  });

  parser(Buffer.from('{"jsonrpc":"2.0","id":1,"method":"ping"}\n', 'utf8'));

  assert.equal(seen.length, 1);
  assert.equal(seen[0].format, CLIENT_FORMAT.JSONL);
  assert.equal(seen[0].message.method, 'ping');
});

test('createMcpMessageParser parses content-length framed messages', () => {
  const seen = [];
  const parser = createMcpMessageParser((message, format) => {
    seen.push({ message, format });
  });

  parser(encodeMcpMessage({ jsonrpc: '2.0', id: 2, method: 'tools/list' }, CLIENT_FORMAT.CONTENT_LENGTH));

  assert.equal(seen.length, 1);
  assert.equal(seen[0].format, CLIENT_FORMAT.CONTENT_LENGTH);
  assert.equal(seen[0].message.method, 'tools/list');
});
