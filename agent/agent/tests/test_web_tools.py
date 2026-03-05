import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web_tools import WebManager


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class _FakeOAuthManager:
    def __init__(self):
        self.calls = []

    def resolve_bearer_token(self, profile_name, force_refresh=False, min_ttl_sec=60):
        self.calls.append((profile_name, force_refresh, min_ttl_sec))
        return "TOKEN_ABC_123"

    @staticmethod
    def mask_token(token):
        token = str(token or "")
        return f"{token[:5]}***" if token else ""


class _FakeHeaders(dict):
    def get_content_charset(self):
        return "utf-8"


class _FakeResponse:
    def __init__(self, body, status=200, url="https://example.com/final", content_type="text/html; charset=utf-8"):
        self._body = body.encode("utf-8")
        self.status = status
        self._url = url
        self.headers = _FakeHeaders({"Content-Type": content_type})

    def read(self, _max_bytes):
        return self._body

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class WebManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.oauth = _FakeOAuthManager()
        self.manager = WebManager(
            to_bool_fn=_to_bool,
            is_private_or_local_host_fn=lambda host: str(host).lower() in {"localhost", "127.0.0.1"},
            oauth_manager=self.oauth,
            web_fetch_allow_private_env="AGENT_WEB_FETCH_ALLOW_PRIVATE_HOSTS",
        )

    async def test_strip_html_and_title_helpers(self):
        html = "<html><head><title>  Test Page </title><style>.x{}</style></head><body><script>x=1</script><h1>Hello</h1></body></html>"
        text = self.manager._strip_html_to_text(html)
        title = self.manager._extract_title(html)
        self.assertEqual(title, "Test Page")
        self.assertIn("Hello", text)
        self.assertNotIn("script", text.lower())

    async def test_web_fetch_blocks_private_host(self):
        raw = await self.manager.web_fetch({"url": "http://localhost:9222/"})
        result = json.loads(raw)
        self.assertEqual(result["status"], "blocked")

    async def test_web_fetch_rejects_invalid_headers(self):
        raw = await self.manager.web_fetch(
            {
                "url": "https://example.com",
                "headers": {"Bad\nHeader": "x"},
            }
        )
        result = json.loads(raw)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Invalid header", result["error"])

    async def test_web_fetch_with_oauth_profile_and_mocked_response(self):
        captured = {}

        def _fake_urlopen(req, timeout=0):
            captured["authorization"] = req.get_header("Authorization")
            captured["timeout"] = timeout
            return _FakeResponse("<html><head><title>Example</title></head><body>hello world</body></html>")

        with patch("web_tools.urllib.request.urlopen", side_effect=_fake_urlopen):
            raw = await self.manager.web_fetch(
                {
                    "url": "https://example.com",
                    "oauth_profile": "default",
                    "extract_text": True,
                }
            )
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["title"], "Example")
        self.assertIn("hello world", result["text"].lower())
        self.assertEqual(result["auth"]["mode"], "oauth_profile")
        self.assertTrue(self.oauth.calls)
        self.assertTrue(str(captured.get("authorization", "")).startswith("Bearer "))


if __name__ == "__main__":
    unittest.main()
