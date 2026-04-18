import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oauth_tools import OAuthManager


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class OAuthManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = OAuthManager(
            to_bool_fn=_to_bool,
            is_private_or_local_host_fn=lambda _host: False,
            web_fetch_allow_private_env="AGENT_WEB_FETCH_ALLOW_PRIVATE_HOSTS",
        )

    async def test_oauth_set_profile_does_not_expose_client_secret(self):
        raw = await self.manager.oauth_set_profile(
            {
                "profile_name": "default",
                "token_url": "https://auth.example.com/oauth/token",
                "client_id": "client-123",
                "client_secret": "super-secret",
            }
        )
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["profile_name"], "default")
        self.assertTrue(result["client_secret_set"])
        self.assertNotIn("client_secret", result)
        self.assertNotIn("super-secret", raw)

    async def test_oauth_get_token_blocks_raw_token_output(self):
        self.manager.fetch_token = lambda _profile: {
            "access_token": "secret-token-value",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        raw = await self.manager.oauth_get_token(
            {
                "token_url": "https://auth.example.com/oauth/token",
                "client_id": "client-123",
                "client_secret": "super-secret",
                "include_access_token": True,
            }
        )
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["raw_token_output_blocked"])
        self.assertFalse(result["access_token_included"])
        self.assertNotIn("access_token", result)
        self.assertNotIn("secret-token-value", raw)
        self.assertNotIn("super-secret", raw)
