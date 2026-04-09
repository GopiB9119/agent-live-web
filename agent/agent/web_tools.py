import json
import os
import re
import html as html_lib
import urllib.request
import urllib.error
from urllib.parse import urlparse
try:
    from runtime_utils import redact_sensitive_data as _redact_sensitive_data
except Exception:
    from .runtime_utils import redact_sensitive_data as _redact_sensitive_data


class WebManager:
    """
    Web-fetch manager with SSRF checks and optional OAuth bearer injection.
    """

    def __init__(
        self,
        to_bool_fn,
        is_private_or_local_host_fn,
        oauth_manager,
        web_fetch_allow_private_env: str,
    ):
        self._to_bool = to_bool_fn
        self._is_private_or_local_host = is_private_or_local_host_fn
        self.oauth_manager = oauth_manager
        self.web_fetch_allow_private_env = str(web_fetch_allow_private_env or "AGENT_WEB_FETCH_ALLOW_PRIVATE_HOSTS")

    @staticmethod
    def _strip_html_to_text(raw_html: str) -> str:
        no_script = re.sub(
            r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>",
            " ",
            raw_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        no_style = re.sub(
            r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>",
            " ",
            no_script,
            flags=re.IGNORECASE | re.DOTALL,
        )
        no_tags = re.sub(r"<[^>]+>", " ", no_style)
        unescaped = html_lib.unescape(no_tags)
        return re.sub(r"\s+", " ", unescaped).strip()

    @staticmethod
    def _extract_title(raw_html: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return re.sub(r"\s+", " ", html_lib.unescape(match.group(1))).strip()

    @staticmethod
    def _json_response(payload, max_chars=50000):
        return json.dumps(_redact_sensitive_data(payload, max_chars=max_chars), ensure_ascii=True)

    async def web_fetch(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        url = str(kwargs.get("url", "")).strip()
        max_chars = int(kwargs.get("max_chars", 50000))
        extract_text = bool(kwargs.get("extract_text", True))
        allow_private_hosts = self._to_bool(kwargs.get("allow_private_hosts", False), False)
        timeout_sec = float(kwargs.get("timeout_sec", 20))
        headers_obj = kwargs.get("headers", {})
        bearer_token = str(kwargs.get("bearer_token", "")).strip()
        oauth_profile = str(kwargs.get("oauth_profile", "")).strip()
        oauth_force_refresh = self._to_bool(kwargs.get("oauth_force_refresh", False), False)
        oauth_min_ttl_sec = int(kwargs.get("oauth_min_ttl_sec", 60))
        oauth_min_ttl_sec = max(1, min(oauth_min_ttl_sec, 1800))
        auth_mode = "none"
        max_chars = max(500, min(max_chars, 500000))

        if not url:
            return self._json_response({"status": "failed", "error": "url is required"})
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            return self._json_response({"status": "failed", "error": "Only http/https URLs are supported."})
        if headers_obj is None:
            headers_obj = {}
        if not isinstance(headers_obj, dict):
            return self._json_response({"status": "failed", "error": "headers must be an object"})

        auth_obj = kwargs.get("auth", {})
        if isinstance(auth_obj, dict):
            auth_type = str(auth_obj.get("type", "")).strip().lower()
            if auth_type == "bearer":
                bearer_token = str(auth_obj.get("token", bearer_token)).strip()
            elif auth_type in {"oauth_profile", "oauth"}:
                oauth_profile = str(auth_obj.get("profile_name", oauth_profile)).strip()
                oauth_force_refresh = self._to_bool(auth_obj.get("force_refresh", oauth_force_refresh), oauth_force_refresh)
                oauth_min_ttl_sec = int(auth_obj.get("min_ttl_sec", oauth_min_ttl_sec) or oauth_min_ttl_sec)
                oauth_min_ttl_sec = max(1, min(oauth_min_ttl_sec, 1800))

        if oauth_profile:
            try:
                bearer_token = self.oauth_manager.resolve_bearer_token(
                    oauth_profile,
                    force_refresh=oauth_force_refresh,
                    min_ttl_sec=oauth_min_ttl_sec,
                )
                auth_mode = "oauth_profile"
            except Exception as e:
                return self._json_response(
                    {
                        "status": "failed",
                        "error": f"OAuth token resolve failed: {e}",
                        "oauth_profile": oauth_profile,
                    }
                )
        elif bearer_token:
            auth_mode = "bearer"

        try:
            parsed = urlparse(url)
        except Exception:
            return self._json_response({"status": "failed", "error": "Invalid URL format.", "url": url})

        hostname = parsed.hostname or ""
        private_fetch_allowed = allow_private_hosts or self._to_bool(os.getenv(self.web_fetch_allow_private_env, "0"), False)
        if not private_fetch_allowed and self._is_private_or_local_host(hostname):
            return self._json_response(
                {
                    "status": "blocked",
                    "error": "Blocked private/local host fetch by SSRF policy.",
                    "url": url,
                    "host": hostname,
                }
            )

        try:
            request_headers = {
                "User-Agent": "AgentLiveWeb/1.0 (+local)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            for key, value in headers_obj.items():
                key_text = str(key).strip()
                value_text = str(value).strip()
                if not key_text:
                    continue
                if any(ch in key_text for ch in ["\n", "\r", ":"]) or any(ch in value_text for ch in ["\n", "\r"]):
                    return self._json_response({"status": "failed", "error": "Invalid header key/value."})
                request_headers[key_text] = value_text
            if bearer_token and "Authorization" not in request_headers:
                request_headers["Authorization"] = f"Bearer {bearer_token}"

            req = urllib.request.Request(
                url,
                headers=request_headers,
            )
            with urllib.request.urlopen(req, timeout=max(1.0, min(timeout_sec, 120.0))) as response:
                body_bytes = response.read(max_chars * 4)
                charset = response.headers.get_content_charset() or "utf-8"
                body = body_bytes.decode(charset, errors="replace")
                body = body[:max_chars]
                text = self._strip_html_to_text(body)[:max_chars] if extract_text else ""
                mask_fn = getattr(self.oauth_manager, "mask_token", None)
                token_preview = mask_fn(bearer_token) if callable(mask_fn) else ""
                return self._json_response(
                    {
                        "status": "ok",
                        "url": response.geturl(),
                        "status_code": response.status,
                        "content_type": response.headers.get("Content-Type", ""),
                        "title": self._extract_title(body),
                        "body": body,
                        "text": text,
                        "auth": {
                            "mode": auth_mode,
                            "oauth_profile": oauth_profile,
                            "token_preview": token_preview,
                        },
                    },
                    max_chars=max_chars,
                )
        except urllib.error.HTTPError as e:
            return self._json_response(
                {
                    "status": "failed",
                    "error": f"HTTPError: {e.code}",
                    "url": url,
                    "reason": str(e),
                }
            )
        except urllib.error.URLError as e:
            return self._json_response(
                {
                    "status": "failed",
                    "error": f"URLError: {e.reason}",
                    "url": url,
                }
            )
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e), "url": url})
