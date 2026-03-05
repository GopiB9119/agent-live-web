import os
import json
import re
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse, urlencode


class OAuthManager:
    """
    In-memory OAuth profile/token manager for the Python agent tool layer.
    """

    def __init__(self, to_bool_fn, is_private_or_local_host_fn, web_fetch_allow_private_env: str):
        self._to_bool = to_bool_fn
        self._is_private_or_local_host = is_private_or_local_host_fn
        self._web_fetch_allow_private_env = str(web_fetch_allow_private_env or "AGENT_WEB_FETCH_ALLOW_PRIVATE_HOSTS")
        self.profile_store = {}
        self.token_cache = {}

    @staticmethod
    def mask_token(token: str, keep_prefix: int = 6, keep_suffix: int = 4) -> str:
        value = str(token or "").strip()
        if not value:
            return ""
        if len(value) <= (keep_prefix + keep_suffix):
            return "*" * len(value)
        return f"{value[:keep_prefix]}...{value[-keep_suffix:]}"

    @staticmethod
    def profile_from_kwargs(kwargs):
        payload = kwargs or {}
        return {
            "token_url": str(payload.get("token_url", "")).strip(),
            "client_id": str(payload.get("client_id", "")).strip(),
            "client_secret": str(payload.get("client_secret", "")).strip(),
            "scope": str(payload.get("scope", "")).strip(),
            "audience": str(payload.get("audience", "")).strip(),
            "grant_type": str(payload.get("grant_type", "client_credentials")).strip() or "client_credentials",
            "refresh_token": str(payload.get("refresh_token", "")).strip(),
            "timeout_sec": float(payload.get("timeout_sec", 20)),
            "extra_params": payload.get("extra_params", {}) if isinstance(payload.get("extra_params", {}), dict) else {},
        }

    @staticmethod
    def cache_key(profile_name: str, profile: dict) -> str:
        if profile_name:
            return f"profile:{profile_name}"
        token_url = str((profile or {}).get("token_url", "")).strip()
        client_id = str((profile or {}).get("client_id", "")).strip()
        return f"direct:{token_url}|{client_id}"

    def get_cached_token(self, cache_key: str, min_ttl_sec: int = 60):
        entry = self.token_cache.get(cache_key)
        if not isinstance(entry, dict):
            return None
        expires_at = float(entry.get("expires_at", 0.0) or 0.0)
        now_ts = time.time()
        if expires_at <= (now_ts + max(1, int(min_ttl_sec))):
            return None
        token = str(entry.get("access_token", "")).strip()
        if not token:
            return None
        return token

    def fetch_token(self, profile: dict):
        profile_obj = profile or {}
        token_url = str(profile_obj.get("token_url", "")).strip()
        client_id = str(profile_obj.get("client_id", "")).strip()
        client_secret = str(profile_obj.get("client_secret", "")).strip()
        grant_type = str(profile_obj.get("grant_type", "client_credentials")).strip() or "client_credentials"
        refresh_token = str(profile_obj.get("refresh_token", "")).strip()
        scope = str(profile_obj.get("scope", "")).strip()
        audience = str(profile_obj.get("audience", "")).strip()
        timeout_sec = float(profile_obj.get("timeout_sec", 20))
        extra_params = profile_obj.get("extra_params", {})
        if not isinstance(extra_params, dict):
            extra_params = {}

        if not token_url:
            raise ValueError("token_url is required.")
        if not re.match(r"^https?://", token_url, flags=re.IGNORECASE):
            raise ValueError("token_url must be http/https.")
        if not client_id:
            raise ValueError("client_id is required.")
        if not client_secret:
            raise ValueError("client_secret is required.")

        parsed = urlparse(token_url)
        host = parsed.hostname or ""
        private_fetch_allowed = self._to_bool(os.getenv(self._web_fetch_allow_private_env, "0"), False)
        if not private_fetch_allowed and self._is_private_or_local_host(host):
            raise ValueError("Blocked private/local token host by SSRF policy.")

        form = {
            "grant_type": grant_type,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        if scope:
            form["scope"] = scope
        if audience:
            form["audience"] = audience
        if grant_type == "refresh_token":
            if not refresh_token:
                raise ValueError("refresh_token grant requires refresh_token.")
            form["refresh_token"] = refresh_token
        for key, value in extra_params.items():
            if value is None:
                continue
            form[str(key)] = str(value)

        body = urlencode(form).encode("utf-8")
        req = urllib.request.Request(
            token_url,
            data=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "AgentLiveWeb/1.0 (+oauth)",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=max(1.0, min(timeout_sec, 120.0))) as response:
            raw = response.read(1024 * 1024).decode("utf-8", errors="replace")
            parsed_json = json.loads(raw)
            access_token = str(parsed_json.get("access_token", "")).strip()
            if not access_token:
                raise ValueError("Token response missing access_token.")
            token_type = str(parsed_json.get("token_type", "Bearer")).strip() or "Bearer"
            expires_in_raw = parsed_json.get("expires_in", 3600)
            try:
                expires_in = int(float(expires_in_raw))
            except Exception:
                expires_in = 3600
            expires_in = max(1, min(expires_in, 86400))
            return {
                "access_token": access_token,
                "token_type": token_type,
                "expires_in": expires_in,
                "scope": str(parsed_json.get("scope", scope)).strip(),
            }

    def resolve_bearer_token(self, profile_name: str, force_refresh: bool = False, min_ttl_sec: int = 60):
        name = str(profile_name or "").strip()
        if not name:
            raise ValueError("profile_name is required.")
        profile = self.profile_store.get(name)
        if not isinstance(profile, dict):
            raise ValueError(f"OAuth profile not found: {name}")

        key = self.cache_key(name, profile)
        if not force_refresh:
            cached = self.get_cached_token(key, min_ttl_sec=min_ttl_sec)
            if cached:
                return cached

        token_payload = self.fetch_token(profile)
        expires_at = time.time() + int(token_payload.get("expires_in", 3600))
        self.token_cache[key] = {
            "access_token": token_payload.get("access_token", ""),
            "token_type": token_payload.get("token_type", "Bearer"),
            "expires_at": float(expires_at),
            "profile_name": name,
        }
        return str(token_payload.get("access_token", "")).strip()

    async def oauth_set_profile(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        profile_name = str(kwargs.get("profile_name", "")).strip()
        if not profile_name:
            return json.dumps({"status": "failed", "error": "profile_name is required"}, ensure_ascii=True)

        try:
            profile = self.profile_from_kwargs(kwargs)
            if not profile.get("token_url"):
                return json.dumps({"status": "failed", "error": "token_url is required"}, ensure_ascii=True)
            if not profile.get("client_id"):
                return json.dumps({"status": "failed", "error": "client_id is required"}, ensure_ascii=True)
            if not profile.get("client_secret"):
                return json.dumps({"status": "failed", "error": "client_secret is required"}, ensure_ascii=True)

            self.profile_store[profile_name] = profile
            self.token_cache.pop(self.cache_key(profile_name, profile), None)
            return json.dumps(
                {
                    "status": "ok",
                    "profile_name": profile_name,
                    "token_url": profile.get("token_url", ""),
                    "client_id": profile.get("client_id", ""),
                    "grant_type": profile.get("grant_type", "client_credentials"),
                    "scope": profile.get("scope", ""),
                    "audience": profile.get("audience", ""),
                    "client_secret_set": bool(profile.get("client_secret")),
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def oauth_get_token(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        profile_name = str(kwargs.get("profile_name", "")).strip()
        force_refresh = self._to_bool(kwargs.get("force_refresh", False), False)
        include_access_token = self._to_bool(kwargs.get("include_access_token", False), False)
        min_ttl_sec = int(kwargs.get("min_ttl_sec", 60))
        min_ttl_sec = max(1, min(min_ttl_sec, 1800))

        try:
            direct_profile = self.profile_from_kwargs(kwargs)
            if profile_name:
                profile = self.profile_store.get(profile_name)
                if not isinstance(profile, dict):
                    return json.dumps({"status": "failed", "error": f"OAuth profile not found: {profile_name}"}, ensure_ascii=True)
            else:
                profile = direct_profile
                if not profile.get("token_url") or not profile.get("client_id") or not profile.get("client_secret"):
                    return json.dumps(
                        {
                            "status": "failed",
                            "error": "Provide profile_name or direct token_url/client_id/client_secret.",
                        },
                        ensure_ascii=True,
                    )

            cache_key = self.cache_key(profile_name, profile)
            if not force_refresh:
                cached = self.get_cached_token(cache_key, min_ttl_sec=min_ttl_sec)
                if cached:
                    payload = {
                        "status": "ok",
                        "source": "cache",
                        "profile_name": profile_name or "",
                        "token_preview": self.mask_token(cached),
                        "token_type": "Bearer",
                    }
                    if include_access_token:
                        payload["access_token"] = cached
                    return json.dumps(payload, ensure_ascii=True)

            token_payload = self.fetch_token(profile)
            access_token = str(token_payload.get("access_token", "")).strip()
            expires_in = int(token_payload.get("expires_in", 3600) or 3600)
            expires_at = time.time() + expires_in
            token_type = str(token_payload.get("token_type", "Bearer")).strip() or "Bearer"
            self.token_cache[cache_key] = {
                "access_token": access_token,
                "token_type": token_type,
                "expires_at": float(expires_at),
                "profile_name": profile_name or "",
            }
            payload = {
                "status": "ok",
                "source": "network",
                "profile_name": profile_name or "",
                "token_type": token_type,
                "token_preview": self.mask_token(access_token),
                "expires_in": expires_in,
                "expires_at_unix": int(expires_at),
            }
            if include_access_token:
                payload["access_token"] = access_token
            return json.dumps(payload, ensure_ascii=True)
        except urllib.error.HTTPError as e:
            return json.dumps(
                {
                    "status": "failed",
                    "error": f"HTTPError: {e.code}",
                    "reason": str(e),
                },
                ensure_ascii=True,
            )
        except urllib.error.URLError as e:
            return json.dumps({"status": "failed", "error": f"URLError: {e.reason}"}, ensure_ascii=True)
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def oauth_profiles(self, kwargs_dict=None):
        kwargs = kwargs_dict or {}
        action = str(kwargs.get("action", "list")).strip().lower() or "list"
        profile_name = str(kwargs.get("profile_name", "")).strip()

        if action in {"delete", "remove"}:
            if not profile_name:
                return json.dumps({"status": "failed", "error": "profile_name is required for delete action."}, ensure_ascii=True)
            removed_profile = self.profile_store.pop(profile_name, None)
            self.token_cache.pop(self.cache_key(profile_name, removed_profile or {}), None)
            return json.dumps(
                {
                    "status": "ok",
                    "action": "delete",
                    "profile_name": profile_name,
                    "deleted": bool(removed_profile),
                },
                ensure_ascii=True,
            )

        if action == "clear":
            profile_count = len(self.profile_store)
            token_count = len(self.token_cache)
            self.profile_store.clear()
            self.token_cache.clear()
            return json.dumps(
                {
                    "status": "ok",
                    "action": "clear",
                    "profiles_removed": profile_count,
                    "tokens_removed": token_count,
                },
                ensure_ascii=True,
            )

        rows = []
        now_ts = time.time()
        for name in sorted(self.profile_store.keys()):
            profile = self.profile_store.get(name) or {}
            cache = self.token_cache.get(self.cache_key(name, profile), {})
            expires_at = float(cache.get("expires_at", 0.0) or 0.0)
            rows.append(
                {
                    "profile_name": name,
                    "token_url": str(profile.get("token_url", "")),
                    "client_id": str(profile.get("client_id", "")),
                    "grant_type": str(profile.get("grant_type", "client_credentials")),
                    "scope": str(profile.get("scope", "")),
                    "audience": str(profile.get("audience", "")),
                    "has_cached_token": bool(cache.get("access_token")),
                    "cached_token_preview": self.mask_token(str(cache.get("access_token", ""))),
                    "cache_expires_in_sec": int(expires_at - now_ts) if expires_at > now_ts else 0,
                }
            )

        return json.dumps(
            {
                "status": "ok",
                "action": "list",
                "count": len(rows),
                "profiles": rows,
            },
            ensure_ascii=True,
        )
