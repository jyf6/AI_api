from __future__ import annotations

import secrets
import re
import time
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from curl_cffi import requests

from utils.pkce import generate_pkce

AUTH_BASE = "https://auth.openai.com"
PLATFORM_BASE = "https://platform.openai.com"
OAUTH_CLIENT_ID = "app_2SKx67EdpoN0G6j64rFvigXD"
OAUTH_REDIRECT_URI = f"{PLATFORM_BASE}/auth/callback"
OAUTH_AUDIENCE = "https://api.openai.com/v1"
AUTH0_CLIENT = "eyJuYW1lIjoiYXV0aDAtc3BhLWpzIiwidmVyc2lvbiI6IjEuMjEuMCJ9"


class OAuthManager:
    """Manages OAuth + PKCE sessions for OpenAI browser authorization."""

    SESSION_TTL_SECONDS = 600

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def start_session(self) -> dict[str, str]:
        """Generate an OAuth + PKCE authorization URL for the user to open in browser."""
        verifier, challenge = generate_pkce()
        session_id = secrets.token_hex(16)
        state = f"{session_id}.{secrets.token_urlsafe(16)}"
        nonce = secrets.token_urlsafe(32)

        params = {
            "issuer": AUTH_BASE,
            "client_id": OAUTH_CLIENT_ID,
            "audience": OAUTH_AUDIENCE,
            "redirect_uri": OAUTH_REDIRECT_URI,
            "device_id": secrets.token_hex(16),
            "screen_hint": "login_or_signup",
            "max_age": "0",
            "scope": "openid profile email offline_access",
            "response_type": "code",
            "response_mode": "query",
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "auth0Client": AUTH0_CLIENT,
        }

        authorize_url = f"{AUTH_BASE}/api/accounts/authorize?{urlencode(params)}"

        now = time.time()
        with self._lock:
            # Clean expired sessions
            self._sessions = {
                k: v for k, v in self._sessions.items()
                if now - v.get("created_at", 0) < self.SESSION_TTL_SECONDS
            }
            self._sessions[session_id] = {
                "code_verifier": verifier,
                "state": state,
                "created_at": now,
            }

        return {"session_id": session_id, "authorize_url": authorize_url}

    def finish_session(
        self, callback_url: str, proxy: str = "", session_id: str = ""
    ) -> dict[str, Any]:
        """Parse callback URL, exchange code for tokens, return token data."""
        raw = callback_url.strip()
        code = ""
        state = ""

        if raw.startswith(("http://", "https://")):
            parsed = parse_qs(urlparse(raw).query)
            code = str((parsed.get("code") or [""])[0]).strip()
            state = str((parsed.get("state") or [""])[0]).strip()
            if not code:
                err = str((parsed.get("error_description") or parsed.get("error") or [""])[0]).strip()
                raise ValueError(err or "Callback URL does not contain code parameter")
        else:
            code = raw

        state_sid = state.split(".", 1)[0] if state else ""
        candidate_sids = [sid for sid in (state_sid, session_id) if sid]

        verifier = ""
        with self._lock:
            for sid in candidate_sids:
                if sid in self._sessions:
                    verifier = self._sessions[sid]["code_verifier"]
                    break

        if not verifier:
            raise ValueError("OAuth session expired or not found, please regenerate authorization link")

        proxy = proxy.strip()
        proxies = {"http": proxy, "https": proxy} if proxy else None
        session = requests.Session(impersonate="chrome124", proxies=proxies)

        token_url = f"{AUTH_BASE}/api/accounts/oauth/token"
        headers = {
            "Content-Type": "application/json",
            "Origin": PLATFORM_BASE,
            "Referer": f"{PLATFORM_BASE}/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        }
        payload = {
            "client_id": OAUTH_CLIENT_ID,
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": OAUTH_REDIRECT_URI,
        }

        try:
            resp = session.post(token_url, headers=headers, json=payload, timeout=30)
        finally:
            session.close()

        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI token exchange failed (HTTP {resp.status_code}): {resp.text[:200]}")

        return resp.json()


class OAuthRefreshError(RuntimeError):
    def __init__(self, status_code: int, error_code: str):
        self.status_code = status_code
        self.error_code = error_code if re.fullmatch(r"[a-zA-Z0-9_]{1,64}", error_code) else "oauth_error"
        super().__init__(f"OAuth refresh failed status={status_code} code={self.error_code}")


async def refresh_access_token(refresh_token: str, proxy: str = "") -> dict[str, Any]:
    """Use refresh_token to get a new access_token via the account's bound proxy."""
    proxy = proxy.strip()
    proxies = {"http": proxy, "https": proxy} if proxy else None
    session = requests.AsyncSession(impersonate="chrome124", proxies=proxies)

    token_url = f"{AUTH_BASE}/oauth/token"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }
    payload = {
        "client_id": OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token.strip(),
    }

    try:
        response = await session.post(token_url, headers=headers, json=payload, timeout=30)
    finally:
        await session.close()

    if response.status_code != 200:
        try:
            error_code = (response.json().get("error") or "oauth_error")
            if isinstance(error_code, dict):
                error_code = error_code.get("code", "oauth_error")
        except Exception:
            error_code = "oauth_error"
        raise OAuthRefreshError(response.status_code, str(error_code))

    return response.json()


oauth_manager = OAuthManager()
