from __future__ import annotations

import base64
import json
import secrets
from datetime import datetime, timezone
from typing import Any

from core.account_pool import BaseAccountPool
from providers.openai.oauth import oauth_manager, refresh_access_token
from utils.log import logger


class OpenAIAccountPool(BaseAccountPool):
    """OpenAI account pool with OAuth token refresh and JWT-based metadata."""

    PROVIDER_NAME = "ChatGPT"

    def __init__(self) -> None:
        super().__init__(platform="gpt")

    # ── OAuth integration ──

    def start_oauth_session(self) -> dict[str, str]:
        return oauth_manager.start_session()

    def finish_oauth_session(self, callback_url: str, proxy: str = "", session_id: str = "") -> dict[str, Any]:
        """Exchange callback URL for tokens and store account in pool."""
        data = oauth_manager.finish_session(callback_url, proxy, session_id)
        access_token = str(data.get("access_token") or "").strip()
        refresh_token = str(data.get("refresh_token") or "").strip()
        id_token = str(data.get("id_token") or "").strip()

        if not access_token or not refresh_token:
            raise RuntimeError("OpenAI response missing access_token or refresh_token")

        id_claims = self._decode_jwt(id_token)
        access_claims = self._decode_jwt(access_token)
        email = str(id_claims.get("email") or access_claims.get("email") or "").strip()
        if not email:
            email = f"user_{secrets.token_hex(4)}@openai.com"

        expires_in = int(data.get("expires_in") or 864000)
        expires_at = int(datetime.now(timezone.utc).timestamp()) + expires_in
        plan_type = (access_claims.get("https://api.openai.com/auth") or {}).get("chatgpt_plan_type", "plus")

        account = {
            "email": email,
            "refresh_token": refresh_token,
            "access_token": access_token,
            "access_token_expires_at": expires_at,
            "proxy": proxy.strip(),
            "plan_type": plan_type,
            "status": "active",
            "inflight": 0,
            "cooldown_until": 0,
            "failure_count": 0,
            "last_used_at": 0,
            "error_message": "",
        }

        with self._condition:
            self._accounts[email] = account
            self._save()
            self._condition.notify_all()

        return account

    # ── Token refresh ──

    def refresh_account(self, email: str) -> dict[str, Any]:
        with self._lock:
            account = self._accounts.get(email)
            if not account:
                raise ValueError(f"Account {email} not found")

        try:
            data = refresh_access_token(account["refresh_token"], account.get("proxy", ""))
            access_token = data.get("access_token")
            if not access_token:
                raise RuntimeError("OAuth response did not contain access_token")
            expires_in = int(data.get("expires_in") or 864000)
            expires_at = int(datetime.now(timezone.utc).timestamp()) + expires_in
            claims = self._decode_jwt(access_token)
            plan_type = (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_plan_type", "plus")

            with self._condition:
                account["access_token"] = access_token
                account["plan_type"] = plan_type
                account["access_token_expires_at"] = expires_at
                account["status"] = "active"
                account["cooldown_until"] = 0
                account["failure_count"] = 0
                account["error_message"] = ""
                self._save()
                self._condition.notify_all()
            return account
        except Exception as exc:
            category = self._classify_error(str(exc), getattr(exc, "status_code", None))
            with self._condition:
                if category == "fatal":
                    account["status"] = "error"
                    account["cooldown_until"] = 0
                account["error_message"] = str(exc)
                self._save()
                self._condition.notify_all()
            raise

    # ── Override get_available_account for auto-refresh ──

    def _prepare_account(self, account_copy: dict[str, Any]) -> dict[str, Any]:
        now = int(datetime.now(timezone.utc).timestamp())

        # Auto-refresh if token expires within 1 hour
        if account_copy.get("access_token_expires_at", 0) - now < 3600:
            try:
                data = refresh_access_token(account_copy["refresh_token"], account_copy.get("proxy", ""))
                new_token = data.get("access_token", "")
                expires_in = int(data.get("expires_in") or 864000)
                expires_at = int(datetime.now(timezone.utc).timestamp()) + expires_in
                if new_token:
                    with self._lock:
                        if account_copy["email"] in self._accounts:
                            self._accounts[account_copy["email"]]["access_token"] = new_token
                            self._accounts[account_copy["email"]]["access_token_expires_at"] = expires_at
                            self._save()
                    account_copy["access_token"] = new_token
            except Exception as exc:
                logger.warning(f"Failed pre-refreshing token for {account_copy['email']}: {exc}")

        return account_copy

    def get_available_account(self, task_type: str = "chat") -> dict[str, Any]:
        return self._prepare_account(super().get_available_account(task_type=task_type))

    def wait_for_available_account(self, task_type: str = "chat") -> dict[str, Any]:
        return self._prepare_account(super().wait_for_available_account(task_type=task_type))

    # ── Hooks ──

    def _mask_sensitive(self, account: dict[str, Any]) -> dict[str, Any]:
        if account.get("refresh_token"):
            account["refresh_token"] = account["refresh_token"][:6] + "..."
        if account.get("access_token"):
            account["access_token"] = account["access_token"][:10] + "..."
        # Compat: expose image_inflight for frontend
        account["image_inflight"] = account.get("inflight", 0)
        return account

    def _classify_error(self, error: str, status_code: int | None = None) -> str:
        lower = error.lower()
        if status_code == 401 or any(kw in lower for kw in (
            "invalid_grant", "invalid refresh token", "invalid token", "token expired",
            "unauthorized", "deactivated", "account disabled",
        )):
            return "fatal"
        if status_code == 429 or any(kw in lower for kw in (
            "quota", "rate limit", "too many", "429", "image_quota_exhausted", "图像生成请求上限", "图片生成请求上限",
        )):
            return "rate_limit"
        return "transient"

    @staticmethod
    def _decode_jwt(token: str) -> dict:
        try:
            payload = str(token or "").split(".")[1]
            payload += "=" * ((4 - len(payload) % 4) % 4)
            return json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        except Exception:
            return {}

    def get_available_models(self) -> list[dict[str, str]]:
        """Probe available ChatGPT models using an active account."""
        with self._lock:
            active = [
                a for a in self._accounts.values()
                if a.get("status") == "active" and a.get("access_token")
            ]
        if not active:
            return []
        return self._discover_models(active[0])

    def _discover_models(self, account: dict[str, Any]) -> list[dict[str, str]]:
        try:
            from providers.openai.backend import OpenAIBackendAPI
            with OpenAIBackendAPI(account.get("access_token", ""), account.get("proxy", "")) as backend:
                path = "/backend-api/models"
                res = backend.session.get(backend.base_url + path, headers=backend._headers(path), timeout=4)
                if res.status_code == 200:
                    data = res.json()
                    models = data.get("models") or []
                    return [
                        {
                            "value": m.get("slug", ""),
                            "label": f"{m.get('title') or m.get('slug')} ({m.get('slug')})",
                            "description": m.get("description") or "",
                        }
                        for m in models
                        if m.get("slug")
                    ]
        except Exception as exc:
            logger.debug(f"Probing ChatGPT models failed: {exc}")
        return []

    def refresh_supported_models(self, email: str) -> list[dict[str, str]]:
        """Discover and persist the exact ChatGPT web model slugs for one account."""
        with self._lock:
            account = self._accounts.get(email)
            if not account:
                raise ValueError(f"Account {email} not found")
            account_copy = dict(account)

        models = self._discover_models(self._prepare_account(account_copy))
        if not models:
            raise RuntimeError("ChatGPT 网页端未返回可用模型")

        with self._condition:
            current = self._accounts.get(email)
            if not current:
                raise ValueError(f"Account {email} not found")
            current["supported_models"] = models
            current["models_updated_at"] = int(datetime.now(timezone.utc).timestamp())
            self._save()
            self._condition.notify_all()
        return models


account_service = OpenAIAccountPool()
