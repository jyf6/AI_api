from __future__ import annotations

import base64
import asyncio
import json
import secrets
from datetime import datetime, timezone
from typing import Any

from core.account_pool import BaseAccountPool
from providers.openai.oauth import oauth_manager
from providers.openai.credentials import GPTCredentials
from core.database import database
from core.blocking import run_blocking
from utils.log import logger


class OpenAIAccountPool(BaseAccountPool):
    """OpenAI account pool with OAuth token refresh and JWT-based metadata."""

    PROVIDER_NAME = "ChatGPT"

    def __init__(self) -> None:
        super().__init__(platform="gpt")
        self.credentials = GPTCredentials(self)
        # 管理端探测使用独立小预算，不挤占业务准入，也不允许无限并发。
        self.management_slots = asyncio.Semaphore(5)
        # 旧账号仅在首次升级时补发设备 ID，后续客户端始终复用该账号身份。
        for email, account in self._accounts.items():
            if not account.get("device_id"):
                account["device_id"] = secrets.token_hex(16)
                version = account.get("credential_version", 0)
                if database.update_credentials("gpt", email, {"device_id": account["device_id"]}, version, account["account_id"]):
                    account["credential_version"] = version + 1
                else:
                    rows = database.list_accounts("gpt", email)
                    if rows:
                        account.update(rows[0])

    async def _prepare_account_async(self, account: dict[str, Any]) -> dict[str, Any]:
        return await self.credentials.prepare(account)

    def _account_ready(self, account: dict[str, Any]) -> bool:
        return self.credentials.ready(account)

    def _account_wait_failure(self) -> Exception:
        from providers.openai.credentials import CredentialUnavailable
        candidates = [account for account in self._accounts.values()
                      if account.get("status") == "active" and account.get("proxy_status") != "disabled"]
        if candidates and not any(self.credentials.ready(account) for account in candidates):
            return CredentialUnavailable("Account credentials are not ready for model invocation")
        return super()._account_wait_failure()

    def _save(self, account_key: str | None = None) -> None:
        # Only enqueue here: callers may hold the account pool lock.
        accounts = self._accounts.items() if account_key is None else (
            ((account_key, self._accounts[account_key]),) if account_key in self._accounts else ()
        )
        for key, account in accounts:
            future = self._health_executor.submit(database.save_account_health, "gpt", key, dict(account))
            def observe(done):
                try:
                    done.result()
                except Exception as exc:
                    logger.error("GPT health persistence failed reason=%s", type(exc).__name__)
            future.add_done_callback(observe)

    # ── OAuth integration ──

    def start_oauth_session(self) -> dict[str, str]:
        return oauth_manager.start_session()

    def finish_oauth_session(self, callback_url: str, proxy: str = "", session_id: str = "", proxy_id: int | None = None) -> dict[str, Any]:
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
            "proxy_id": proxy_id,
            "proxy_status": "active" if proxy_id else None,
            "device_id": secrets.token_hex(16),
            "plan_type": plan_type,
            "status": "active",
            "inflight": 0,
            "cooldown_until": 0,
            "failure_count": 0,
            "last_used_at": 0,
            "error_message": "",
        }

        # 登录发布与人工删除串行；OAuth 网络交换不持有管理锁或业务池锁。
        with self._management_lock:
            with self._lock:
                previous = self._accounts.get(email, {})
                previous_id = previous.get("account_id")
                if not proxy and not proxy_id:
                    account["proxy"] = previous.get("proxy", "")
                    account["proxy_id"] = previous.get("proxy_id")
                    account["proxy_status"] = previous.get("proxy_status")
                account["device_id"] = previous.get("device_id") or account["device_id"]
            fields = {key: value for key, value in account.items() if key in {
                "access_token", "refresh_token", "access_token_expires_at", "device_id", "plan_type"
            }}
            stored_identity = database.import_account("gpt", email, fields, account["proxy"], proxy_id=account["proxy_id"])
            rows = database.list_accounts("gpt", email)
            if not rows:
                raise RuntimeError("Saved account could not be reloaded")
            replacement = rows[0]
            if replacement["account_id"] != stored_identity["account_id"]:
                raise RuntimeError("Account identity changed during login persistence")
            with self._condition:
                current = self._accounts.get(email)
                # 查询之后可能已有较新刷新/重建结果发布，登录的迟到快照不能回退它。
                if current is not None and (
                    (current["account_id"] == replacement["account_id"]
                     and current["credential_version"] > replacement["credential_version"])
                    or current["account_id"] not in {previous_id, replacement["account_id"]}
                ):
                    return dict(current)
                account = {**(current if current is not None and current["account_id"] == replacement["account_id"] else account),
                           **replacement}
                self._accounts[email] = account
                self._condition.notify_all()
            return dict(account)

    async def refresh_account_async(self, email: str, force: bool = False) -> dict[str, Any]:
        return await self.credentials.refresh(email, force=force)

    async def prepare_request_account(self, account: dict[str, Any], *, allow_inactive: bool = False) -> dict[str, Any]:
        """长调用的后续认证请求重新检查最新凭证，保持设备和出口身份一致。"""
        with self._lock:
            current = self._accounts.get(account["email"])
            if (current is None or (not allow_inactive and current.get("status") != "active")
                    or current.get("proxy_status") == "disabled"):
                from providers.openai.credentials import CredentialUnavailable
                raise CredentialUnavailable("Account is no longer active")
            current = dict(current)
        if (current["account_id"] != account["account_id"]
                or current.get("proxy", "") != account.get("proxy", "")
                or current.get("device_id", "") != account.get("device_id", "")):
            from providers.openai.credentials import CredentialUnavailable
            raise CredentialUnavailable("Account identity changed during model execution")
        prepared = await self.credentials.prepare(current)
        # 刷新等待期间也可能发生删除重建或代理重绑，返回前再次校验身份。
        if (prepared["account_id"] != account["account_id"]
                or prepared.get("proxy", "") != account.get("proxy", "")
                or prepared.get("device_id", "") != account.get("device_id", "")):
            from providers.openai.credentials import CredentialUnavailable
            raise CredentialUnavailable("Account identity changed during credential preparation")
        # 回调按最后一次实际使用的凭证版本判断健康状态，不沿用领取时的旧令牌版本。
        account.update(prepared)
        return prepared

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

    async def get_available_models(self) -> list[dict[str, str]]:
        """Probe available ChatGPT models using an active account."""
        with self._lock:
            active = [
                dict(a) for a in self._accounts.values()
                if a.get("status") == "active" and a.get("access_token")
            ]
        if not active:
            return []
        return await self._discover_models(active[0])

    async def _discover_models(self, account: dict[str, Any]) -> list[dict[str, str]]:
        try:
            from providers.openai.backend import OpenAIBackendAPI
            async with self.management_slots, OpenAIBackendAPI(
                account.get("access_token", ""), account.get("proxy", ""), account.get("device_id", ""),
                credential_provider=lambda: self.prepare_request_account(account),
            ) as backend:
                # 模型列表读取同样不能使用过期凭证或身份已改变的账号快照。
                path = "/backend-api/models"
                res = await backend._request("GET", backend.base_url + path, headers=backend._headers(path), timeout=4)
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

    async def refresh_supported_models(self, email: str) -> list[dict[str, str]]:
        """Discover and persist the exact ChatGPT web model slugs for one account."""
        with self._lock:
            account = self._accounts.get(email)
            if not account:
                raise ValueError(f"Account {email} not found")
            account_copy = dict(account)

        models = await self._discover_models(account_copy)
        if not models:
            raise RuntimeError("ChatGPT 网页端未返回可用模型")

        patch = {"supported_models": models, "models_updated_at": int(datetime.now(timezone.utc).timestamp())}
        version = account_copy.get("credential_version", 0)
        saved = await run_blocking(database.update_supported_models, "gpt", email, models,
                                   patch["models_updated_at"], version, account_copy["account_id"])
        if saved:
            with self._condition:
                current = self._accounts.get(email)
                if current is not None and current.get("credential_version", 0) == version and current["account_id"] == account_copy["account_id"]:
                    current.update(patch)
                    self._condition.notify_all()
        return models


account_service = OpenAIAccountPool()
