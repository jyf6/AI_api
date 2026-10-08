"""GPT 凭证刷新：同账号合并，先保存，再发布可调度凭证�?""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from core.blocking import run_blocking
from core.database import database
from providers.openai.oauth import refresh_access_token
from utils.log import logger, stable_log_ref


class CredentialUnavailable(RuntimeError):
    """凭证准备失败；不能作为模型失败重试或容量拒绝�?""


class GPTCredentials:
    def __init__(self, pool, concurrency: int | None = None):
        self.pool = pool
        limit = concurrency if concurrency is not None else int(os.getenv("GPT_REFRESH_CONCURRENCY") or 5)
        if limit <= 0:
            raise ValueError("GPT_REFRESH_CONCURRENCY must be positive")
        self._slots = asyncio.Semaphore(limit)
        self._tasks: dict[str, asyncio.Task] = {}
        self._pending: dict[str, tuple[int, dict[str, Any], int]] = {}
        self._backoff: dict[str, tuple[int, float]] = {}

    def ready(self, account: dict) -> bool:
        return (account.get("error_message") != "credential_auth_invalid"
                and not account.get("credential_pending") and bool(account.get("access_token"))
                and account.get("access_token_expires_at", 0) - time.time() >= 360)

    async def refresh(self, email: str, force: bool = False) -> dict:
        # �?await 的查�?创建在单事件循环内合并成同一个任务�?
        task = self._tasks.get(email)
        if task is None or task.done():
            task = asyncio.create_task(self._refresh(email, force))
            self._tasks[email] = task
            def observe(done):
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(observe)
        return await asyncio.shield(task)

    async def _refresh(self, email: str, force: bool) -> dict:
        # 待写库凭证必须优先保存；即使手动 force，也不再�?OAuth�?
        if email in self._pending:
            return await self._persist(email)
        failures, retry_at = self._backoff.get(email, (0, 0))
        if not force and time.monotonic() < retry_at:
            raise CredentialUnavailable("Credential refresh is in backoff")
        async with self._slots:
            with self.pool._lock:
                current = self.pool._accounts.get(email)
                if current is None:
                    raise ValueError("Account not found")
                account = dict(current)
            if account.get("proxy_status") == "disabled":
                raise CredentialUnavailable("Account proxy is disabled")
            version = account.get("credential_version", 0)
            try:
                data = await refresh_access_token(account["refresh_token"], account.get("proxy", ""))
                if not data.get("access_token"):
                    raise CredentialUnavailable("OAuth response missing access_token")
            except Exception as exc:
                delay = min(300, 30 * 2 ** min(failures, 4))
                self._backoff[email] = (failures + 1, time.monotonic() + delay)
                fatal = self.pool._classify_error(str(exc), getattr(exc, "status_code", None)) == "fatal"
                # 不记�?OAuth 原始响应或异常正文，后台刷新失败仍留下可追踪事件�?
                logger.warning("GPT credential refresh failed account_ref=%s version=%s fatal=%s reason=%s",
                               stable_log_ref("gpt-account", email), version, fatal, type(exc).__name__)
                with self.pool._condition:
                    current = self.pool._accounts.get(email)
                    if current is not None and current.get("credential_version", 0) == version and current["account_id"] == account["account_id"]:
                        if fatal:
                            current["status"] = "error"
                        current["error_message"] = "credential_auth_invalid" if fatal else "credential_refresh_transient"
                        state = dict(current)
                        self.pool._condition.notify_all()
                    else:
                        state = None
                if state is not None:
                    try:
                        await run_blocking(database.save_account_health, "gpt", email, state)
                    except Exception:
                        logger.error("GPT credential failure status persistence failed account_ref=%s",
                                     stable_log_ref("gpt-account", email))
                raise CredentialUnavailable("Account requires login" if fatal else "Credential refresh temporarily failed") from exc
            claims = self.pool._decode_jwt(data["access_token"])
            patch = {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token") or account["refresh_token"],
                "access_token_expires_at": int(time.time()) + int(data.get("expires_in") or 864000),
                "plan_type": (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_plan_type", account.get("plan_type", "plus")),
            }
            # �?OAuth 返回开始保留整套新令牌。网络刷新与写库都不持账号池锁�?
            self._pending[email] = (version, patch, account["account_id"])
            with self.pool._condition:
                current = self.pool._accounts.get(email)
                if current is not None and current.get("credential_version", 0) == version and current["account_id"] == account["account_id"]:
                    current["credential_pending"] = True
                    self.pool._condition.notify_all()
            return await self._persist(email)

    async def _persist(self, email: str) -> dict:
        version, patch, account_id = self._pending[email]
        try:
            saved = await run_blocking(database.update_credentials, "gpt", email, patch, version, account_id)
            if not saved:
                rows = await run_blocking(database.list_accounts, "gpt", email)
                replacement = rows[0] if rows else None
            else:
                replacement = {**patch, "account_id": account_id, "credential_version": version + 1}
        except Exception as exc:
            self._backoff[email] = (0, time.monotonic() + 30)
            logger.error("GPT rotated credentials pending persistence account_ref=%s version=%s reason=%s",
                         stable_log_ref("gpt-account", email), version, type(exc).__name__)
            raise CredentialUnavailable("New credentials pending database persistence") from exc
        with self.pool._condition:
            current = self.pool._accounts.get(email)
            if current is not None:
                # 同名删除重建属于新身份，即使版本相同也不能发布旧凭证�?
                if replacement is not None and (
                    (current["account_id"] == account_id and current.get("credential_version", 0) == version)
                    or (current["account_id"] == replacement["account_id"] and current.get("credential_version", 0) <= replacement.get("credential_version", 0))
                ):
                    current.update(replacement)
                    if saved and current.get("error_message") == "credential_auth_invalid":
                        # 成功保存了新凭证后允许再次验证；仍由健康探测决定恢复派单�?                        current["error_message"] = ""
                elif replacement is None and current["account_id"] == account_id:
                    current["status"] = "error"
                if current["account_id"] == account_id:
                    current.pop("credential_pending", None)
                result = dict(current)
                self.pool._condition.notify_all()
            else:
                result = {}
        self._pending.pop(email, None)
        self._backoff.pop(email, None)
        return result

    async def prepare(self, account: dict) -> dict:
        if account.get("error_message") == "credential_auth_invalid":
            raise CredentialUnavailable("Account requires login")
        email = account["email"]
        remaining = account.get("access_token_expires_at", 0) - time.time()
        if email in self._pending or remaining < 360:
            account = await self.refresh(email)
        elif remaining < 3600:
            # 旧访问令牌仍安全可用，后台预刷新不拖住本次调用�?
            task = asyncio.create_task(self.refresh(email))
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        if not self.ready(account):
            raise CredentialUnavailable("Access token is not safe for model invocation")
        return account

    async def maintain(self):
        while True:
            with self.pool._lock:
                accounts = [dict(account) for account in self.pool._accounts.values()]
            now = time.monotonic()
            for account in accounts:
                email = account["email"]
                if (email in self._pending or (account.get("status") == "active" and account.get("proxy_status") != "disabled")) and (
                    email in self._pending or account.get("access_token_expires_at", 0) - time.time() < 3600
                ) and now >= self._backoff.get(email, (0, 0))[1]:
                    task = asyncio.create_task(self.refresh(email))
                    task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            await asyncio.sleep(1)

    async def drain(self):
        # 关闭期间允许已轮换凭证完成保存，不能取消 OAuth 后丢弃返回值�?
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
