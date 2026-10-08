from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from curl_cffi.requests import Cookies

from core.account_pool import BaseAccountPool, serialized_account_edit
from core.database import database
from core.blocking import run_blocking
from providers.gemini.webapi import AuthError
from utils.helper import parse_cookie_string
from utils.log import logger, stable_log_ref


_AUTH_COOKIE_NAMES = ("__Secure-1PSID", "__Secure-1PSIDTS")


def _parse_auth_cookie(cookie: str) -> tuple[str, str]:
    cookies = parse_cookie_string(cookie)
    psid = cookies.get(_AUTH_COOKIE_NAMES[0])
    if not psid:
        raise ValueError("Gemini Cookie 中缺少 __Secure-1PSID")
    psidts = cookies.get(_AUTH_COOKIE_NAMES[1])
    if not psidts:
        raise ValueError("Gemini Cookie 中缺少 __Secure-1PSIDTS，请粘贴完整 Cookie Header")
    return psid, psidts


def _auth_cookie_header(psid: str, psidts: str) -> str:
    return f"__Secure-1PSID={psid}; __Secure-1PSIDTS={psidts}"


def _refresh_interval() -> int:
    try:
        return max(60, int(os.getenv("GEMINI_REFRESH_INTERVAL", "600")))
    except ValueError:
        return 600


def _stored_cookies(cookies: Any) -> list[dict[str, Any]]:
    """Keep the authenticated client's Google cookies in account credentials."""
    jar = getattr(cookies, "jar", None)
    if jar is None:
        return []
    now = time.time()
    return [
        {"name": c.name, "value": c.value, "domain": c.domain,
         "path": c.path, "expires": c.expires}
        for c in jar
        if c.value and ((c.domain or "").lstrip(".").lower() == "google.com"
                        or (c.domain or "").lower().endswith(".google.com"))
        and (c.expires is None or c.expires > now)
    ]


def _restore_cookies(account: dict[str, Any]) -> Cookies | None:
    entries = account.get("cookie_jar") or []
    if not isinstance(entries, list):
        return None
    jar = Cookies()
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("value"):
            continue
        if entry.get("expires") and entry["expires"] <= time.time():
            continue
        domain = entry.get("domain") or ".google.com"
        if domain.lstrip(".").lower() != "google.com" and not domain.lower().endswith(".google.com"):
            continue
        jar.set(entry["name"], entry["value"], domain=domain,
                path=entry.get("path") or "/", secure=True)
    if not any(c.name == "__Secure-1PSID" and c.value == account.get("psid") for c in jar.jar):
        return None
    if not any(c.name == "__Secure-1PSIDTS" and c.value == account.get("psidts") for c in jar.jar):
        return None
    return jar


class GeminiAccountPool(BaseAccountPool):
    """Gemini 账号池：续期交给 Gemini-API，代理只验证并持久化续期结果。"""

    PROVIDER_NAME = "Gemini"

    @staticmethod
    def _client_ready(client: Any) -> bool:
        if client is None or not client._running or not client._check_account_status():
            return False
        if not hasattr(client, "refresh_task"):
            return True
        return client.refresh_task is not None and not client.refresh_task.done()

    def __init__(self) -> None:
        super().__init__(platform="gemini")
        self._clients: dict[str, Any] = {}
        self._client_locks: dict[str, asyncio.Lock] = {}
        # 业务请求独占客户端，空闲时复用；常驻 _clients 只负责账号续期与模型发现。
        self._idle_request_clients: dict[str, list[Any]] = {}
        self._request_generations: dict[str, int] = {}
        self._warmup_limit = 5
        self._pending_cookie_saves: set[str] = set()
        self._cookie_save_locks: dict[str, asyncio.Lock] = {}

    # ── Cookie-based Account Integration ──

    @serialized_account_edit
    def add_account(self, name: str, cookie: str, proxy: str = "", proxy_id: int | None = None) -> dict[str, Any]:
        psid, psidts = _parse_auth_cookie(cookie)
        account_name = name.strip() or f"gemini-{int(time.time())}"
        account = {
            "name": account_name,
            "cookie": _auth_cookie_header(psid, psidts),
            "psid": psid,
            "psidts": psidts,
            "cookie_jar": [],
            "proxy": proxy.strip(),
            "proxy_id": proxy_id,
            "proxy_status": "active" if proxy_id else None,
            "status": "active",
            "inflight": 0,
            "cooldown_until": 0,
            "failure_count": 0,
            "last_used_at": 0,
            "error_message": "",
        }
        with self._condition:
            # 同名账号重新提交 Cookie 属于人工恢复，旧缓存绝不能反向覆盖新凭据。
            duplicate = next(
                (existing_name for existing_name, existing in self._accounts.items()
                 if existing_name != account_name and existing.get("psid") == psid),
                None,
            )
            if duplicate:
                raise ValueError(f"该 __Secure-1PSID 已被账号 [{duplicate}] 使用")
        return self._register_account(account["name"], account)

    # ── Gemini-specific: cookie management ──

    @serialized_account_edit
    def update_cookie(self, name: str, cookie: str) -> bool:
        psid, psidts = _parse_auth_cookie(cookie)
        with self._condition:
            account = self._accounts.get(name)
            if account is None:
                return False
            duplicate = next(
                (existing_name for existing_name, existing in self._accounts.items()
                 if existing_name != name and existing.get("psid") == psid),
                None,
            )
            if duplicate:
                raise ValueError(f"该 __Secure-1PSID 已被账号 [{duplicate}] 使用")
        return self._replace_cookie_fields(name, {
            "cookie": _auth_cookie_header(psid, psidts), "psid": psid, "psidts": psidts, "cookie_jar": []
        })

    async def merge_cookie(self, name: str, updates: dict[str, str], jar: Any = None,
                           expected_generation: int | None = None) -> None:
        """续期写库在池锁外执行；版本校验阻止旧客户端覆盖人工更新的凭证。"""
        async with self._cookie_save_locks.setdefault(name, asyncio.Lock()):
            with self._condition:
                account = self._accounts.get(name)
                if account is None:
                    self._pending_cookie_saves.discard(name)
                    return
                if (expected_generation is not None
                        and self._request_generations.get(name, 0) != expected_generation):
                    return
                psid = updates.get("__Secure-1PSID") or account["psid"]
                psidts = updates.get("__Secure-1PSIDTS") or account["psidts"]
                account.update(cookie=_auth_cookie_header(psid, psidts), psid=psid, psidts=psidts)
                if jar is not None:
                    account["cookie_jar"] = _stored_cookies(jar)
                snapshot = dict(account)
                self._pending_cookie_saves.add(name)
            if self._platform:
                patch = {field: snapshot[field] for field in ("cookie", "psid", "psidts", "cookie_jar")}
                saved = await run_blocking(database.update_credentials, "gemini", name, patch,
                                           snapshot["credential_version"], snapshot["account_id"])
                replacement = None if saved else await run_blocking(database.list_accounts, "gemini", name)
                with self._condition:
                    current = self._accounts.get(name)
                    if (saved and current is not None and current["account_id"] == snapshot["account_id"]
                            and current["credential_version"] == snapshot["credential_version"]):
                        current["credential_version"] += 1
                        self._pending_cookie_saves.discard(name)
                        self._save_health(name)
                        self._condition.notify_all()
                    elif not saved:
                        # 数据库中的人工登录/代理更新优先，旧客户端的续期结果不能回写。
                        if (current is not None and current["account_id"] == snapshot["account_id"]
                                and current["credential_version"] == snapshot["credential_version"] and replacement):
                            current.update(replacement[0])
                        self._pending_cookie_saves.discard(name)
                        self._condition.notify_all()
                        logger.warning("Gemini Cookie persistence version changed; current credentials retained")
            else:
                await run_blocking(self._save, name)
                self._pending_cookie_saves.discard(name)

    async def _retry_pending_cookie_saves(self) -> None:
        with self._condition:
            pending = list(self._pending_cookie_saves)
        for name in pending:
            try:
                await self.merge_cookie(name, {})
            except Exception as exc:
                logger.warning("Gemini Cookie persistence retry failed reason=%s", type(exc).__name__)

    @staticmethod
    def is_auth_error(error: Exception | str) -> bool:
        """仅认证失效才停止账号，临时上游错误继续保留在池中。"""
        text = str(error).lower()
        return type(error).__name__ == "AuthError" or any(
            marker in text for marker in ("unauthenticated", "unauthorized", "cookie", "expired", "401", "未认证")
        )

    def mark_auth_failed(self, name: str, error: Exception | str, expected_account: dict | None = None) -> None:
        """认证失效时停用账号，但保留最近一次成功轮换的本地缓存。"""
        with self._condition:
            account = self._accounts.get(name)
            if account is None or (expected_account is not None and any(
                account.get(field) != expected_account.get(field) for field in ("account_id", "credential_version")
            )):
                return
            old_status = account.get("status")
            account["status"] = "error"
            account["failure_count"] = account.get("failure_count", 0) + 1
            account["error_message"] = str(error)[:500] or "Gemini Cookie 未认证，请更新 Cookie"
            self._save_health(name)
            logger.warning("event=account_marked_error platform=gemini account_ref=%s old_status=%s new_status=error reason_code=AUTH_FAILED",
                           stable_log_ref("gemini-account", name), old_status)
            self._condition.notify_all()

    def mark_refresh_verification_failed(self, name: str, error: Exception | str, expected_account: dict | None = None) -> None:
        """续期后验证失败时暂停账号；下次 Gemini-API 续期验证成功后自动恢复。"""
        if self.is_auth_error(error):
            self.mark_auth_failed(name, error, expected_account)
            return
        with self._condition:
            account = self._accounts.get(name)
            if account is None or (expected_account is not None and any(
                account.get(field) != expected_account.get(field) for field in ("account_id", "credential_version")
            )):
                return
            old_status = account.get("status")
            account["status"] = "cooldown"
            account["cooldown_until"] = int(time.time() + 300)
            account["failure_count"] = account.get("failure_count", 0) + 1
            account["error_message"] = str(error)[:500]
            self._save_health(name)
            logger.warning("event=account_cooldown_started platform=gemini account_ref=%s old_status=%s new_status=cooldown reason_code=REFRESH_VERIFICATION_FAILED cooldown_seconds=300",
                           stable_log_ref("gemini-account", name), old_status)
            self._condition.notify_all()

    async def verify_refreshed_client(self, name: str, client: Any) -> None:
        """续期后只校验账号认证状态，避免周期性生成请求误判账号并消耗额度。"""
        with self._condition:
            if self._clients.get(name) is not client:
                return
            expected_account = dict(self._accounts[name])
        try:
            await client._fetch_user_status()
            if not client._check_account_status():
                raise RuntimeError("Gemini Cookie 未认证，请更新完整 Cookie Header")
        except Exception as exc:
            self.mark_refresh_verification_failed(name, exc, expected_account)
            logger.warning("event=account_refresh_verification_failed platform=gemini account_ref=%s reason_code=%s",
                           stable_log_ref("gemini-account", name), type(exc).__name__)
            return

        with self._condition:
            if (self._clients.get(name) is not client
                    or self._accounts.get(name, {}).get("status") not in {"active", "cooldown"}
                    or any(self._accounts[name].get(field) != expected_account.get(field)
                           for field in ("account_id", "credential_version"))):
                return
            account = self._accounts[name]
            old_status = account.get("status")
            account["status"] = "active"
            account["cooldown_until"] = 0
            account["failure_count"] = 0
            account["error_message"] = ""
            generation = self._request_generations.get(name, 0)
        try:
            await self.merge_cookie(name, dict(client.cookies), client.cookies, generation)
        except Exception as exc:
            logger.error("event=account_refresh_persist_failed platform=gemini account_ref=%s reason_code=%s",
                         stable_log_ref("gemini-account", name), type(exc).__name__)
            return
        with self._condition:
            self._condition.notify_all()
        logger.info("event=account_refresh_verified platform=gemini account_ref=%s old_status=%s new_status=active reason_code=refresh_verified",
                    stable_log_ref("gemini-account", name), old_status)

    async def get_client(self, account: dict[str, Any]) -> Any:
        """Return one authenticated refresh client per account."""
        name = account["name"]
        with self._condition:
            lock = self._client_locks.setdefault(name, asyncio.Lock())
        async with lock:
            with self._condition:
                account = dict(self._accounts[name])
                generation = self._request_generations.get(name, 0)
            client = self._clients.get(name)
            if account.get("status") == "active" and self._client_ready(client):
                return client
            if client is not None:
                await self.discard_client(name)
                with self._condition:
                    account = dict(self._accounts[name])
                    generation = self._request_generations.get(name, 0)

            from providers.gemini.webapi import GeminiClient

            psid = account["psid"]
            psidts = account.get("psidts") or None

            async def on_cookie_refreshed(refreshed_client: Any) -> None:
                await self.verify_refreshed_client(name, refreshed_client)

            client = GeminiClient(
                psid,
                psidts,
                proxy=account.get("proxy") or None,
                on_cookie_refreshed=on_cookie_refreshed,
            )
            stored = _restore_cookies(account)
            if stored is not None:
                client.cookies = stored
            try:
                await client.init(
                    timeout=180,
                    auto_refresh=True,
                    refresh_interval=_refresh_interval(),
                    impersonate="chrome145",
                )
                if not client._check_account_status():
                    raise RuntimeError("Gemini Cookie 未认证，请更新完整 Cookie Header")
                with self._condition:
                    current = self._accounts.get(name)
                    if (current is None or current.get("status") != "active"
                            or self._request_generations.get(name, 0) != generation):
                        raise RuntimeError("Gemini 账号在初始化期间已变更")
                    current["status"] = "active"
                    current["cooldown_until"] = 0
                    current["failure_count"] = 0
                    current["error_message"] = ""
                await self.merge_cookie(name, dict(client.cookies), client.cookies, generation)
                with self._condition:
                    current = self._accounts[name]
                    # 锁外保存期间可能人工替换 Cookie/代理，旧客户端不能重新进入缓存。
                    if (self._request_generations.get(name, 0) != generation
                            or current.get("account_id") != account.get("account_id")
                            or current["psid"] != account["psid"]
                            or current.get("proxy", "") != account.get("proxy", "")):
                        raise RuntimeError("Gemini 账号在凭证保存期间已变更")
                    self._clients[name] = client
                    self._condition.notify_all()
                logger.info("event=account_client_ready platform=gemini account_ref=%s",
                            stable_log_ref("gemini-account", name))
                return client
            except AuthError as exc:
                await client.close()
                raise RuntimeError("Gemini Cookie 未认证，请更新 Cookie") from exc
            except BaseException:
                await client.close()
                raise

    async def warmup_clients(self) -> None:
        """Eagerly warm up active Gemini accounts on startup to keep RotateCookies running 24/7."""
        with self._condition:
            # 服务重启后先恢复已到期冷却，再预热客户端。
            self._restore_expired_cooldowns(time.time())
            active_accounts = [
                dict(acc) for acc in self._accounts.values()
                if acc.get("status") == "active" and acc.get("cookie")
                and not self._client_ready(self._clients.get(acc["name"]))
            ]

        if not active_accounts:
            return

        logger.info(f"[Gemini Warmup] 以最多 {self._warmup_limit} 个并发初始化 {len(active_accounts)} 个活跃账号")
        semaphore = asyncio.Semaphore(self._warmup_limit)

        async def warm_one(acc: dict[str, Any]) -> None:
            async with semaphore:
                with self._condition:
                    current = self._accounts.get(acc["name"])
                    if current is None or current.get("status") != "active":
                        return
                try:
                    await self.get_client(acc)
                except Exception as exc:
                    if self.is_auth_error(exc):
                        self.mark_auth_failed(acc["name"], exc, acc)
                    logger.warning("event=account_warmup_failed platform=gemini account_ref=%s reason_code=%s",
                                   stable_log_ref("gemini-account", acc["name"]), type(exc).__name__)

        await asyncio.gather(*(warm_one(acc) for acc in active_accounts))

    async def maintain_clients(self) -> None:
        """Retry failed initializations and recreate stopped refresh tasks."""
        while True:
            try:
                await self._retry_pending_cookie_saves()
                await self.warmup_clients()
            except Exception as exc:
                logger.error("event=account_maintenance_failed platform=gemini reason_code=%s", type(exc).__name__)
            await asyncio.sleep(60)

    async def get_request_client(self, account: dict[str, Any]) -> tuple[Any, int]:
        """为一次业务请求独占一条连接，同账号其他请求不会因它关闭而中断。"""
        name = account["name"]
        master = await self.get_client(account)
        generation = self._request_generations.get(name, 0)
        idle = self._idle_request_clients.setdefault(name, [])
        while idle:
            client = idle.pop()
            if client._running and client._check_account_status():
                client.cookies = dict(master.cookies)
                return client, generation
            await client.close()

        from providers.gemini.webapi import GeminiClient

        cookies = dict(master.cookies)
        with self._condition:
            current = dict(self._accounts[name])
        client = GeminiClient(
            current["psid"], current["psidts"],
            proxy=current.get("proxy") or None,
        )
        if cookies:
            client.cookies = cookies
        try:
            await client.init(timeout=180, auto_refresh=False, impersonate="chrome145")
            if not client._check_account_status():
                raise RuntimeError("Gemini Cookie 未认证，请更新完整 Cookie Header")
            return client, generation
        except BaseException:
            await client.close()
            raise

    async def release_request_client(self, name: str, client: Any, generation: int, success: bool) -> None:
        """只回收本次请求的连接；人工更新账号后旧连接不会重新入池。"""
        if success and generation == self._request_generations.get(name, 0) and client._running and client._check_account_status():
            self._idle_request_clients.setdefault(name, []).append(client)
        else:
            await client.close()

    async def close_clients(self) -> None:
        for name in set(self._clients) | set(self._idle_request_clients):
            await self.discard_client(name)

    def _account_ready(self, account: dict[str, Any]) -> bool:
        return self._client_ready(self._clients.get(account["name"]))

    async def discard_client(self, name: str, expected_account: dict | None = None) -> None:
        # 只关闭空闲连接；正在使用的请求在退出时按代次自行关闭。
        with self._condition:
            current = self._accounts.get(name)
            if expected_account is not None and (current is None or any(
                current.get(field) != expected_account.get(field) for field in ("account_id", "credential_version")
            )):
                return
            self._request_generations[name] = self._request_generations.get(name, 0) + 1
            idle = self._idle_request_clients.pop(name, [])
            client = self._clients.pop(name, None)
            self._condition.notify_all()
        for request_client in idle:
            try:
                await request_client.close()
            except Exception:
                pass
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass

    # ── Hooks & Self-Healing ──

    def _mask_sensitive(self, account: dict[str, Any]) -> dict[str, Any]:
        safe = {
            k: v for k, v in account.items()
            if k not in {"cookie", "psid", "psidts", "cookie_jar"}
        }
        if account.get("cookie"):
            safe["cookie"] = account.get("cookie", "")[:18] + "..."
        return safe

    def get_available_models(self) -> list[dict[str, str]]:
        """Return discovered available models from active Gemini client registries."""
        models: list[dict[str, str]] = []
        seen: set[str] = set()
        for client in list(self._clients.values()):
            registry = getattr(client, "_model_registry", {})
            if isinstance(registry, dict):
                for m in registry.values():
                    name = getattr(m, "model_name", "")
                    if name and name not in seen:
                        seen.add(name)
                        display = getattr(m, "display_name", "") or name
                        desc = getattr(m, "description", "") or ""
                        models.append({
                            "value": name,
                            "label": f"{display} ({name})",
                            "description": desc,
                        })
        return models

    async def refresh_supported_models(self, name: str) -> list[dict[str, str]]:
        """Discover and persist the exact Gemini web models for one account."""
        with self._lock:
            account = self._accounts.get(name)
            if not account:
                raise ValueError(f"Account {name} not found")
            account_copy = dict(account)

        try:
            client = await self.get_client(account_copy)
        except Exception as exc:
            if self.is_auth_error(exc):
                self.mark_auth_failed(name, exc, account_copy)
            raise

        registry = getattr(client, "_model_registry", {})
        models = [
            {
                "value": model.model_name,
                "label": f"{model.display_name or model.model_name} ({model.model_name})",
                "description": model.description or "",
            }
            for model in registry.values()
            if getattr(model, "model_name", "")
        ]
        if not models:
            raise RuntimeError("Gemini 网页端未返回可用模型")

        with self._condition:
            current = self._accounts.get(name)
            if current is None or self._clients.get(name) is not client:
                raise ValueError("账号已更新，请重新获取模型列表")
            snapshot = dict(current)
        updated_at = int(time.time())
        saved = not self._platform or await run_blocking(
            database.update_supported_models, "gemini", name, models, updated_at,
            snapshot["credential_version"], snapshot["account_id"]
        )
        with self._condition:
            current = self._accounts.get(name)
            if saved and current is not None and all(current.get(field) == snapshot.get(field)
                                                    for field in ("account_id", "credential_version")):
                current.update(supported_models=models, models_updated_at=updated_at)
                self._condition.notify_all()
        if not self._platform:
            await run_blocking(self._save, name)
        return models


gemini_account_service = GeminiAccountPool()
