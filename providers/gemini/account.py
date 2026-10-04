from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from curl_cffi.requests import Cookies

from core.account_pool import BaseAccountPool
from providers.gemini.webapi import AuthError
from utils.helper import parse_cookie_string
from utils.log import logger


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

    # ── Cookie-based Account Integration ──

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
            self._accounts[account["name"]] = account
            self._save(account["name"])
            self._condition.notify_all()
        return account

    # ── Gemini-specific: cookie management ──

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
            account["cookie"] = _auth_cookie_header(psid, psidts)
            account["psid"] = psid
            account["psidts"] = psidts
            account["cookie_jar"] = []
            # 更新 Cookie 即视为人工恢复账号，代理与在途请求计数保持不变。
            account["status"] = "active"
            account["cooldown_until"] = 0
            account["failure_count"] = 0
            account["error_message"] = ""
            self._save(name)
            self._condition.notify_all()
            return True

    def merge_cookie(self, name: str, updates: dict[str, str], jar: Any = None) -> None:
        with self._lock:
            account = self._accounts.get(name)
            if not account or not updates:
                return
            psid = updates.get("__Secure-1PSID") or account.get("psid", "")
            psidts = updates.get("__Secure-1PSIDTS") or account.get("psidts", "")
            if not psid or not psidts:
                return
            account["cookie"] = _auth_cookie_header(psid, psidts)
            account["psid"] = psid
            account["psidts"] = psidts
            if jar is not None:
                account["cookie_jar"] = _stored_cookies(jar)
            try:
                self._save(name)
            except Exception:
                self._pending_cookie_saves.add(name)
                raise
            else:
                self._pending_cookie_saves.discard(name)

    def _retry_pending_cookie_saves(self) -> None:
        with self._condition:
            for name in list(self._pending_cookie_saves):
                if name not in self._accounts:
                    self._pending_cookie_saves.discard(name)
                    continue
                try:
                    self._save(name)
                except Exception as exc:
                    logger.warning(f"[Gemini Cookie] 账号 [{name}] 续期凭证写库仍失败: {exc}")
                else:
                    self._pending_cookie_saves.discard(name)

    @staticmethod
    def is_auth_error(error: Exception | str) -> bool:
        """仅认证失效才停止账号，临时上游错误继续保留在池中。"""
        text = str(error).lower()
        return type(error).__name__ == "AuthError" or any(
            marker in text for marker in ("unauthenticated", "unauthorized", "cookie", "expired", "401", "未认证")
        )

    def mark_auth_failed(self, name: str, error: Exception | str) -> None:
        """认证失效时停用账号，但保留最近一次成功轮换的本地缓存。"""
        with self._condition:
            account = self._accounts[name]
            account["status"] = "error"
            account["failure_count"] = account.get("failure_count", 0) + 1
            account["error_message"] = str(error)[:500] or "Gemini Cookie 未认证，请更新 Cookie"
            self._save(name)
            self._condition.notify_all()

    def mark_refresh_verification_failed(self, name: str, error: Exception | str) -> None:
        """续期后验证失败时暂停账号；下次 Gemini-API 续期验证成功后自动恢复。"""
        if self.is_auth_error(error):
            self.mark_auth_failed(name, error)
            return
        with self._condition:
            account = self._accounts[name]
            account["status"] = "cooldown"
            account["cooldown_until"] = int(time.time() + 300)
            account["failure_count"] = account.get("failure_count", 0) + 1
            account["error_message"] = str(error)[:500]
            self._save(name)
            self._condition.notify_all()

    async def verify_refreshed_client(self, name: str, client: Any) -> None:
        """续期后只校验账号认证状态，避免周期性生成请求误判账号并消耗额度。"""
        try:
            await client._fetch_user_status()
            if not client._check_account_status():
                raise RuntimeError("Gemini Cookie 未认证，请更新完整 Cookie Header")
        except Exception as exc:
            self.mark_refresh_verification_failed(name, exc)
            logger.warning(f"[Gemini Refresh Verify] 账号 [{name}] 续期后测试失败: {exc}")
            return

        with self._condition:
            if (self._clients.get(name) is not client
                    or self._accounts.get(name, {}).get("status") not in {"active", "cooldown"}):
                return
            account = self._accounts[name]
            account["status"] = "active"
            account["cooldown_until"] = 0
            account["failure_count"] = 0
            account["error_message"] = ""
        try:
            self.merge_cookie(name, dict(client.cookies), client.cookies)
        except Exception as exc:
            logger.error(f"[Gemini Cookie] 账号 [{name}] 登录态有效但写库失败，将继续重试: {exc}")
            return
        with self._condition:
            self._condition.notify_all()
        logger.info(f"[Gemini Refresh Verify] 账号 [{name}] 续期后真实测试成功，已同步 Cookie")

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
                self.merge_cookie(name, dict(client.cookies), client.cookies)
                with self._condition:
                    self._clients[name] = client
                    self._condition.notify_all()
                logger.info(f"[Gemini Client] 账号 [{name}] 已认证，后台续期已启动")
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
                        self.mark_auth_failed(acc["name"], exc)
                    logger.warning(f"[Gemini Warmup] 账号 [{acc['name']}] 预热失败: {exc}")

        await asyncio.gather(*(warm_one(acc) for acc in active_accounts))

    async def maintain_clients(self) -> None:
        """Retry failed initializations and recreate stopped refresh tasks."""
        while True:
            try:
                self._retry_pending_cookie_saves()
                await self.warmup_clients()
            except Exception as exc:
                logger.error(f"[Gemini Warmup] 后台维护失败，将在一分钟后重试: {exc}")
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

    async def discard_client(self, name: str) -> None:
        # 只关闭空闲连接；正在使用的请求在退出时按代次自行关闭。
        self._request_generations[name] = self._request_generations.get(name, 0) + 1
        idle = self._idle_request_clients.pop(name, [])
        with self._condition:
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
                self.mark_auth_failed(name, exc)
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
            if not current:
                raise ValueError(f"Account {name} not found")
            current["supported_models"] = models
            current["models_updated_at"] = int(time.time())
            self._save(name)
            self._condition.notify_all()
        return models


gemini_account_service = GeminiAccountPool()
