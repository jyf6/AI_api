from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from core.account_pool import BaseAccountPool
from providers.gemini.webapi import AuthError
from utils.helper import parse_cookie_string
from utils.log import logger


def _get_cached_cookies_for_psid(psid: str) -> dict[str, str]:
    """Read latest rotated cookies saved by gemini_webapi from local disk cache."""
    cache_dir = Path(os.getenv("GEMINI_COOKIE_PATH") or (tempfile.gettempdir() + "/gemini_webapi"))
    cache_file = cache_dir / f".cached_cookies_{psid}.json"
    if not cache_file.is_file():
        return {}
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            cookies = {
                item["name"]: item["value"]
                for item in data
                if isinstance(item, dict) and "name" in item and "value" in item
            }
            return cookies
    except Exception as exc:
        logger.warning(f"Failed to read cached cookies for {psid}: {exc}")
    return {}


def _clear_cached_cookies_for_psids(*psids: str) -> None:
    """人工更新 Cookie 时删除旧会话缓存，避免缓存覆盖新凭据。"""
    cache_dir = Path(os.getenv("GEMINI_COOKIE_PATH") or (tempfile.gettempdir() + "/gemini_webapi"))
    for psid in set(filter(None, psids)):
        (cache_dir / f".cached_cookies_{psid}.json").unlink(missing_ok=True)


class GeminiAccountPool(BaseAccountPool):
    """Gemini 账号池：续期交给 Gemini-API，代理只验证并持久化续期结果。"""

    PROVIDER_NAME = "Gemini"

    def __init__(self) -> None:
        super().__init__(platform="gemini")
        self._clients: dict[str, Any] = {}
        self._client_locks: dict[str, asyncio.Lock] = {}

    # ── Cookie-based Account Integration ──

    def add_account(self, name: str, cookie: str, proxy: str = "") -> dict[str, Any]:
        cookies = parse_cookie_string(cookie)
        psid = cookies.get("__Secure-1PSID")
        if not psid:
            raise ValueError("Gemini Cookie 中缺少 __Secure-1PSID")
        psidts = cookies.get("__Secure-1PSIDTS")
        if not psidts:
            raise ValueError("Gemini Cookie 中缺少 __Secure-1PSIDTS，请粘贴完整 Cookie Header")
        normalized_cookie = "; ".join(f"{key}={value}" for key, value in cookies.items())
        account = {
            "name": name.strip() or f"gemini-{int(time.time())}",
            "cookie": normalized_cookie,
            "psid": psid,
            "psidts": psidts,
            "proxy": proxy.strip(),
            "status": "active",
            "inflight": 0,
            "cooldown_until": 0,
            "failure_count": 0,
            "last_used_at": 0,
            "error_message": "",
        }
        with self._condition:
            # 同名账号重新提交 Cookie 属于人工恢复，旧缓存绝不能反向覆盖新凭据。
            previous = self._accounts.get(account["name"], {})
            _clear_cached_cookies_for_psids(previous.get("psid", ""), psid)
            self._accounts[account["name"]] = account
            self._save()
            self._condition.notify_all()
        return account

    # ── Gemini-specific: cookie management ──

    def update_cookie(self, name: str, cookie: str) -> None:
        with self._condition:
            if name in self._accounts and cookie:
                old_psid = self._accounts[name].get("psid", "")
                self._accounts[name]["cookie"] = cookie
                parsed = parse_cookie_string(cookie)
                self._accounts[name]["psid"] = parsed.get("__Secure-1PSID", self._accounts[name]["psid"])
                self._accounts[name]["psidts"] = parsed.get("__Secure-1PSIDTS", self._accounts[name].get("psidts", ""))
                # 手工更新值优先，清理同账号旧 PSID 的磁盘会话缓存。
                _clear_cached_cookies_for_psids(old_psid, self._accounts[name]["psid"])
                # 更新 Cookie 即视为人工恢复账号，重新加入调度。
                self._accounts[name]["status"] = "active"
                self._accounts[name]["failure_count"] = 0
                self._accounts[name]["error_message"] = ""
                self._save()
                self._condition.notify_all()

    def merge_cookie(self, name: str, updates: dict[str, str]) -> None:
        with self._lock:
            account = self._accounts.get(name)
            if not account or not updates:
                return
            merged = parse_cookie_string(account.get("cookie", ""))
            merged.update({key: value for key, value in updates.items() if value})
            account["cookie"] = "; ".join(f"{key}={value}" for key, value in merged.items())
            account["psid"] = merged.get("__Secure-1PSID", account.get("psid", ""))
            account["psidts"] = merged.get("__Secure-1PSIDTS", account.get("psidts", ""))
            self._save()

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
            self._save()
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
            self._save()
            self._condition.notify_all()

    async def verify_refreshed_client(self, name: str, client: Any) -> None:
        """Gemini-API 完成续期后，使用同一会话真实验证再持久化 Cookie。"""
        try:
            if not client._check_account_status():
                raise RuntimeError("Gemini Cookie 未认证，请更新完整 Cookie Header")
            result = await client.generate_content("请只回复 OK。", temporary=True)
            if not result.text.strip():
                raise RuntimeError("Gemini 续期后的测试指令未返回文本")
            self.merge_cookie(name, dict(client.cookies))
            with self._condition:
                account = self._accounts[name]
                account["status"] = "active"
                account["cooldown_until"] = 0
                account["failure_count"] = 0
                account["error_message"] = ""
                self._save()
                self._condition.notify_all()
            logger.info(f"[Gemini Refresh Verify] 账号 [{name}] 续期后真实测试成功，已同步 Cookie")
        except Exception as exc:
            self.mark_refresh_verification_failed(name, exc)
            logger.warning(f"[Gemini Refresh Verify] 账号 [{name}] 续期后测试失败: {exc}")

    async def get_client(self, account: dict[str, Any]) -> Any:
        """Return a long-lived client with latest cached cookies so the Web API can rotate PSIDTS."""
        name = account["name"]
        with self._lock:
            lock = self._client_locks.setdefault(name, asyncio.Lock())
        async with lock:
            client = self._clients.get(name)
            if client is not None:
                return client

            from providers.gemini.webapi import GeminiClient

            psid = account["psid"]
            # 数据库为人工维护的权威 Cookie，磁盘缓存只补充轮换出的额外字段。
            current_cookies = parse_cookie_string(account.get("cookie", ""))
            cached = _get_cached_cookies_for_psid(psid)
            if cached:
                # 数据库是人工维护的权威来源；缓存只补全其没有的 Cookie 字段。
                cached.update(current_cookies)
                current_cookies = cached

            psidts = current_cookies.get("__Secure-1PSIDTS") or account.get("psidts") or None

            async def on_cookie_refreshed(refreshed_client: Any) -> None:
                await self.verify_refreshed_client(name, refreshed_client)

            client = GeminiClient(
                psid,
                psidts,
                proxy=account.get("proxy") or None,
                # Gemini-API 是唯一发起 Cookie 轮换的组件。
                on_cookie_refreshed=on_cookie_refreshed,
            )
            if current_cookies:
                client.cookies = current_cookies

            try:
                await client.init(
                    timeout=180,
                    auto_refresh=True,
                    refresh_interval=300,
                    impersonate="chrome145",
                )
                # 网页 API 即使拿到访客 token 也会完成 init，必须显式拒绝未认证会话。
                if not client._check_account_status():
                    raise RuntimeError("Gemini Cookie 未认证，请更新完整 Cookie Header")
                # 初始化仅保存当前已验证会话，不额外轮换 Cookie。
                self.merge_cookie(name, dict(client.cookies))
            except AuthError as exc:
                await client.close()
                raise RuntimeError("Gemini Cookie 未认证，请更新 Cookie") from exc
            except Exception:
                await client.close()
                raise
            self._clients[name] = client

            return client

    async def warmup_clients(self) -> None:
        """Eagerly warm up active Gemini accounts on startup to keep RotateCookies running 24/7."""
        with self._lock:
            active_accounts = [
                dict(acc) for acc in self._accounts.values()
                if acc.get("status") == "active" and acc.get("cookie")
            ]

        if not active_accounts:
            logger.info("[Gemini Warmup] 没有处于 active 状态的账号，跳过预热")
            return

        logger.info(f"[Gemini Warmup] 开始并发预热 {len(active_accounts)} 个活跃 Gemini 账号...")
        for acc in active_accounts:
            try:
                await self.get_client(acc)
                logger.info(f"[Gemini Warmup] 账号 [{acc['name']}] 预热成功，后台自动续期已常驻运行")
            except Exception as exc:
                if self.is_auth_error(exc):
                    self.mark_auth_failed(acc["name"], exc)
                logger.warning(f"[Gemini Warmup] 账号 [{acc['name']}] 预热失败: {exc}")

    async def close_clients(self) -> None:
        clients = list(self._clients.items())
        self._clients.clear()
        for name, client in clients:
            try:
                self.merge_cookie(name, dict(client.cookies))
                await client.close()
            except Exception:
                pass

    async def discard_client(self, name: str) -> None:
        client = self._clients.pop(name, None)
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass

    def release_account(
        self,
        key: str,
        success: bool,
        error: str = "",
        status_code: int | None = None,
        retry_after: int | None = None,
        task_type: str = "chat",
    ) -> None:
        """临时失败进入短冷却；认证失效才停止账号并要求人工更新 Cookie。"""
        if not success and not self.is_auth_error(error):
            norm_type = "image" if task_type == "image" else "chat"
            with self._condition:
                account = self._accounts.get(key)
                if not account:
                    return
                account[f"inflight_{norm_type}"] = max(0, account.get(f"inflight_{norm_type}", 1) - 1)
                account["inflight"] = account.get("inflight_image", 0) + account.get("inflight_chat", 0)
                account["failure_count"] = account.get("failure_count", 0) + 1
                # 连续失败按 30、60、120 秒退避，既避免反复命中，又不把偶发网络抖动永久拉黑。
                cooldown_seconds = min(120, 30 * (2 ** (account["failure_count"] - 1)))
                account["cooldown_until"] = int(time.time() + cooldown_seconds)
                account["error_message"] = error[:500]
                self._save()
                self._condition.notify_all()
            logger.warning(f"[Gemini Account] 账号 [{key}] 临时调用失败，冷却 {cooldown_seconds}s: {error}")
            return
        super().release_account(key, success, error, status_code, retry_after, task_type)

    # ── Hooks & Self-Healing ──

    def _mask_sensitive(self, account: dict[str, Any]) -> dict[str, Any]:
        safe = {
            k: v for k, v in account.items()
            if k not in {"cookie", "psid", "psidts"}
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
            self._save()
            self._condition.notify_all()
        return models


gemini_account_service = GeminiAccountPool()
