from __future__ import annotations

import time
from typing import Any

from core.account_pool import BaseAccountPool
from utils.helper import parse_cookie_string


class DoubaoAccountPool(BaseAccountPool):
    """Doubao account pool: Cookie-based authentication."""

    PROVIDER_NAME = "Doubao"
    # 豆包与其他平台一样每个账号最多四个在途请求，账号调度起始间隔为两秒。
    MAX_INFLIGHT_TOTAL = 4
    MIN_DISPATCH_INTERVAL_SECONDS = 2.0

    def __init__(self) -> None:
        super().__init__(platform="doubao")

    def add_account(self, name: str, cookie: str, proxy: str = "") -> dict[str, Any]:
        cookies = parse_cookie_string(cookie)
        if not cookies.get("sessionid"):
            raise ValueError("Cookie 中缺少 sessionid")
        account = {
            "name": name.strip() or f"doubao-{int(time.time())}",
            "cookies": cookies,
            "proxy": proxy.strip(),
            "status": "active",
            "inflight": 0,
            "cooldown_until": 0,
            "failure_count": 0,
            "last_used_at": 0,
            "error_message": "",
        }
        with self._condition:
            self._accounts[account["name"]] = account
            self._save(account["name"])
            self._condition.notify_all()
        return account

    def update_cookie(self, name: str, cookie: str) -> bool:
        """更新豆包 Cookie 并保留当前代理及在途请求计数。"""
        cookies = parse_cookie_string(cookie)
        if not cookies.get("sessionid"):
            raise ValueError("Cookie 中缺少 sessionid")
        with self._condition:
            account = self._accounts.get(name)
            if account is None:
                return False
            account["cookies"] = cookies
            account["status"] = "active"
            account["cooldown_until"] = 0
            account["failure_count"] = 0
            account["error_message"] = ""
            self._save(name)
            self._condition.notify_all()
            return True

    # ── Hooks ──

    def _mask_sensitive(self, account: dict[str, Any]) -> dict[str, Any]:
        if account.get("cookies"):
            account["cookies"] = {
                k: (v[:4] + "..." if len(v) > 4 else "...")
                for k, v in account["cookies"].items()
            }
        return account

    def _classify_error(self, error: str, status_code: int | None = None) -> str:
        lower = error.lower()
        if status_code == 401 or any(word in lower for word in (
            "login failed", "login required", "session expired", "cookie expired", "login expired",
            "unauthorized", "401", "cookie invalid",
            "invalid cookie", "\u767b\u5f55\u5931\u6548", "\u767b\u9646\u5931\u6548",
            "\u5df2\u5931\u6548",
            "\u672a\u767b\u5f55", "\u4f1a\u8bdd\u8fc7\u671f", "\u8ba4\u8bc1\u5931\u8d25",
            "cookie \u65e0\u6548", "cookie\u65e0\u6548", "\u767b\u5f55\u8fc7\u671f",
            "\u767b\u9646\u8fc7\u671f",
        )):
            return "fatal"
        if status_code == 429 or any(word in lower for word in (
            "rate limit", "too many", "429", "quota", "credits exhausted",
            "\u989d\u5ea6", "\u8bf7\u6c42\u8fc7\u4e8e\u9891\u7e41",
        )):
            return "rate_limit"
        return "transient"


doubao_account_service = DoubaoAccountPool()
