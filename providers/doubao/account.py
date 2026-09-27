from __future__ import annotations

import time
from typing import Any

from core.account_pool import BaseAccountPool
from utils.helper import parse_cookie_string


class DoubaoAccountPool(BaseAccountPool):
    """Doubao account pool: Cookie-based authentication."""

    PROVIDER_NAME = "Doubao"
    # 严格单并发：同一个 Cookie 禁止多线程同时请求，必须排队串行
    MAX_INFLIGHT_TOTAL = 1
    # 调度最小间隔：同一账号连续两次被调度之间强制保持在 2.0 秒以上
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
            self._save()
            self._condition.notify_all()
        return account

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
            "login", "session", "unauthorized", "expired", "401", "cookie invalid",
            "invalid cookie", "\u767b\u5f55\u5931\u6548", "\u767b\u9646\u5931\u6548",
            "\u5df2\u5931\u6548",
            "\u672a\u767b\u5f55", "\u4f1a\u8bdd\u8fc7\u671f", "\u8ba4\u8bc1\u5931\u8d25",
            "cookie \u65e0\u6548", "cookie\u65e0\u6548", "\u767b\u5f55\u8fc7\u671f",
            "\u767b\u9646\u8fc7\u671f",
        )):
            return "fatal"
        if status_code == 429 or any(word in lower for word in ("rate limit", "too many", "429", "\u8bf7\u6c42\u8fc7\u4e8e\u9891\u7e41")):
            return "rate_limit"
        return "transient"


doubao_account_service = DoubaoAccountPool()
