from __future__ import annotations

import time
from typing import Any

from core.account_pool import BaseAccountPool
from utils.helper import parse_cookie_string


class DoubaoAccountPool(BaseAccountPool):
    """Doubao account pool: Cookie-based authentication."""

    PROVIDER_NAME = "Doubao"

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
        if any(word in lower for word in ("login", "session", "unauthorized", "expired", "401")):
            return "fatal"
        if status_code == 429 or any(word in lower for word in ("rate limit", "too many")):
            return "rate_limit"
        return "transient"


doubao_account_service = DoubaoAccountPool()
