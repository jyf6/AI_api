from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ACCOUNTS_FILE = DATA_DIR / "gemini_accounts.json"


def parse_cookie_header(value: str) -> dict[str, str]:
    return {
        name.strip(): cookie.strip()
        for part in value.split(";")
        if "=" in part
        for name, cookie in [part.strip().split("=", 1)]
        if name.strip() and cookie.strip()
    }


class GeminiAccountService:
    MAX_INFLIGHT_PER_ACCOUNT = 2
    COOLDOWN_SECONDS = 60

    def __init__(self) -> None:
        self._lock = Lock()
        self._accounts = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not ACCOUNTS_FILE.exists():
            return {}
        try:
            data = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self) -> None:
        ACCOUNTS_FILE.write_text(json.dumps(self._accounts, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_account(self, name: str, cookie: str, proxy: str = "") -> dict[str, Any]:
        cookies = parse_cookie_header(cookie)
        psid = cookies.get("__Secure-1PSID")
        if not psid:
            raise ValueError("Gemini Cookie 中缺少 __Secure-1PSID")
        account = {"name": name.strip() or f"gemini-{int(time.time())}", "cookie": cookie.strip(),
                   "psid": psid, "psidts": cookies.get("__Secure-1PSIDTS", ""), "proxy": proxy.strip(),
                   "status": "active", "inflight": 0, "cooldown_until": 0, "failure_count": 0,
                   "error_message": ""}
        with self._lock:
            self._accounts[account["name"]] = account
            self._save()
        return account

    def list_accounts(self) -> list[dict[str, Any]]:
        with self._lock:
            result = []
            for account in self._accounts.values():
                item = {k: v for k, v in account.items() if k not in {"cookie", "psid", "psidts"}}
                item["cookie"] = account.get("cookie", "")[:18] + "..."
                result.append(item)
            return result

    def delete_account(self, name: str) -> bool:
        with self._lock:
            if name not in self._accounts:
                return False
            del self._accounts[name]
            self._save()
            return True

    def update_cookie(self, name: str, cookie: str) -> None:
        with self._lock:
            if name in self._accounts and cookie:
                self._accounts[name]["cookie"] = cookie
                parsed = parse_cookie_header(cookie)
                self._accounts[name]["psid"] = parsed.get("__Secure-1PSID", self._accounts[name]["psid"])
                self._accounts[name]["psidts"] = parsed.get("__Secure-1PSIDTS", self._accounts[name].get("psidts", ""))
                self._save()

    def get_available_account(self) -> dict[str, Any]:
        now = int(time.time())
        with self._lock:
            candidates = [a for a in self._accounts.values() if a.get("status") == "active"
                          and a.get("inflight", 0) < self.MAX_INFLIGHT_PER_ACCOUNT
                          and a.get("cooldown_until", 0) <= now]
            if not candidates:
                raise RuntimeError("No available Gemini accounts")
            account = min(candidates, key=lambda a: a.get("last_used_at", 0))
            account["inflight"] = account.get("inflight", 0) + 1
            account["last_used_at"] = now
            self._save()
            return dict(account)

    def release_account(self, name: str, success: bool, error: str = "") -> None:
        with self._lock:
            account = self._accounts.get(name)
            if not account:
                return
            account["inflight"] = max(0, account.get("inflight", 1) - 1)
            if success:
                account["failure_count"] = 0
                account["cooldown_until"] = 0
                account["error_message"] = ""
            else:
                account["failure_count"] = account.get("failure_count", 0) + 1
                account["cooldown_until"] = int(time.time()) + self.COOLDOWN_SECONDS
                account["error_message"] = error[:300]
                if any(word in error.lower() for word in ("cookie", "auth", "unauthorized", "401", "expired")):
                    account["status"] = "error"
            self._save()

    def stats(self) -> dict[str, int]:
        with self._lock:
            now = int(time.time())
            return {"total_accounts": len(self._accounts),
                    "active_accounts": sum(a.get("status") == "active" for a in self._accounts.values()),
                    "cooldown_accounts": sum(a.get("cooldown_until", 0) > now for a in self._accounts.values()),
                    "total_inflight_tasks": sum(a.get("inflight", 0) for a in self._accounts.values())}


gemini_account_service = GeminiAccountService()
