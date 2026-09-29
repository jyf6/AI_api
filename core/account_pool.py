from __future__ import annotations

import json
import random
import time
from pathlib import Path
from threading import Condition, Lock
from typing import Any

from utils.log import logger
from core.database import database


class BaseAccountPool:
    """账号池：负责账号占用、等待、释放与失效，不设置调用冷却。

    Subclasses override hooks to customize behavior per provider.
    """

    MAX_INFLIGHT_TOTAL: int = 4
    MIN_DISPATCH_INTERVAL_SECONDS: float = 1.0
    RATE_LIMIT_COOLDOWN_SECONDS: int = 300
    PROVIDER_NAME: str = "base"

    def __init__(self, data_file: Path | None = None, platform: str = "") -> None:
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._data_file = data_file
        self._platform = platform
        self._accounts: dict[str, dict[str, Any]] = self._load()
        # 同一账号当前并发批次的结果只放内存；账号健康状态在批次收敛时持久化。
        self._batches: dict[str, dict[str, Any]] = {}

    # ── Persistence ──

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._platform:
            rows = database.list_accounts(self._platform)
            if rows:
                return {
                    str(row.get("email") or row.get("name")): {
                        **row,
                        "inflight": 0,
                        "inflight_image": 0,
                        "inflight_chat": 0,
                        "last_used_at": 0,
                        "last_dispatched_at": 0.0,
                    }
                    for row in rows
                }
        if self._data_file is None:
            return {}
        self._data_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._data_file.exists():
            return {}
        try:
            data = json.loads(self._data_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.error(f"Failed to load {self._data_file.name}: {exc}")
            return {}

    def _save(self, account_key: str | None = None) -> None:
        if self._platform:
            if account_key is None:
                accounts = self._accounts.items()
            elif account_key in self._accounts:
                accounts = ((account_key, self._accounts[account_key]),)
            else:
                return
            for key, account in accounts:
                credentials = {
                    k: v for k, v in account.items()
                    if k not in {
                        "name", "email", "proxy", "status",
                        "inflight", "inflight_image", "inflight_chat",
                        "last_used_at", "last_dispatched_at", "cooldown_until", "failure_count", "error_message"
                    }
                }
                database.import_account(
                    self._platform, key, credentials,
                    account.get("proxy", ""), account.get("status", "active"),
                    int(account.get("cooldown_until", 0)), int(account.get("failure_count", 0)),
                    account.get("error_message", "")
                )
            return
        if self._data_file is None:
            return
        try:
            self._data_file.write_text(
                json.dumps(self._accounts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error(f"Failed to save {self._data_file.name}: {exc}")

    # ── CRUD ──

    def delete_account(self, key: str) -> bool:
        with self._lock:
            if key not in self._accounts:
                return False
            del self._accounts[key]
            self._batches.pop(key, None)
            if self._platform:
                database.delete_account(self._platform, key)
            elif self._data_file is not None:
                self._save()
            return True

    def set_account_health(self, key: str, healthy: bool, error: str = "") -> None:
        """Update verified account health without changing task in-flight counters."""
        with self._condition:
            account = self._accounts.get(key)
            if not account:
                return
            if healthy:
                account["status"] = "active"
                account["cooldown_until"] = 0
                account["failure_count"] = 0
                account["error_message"] = ""
                self._batches.pop(key, None)
            else:
                account["status"] = "error"
                account["cooldown_until"] = 0
                account["failure_count"] = account.get("failure_count", 0) + 1
                account["error_message"] = error[:500]
            self._save(key)
            self._condition.notify_all()

    def list_accounts(self) -> list[dict[str, Any]]:
        with self._condition:
            self._restore_expired_cooldowns(time.time())
            return [self._mask_sensitive(dict(acc)) for acc in self._accounts.values()]

    def _restore_expired_cooldowns(self, now: float) -> None:
        """恢复已到期的持久冷却账号，调用方需持有账号池锁。"""
        restored_keys = []
        for key, account in self._accounts.items():
            if account.get("status") == "cooldown" and int(account.get("cooldown_until", 0) or 0) <= now:
                account["status"] = "active"
                account["cooldown_until"] = 0
                restored_keys.append(key)
        if restored_keys:
            for key in restored_keys:
                self._save(key)
            self._condition.notify_all()

    # ── Scheduling ──

    def _reserve_available_account(self, task_type: str) -> dict[str, Any] | None:
        norm_type = "image" if task_type == "image" else "chat"
        now = time.time()
        self._restore_expired_cooldowns(now)
        candidates = []
        for account in self._accounts.values():
            if account.get("status") != "active" or account.get("inflight", 0) >= self.MAX_INFLIGHT_TOTAL:
                continue
            if self._batches.get(account.get("name") or account.get("email"), {}).get("probing"):
                continue
            # 临时故障账号在冷却期内不参与普通调度，避免随机再次命中同一故障出口。
            if account.get("cooldown_until", 0) > now:
                continue
            if now - account.get("last_dispatched_at", 0.0) < self.MIN_DISPATCH_INTERVAL_SECONDS:
                continue
            candidates.append(account)
        if not candidates:
            return None
        selected = self._select_strategy(candidates)
        key = selected.get("name") or selected.get("email")
        if selected.get("inflight", 0) == 0:
            self._batches[key] = {"success": False, "failures": 0, "probing": False, "explicit": False, "error": ""}
        selected[f"inflight_{norm_type}"] = selected.get(f"inflight_{norm_type}", 0) + 1
        selected["inflight"] = selected.get("inflight_image", 0) + selected.get("inflight_chat", 0)
        selected["last_used_at"] = int(now)
        selected["last_dispatched_at"] = now
        return dict(selected)

    def get_available_account(self, task_type: str = "chat") -> dict[str, Any]:
        """立即获取账号；管理端探测等非任务调用可据此得到明确的无可用账号错误。"""
        with self._condition:
            account = self._reserve_available_account(task_type)
            if account is None:
                raise RuntimeError(f"No available {self.PROVIDER_NAME} accounts for task_type '{task_type}'")
            return account

    def wait_for_available_account(self, task_type: str = "chat") -> dict[str, Any]:
        """业务请求在账号忙碌时等待释放，不因正常占用而失败。"""
        with self._condition:
            while True:
                account = self._reserve_available_account(task_type)
                if account is not None:
                    return account
                now = time.time()
                deadlines = []
                for candidate in self._accounts.values():
                    if candidate.get("status") not in {"active", "cooldown"} or candidate.get("inflight", 0) >= self.MAX_INFLIGHT_TOTAL:
                        continue
                    deadline = max(
                        float(candidate.get("cooldown_until", 0)),
                        float(candidate.get("last_dispatched_at", 0)) + self.MIN_DISPATCH_INTERVAL_SECONDS,
                    )
                    if deadline > now:
                        deadlines.append(deadline)
                # 没有可用账号时，既等待释放，也在最早冷却或分配间隔到期时自动重新调度。
                timeout = max(0.01, min(deadlines) - now) if deadlines else None
                self._condition.wait(timeout)

    def release_account(
        self,
        key: str,
        success: bool,
        error: str = "",
        status_code: int | None = None,
        retry_after: int | None = None,
        task_type: str = "chat",
    ) -> None:
        """释放单次请求；普通故障在当前并发批次全部结束后才判定账号健康。"""
        norm_type = "image" if task_type == "image" else "chat"
        with self._condition:
            account = self._accounts.get(key)
            if not account:
                return

            previous_health = (
                account.get("status"), account.get("cooldown_until", 0),
                account.get("failure_count", 0), account.get("error_message", ""),
            )

            account[f"inflight_{norm_type}"] = max(0, account.get(f"inflight_{norm_type}", 1) - 1)
            account["inflight"] = account.get("inflight_image", 0) + account.get("inflight_chat", 0)
            batch = self._batches.setdefault(key, {"success": False, "failures": 0, "probing": False, "explicit": False, "error": ""})
            if success:
                batch["success"] = True
            elif error:
                category = self._classify_error(error, status_code)
                if category == "fatal":
                    account["status"] = "error"
                    account["cooldown_until"] = 0
                    account["error_message"] = error[:500]
                    account["failure_count"] = account.get("failure_count", 0) + 1
                    batch["explicit"] = True
                    self._save(key)
                elif category == "rate_limit":
                    account["status"] = "active"
                    account["cooldown_until"] = int(time.time()) + (retry_after if retry_after and retry_after > 0 else self.RATE_LIMIT_COOLDOWN_SECONDS)
                    account["error_message"] = error[:500]
                    account["failure_count"] = account.get("failure_count", 0) + 1
                    batch["explicit"] = True
                    self._save(key)
                else:
                    # 暂停新分配，等待本批已在执行的请求给出结果。
                    batch["failures"] += 1
                    batch["probing"] = True
                    batch["error"] = error[:500]
            if account["inflight"] == 0:
                if not batch["explicit"]:
                    if batch["success"]:
                        account["status"] = "active"
                        account["cooldown_until"] = 0
                        account["failure_count"] = 0
                        account["error_message"] = ""
                    elif batch["failures"]:
                        account["status"] = "active"
                        account["cooldown_until"] = 0
                        account["failure_count"] = account.get("failure_count", 0) + 1
                        account["error_message"] = batch["error"]
                    current_health = (
                        account.get("status"), account.get("cooldown_until", 0),
                        account.get("failure_count", 0), account.get("error_message", ""),
                    )
                    if current_health != previous_health:
                        self._save(key)
                self._batches.pop(key, None)
            self._condition.notify_all()

    def stats(self) -> dict[str, int]:
        with self._lock:
            now = time.time()
            self._restore_expired_cooldowns(now)
            return {
                "total_accounts": len(self._accounts),
                "active_accounts": sum(
                    1 for a in self._accounts.values() if a.get("status") == "active"
                ),
                "cooldown_accounts": sum(
                    1 for account in self._accounts.values()
                    if int(account.get("cooldown_until", 0) or 0) > now
                ),
                "total_inflight_tasks": sum(
                    a.get("inflight", 0) for a in self._accounts.values()
                ),
                "total_inflight_image": sum(
                    a.get("inflight_image", 0) for a in self._accounts.values()
                ),
                "total_inflight_chat": sum(
                    a.get("inflight_chat", 0) for a in self._accounts.values()
                ),
            }

    def capacity(self) -> dict[str, int | None]:
        """Return the current shared account capacity without reserving a slot."""
        with self._lock:
            now = time.time()
            self._restore_expired_cooldowns(now)
            usable = [
                account for account in self._accounts.values()
                if account.get("status") == "active" and account.get("cooldown_until", 0) <= now
                and not self._batches.get(account.get("name") or account.get("email"), {}).get("probing")
            ]
            cooldowns = [
                float(account.get("cooldown_until", 0))
                for account in self._accounts.values()
                if float(account.get("cooldown_until", 0)) > now
            ]
            return {
                "available_slots": sum(max(0, self.MAX_INFLIGHT_TOTAL - account.get("inflight", 0)) for account in usable),
                "total_slots": len(usable) * self.MAX_INFLIGHT_TOTAL,
                "cooldown_accounts": len(cooldowns),
                "next_available_at": int(min(cooldowns)) if cooldowns else None,
            }

    # ── Hooks (override in subclasses) ──

    def _select_strategy(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """每次调用从当前可用账号中随机选择，不与任务绑定。"""
        return random.choice(candidates)

    def _mask_sensitive(self, account: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive fields for list display. Override per provider."""
        return account

    def _classify_error(self, error: str, status_code: int | None = None) -> str:
        """Classify an error. Return 'fatal', 'rate_limit', or 'transient'.
        Override per provider for provider-specific error keywords.
        """
        lower = error.lower()
        if status_code == 401 or any(kw in lower for kw in (
            "invalid_grant", "invalid token", "token expired", "expired token", "unauthorized",
            "unauthenticated", "authentication failed", "account disabled", "deactivated", "401",
            "未认证", "登录失效", "cookie 无效", "cookie无效",
        )):
            return "fatal"
        if status_code == 429 or any(kw in lower for kw in (
            "quota", "rate limit", "too many", "429", "insufficient_quota", "quota exhausted",
            "quota exceeded", "credits exhausted", "额度不足", "额度耗尽",
        )):
            return "rate_limit"
        return "transient"
