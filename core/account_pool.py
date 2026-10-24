from __future__ import annotations

import asyncio
import json
import math
import time
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Condition, Event, Lock
from typing import Any, Callable, TypeVar

from utils.log import logger, proxy_log_ref, stable_log_ref
from core.database import database
from core.admission import CapacityUnavailable, current_permit


class AccountWaitCancelled(Exception):
    """等待中的请求已取消，账号尚未交给业务调用。"""


class AccountUnavailable(RuntimeError):
    """指定账号需要人工恢复或已移除，调用侧应更换账号。"""


class BoundAccountCooldownError(RuntimeError):
    """绑定账号处于冷却中，直接向上层抛出剩余时间。"""
    def __init__(self, account_id: int, remaining_seconds: int, error_message: str = "",
                 cooldown_until: float | None = None):
        self.account_id = account_id
        self.remaining_seconds = remaining_seconds
        self.error_message = error_message
        # 保留账号池的真实到期点，刷新页面不能从历史相对秒数重新倒计时。
        self.cooldown_until = cooldown_until if cooldown_until is not None else time.time() + remaining_seconds
        super().__init__(f"Bound account {account_id} is in cooldown for {remaining_seconds}s")


T = TypeVar("T")


def serialized_account_edit(method):
    """人工修改彼此串行；这把锁不参与业务派单，数据库慢时账号池仍可调度。"""
    @wraps(method)
    def edit(self, *args, **kwargs):
        with self._management_lock:
            return method(self, *args, **kwargs)
    return edit


async def await_thread_result(call: Callable[[], T]) -> T:
    """取消 HTTP 请求后仍等待工作线程自行收尾，避免提前释放账号。"""
    worker = asyncio.create_task(asyncio.to_thread(call))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # shield 保证工作线程执行 finally；取走后台异常以免丢失失败记录。
        def observe_completion(done: asyncio.Task) -> None:
            try:
                done.result()
            except Exception as exc:
                logger.warning("已取消请求的上游线程失败 reason=%s", type(exc).__name__)

        worker.add_done_callback(observe_completion)
        raise


class BaseAccountPool:
    """账号池：负责账号占用、等待、释放与失效，不设置调用冷却。

    Subclasses override hooks to customize behavior per provider.
    """

    MAX_INFLIGHT_IMAGE: int = 4
    MAX_INFLIGHT_CHAT: int = 4
    MAX_INFLIGHT_TOTAL: int = 4
    MIN_DISPATCH_INTERVAL_SECONDS: float = 2.5
    RATE_LIMIT_COOLDOWN_SECONDS: int = 120
    PROVIDER_NAME: str = "base"


    def _account_has_capacity(self, account: dict[str, Any], norm_type: str = "chat") -> bool:
        """检查账号是否具有当前任务类型（image/chat）及全局总并发的可用配额。"""
        if account.get("inflight", 0) >= self.MAX_INFLIGHT_TOTAL:
            return False
        if norm_type == "image" and account.get("inflight_image", 0) >= self.MAX_INFLIGHT_IMAGE:
            return False
        if norm_type == "chat" and account.get("inflight_chat", 0) >= self.MAX_INFLIGHT_CHAT:
            return False
        return True

    def __init__(self, data_file: Path | None = None, platform: str = "") -> None:
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._management_lock = Lock()
        self._data_file = data_file
        self._platform = platform
        self._accounts: dict[str, dict[str, Any]] = self._load()
        self._health_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{platform}-health")
        # 同一账号当前并发批次的结果只放内存；账号健康状态在批次收敛时持久化。
        self._batches: dict[str, dict[str, Any]] = {}

    # ── Persistence ──

    def _persist_account_snapshot(self, key: str, snapshot: dict) -> dict:
        """只保存调用方快照，不访问实时账号池；调用方在池锁外执行。"""
        account = dict(snapshot)
        if not self._platform:
            return account
        if account.get("proxy") and not account.get("proxy_id"):
            proxy_id = database.ensure_proxy_node(account["proxy"])
            node = database.get_proxy_node(proxy_id)
            account.update(proxy_id=proxy_id, proxy_status=node["status"])
        credentials = {k: v for k, v in account.items() if k not in {
            "account_id", "name", "email", "proxy", "proxy_id", "proxy_status", "status",
            "credential_version", "credential_pending", "inflight", "inflight_image", "inflight_chat",
            "last_used_at", "last_dispatched_at", "cooldown_until", "failure_count", "error_message"
        }}
        account.update(database.import_account(
            self._platform, key, credentials, account.get("proxy", ""), account["status"],
            int(account.get("cooldown_until", 0)), int(account.get("failure_count", 0)),
            account.get("error_message", ""), account.get("proxy_id")
        ))
        return account

    def _register_account(self, key: str, snapshot: dict) -> dict:
        """人工录入先持久化，再发布；同身份现有在途计数不随重新登录归零。"""
        stored = self._persist_account_snapshot(key, snapshot)
        with self._condition:
            current = self._accounts.get(key)
            if (current is not None and current.get("account_id") == stored.get("account_id")
                    and current.get("credential_version", 0) > stored.get("credential_version", 0)):
                return dict(current)
            if current is not None and current.get("account_id") == stored.get("account_id"):
                for field in ("inflight", "inflight_image", "inflight_chat", "last_used_at", "last_dispatched_at"):
                    if field in current:
                        stored[field] = current[field]
            self._accounts[key] = stored
            self._condition.notify_all()
        if not self._platform:
            self._save()
        return stored

    def _replace_cookie_fields(self, key: str, fields: dict) -> bool:
        """人工 Cookie 更新按读取时的身份版本保存，后台续期抢先更新时要求重试。"""
        with self._condition:
            current = self._accounts.get(key)
            if current is None:
                return False
            snapshot = dict(current)
        if self._platform and not database.update_account_cookie(
            self._platform, key, fields, snapshot["credential_version"], snapshot["account_id"]
        ):
            raise RuntimeError("账号凭证已被并发更新，请重试")
        with self._condition:
            current = self._accounts.get(key)
            if (self._platform and current is not None and current["account_id"] == snapshot["account_id"]
                    and current["credential_version"] == snapshot["credential_version"] + 1
                    and all(current.get(field) == value for field, value in fields.items())):
                # 后台 CAS 冲突重载可能已经发布了本次人工写入，不重复推进版本。
                return True
            if current is None or any(current.get(field) != snapshot.get(field)
                                      for field in ("account_id", "credential_version")):
                raise RuntimeError("账号身份已被并发更新，请重试")
            current.update(fields, status="active", cooldown_until=0, failure_count=0, error_message="")
            if self._platform:
                current["credential_version"] += 1
            self._condition.notify_all()
        if not self._platform:
            self._save()
        return True

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
                account.update(self._persist_account_snapshot(key, account))
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

    def _save_health(self, key: str) -> None:
        """只提交健康字段的快照；数据库线程不持有账号池锁，也不回写 Cookie。"""
        if self._platform == "gpt" or not self._platform:
            self._save(key)
            return
        account = self._accounts[key]
        state = {field: account.get(field, 0) for field in
                 ("account_id", "credential_version", "cooldown_until", "failure_count")}
        state.update(status=account["status"], error_message=account.get("error_message", ""))
        future = self._health_executor.submit(database.save_account_health, self._platform, key, state)
        def observe(done):
            try:
                done.result()
            except Exception as exc:
                logger.error("Account health persistence failed platform=%s account_ref=%s reason=%s",
                             self._platform, stable_log_ref(f"{self._platform}-account", key), type(exc).__name__)
        future.add_done_callback(observe)

    # ── CRUD ──

    @serialized_account_edit
    def delete_account(self, key: str) -> bool:
        with self._condition:
            if key not in self._accounts:
                return False
            snapshot = dict(self._accounts[key])
        # 删除的数据库 I/O 在锁外；并发重新登录或刷新后的身份不能被旧删除覆盖。
        if self._platform and not database.delete_account(
            self._platform, key, snapshot["account_id"], snapshot["credential_version"]
        ):
            return False
        with self._condition:
            current = self._accounts.get(key)
            if current is not None and all(current.get(field) == snapshot.get(field)
                                           for field in ("account_id", "credential_version")):
                del self._accounts[key]
                self._batches.pop(key, None)
                self._condition.notify_all()
        if not self._platform and self._data_file is not None:
            self._save()
        return True

    @serialized_account_edit
    def set_account_proxy(self, key: str, proxy_id: int | None) -> bool:
        """固定或清除账号的代理绑定，并立即更新运行时账号。"""
        replacement = database.bind_account_proxy(self._platform, key, proxy_id)
        if replacement is None:
            return False
        with self._condition:
            current = self._accounts.get(key)
            if current is None or current["account_id"] != replacement["account_id"]:
                return False
            if current["credential_version"] <= replacement["credential_version"]:
                current.update(replacement)
            self._condition.notify_all()
            return True

    def refresh_proxy_node(self, proxy_id: int, proxy_url: str, status: str) -> None:
        """同步运行中的账号所绑定节点的最新地址和启停状态。"""
        rows = database.list_accounts(self._platform)
        with self._condition:
            for replacement in rows:
                key = replacement.get("email") or replacement.get("name")
                account = self._accounts.get(key)
                if (account is not None and replacement.get("proxy_id") == proxy_id
                        and account["account_id"] == replacement["account_id"]
                        and account["credential_version"] <= replacement["credential_version"]):
                    account.update(replacement)
            self._condition.notify_all()

    def refresh_proxy_status(self, proxy_id: int, status: str) -> None:
        """节点启停变更立即对当前账号调度生效。"""
        with self._condition:
            previous = {account.get("proxy_status") for account in self._accounts.values() if account.get("proxy_id") == proxy_id}
            for account in self._accounts.values():
                if account.get("proxy_id") == proxy_id:
                    account["proxy_status"] = status
            if previous and previous != {status}:
                logger.info("event=proxy_status_changed platform=%s proxy_ref=%s old_status=%s new_status=%s",
                            self._platform, stable_log_ref("proxy-node", str(proxy_id)), ",".join(sorted(map(str, previous))), status)
            self._condition.notify_all()

    def set_account_health(self, key: str, healthy: bool, error: str = "", *, expected_account: dict | None = None) -> None:
        """Update verified account health without changing task in-flight counters."""
        with self._condition:
            account = self._accounts.get(key)
            if not account:
                return
            # GPT 管理端探测结束时可能已重新登录；旧探测不能修改新身份的健康状态。
            if expected_account is not None and any(
                account.get(field) != expected_account.get(field)
                for field in ("account_id", "credential_version")
            ):
                return
            old_status = account.get("status")
            if healthy:
                # 健康探测不能提前解除限流、额度或网络冷却；仅冷却到期流程恢复派单。
                if account.get("status") == "cooldown" and account.get("cooldown_until", 0) > time.time():
                    self._condition.notify_all()
                    return
                account["status"] = "active"
                account["cooldown_until"] = 0
                account["failure_count"] = 0
                account["error_message"] = ""
                self._batches.pop(key, None)
                logger.info("event=account_health_verified platform=%s account_ref=%s old_status=%s new_status=active reason_code=verification_passed",
                            self._platform, stable_log_ref(f"{self._platform}-account", key), old_status)
            else:
                account["status"] = "error"
                account["cooldown_until"] = 0
                account["failure_count"] = account.get("failure_count", 0) + 1
                account["error_message"] = error[:500]
                logger.warning("event=account_health_verification_failed platform=%s account_ref=%s old_status=%s new_status=error reason_code=%s",
                               self._platform, stable_log_ref(f"{self._platform}-account", key),
                               old_status, "verification_failed")
            self._save_health(key)
            self._condition.notify_all()

    def list_accounts(self) -> list[dict[str, Any]]:
        with self._condition:
            now = time.time()
            self._restore_expired_cooldowns(now)
            result = []
            for acc in self._accounts.values():
                safe = dict(acc)
                cd = int(safe.get("cooldown_until", 0) or 0)
                safe["cooldown_remaining_seconds"] = max(0, cd - int(now)) if cd > 0 else 0
                for secret_identity in ("device_id", "web_id", "fp"):
                    safe.pop(secret_identity, None)
                if safe.get("proxy"):
                    parts = urlsplit(safe["proxy"])
                    host = parts.hostname or ""
                    if ":" in host:
                        host = f"[{host}]"
                    if parts.port:
                        host = f"{host}:{parts.port}"
                    safe["proxy"] = urlunsplit((parts.scheme, host, "", "", ""))
                result.append(self._mask_sensitive(safe))
            return result

    def _restore_expired_cooldowns(self, now: float) -> None:
        """恢复已到期的冷却账号，仅依据系统时钟到期被动自动翻转，严禁发起网络探活。"""
        restored_keys = []
        for key, account in self._accounts.items():
            cd_until = float(account.get("cooldown_until", 0) or 0)
            if cd_until > 0 and cd_until <= now:
                account["status"] = "active"
                account["cooldown_until"] = 0
                account["failure_count"] = 0
                account["error_message"] = ""
                restored_keys.append(key)
        if restored_keys:
            for key in restored_keys:
                self._save_health(key)
                logger.info("event=account_cooldown_expired platform=%s account_ref=%s old_status=cooldown new_status=active reason_code=cooldown_expired",
                            self._platform, stable_log_ref(f"{self._platform}-account", key))
            self._condition.notify_all()

    # ── Scheduling ──

    def _reserve_available_account(self, task_type: str, account_id: int | None = None,
                                   excluded_account_ids: set[int] | None = None) -> dict[str, Any] | None:
        norm_type = "image" if task_type == "image" else "chat"
        now = time.time()
        self._restore_expired_cooldowns(now)
        candidates = []
        for account in self._accounts.values():
            if account_id is not None and account.get("account_id") != account_id:
                continue
            if account.get("account_id") in (excluded_account_ids or ()):
                continue
            if account.get("status") != "active" or account.get("proxy_status") == "disabled" or not self._account_has_capacity(account, norm_type):
                continue
            # 定向账号允许在领取后执行自动刷新；人工失效已在领取入口排除。
            if account_id is None and not self._account_ready(account):
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

    def wait_for_available_account(self, task_type: str = "chat", cancelled: Event | None = None) -> dict[str, Any]:
        """业务请求在账号忙碌时等待释放，不因正常占用而失败。"""
        with self._condition:
            while True:
                # 取消检查与预订处于同一把锁下，避免唤醒后仍占用账号。
                if cancelled is not None and cancelled.is_set():
                    raise AccountWaitCancelled()
                account = self._reserve_available_account(task_type)
                if account is not None:
                    return account
                if not any(
                    candidate.get("status") in {"active", "cooldown"}
                    and candidate.get("proxy_status") != "disabled"
                    for candidate in self._accounts.values()
                ):
                    raise RuntimeError(f"No available {self.PROVIDER_NAME} accounts for task_type '{task_type}'")
                now = time.time()
                deadlines = []
                for candidate in self._accounts.values():
                    if candidate.get("status") not in {"active", "cooldown"} or not self._account_has_capacity(candidate, "image" if task_type == "image" else "chat"):
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

    async def acquire_account(self, task_type: str = "chat", deadline: float | None = None,
                              *, account_id: int | None = None,
                              excluded_account_ids: set[int] | None = None) -> dict[str, Any]:
        """Wait without blocking a worker thread; reservation and cancellation cannot interleave."""
        loop = asyncio.get_running_loop()
        permit = current_permit.get()
        if deadline is None:
            deadline = permit.deadline if permit is not None and not permit.upstream_started else loop.time() + 30
        if permit is not None:
            await permit.begin_account_wait()
        try:
            await self._refresh_model_call_counts()
            while True:
                if account_id is not None and not self.account_is_recoverable(account_id):
                    raise AccountUnavailable("Bound account requires manual recovery")
                if account_id is not None:
                    with self._condition:
                        bound = next((a for a in self._accounts.values() if a.get("account_id") == account_id), None)
                        if bound is not None:
                            cd_until = float(bound.get("cooldown_until", 0) or 0)
                            rem = cd_until - time.time()
                            if rem > 0:
                                raise BoundAccountCooldownError(account_id, max(1, math.ceil(rem)), bound.get("error_message", ""),
                                                                cooldown_until=cd_until)
                if loop.time() >= deadline:
                    with self._condition:
                        failure = self._account_wait_failure()
                    raise failure
                if permit is not None:
                    await permit.ensure_reserved(deadline)
                with self._condition:
                    account = (self._reserve_available_account(task_type)
                               if account_id is None and not excluded_account_ids
                               else self._reserve_available_account(task_type, account_id, excluded_account_ids))
                    healthy = any(
                        candidate.get("status") in {"active", "cooldown"}
                        and candidate.get("proxy_status") != "disabled"
                        and candidate.get("account_id") not in (excluded_account_ids or ())
                        for candidate in self._accounts.values()
                    )
                if account is not None:
                    try:
                        return await self._prepare_account_async(account)
                    except BaseException as exc:
                        self.release_account(account.get("name") or account.get("email"), False, task_type=task_type, acquired_account=account)
                        # 首次选中的账号也可能在自动刷新时确认失效，交给槽位调用侧继续换号。
                        if (isinstance(exc, Exception) and account.get("account_id") is not None
                                and not self.account_is_recoverable(account["account_id"])):
                            raise AccountUnavailable("Selected account requires manual recovery") from exc
                        raise
                if not healthy:
                    raise CapacityUnavailable("NO_HEALTHY_ACCOUNT")
                if permit is not None:
                    await permit.park_for_account()
                # Account dispatch intervals and cooldowns can expire without a release event.
                await asyncio.sleep(min(.05, max(0, deadline - loop.time())))
        finally:
            if permit is not None:
                await permit.end_account_wait()

    def account_is_recoverable(self, account_id: int) -> bool:
        """只判断账号能否自动恢复，忙碌与冷却不属于失效。"""
        with self._lock:
            return any(account.get("account_id") == account_id
                       and account.get("status") in {"active", "cooldown"}
                       and account.get("proxy_status") != "disabled"
                       and account.get("error_message") != "credential_auth_invalid"
                       for account in self._accounts.values())

    async def wait_for_model_send(self, account: dict[str, Any]) -> None:
        """所有请求客户端共用账号发送门槛，分析和生图在实际POST之前计时。"""
        key = account.get("account_id") or account.get("name") or account["email"]
        locks = self.__dict__.setdefault("_model_send_locks", {})
        times = self.__dict__.setdefault("_model_send_times", {})
        loop = asyncio.get_running_loop()
        permit = current_permit.get()
        deadline = (permit.deadline if permit is not None and not permit.upstream_started
                    and permit.deadline is not None else loop.time() + 30)
        async with locks.setdefault(key, asyncio.Lock()):
            while True:
                # 领取后其他在途请求仍可能触发冷却，发送前必须使用池中最新状态。
                if account.get("account_id") is not None and not self.account_is_recoverable(account["account_id"]):
                    raise AccountUnavailable("Account became unavailable before model send")
                with self._lock:
                    current = next((candidate for candidate in self._accounts.values()
                                    if candidate.get("account_id") == account.get("account_id")), account)
                    cooldown = current.get("cooldown_until", 0) - time.time()
                    if cooldown > 0:
                        raise BoundAccountCooldownError(
                            current["account_id"], max(1, math.ceil(cooldown)),
                            current.get("error_message", ""), cooldown_until=current["cooldown_until"],
                        )
                remaining = max(cooldown, times.get(key, 0) + self.MIN_DISPATCH_INTERVAL_SECONDS - loop.time())
                if remaining <= 0:
                    break
                budget = deadline - loop.time()
                if budget <= 0:
                    raise CapacityUnavailable()
                await asyncio.sleep(min(remaining, budget, .25))
            # 连接建联可能失败且不应计入调用量；计数在上游接受模型 POST 后独立写入。
            times[key] = loop.time()

    async def record_model_call(self, account: dict[str, Any]) -> None:
        """只在模型 POST 已建立响应后计数；读取、上传、下载与失败建联不计。"""
        key = account.get("account_id") or account.get("name") or account["email"]
        window = int(time.time() // 600)
        from core.slot_sessions import slot_sessions
        try:
            if self._platform in {"gpt", "gemini"} and slot_sessions.redis is not None:
                redis_key = f"flexi:ai:account-model-calls:{self._platform}:{window}"
                total = await slot_sessions.redis.hincrby(redis_key, str(key), 1)
                await slot_sessions.redis.expire(redis_key, 1200)
                with self._lock:
                    self.__dict__.setdefault("_model_call_counts", {})[key] = int(total)
            else:
                with self._lock:
                    counts = self.__dict__.setdefault("_model_call_counts", {})
                    counts[key] = counts.get(key, 0) + 1
        except Exception as exc:
            logger.warning("event=account_call_count_write_failed platform=%s reason_code=%s",
                           self._platform, type(exc).__name__)

    async def _refresh_model_call_counts(self) -> None:
        """每十分钟滚动计数由 Redis 保存；读取失败时继续使用进程内计数。"""
        from core.slot_sessions import slot_sessions
        if self._platform not in {"gpt", "gemini"} or slot_sessions.redis is None:
            return
        try:
            window = int(time.time() // 600)
            stored = await asyncio.wait_for(slot_sessions.redis.hgetall(
                f"flexi:ai:account-model-calls:{self._platform}:{window}"), 1)
            with self._lock:
                counts = self.__dict__.setdefault("_model_call_counts", {})
                counts.clear()
                for key, value in stored.items():
                    account_id = int(key)
                    counts[account_id] = int(value)
        except Exception as exc:
            logger.warning("event=account_call_count_read_failed platform=%s reason_code=%s",
                           self._platform, type(exc).__name__)

    def _account_wait_failure(self) -> Exception:
        """普通账号忙返回容量错误；平台可区分自己的凭证准备失败。"""
        return CapacityUnavailable()


    async def _prepare_account_async(self, account: dict[str, Any]) -> dict[str, Any]:
        return account

    def release_account(
        self,
        key: str,
        success: bool,
        error: str = "",
        status_code: int | None = None,
        retry_after: int | None = None,
        task_type: str = "chat",
        failure_scope: str = "account",
        acquired_account: dict[str, Any] | None = None,
    ) -> None:
        """释放单次请求；普通故障在当前并发批次全部结束后才判定账号健康。"""
        norm_type = "image" if task_type == "image" else "chat"
        with self._condition:
            account = self._accounts.get(key)
            if not account:
                return

            # 删除重建后的同名账号不承接旧身份的计数；同 ID 新凭证仍承接旧在途。
            if self._platform and acquired_account is not None:
                if account["account_id"] != acquired_account["account_id"]:
                    return
                if account["credential_version"] != acquired_account["credential_version"]:
                    success, error = False, ""
                batch = self._batches.get(key)
                if batch is not None and batch.get("credential_version") != account["credential_version"]:
                    self._batches.pop(key)

            previous_health = (
                account.get("status"), account.get("cooldown_until", 0),
                account.get("failure_count", 0), account.get("error_message", ""),
            )

            account[f"inflight_{norm_type}"] = max(0, account.get(f"inflight_{norm_type}", 1) - 1)
            account["inflight"] = account.get("inflight_image", 0) + account.get("inflight_chat", 0)
            batch = self._batches.setdefault(key, {"success": False, "failures": 0, "probing": False, "explicit": False, "error": ""})
            if self._platform and acquired_account is not None:
                batch["credential_version"] = account["credential_version"]
            if success:
                batch["success"] = True
            elif error:
                # 明确的上游认证/限流响应仍归账号；代理握手等传输异常不按错误正文误判账号。
                category = (
                    self._classify_error(error, status_code)
                    if failure_scope != "transport" or status_code in {401, 429}
                    else "transient"
                )
                if category == "fatal":
                    account["status"] = "error"
                    account["cooldown_until"] = 0
                    account["error_message"] = error[:500]
                    account["failure_count"] = account.get("failure_count", 0) + 1
                    batch["explicit"] = True
                    self._save_health(key)
                    logger.error("event=account_marked_error platform=%s account_ref=%s old_status=%s new_status=error status_code=%s reason_code=AUTH_OR_ACCOUNT_INVALID",
                                 self._platform, stable_log_ref(f"{self._platform}-account", key), previous_health[0], status_code)
                elif category == "rate_limit":
                    # 频控限流：强制冷却 2 分钟 (120 秒)，绝不记录代理失败
                    cooldown_secs = 120
                    account["status"] = "cooldown"
                    account["cooldown_until"] = int(time.time()) + cooldown_secs
                    account["error_message"] = error[:500]
                    account["failure_count"] = account.get("failure_count", 0) + 1
                    batch["explicit"] = True
                    self._save_health(key)
                    logger.warning("event=account_rate_limited platform=%s account_ref=%s old_status=%s new_status=cooldown status_code=%s cooldown_seconds=%d cooldown_until=%d reason_code=RATE_LIMIT",
                                   self._platform, stable_log_ref(f"{self._platform}-account", key), previous_health[0], status_code,
                                   cooldown_secs, account["cooldown_until"])
                elif category == "quota_exhausted":
                    # 额度耗尽 / 文件保护：优先用解析到的重置时间，缺省 5 小时 (18000 秒)，绝不记录代理失败
                    from utils.helper import QUOTA_RESET_DEFAULT_SECONDS, detect_quota_exhausted
                    _, detected_seconds = detect_quota_exhausted(error)
                    cooldown_secs = (
                        retry_after if retry_after and retry_after > 0
                        else detected_seconds if detected_seconds and detected_seconds > 0
                        else QUOTA_RESET_DEFAULT_SECONDS
                    )
                    account["status"] = "cooldown"
                    account["cooldown_until"] = int(time.time()) + cooldown_secs
                    account["error_message"] = error[:500]
                    account["failure_count"] = account.get("failure_count", 0) + 1
                    batch["explicit"] = True
                    self._save_health(key)
                    logger.warning("event=account_quota_exhausted platform=%s account_ref=%s old_status=%s new_status=cooldown status_code=%s cooldown_seconds=%d cooldown_until=%d reason_code=QUOTA_EXHAUSTED",
                                   self._platform, stable_log_ref(f"{self._platform}-account", key), previous_health[0], status_code,
                                   cooldown_secs, account["cooldown_until"])
                elif failure_scope == "timeout" or status_code == 504:
                    # 思考超时/读取超时：不扣罚账号健康度，不增加账号 failure_count，不增加代理失败，不进入冷却，保留会话绑定
                    logger.info("event=thinking_timeout_exempted platform=%s account_ref=%s status_code=504",
                                self._platform, stable_log_ref(f"{self._platform}-account", key))
                elif failure_scope == "transport":
                    # DNS/TCP/代理建联失败在一次同账号重试后冷却 5 秒；已领取请求仍继续收尾。
                    account["status"] = "cooldown"
                    account["cooldown_until"] = max(
                        float(account.get("cooldown_until", 0) or 0), time.time() + 5
                    )
                    account["error_message"] = "network_connect_failed"
                    account["failure_count"] = account.get("failure_count", 0) + 1
                    batch["explicit"] = True
                    self._save_health(key)
                    if account.get("proxy_id"):
                        proxy_ref = proxy_log_ref(account)
                        future = self._health_executor.submit(database.record_proxy_failure, account["proxy_id"])
                        def observe_proxy_failure(done):
                            # 后台记录失败必须可见，日志不包含代理地址或认证信息。
                            try:
                                done.result()
                            except Exception as exc:
                                logger.error("event=proxy_failure_persist_failed platform=%s proxy_ref=%s reason_code=%s",
                                             self._platform, proxy_ref, type(exc).__name__)
                        future.add_done_callback(observe_proxy_failure)
                    logger.warning("event=proxy_transport_failure platform=%s account_ref=%s proxy_ref=%s status_code=%s",
                                   self._platform, stable_log_ref(f"{self._platform}-account", key),
                                   proxy_log_ref(account), status_code)
                else:
                    # 暂停新分配，等待本批已在执行的请求给出结果。
                    batch["failures"] += 1
                    batch["probing"] = True
                    batch["error"] = error[:500]
                    logger.warning("event=account_transient_failure platform=%s account_ref=%s status_code=%s",
                                   self._platform, stable_log_ref(f"{self._platform}-account", key), status_code)
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
                        self._save_health(key)
                        if account.get("status") == "active" and batch["success"]:
                            logger.info("event=%s platform=%s account_ref=%s old_status=%s new_status=active reason_code=request_succeeded",
                                        "account_recovered" if previous_health[0] != "active" else "account_health_cleared",
                                        self._platform, stable_log_ref(f"{self._platform}-account", key), previous_health[0])
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
                and account.get("proxy_status") != "disabled"
                and self._account_ready(account)
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

    def _account_ready(self, account: dict[str, Any]) -> bool:
        return True

    def _select_strategy(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """按十分钟调用量、在途数、失败数、最久未派单依次选择。"""
        if self._platform == "doubao":
            return min(candidates, key=lambda account: (
                account.get("last_dispatched_at", 0.0), account.get("inflight", 0),
            ))
        counts = self.__dict__.get("_model_call_counts", {})
        return min(candidates, key=lambda account: (
            counts.get(account.get("account_id") or account.get("name") or account.get("email"), 0),
            account.get("inflight", 0), account.get("failure_count", 0),
            account.get("last_dispatched_at", 0.0),
        ))

    def _mask_sensitive(self, account: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive fields for list display. Override per provider."""
        return account

    def _classify_error(self, error: str, status_code: int | None = None) -> str:
        """Classify an error. Return 'fatal', 'rate_limit', 'quota_exhausted', or 'transient'.
        Override per provider for provider-specific error keywords.
        """
        lower = error.lower()
        if status_code == 401 or any(kw in lower for kw in (
            "invalid_grant", "invalid token", "token expired", "expired token", "unauthorized",
            "unauthenticated", "authentication failed", "account disabled", "deactivated", "401",
            "未认证", "登录失效", "cookie 无效", "cookie无效",
        )):
            return "fatal"
        # 真额度耗尽、图片生成上限或上传文件保护：统一按文案识别，避免只命中有限关键词。
        from utils.helper import detect_quota_exhausted
        is_quota, _ = detect_quota_exhausted(error)
        if is_quota:
            return "quota_exhausted"
        # 普通 429 频控限流
        if status_code == 429 or any(kw in lower for kw in (
            "rate limit", "too many", "429", "throttled", "temporarily blocked",
        )):
            return "rate_limit"
        return "transient"
