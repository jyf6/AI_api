"""单进程模型执行额度；平台与全局额度在同一条件锁下领取。"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from contextvars import ContextVar


class CapacityUnavailable(Exception):
    """仅在模型请求尚未发出时由准入阶段抛出。"""

    def __init__(self, code: str = "LOCAL_CAPACITY_BUSY") -> None:
        super().__init__(code)
        self.code = code

    def payload(self, platform: str, request_id: str) -> dict:
        return {"code": self.code, "platform": platform, "request_id": request_id,
                "upstream_started": False, "retry_after_seconds": 5}


current_permit: ContextVar[ExecutionPermit | None] = ContextVar("model_execution_permit", default=None)


async def mark_model_request_started() -> None:
    permit = current_permit.get()
    if permit is not None:
        await permit.mark_upstream_started()


class ExecutionPermit:
    def __init__(self, owner: ModelAdmission, platform: str) -> None:
        self.owner = owner
        self.platform = platform
        self.released = False
        self._reserved = True
        self.upstream_started = False
        self.deadline: float | None = None

    async def begin_account_wait(self) -> None:
        """账号等待计数由准入器统一维护，不暴露内部条件锁给账号池。"""
        async with self.owner._condition:
            self.owner._waiting[self.platform] += 1

    async def end_account_wait(self) -> None:
        async with self.owner._condition:
            self.owner._waiting[self.platform] -= 1

    async def park_for_account(self) -> None:
        """首次模型发送前未领到账号时交还额度，其他平台可继续执行。"""
        async with self.owner._condition:
            if self._reserved and not self.upstream_started:
                self._reserved = False
                self.owner._reserved[self.platform] -= 1
                self.owner._global_reserved -= 1
                self.owner._condition.notify_all()

    async def ensure_reserved(self, deadline: float) -> None:
        """再次尝试账号前重新原子取得两种额度，保留最初准入截止时间。"""
        async with self.owner._condition:
            while not self._reserved:
                if self.released:
                    raise RuntimeError("Cannot reserve a released permit")
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise CapacityUnavailable()
                if (self.owner._reserved[self.platform] < self.owner.limits[self.platform]
                        and self.owner._global_reserved < self.owner.global_limit):
                    self.owner._reserved[self.platform] += 1
                    self.owner._global_reserved += 1
                    self._reserved = True
                    return
                try:
                    await asyncio.wait_for(self.owner._condition.wait(), remaining)
                except TimeoutError as exc:
                    raise CapacityUnavailable() from exc

    async def mark_upstream_started(self) -> None:
        async with self.owner._condition:
            if self.released or not self._reserved:
                raise RuntimeError("Cannot start upstream without a reserved permit")
            if not self.upstream_started:
                self.upstream_started = True
                self.owner._running[self.platform] += 1

    async def release(self) -> None:
        async with self.owner._condition:
            if self.released:
                return
            self.released = True
            if self._reserved:
                self._reserved = False
                self.owner._reserved[self.platform] -= 1
                self.owner._global_reserved -= 1
            if self.upstream_started:
                self.owner._running[self.platform] -= 1
            self.owner._condition.notify_all()


class ModelAdmission:
    PLATFORMS = ("gpt", "gemini", "doubao")

    def __init__(self, limits: Mapping[str, int], global_limit: int) -> None:
        if set(limits) != set(self.PLATFORMS):
            raise ValueError("All three platform limits must be configured")
        if any(type(value) is not int or value <= 0 for value in (*limits.values(), global_limit)):
            raise ValueError("Model execution limits must be positive integers")
        self.limits = dict(limits)
        self.global_limit = global_limit
        self._condition = asyncio.Condition()
        self._reserved = dict.fromkeys(self.PLATFORMS, 0)
        self._running = dict.fromkeys(self.PLATFORMS, 0)
        self._waiting = dict.fromkeys(self.PLATFORMS, 0)
        self._global_reserved = 0

    @classmethod
    def from_environment(cls) -> ModelAdmission:
        # 缺失或非法配置应使启动失败，不能退回无限并发。
        return cls({platform: int(os.environ[f"{platform.upper()}_LIMIT"])
                    for platform in cls.PLATFORMS}, int(os.environ["MODEL_GLOBAL_LIMIT"]))

    async def acquire(self, platform: str, deadline: float) -> ExecutionPermit:
        if platform not in self.limits:
            raise ValueError(f"Unknown platform: {platform}")
        async with self._condition:
            self._waiting[platform] += 1
            try:
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise CapacityUnavailable()
                    if (self._reserved[platform] < self.limits[platform]
                            and self._global_reserved < self.global_limit):
                        self._reserved[platform] += 1
                        self._global_reserved += 1
                        permit = ExecutionPermit(self, platform)
                        permit.deadline = deadline
                        return permit
                    try:
                        await asyncio.wait_for(self._condition.wait(), remaining)
                    except TimeoutError as exc:
                        raise CapacityUnavailable() from exc
            finally:
                self._waiting[platform] -= 1

    def snapshot(self, platform: str, account_capacity: Mapping) -> dict:
        # 调用方在事件循环内同步读取；中途没有 await，计数保持一致。
        account_free = account_capacity["available_slots"]
        global_free = self.global_limit - self._global_reserved
        return {**account_capacity,
                "account_free": account_free,
                "platform_limit": self.limits[platform],
                "platform_reserved": self._reserved[platform],
                "global_free": global_free,
                "upstream_running": self._running[platform],
                "waiting_requests": self._waiting[platform],
                "available_slots": min(account_free, self.limits[platform] - self._reserved[platform], global_free),
                "updated_at": int(time.time() * 1000)}


model_admission: ModelAdmission | None = None


def get_model_admission() -> ModelAdmission:
    if model_admission is None:
        raise RuntimeError("Model admission has not been initialized by application startup")
    return model_admission


def initialize_model_admission() -> None:
    global model_admission
    model_admission = ModelAdmission.from_environment()
