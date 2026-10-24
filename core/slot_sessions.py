"""生图调用侧的当前会话缓存；不保存图片来源，不参与图片转存。"""
from __future__ import annotations

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from core.admission import CapacityUnavailable
from utils.log import logger


CURRENT_SESSION: ContextVar[SlotSession | None] = ContextVar("current_slot_session", default=None)
SESSION_TTL_SECONDS = 30 * 24 * 60 * 60


class ConversationUnavailable(RuntimeError):
    """账号正常，但旧会话明确无法续用；可以在原账号建立新会话。"""


@dataclass
class SlotSession:
    """一次真实调用持有的当前状态；只更新该槽位该平台的一份缓存。"""
    owner: SlotSessions
    key: str
    platform: str
    data: dict[str, Any] | None = None
    account_hint: int | None = None
    excluded_account_ids: set[int] = field(default_factory=set)
    initial_existed: bool = False
    abandoned: bool = False

    @property
    def account_id(self) -> int | None:
        return self.data["accountId"] if self.data else self.account_hint

    @property
    def conversation_ref(self) -> dict | None:
        return self.data["conversationRef"] if self.data else None

    async def save(self, account_id: int, reference: dict) -> None:
        """上游已确认会话即可更新，图片下载/上传失败不会回滚这份状态。"""
        value = {"version": 1, "accountId": account_id, "conversationRef": copy.deepcopy(reference)}
        if not self.owner.valid(value, self.platform):
            return
        if self.data == value:
            return
        if self.abandoned:
            return
        self.data = value
        self.account_hint = None
        if self.owner.redis is not None:
            try:
                # 若换号接口已清除键，防止旧请求在途写回复活旧会话
                if self.initial_existed and hasattr(self.owner.redis, "exists"):
                    exists = await asyncio.wait_for(self.owner.redis.exists(self.key), 1)
                    if not exists:
                        logger.info("event=slot_session_writeback_skipped_due_to_switch key=%s", self.key)
                        return
                await asyncio.wait_for(self.owner.redis.set(
                    self.key, json.dumps(value, ensure_ascii=False), ex=SESSION_TTL_SECONDS), 1)
            except Exception as exc:
                logger.warning("event=slot_session_cache_write_failed platform=%s reason_code=%s", self.platform, type(exc).__name__)

    async def forget(self) -> None:
        """确定旧账号/会话失效后移除首选，不删除上游历史会话。"""
        self.data = None
        self.account_hint = None
        self.initial_existed = False
        if self.owner.redis is not None:
            try:
                await asyncio.wait_for(self.owner.redis.delete(self.key), 1)
            except Exception as exc:
                logger.warning("event=slot_session_cache_delete_failed platform=%s reason_code=%s", self.platform, type(exc).__name__)


class SlotSessions:
    """复用应用Redis连接；同键占用只覆盖实际Python worker生命周期。"""
    def __init__(self) -> None:
        self.redis = None
        self._active: set[str] = set()

    @staticmethod
    def valid(value: Any, platform: str) -> bool:
        """缓存是外部输入，损坏或过期格式按未命中处理。"""
        if not isinstance(value, dict) or value.get("version") != 1:
            return False
        if type(value.get("accountId")) is not int or value["accountId"] <= 0:
            return False
        ref = value.get("conversationRef")
        if not isinstance(ref, dict):
            return False
        if platform == "gpt":
            return isinstance(ref.get("conversationId"), str) and bool(ref["conversationId"])
        metadata = ref.get("metadata")
        return (isinstance(ref.get("cid"), str) and bool(ref["cid"])
                and isinstance(metadata, list) and len(metadata) >= 3
                and metadata[0] == ref["cid"] and bool(metadata[1]) and bool(metadata[2]))

    @asynccontextmanager
    async def claim(self, task_code: str, item_id: int | None, platform: str):
        """无槽位或豆包保持原调用；忙碌同键用现有容量信号退队。"""
        if platform not in {"gpt", "gemini"} or not task_code or item_id is None:
            yield None
            return
        key = f"flexi:ai:slot-session:{task_code}:{item_id}:{platform}"
        # 单事件循环内检查与占用之间没有await，不存在两次同时取得执行权。
        if key in self._active:
            raise CapacityUnavailable()
        self._active.add(key)
        session = SlotSession(self, key, platform)
        token = CURRENT_SESSION.set(session)
        try:
            if self.redis is not None:
                try:
                    raw = await asyncio.wait_for(self.redis.getex(key, ex=SESSION_TTL_SECONDS), 1)
                    if raw:
                        session.initial_existed = True
                        value = json.loads(raw)
                        if self.valid(value, platform):
                            session.data = value
                except Exception as exc:
                    logger.warning("event=slot_session_cache_read_failed platform=%s reason_code=%s", platform, type(exc).__name__)
            yield session
        finally:
            CURRENT_SESSION.reset(token)
            self._active.discard(key)


slot_sessions = SlotSessions()
