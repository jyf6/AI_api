"""调用侧会话缓存的生命周期、隔离与降级测试。"""
import asyncio

import pytest

from core.admission import CapacityUnavailable
from core.slot_sessions import CURRENT_SESSION, SlotSessions


class RedisMemory:
    def __init__(self):
        self.values = {}
        self.writes = 0

    async def getex(self, key, ex):
        return self.values.get(key)

    async def set(self, key, value, ex):
        self.writes += 1
        self.values[key] = value

    async def delete(self, key):
        self.values.pop(key, None)

    async def exists(self, key):
        return key in self.values


def test_current_session_survives_image_failure_and_new_cache_instance():
    async def scenario():
        redis = RedisMemory()
        cache = SlotSessions()
        cache.redis = redis
        with pytest.raises(RuntimeError):
            async with cache.claim("task", 1, "gpt") as session:
                assert CURRENT_SESSION.get() is session
                await session.save(2, {"conversationId": "first"})
                raise RuntimeError("Image download failed")
        replacement = SlotSessions()
        replacement.redis = redis
        async with replacement.claim("task", 1, "gpt") as session:
            assert session.account_id == 2
            assert session.conversation_ref == {"conversationId": "first"}
            await session.save(3, {"conversationId": "switched"})
        assert len(redis.values) == 1
        assert "switched" in next(iter(redis.values.values()))
        assert CURRENT_SESSION.get() is None
    asyncio.run(scenario())


def test_platform_task_and_slot_are_isolated_and_busy_claim_released():
    async def scenario():
        cache = SlotSessions()
        async with cache.claim("task", 1, "gpt"):
            with pytest.raises(CapacityUnavailable):
                async with cache.claim("task", 1, "gpt"):
                    pytest.fail("Must not enter a busy slot")
            for task, item, platform in (("task", 1, "gemini"), ("task", 2, "gpt"), ("other", 1, "gpt")):
                async with cache.claim(task, item, platform) as session:
                    assert session.account_id is None
            async with cache.claim("task", 1, "doubao") as session:
                assert session is None
        assert cache._active == set()
    asyncio.run(scenario())


def test_cache_errors_do_not_fail_images_and_metadata_is_copied():
    async def scenario():
        class BrokenRedis(RedisMemory):
            async def getex(self, key, ex):
                raise ConnectionError()
            async def set(self, key, value, ex):
                raise ConnectionError()
        cache = SlotSessions()
        cache.redis = BrokenRedis()
        async with cache.claim("task", 1, "gemini") as session:
            metadata = ["c_1", "r_1", "rc_1", None, None, None, None, None, None, "context"]
            await session.save(1, {"cid": "c_1", "metadata": metadata})
            metadata[1] = "changed"
            assert session.conversation_ref["metadata"][1] == "r_1"
            await session.save(1, {"cid": "c_1", "metadata": []})
            assert session.conversation_ref["metadata"][1] == "r_1"
    asyncio.run(scenario())


def test_repeated_stream_metadata_does_not_repeat_redis_writes():
    async def scenario():
        cache = SlotSessions()
        cache.redis = RedisMemory()
        async with cache.claim("task", 1, "gemini") as session:
            ref = {"cid": "c_1", "metadata": ["c_1", "r_1", "rc_1"]}
            for _ in range(20):
                await session.save(1, ref)
            assert cache.redis.writes == 1
    asyncio.run(scenario())


def test_worker_keeps_slot_occupied_after_http_cancellation(monkeypatch):
    async def scenario():
        from api.model_execution import execute_model_request
        from core.admission import ModelAdmission, mark_model_request_started
        owner = ModelAdmission({"gpt": 2, "gemini": 1, "doubao": 1}, 2)
        monkeypatch.setattr("core.admission.model_admission", owner)
        cache = SlotSessions()
        started, finish, ended = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def work():
            async with cache.claim("task", 1, "gpt"):
                await mark_model_request_started()
                started.set()
                await finish.wait()
            ended.set()
        request = asyncio.create_task(execute_model_request("gpt", "detached-slot", work))
        await started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        with pytest.raises(CapacityUnavailable):
            async with cache.claim("task", 1, "gpt"):
                pytest.fail("HTTP断开不能让仍在执行的槽位被再次占用")
        finish.set()
        await ended.wait()
        async with cache.claim("task", 1, "gpt"):
            pass
    asyncio.run(scenario())
