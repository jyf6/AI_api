from core.account_pool import BaseAccountPool
import asyncio
import time
from threading import Condition, Lock

import pytest

from core.account_pool import AccountUnavailable, BoundAccountCooldownError
from core.admission import CapacityUnavailable


def make_pool():
    """使用真实领取方法与隔离账号验证分类型并发，不触碰数据库。"""
    pool = BaseAccountPool.__new__(BaseAccountPool)
    pool._lock = Lock()
    pool._condition = Condition(pool._lock)
    pool._platform = ""
    pool._data_file = None
    pool._batches = {}
    pool.MAX_INFLIGHT_TOTAL = 8
    pool.MIN_DISPATCH_INTERVAL_SECONDS = 0
    pool._accounts = {"a": {"name": "a", "account_id": 1, "status": "active", "inflight": 0},
                      "b": {"name": "b", "account_id": 2, "status": "active", "inflight": 0}}
    return pool


def test_four_images_and_four_chats_have_independent_limits():
    pool = make_pool()
    pool._accounts.pop("b")
    assert pool._account_has_capacity(pool._accounts["a"], "image")
    for _ in range(4):
        assert pool._reserve_available_account("image") is not None
    assert pool._reserve_available_account("image") is None
    for _ in range(4):
        assert pool._reserve_available_account("chat") is not None
    assert pool._reserve_available_account("chat") is None
    assert pool._accounts["a"]["inflight"] == 8


def test_preferred_account_waits_when_busy_instead_of_selecting_other():
    async def scenario():
        pool = make_pool()
        pool._accounts["a"].update(inflight_image=4, inflight=4)
        with pytest.raises(CapacityUnavailable):
            await pool.acquire_account("image", deadline=asyncio.get_running_loop().time() + .01, account_id=1)
        assert pool._accounts["b"]["inflight"] == 0
        pool._accounts["a"].update(inflight_image=3, inflight=3)
        assert (await pool.acquire_account("image", account_id=1))["account_id"] == 1
    asyncio.run(scenario())


def test_preferred_disabled_or_missing_account_is_not_a_capacity_wait():
    async def scenario():
        pool = make_pool()
        pool._accounts["a"]["status"] = "error"
        for account_id in (1, 99):
            with pytest.raises(AccountUnavailable):
                await pool.acquire_account("image", account_id=account_id)
    asyncio.run(scenario())


def test_preferred_cooldown_returns_reason_without_sending():
    async def scenario():
        pool = make_pool()
        pool._accounts["a"]["cooldown_until"] = time.time() + .08
        started = asyncio.get_running_loop().time()
        with pytest.raises(BoundAccountCooldownError) as caught:
            await pool.acquire_account("image", account_id=1)
        assert caught.value.remaining_seconds >= 1
        assert asyncio.get_running_loop().time() - started < .05
        assert pool._accounts["b"]["inflight"] == 0
    asyncio.run(scenario())


def test_actual_model_send_is_spaced_across_image_and_chat():
    async def scenario():
        pool = make_pool()
        pool.MIN_DISPATCH_INTERVAL_SECONDS = .03
        sent = []
        async def send():
            await pool.wait_for_model_send(pool._accounts["a"])
            sent.append(asyncio.get_running_loop().time())
        await asyncio.gather(*(send() for _ in range(4)))
        assert all(right - left >= .025 for left, right in zip(sent, sent[1:]))
    asyncio.run(scenario())


def test_cooldown_set_after_acquisition_blocks_actual_send():
    async def scenario():
        pool = make_pool()
        account = await pool.acquire_account("image", account_id=1)
        pool._accounts["a"]["cooldown_until"] = time.time() + .08
        with pytest.raises(BoundAccountCooldownError):
            await pool.wait_for_model_send(account)
    asyncio.run(scenario())


def test_scheduler_prefers_low_window_calls_then_inflight_then_failures():
    pool = make_pool()
    pool._model_call_counts = {1: 10, 2: 3}
    a, b = pool._accounts.values()
    a.update(failure_count=0, inflight=3)
    b.update(failure_count=1, inflight=0)
    assert pool._select_strategy([a, b]) is b
    b["failure_count"] = 0
    assert pool._select_strategy([a, b]) is b
    pool._model_call_counts[1] = 3
    assert pool._select_strategy([a, b]) is b
    a["inflight"] = 0
    a["last_dispatched_at"], b["last_dispatched_at"] = 1, 2
    assert pool._select_strategy([a, b]) is a


def test_gpt_and_gemini_use_two_point_five_seconds_at_both_gates():
    assert BaseAccountPool.MIN_DISPATCH_INTERVAL_SECONDS == 2.5
    pool = make_pool()
    pool.MIN_DISPATCH_INTERVAL_SECONDS = 2.5
    pool._accounts.pop("b")
    assert pool._reserve_available_account("chat") is not None
    assert pool._reserve_available_account("image") is None


def test_long_cooldown_after_reservation_returns_reason_without_sending():
    async def scenario():
        pool = make_pool()
        account = await pool.acquire_account("image", account_id=1)
        until = time.time() + 60.2
        pool._accounts["a"].update(cooldown_until=until, error_message="rate limit 429")
        with pytest.raises(BoundAccountCooldownError) as caught:
            await pool.wait_for_model_send(account)
        assert caught.value.cooldown_until == until
        assert caught.value.remaining_seconds == 61
        assert caught.value.error_message == "rate limit 429"
        assert not pool.__dict__.get("_model_call_counts")
    asyncio.run(scenario())


@pytest.mark.parametrize("platform", ["gpt", "gemini"])
def test_total_model_calls_survive_pool_restart_and_do_not_count_reservations(monkeypatch, platform):
    from core.slot_sessions import slot_sessions

    class RedisCounters:
        def __init__(self):
            self.hashes = {}

        async def hgetall(self, key):
            return {k.encode(): str(v).encode() for k, v in self.hashes.get(key, {}).items()}

        async def hincrby(self, key, field, increment):
            values = self.hashes.setdefault(key, {})
            values[field] = values.get(field, 0) + increment
            return values[field]

        async def expire(self, *_):
            return True

    async def scenario():
        redis = RedisCounters()
        monkeypatch.setattr(slot_sessions, "redis", redis)
        first = make_pool()
        first._platform = platform
        account = await first.acquire_account("chat", account_id=1)
        assert not redis.hashes
        await first.wait_for_model_send(account)
        assert not redis.hashes
        await first.record_model_call(account)
        await first.record_model_call(account)
        window_key = f"flexi:ai:account-model-calls:{platform}:{int(time.time() // 600)}"
        assert redis.hashes[window_key] == {"1": 2}
        restarted = make_pool()
        restarted._platform = platform
        assert (await restarted.acquire_account("image"))["account_id"] == 2
        assert restarted._model_call_counts[1] == 2
        # 旧会话仍定向账号1，不因调用量较高切换账号。
        assert (await restarted.acquire_account("image", account_id=1))["account_id"] == 1
    asyncio.run(scenario())
