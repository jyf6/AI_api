from core.account_pool import BaseAccountPool
import asyncio
import time
from threading import Condition, Lock

import pytest

from core.account_pool import AccountUnavailable
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


def test_preferred_cooldown_waits_and_recovers_without_switch():
    async def scenario():
        pool = make_pool()
        pool._accounts["a"]["cooldown_until"] = time.time() + .08
        started = asyncio.get_running_loop().time()
        result = await pool.acquire_account("image", account_id=1)
        assert result["account_id"] == 1
        assert asyncio.get_running_loop().time() - started >= .07
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


def test_cooldown_set_after_acquisition_still_blocks_actual_send():
    async def scenario():
        pool = make_pool()
        account = await pool.acquire_account("image", account_id=1)
        pool._accounts["a"]["cooldown_until"] = time.time() + .08
        started = asyncio.get_running_loop().time()
        await pool.wait_for_model_send(account)
        assert asyncio.get_running_loop().time() - started >= .07
    asyncio.run(scenario())


def test_scheduler_prefers_least_recently_dispatched_then_lower_inflight():
    pool = BaseAccountPool.__new__(BaseAccountPool)
    accounts = [
        {"name": "recent", "last_dispatched_at": 20.0, "inflight": 0},
        {"name": "older_busy", "last_dispatched_at": 10.0, "inflight": 2},
        {"name": "older_idle", "last_dispatched_at": 10.0, "inflight": 0},
    ]

    assert pool._select_strategy(accounts)["name"] == "older_idle"
