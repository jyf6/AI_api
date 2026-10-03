"""请求取消和代理节点更新的上线边界测试。"""

import asyncio
from threading import Condition, Event, Lock

import pytest

from core.account_pool import BaseAccountPool, await_thread_result
from core.database import Database


def make_pool() -> BaseAccountPool:
    """建立不连接数据库的单账号池，用于验证实际占用数。"""
    pool = BaseAccountPool.__new__(BaseAccountPool)
    pool._lock = Lock()
    pool._condition = Condition(pool._lock)
    pool._data_file = None
    pool._platform = ""
    pool._batches = {}
    pool.MAX_INFLIGHT_TOTAL = 1
    pool.MIN_DISPATCH_INTERVAL_SECONDS = 0
    pool._accounts = {"one": {"name": "one", "status": "active", "inflight": 0}}
    return pool


def test_cancel_waiting_request_does_not_take_next_slot():
    async def scenario() -> None:
        pool = make_pool()
        pool.get_available_account("image")
        waiting = Event()
        original = pool.wait_for_available_account

        def wait(task_type: str, cancelled: Event):
            waiting.set()
            return original(task_type, cancelled)

        pool.wait_for_available_account = wait
        request = asyncio.create_task(pool.acquire_account("image"))
        assert await asyncio.to_thread(waiting.wait, 1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        pool.release_account("one", True, task_type="image")
        await asyncio.sleep(0.02)
        assert pool.stats()["total_inflight_tasks"] == 0

    asyncio.run(scenario())


def test_cancel_after_reservation_returns_abandoned_slot():
    async def scenario() -> None:
        pool = make_pool()
        reserved = Event()
        finish = Event()

        def wait(_task_type: str, _cancelled: Event):
            account = pool.get_available_account("image")
            reserved.set()
            finish.wait(1)
            return account

        pool.wait_for_available_account = wait
        request = asyncio.create_task(pool.acquire_account("image"))
        assert await asyncio.to_thread(reserved.wait, 1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert pool.stats()["total_inflight_tasks"] == 1
        finish.set()
        for _ in range(50):
            if pool.stats()["total_inflight_tasks"] == 0:
                break
            await asyncio.sleep(0.01)
        assert pool.stats()["total_inflight_tasks"] == 0

    asyncio.run(scenario())


def test_cancel_does_not_release_running_thread_early():
    async def scenario() -> None:
        pool = make_pool()
        pool.get_available_account("image")
        started = Event()
        finish = Event()

        def run() -> None:
            started.set()
            finish.wait(1)
            pool.release_account("one", True, task_type="image")

        request = asyncio.create_task(await_thread_result(run))
        assert await asyncio.to_thread(started.wait, 1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert pool.stats()["total_inflight_tasks"] == 1
        finish.set()
        for _ in range(50):
            if pool.stats()["total_inflight_tasks"] == 0:
                break
            await asyncio.sleep(0.01)
        assert pool.stats()["total_inflight_tasks"] == 0

    asyncio.run(scenario())


def test_proxy_node_update_rolls_back_if_account_update_fails(monkeypatch):
    """第二条 SQL 失败时，节点地址也必须回滚。"""
    state = {"url": "http://old:8080"}

    class Connection:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def cursor(self):
            return self

        def begin(self):
            self.before = state["url"]

        def execute(self, sql, params):
            if sql.startswith("UPDATE ai_proxy_node"):
                state["url"] = params[1]
            elif sql.startswith("UPDATE ai_proxy_account"):
                raise RuntimeError("account update failed")

        def rollback(self):
            state["url"] = self.before

        def commit(self):
            raise AssertionError("失败后不应提交")

    db = Database()
    monkeypatch.setattr(db, "_connect", Connection)
    with pytest.raises(RuntimeError, match="account update failed"):
        db.update_proxy_node(1, "node", "http://new:8080", "active")
    assert state["url"] == "http://old:8080"


def test_restart_loads_bound_node_url_instead_of_stale_account_url(monkeypatch):
    """旧冗余字段即使未同步，也不能在重启时重新启用旧出口。"""
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def cursor(self):
            return self

        def execute(self, _sql, _params):
            pass

        def fetchall(self):
            return [{
                "account_name": "one", "credentials": "{}", "proxy": "http://old:8080",
                "proxy_id": 7, "bound_proxy": "http://new:8080", "proxy_status": "active",
                "status": "active", "cooldown_until": 0, "failure_count": 0,
                "error_message": "",
            }]

    db = Database()
    monkeypatch.setattr(db, "_connect", Connection)
    assert db.list_accounts("gpt")[0]["proxy"] == "http://new:8080"
