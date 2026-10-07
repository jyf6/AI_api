"""请求取消和代理节点更新的上线边界测试。"""

import asyncio
from threading import Condition, Event, Lock

import pytest

from core.account_pool import BaseAccountPool, await_thread_result
from core.database import Database
from core.admission import CapacityUnavailable


def make_pool() -> BaseAccountPool:
    """建立不连接数据库的单账号池，用于验证实际占用数。"""
    pool = BaseAccountPool.__new__(BaseAccountPool)
    pool._lock = Lock()
    pool._management_lock = Lock()
    pool._condition = Condition(pool._lock)
    pool._data_file = None
    pool._platform = ""
    pool._batches = {}
    pool.MAX_INFLIGHT_TOTAL = 1
    pool.MIN_DISPATCH_INTERVAL_SECONDS = 0
    pool._accounts = {"one": {"name": "one", "status": "active", "inflight": 0}}
    return pool


def test_account_delete_database_io_is_outside_lock_and_preserves_recreated_account(monkeypatch):
    owner = make_pool()
    owner._platform = "gpt"
    owner._accounts["one"].update(account_id=11, credential_version=7)
    def delete(platform, key, account_id, version):
        assert not owner._lock.locked()
        assert (account_id, version) == (11, 7)
        with owner._condition:
            owner._accounts[key] = {"name": key, "account_id": 12, "credential_version": 0, "status": "active"}
        return True
    monkeypatch.setattr("core.account_pool.database.delete_account", delete)
    assert owner.delete_account("one")
    assert owner._accounts["one"]["account_id"] == 12


def test_cancel_waiting_request_does_not_take_next_slot():
    async def scenario() -> None:
        pool = make_pool()
        pool.get_available_account("image")
        waiting = asyncio.Event()
        original = pool._reserve_available_account

        def reserve(task_type: str):
            waiting.set()
            return original(task_type)

        pool._reserve_available_account = reserve
        request = asyncio.create_task(pool.acquire_account("image"))
        await asyncio.wait_for(waiting.wait(), 1)
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
        reserved = asyncio.Event()
        finish = asyncio.Event()

        async def prepare(account):
            reserved.set()
            await finish.wait()
            return account

        pool._prepare_account_async = prepare
        request = asyncio.create_task(pool.acquire_account("image"))
        await asyncio.wait_for(reserved.wait(), 1)
        assert pool.stats()["total_inflight_tasks"] == 1
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        finish.set()
        for _ in range(50):
            if pool.stats()["total_inflight_tasks"] == 0:
                break
            await asyncio.sleep(0.01)
        assert pool.stats()["total_inflight_tasks"] == 0

    asyncio.run(scenario())


def test_100_account_waiters_do_not_use_executor_and_expire(monkeypatch):
    async def scenario():
        pool = make_pool()
        pool.get_available_account("image")
        loop = asyncio.get_running_loop()

        def unexpected_executor(*_args, **_kwargs):
            raise AssertionError("Account waiting must not consume an executor thread")

        monkeypatch.setattr(loop, "run_in_executor", unexpected_executor)
        deadline = loop.time() + .03
        results = await asyncio.gather(
            *(pool.acquire_account("image", deadline=deadline) for _ in range(100)),
            return_exceptions=True,
        )
        assert all(isinstance(result, CapacityUnavailable) for result in results)
        assert pool.stats()["total_inflight_tasks"] == 1
        pool.release_account("one", True, task_type="image")
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
                "account_id": 11, "account_name": "one", "credentials": "{}", "proxy": "http://old:8080",
                "proxy_id": 7, "bound_proxy": "http://new:8080", "proxy_status": "active",
                "status": "active", "cooldown_until": 0, "failure_count": 0,
                "error_message": "",
            }]

    db = Database()
    monkeypatch.setattr(db, "_connect", Connection)
    assert db.list_accounts("gpt")[0]["proxy"] == "http://new:8080"


def test_old_gpt_result_cannot_change_recreated_account_identity():
    """同名重建后，旧 401 不修改新账号健康或计数。"""
    owner = make_pool()
    owner._platform = "gpt"
    current = {"email":"one", "account_id":22, "credential_version":7,
               "status":"active", "inflight":1, "inflight_image":1, "inflight_chat":0}
    owner._accounts["one"] = current
    old = {**current, "account_id":11}
    owner.release_account("one",False,"unauthorized",status_code=401,task_type="image",acquired_account=old)
    assert current["status"] == "active"
    assert current["inflight"] == 1
    assert not owner._batches


def test_old_gpt_auth_failure_releases_own_slot_without_poisoning_new_credentials():
    """同 ID 新登录保留在途数；旧版本结果仅归还自己的占用。"""
    owner = make_pool()
    owner._platform = "gpt"
    owner._save = lambda *_args: None
    owner._classify_error = lambda *_args: "fatal"
    current = {"email":"one", "account_id":11, "credential_version":8,
               "status":"active", "inflight":2, "inflight_image":2, "inflight_chat":0}
    owner._accounts["one"] = current
    fresh = dict(current)
    old = {**current,"credential_version":7}
    owner.release_account("one",True,task_type="image",acquired_account=fresh)
    owner.release_account("one",False,"unauthorized",status_code=401,task_type="image",acquired_account=old)
    assert current["inflight"] == 0
    assert current["status"] == "active"
    assert current["failure_count"] == 0
    assert not owner._batches


def test_old_gpt_success_does_not_reactivate_new_invalid_credentials():
    owner = make_pool()
    owner._platform = "gpt"
    current = {"email":"one", "account_id":11, "credential_version":8,
               "status":"error", "inflight":1, "inflight_image":1, "inflight_chat":0}
    owner._accounts["one"] = current
    owner._batches["one"] = {"credential_version":7,"success":True}
    old = {**current,"credential_version":7}
    owner.release_account("one",True,task_type="image",acquired_account=old)
    assert current["inflight"] == 0
    assert current["status"] == "error"
    assert not owner._batches


def test_slow_health_database_does_not_block_account_dispatch(monkeypatch):
    """冷却恢复写库尚未完成时，事件循环仍能交付账号。"""
    async def scenario():
        import time
        from concurrent.futures import ThreadPoolExecutor
        pool = make_pool()
        pool._platform = "gemini"
        pool._health_executor = ThreadPoolExecutor(max_workers=1)
        pool._accounts["one"].update(account_id=11,credential_version=3,
                                     status="cooldown",cooldown_until=int(time.time())-1)
        entered,release = Event(),Event()
        writes=[]
        def write(platform,key,state):
            writes.append(dict(state))
            entered.set()
            release.wait(2)
        def old_import(*args):
            write("gemini","one",{})
            return {"account_id":11,"credential_version":4}
        monkeypatch.setattr("core.account_pool.database.save_account_health",write)
        monkeypatch.setattr("core.account_pool.database.import_account",old_import)
        try:
            before=time.monotonic()
            account=await pool.acquire_account("image")
            assert time.monotonic()-before < .2
            assert account["inflight"] == 1
            assert await asyncio.to_thread(entered.wait,1)
            assert not pool._lock.locked()
            assert not release.is_set()
            assert writes[0]["credential_version"] == 3
            assert "cookie" not in writes[0]
            assert "access_token" not in writes[0]
        finally:
            release.set()
            pool._health_executor.shutdown(wait=True)
    asyncio.run(scenario())
