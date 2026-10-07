"""人工账号修改在慢数据库下不阻塞业务池，且不覆盖较新的凭证。"""
import asyncio
from threading import Event

import pytest

from core.blocking import run_blocking
from core.database import database


def make_pool(platform, monkeypatch):
    monkeypatch.setattr(database, "list_accounts", lambda *args: [])
    if platform == "gemini":
        from providers.gemini.account import GeminiAccountPool
        owner = GeminiAccountPool()
        account = {"name": "one", "cookie": "__Secure-1PSID=p; __Secure-1PSIDTS=old",
                   "psid": "p", "psidts": "old", "cookie_jar": []}
    else:
        from providers.doubao.account import DoubaoAccountPool
        owner = DoubaoAccountPool()
        account = {"name": "one", "cookies": {"sessionid": "old"}}
    account.update(account_id=11, credential_version=7, status="active", inflight=2, proxy="", proxy_id=None)
    owner._accounts = {"one": account}
    return owner


@pytest.mark.parametrize("platform", ["gemini", "doubao"])
def test_manual_cookie_slow_database_does_not_hold_pool_lock(platform, monkeypatch):
    owner = make_pool(platform, monkeypatch)
    entered, finish = Event(), Event()
    def persist(*args):
        assert not owner._lock.locked()
        assert args[-2:] == (7, 11)
        entered.set()
        assert finish.wait(2)
        return True
    monkeypatch.setattr(database, "update_account_cookie", persist)
    cookie = "__Secure-1PSID=p; __Secure-1PSIDTS=new" if platform == "gemini" else "sessionid=new"
    async def scenario():
        task = asyncio.create_task(run_blocking(owner.update_cookie, "one", cookie))
        try:
            deadline = asyncio.get_running_loop().time() + 1
            while not entered.is_set():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(.001)
            with owner._condition:
                assert owner._accounts["one"]["credential_version"] == 7
            assert not task.done()
        finally:
            finish.set()
        assert await task
        assert owner._accounts["one"]["credential_version"] == 8
        assert owner._accounts["one"]["inflight"] == 2
    asyncio.run(scenario())
    owner._health_executor.shutdown(wait=True)


@pytest.mark.parametrize("platform", ["gemini", "doubao"])
def test_manual_cookie_cas_conflict_preserves_current_credentials(platform, monkeypatch):
    owner = make_pool(platform, monkeypatch)
    monkeypatch.setattr(database, "update_account_cookie", lambda *args: False)
    cookie = "__Secure-1PSID=p; __Secure-1PSIDTS=new" if platform == "gemini" else "sessionid=new"
    with pytest.raises(RuntimeError, match="并发更新"):
        owner.update_cookie("one", cookie)
    assert owner._accounts["one"]["credential_version"] == 7
    if platform == "gemini":
        assert owner._accounts["one"]["psidts"] == "old"
    else:
        assert owner._accounts["one"]["cookies"]["sessionid"] == "old"
    owner._health_executor.shutdown(wait=True)


@pytest.mark.parametrize("platform", ["gemini", "doubao"])
def test_registration_database_io_is_outside_pool_lock_and_keeps_inflight(platform, monkeypatch):
    owner = make_pool(platform, monkeypatch)
    def persist(*args):
        assert not owner._lock.locked()
        return {"account_id": 11, "credential_version": 8}
    monkeypatch.setattr(database, "import_account", persist)
    cookie = "__Secure-1PSID=p; __Secure-1PSIDTS=new" if platform == "gemini" else "sessionid=new"
    stored = owner.add_account("one", cookie)
    assert stored["inflight"] == 2
    assert stored["credential_version"] == 8
    owner._health_executor.shutdown(wait=True)


def test_proxy_binding_queries_database_without_pool_lock(monkeypatch):
    owner = make_pool("gemini", monkeypatch)
    def bind(*args):
        assert not owner._lock.locked()
        return dict(owner._accounts["one"], proxy_id=3, proxy="bound", credential_version=8)
    monkeypatch.setattr(database, "bind_account_proxy", bind)
    assert owner.set_account_proxy("one", 3)
    assert owner._accounts["one"]["proxy_id"] == 3
    assert owner._accounts["one"]["inflight"] == 2
    owner._health_executor.shutdown(wait=True)


@pytest.mark.parametrize("change", ["new-version", "new-identity"])
def test_stale_proxy_query_cannot_roll_back_current_credentials(change, monkeypatch):
    owner = make_pool("gemini", monkeypatch)
    old = dict(owner._accounts["one"], proxy_id=3)
    def query(*args):
        assert not owner._lock.locked()
        current = owner._accounts["one"]
        current.update(proxy_id=3, credential_version=8, psidts="new-credential")
        if change == "new-identity":
            current["account_id"] = 12
        return [old]
    monkeypatch.setattr(database, "list_accounts", query)
    owner.refresh_proxy_node(3, "bound", "active")
    assert owner._accounts["one"]["credential_version"] == 8
    assert owner._accounts["one"]["psidts"] == "new-credential"
    owner._health_executor.shutdown(wait=True)


@pytest.mark.parametrize("platform", ["gemini", "doubao"])
@pytest.mark.parametrize("change", ["new-cookie", "recreated"])
def test_old_model_failure_cannot_disable_updated_cookie_account(platform, change, monkeypatch):
    owner = make_pool(platform, monkeypatch)
    old = dict(owner._accounts["one"])
    current = owner._accounts["one"]
    current.update(credential_version=8, inflight_image=1, inflight_chat=0, inflight=1)
    if change == "recreated":
        current["account_id"] = 12
    owner.release_account("one", False, "unauthorized", status_code=401,
                          task_type="image", acquired_account=old)
    assert current["status"] == "active"
    assert current["inflight"] == (1 if change == "recreated" else 0)
    owner.set_account_health("one", False, "old-probe-401", expected_account=old)
    assert current["status"] == "active"
    owner._health_executor.shutdown(wait=True)


def test_old_gemini_probe_does_not_close_new_client(monkeypatch):
    owner = make_pool("gemini", monkeypatch)
    old = dict(owner._accounts["one"])
    owner._accounts["one"]["credential_version"] = 8
    client = object()
    owner._clients["one"] = client
    asyncio.run(owner.discard_client("one", expected_account=old))
    assert owner._clients["one"] is client
    assert owner._request_generations.get("one", 0) == 0
    owner._health_executor.shutdown(wait=True)


def test_old_gemini_refresh_verification_failure_cannot_disable_new_cookie(monkeypatch):
    owner = make_pool("gemini", monkeypatch)
    async def scenario():
        entered, finish = asyncio.Event(), asyncio.Event()
        class Client:
            async def _fetch_user_status(self):
                entered.set()
                await finish.wait()
                raise RuntimeError("Cookie unauthorized")
        client = Client()
        owner._clients["one"] = client
        task = asyncio.create_task(owner.verify_refreshed_client("one", client))
        await entered.wait()
        owner._accounts["one"].update(credential_version=8, psidts="new-cookie")
        finish.set()
        await task
        assert owner._accounts["one"]["status"] == "active"
    asyncio.run(scenario())
    owner._health_executor.shutdown(wait=True)
