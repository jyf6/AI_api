"""GPT 管理探测的凭证和独立并发预算回归。"""
import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
import time
import pytest

from core.database import database


def test_model_discovery_uses_fresh_token_and_bounded_management_budget(monkeypatch):
    # 导入账号模块时禁止访问真实账号库。
    monkeypatch.setattr(database, "list_accounts", lambda *args: [])
    account_module = importlib.import_module("providers.openai.account")
    backend_module = importlib.import_module("providers.openai.backend")

    async def scenario():
        active = peak = 0
        class Session:
            def __init__(self, **kwargs):
                self.headers = {}
            async def close(self):
                pass
            async def get(self, url, **kwargs):
                nonlocal active, peak
                assert kwargs["headers"]["Authorization"] == "Bearer fresh"
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(.01)
                active -= 1
                return SimpleNamespace(status_code=200, json=lambda: {"models": [{"slug": "model"}]})
        monkeypatch.setattr(backend_module.requests, "AsyncSession", Session)
        owner = account_module.OpenAIAccountPool.__new__(account_module.OpenAIAccountPool)
        owner.management_slots = asyncio.Semaphore(5)
        owner.prepare_request_account = AsyncMock(return_value={"access_token": "fresh"})
        results = await asyncio.gather(*(owner._discover_models({"access_token": "expired"}) for _ in range(20)))
        assert all(result[0]["value"] == "model" for result in results)
        assert peak == 5
        assert owner.prepare_request_account.await_count == 20
    asyncio.run(scenario())


def test_old_management_probe_does_not_change_new_login_health():
    from tests.test_cancellation_and_proxy_transaction import make_pool
    owner = make_pool()
    current = owner._accounts["one"]
    current.update(account_id=12, credential_version=8, status="error")
    owner.set_account_health("one", True, expected_account={"account_id": 11, "credential_version": 8})
    assert current["status"] == "error"
    owner.set_account_health("one", True, expected_account={"account_id": 12, "credential_version": 7})
    assert current["status"] == "error"


def test_model_metadata_refresh_does_not_discard_inflight_oauth_rotation(monkeypatch):
    monkeypatch.setattr(database, "list_accounts", lambda *args: [])
    from providers.openai.account import OpenAIAccountPool
    from providers.openai.credentials import GPTCredentials
    from tests.test_gpt_credentials import pool
    async def scenario():
        owner = OpenAIAccountPool.__new__(OpenAIAccountPool)
        owner.__dict__.update(pool().__dict__)
        owner.credentials = GPTCredentials(owner)
        owner._discover_models = AsyncMock(return_value=[{"value": "model"}])
        started, finish = asyncio.Event(), asyncio.Event()
        async def oauth(*args):
            started.set()
            await finish.wait()
            return {"access_token": "new", "refresh_token": "rotated"}
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", oauth)
        monkeypatch.setattr(database, "update_credentials", lambda *args: True)
        monkeypatch.setattr(database, "update_supported_models", lambda *args: True)
        refresh = asyncio.create_task(owner.credentials.refresh("one"))
        await started.wait()
        await owner.refresh_supported_models("one")
        assert owner._accounts["one"]["credential_version"] == 7
        finish.set()
        await refresh
        assert owner._accounts["one"]["refresh_token"] == "rotated"
        assert owner._accounts["one"]["credential_version"] == 8
    asyncio.run(scenario())


def test_unready_credentials_are_not_reported_as_busy_accounts(monkeypatch):
    monkeypatch.setattr(database, "list_accounts", lambda *args: [])
    from providers.openai.account import OpenAIAccountPool
    from providers.openai.credentials import GPTCredentials, CredentialUnavailable
    from tests.test_gpt_credentials import pool
    async def scenario():
        owner = OpenAIAccountPool.__new__(OpenAIAccountPool)
        owner.__dict__.update(pool().__dict__)
        owner._batches = {}
        owner.credentials = GPTCredentials(owner)
        owner._accounts["one"].update(access_token_expires_at=time.time() - 1)
        with pytest.raises(CredentialUnavailable):
            await owner.acquire_account(deadline=asyncio.get_running_loop().time() + .01)
        assert owner._accounts["one"].get("inflight", 0) == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["new-version", "new-identity"])
def test_late_login_query_cannot_overwrite_newer_account_state(change, monkeypatch):
    monkeypatch.setattr(database, "list_accounts", lambda *args: [])
    from providers.openai.account import OpenAIAccountPool
    from providers.openai.oauth import oauth_manager
    from tests.test_gpt_credentials import pool
    owner = OpenAIAccountPool.__new__(OpenAIAccountPool)
    owner.__dict__.update(pool().__dict__)
    owner._decode_jwt = lambda token: {"email": "one"}
    monkeypatch.setattr(oauth_manager, "finish_session", lambda *args: {
        "access_token": "login-token", "refresh_token": "login-refresh", "id_token": "id",
    })
    monkeypatch.setattr(database, "import_account", lambda *args, **kwargs: {"account_id": 11, "credential_version": 8})
    def query(*args):
        current = owner._accounts["one"]
        current.update(access_token="newer-token", credential_version=9)
        if change == "new-identity":
            current.update(account_id=12, credential_version=0)
        return [{"email": "one", "account_id": 11, "credential_version": 8, "access_token": "old-query"}]
    monkeypatch.setattr(database, "list_accounts", query)
    result = owner.finish_oauth_session("test-callback")
    assert result["access_token"] == "newer-token"
    assert owner._accounts["one"]["access_token"] == "newer-token"


@pytest.mark.parametrize("existing", [True, False])
def test_login_publication_and_account_deletion_are_serialized(existing, monkeypatch):
    """登录查库后的删除等待发布完成，删除后不能残留可调度的内存身份。"""
    from threading import Event
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(database, "list_accounts", lambda *args: [])
    from providers.openai.account import OpenAIAccountPool
    from providers.openai.oauth import oauth_manager
    from tests.test_gpt_credentials import pool
    owner = OpenAIAccountPool.__new__(OpenAIAccountPool)
    owner.__dict__.update(pool().__dict__)
    owner._platform, owner._batches = "gpt", {}
    if not existing:
        owner._accounts.clear()
    owner._decode_jwt = lambda token: {"email": "one"}
    monkeypatch.setattr(oauth_manager, "finish_session", lambda *args: {
        "access_token": "login-token", "refresh_token": "login-refresh", "id_token": "id",
    })
    monkeypatch.setattr(database, "import_account", lambda *args, **kwargs: {
        "account_id": 11, "credential_version": 8,
    })
    queried, publish, delete_started, deleted = Event(), Event(), Event(), Event()
    def query(*args):
        assert not owner._lock.locked()
        queried.set()
        assert publish.wait(2)
        return [{"email": "one", "account_id": 11, "credential_version": 8}]
    def delete(*args):
        assert args[-2:] == (11, 8)
        assert not owner._lock.locked()
        deleted.set()
        return True
    def remove():
        delete_started.set()
        return owner.delete_account("one")
    monkeypatch.setattr(database, "list_accounts", query)
    monkeypatch.setattr(database, "delete_account", delete)
    with ThreadPoolExecutor(max_workers=2) as executor:
        login = executor.submit(owner.finish_oauth_session, "test-callback")
        try:
            assert queried.wait(2)
            removal = executor.submit(remove)
            assert delete_started.wait(2)
            assert not deleted.wait(.05)
        finally:
            publish.set()
        assert login.result(timeout=2)["account_id"] == 11
        assert removal.result(timeout=2)
    assert deleted.is_set()
    assert "one" not in owner._accounts
