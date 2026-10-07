"""显式开启的隔离 MySQL 凭证轮换验收；OAuth 为替身，不使用真实账号。"""
import asyncio
import os
import time
import uuid
from unittest.mock import AsyncMock

import pytest


@pytest.mark.skipif(not os.getenv("MODEL_CREDENTIAL_TEST_DATABASE"), reason="需要显式指定隔离测试库")
def test_rotation_survives_reload_while_model_metadata_is_saved(monkeypatch):
    assert os.environ["MYSQL_PORT"] == "33306"
    assert os.environ["MYSQL_DATABASE"] == os.environ["MODEL_CREDENTIAL_TEST_DATABASE"]
    assert os.environ["MYSQL_DATABASE"].startswith("flexi_model_")
    from core.database import database
    from providers.openai.account import OpenAIAccountPool
    email = "credential-test-" + uuid.uuid4().hex
    database.import_account("gpt", email, {
        "access_token": "old", "refresh_token": "old-refresh", "device_id": "test-device",
        "access_token_expires_at": int(time.time()) + 10000,
    })
    owner = OpenAIAccountPool()
    reloaded = None
    async def scenario():
        started, finish = asyncio.Event(), asyncio.Event()
        async def oauth(*args):
            started.set()
            await finish.wait()
            return {"access_token": "fresh", "refresh_token": "rotated-refresh", "expires_in": 10000}
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", oauth)
        owner._discover_models = AsyncMock(return_value=[{"value": "test-model"}])
        version = owner._accounts[email]["credential_version"]
        task = asyncio.create_task(owner.credentials.refresh(email))
        await started.wait()
        try:
            await owner.refresh_supported_models(email)
            assert database.list_accounts("gpt", email)[0]["credential_version"] == version
        finally:
            finish.set()
        await task
    try:
        asyncio.run(scenario())
        # 新建池只读取持久层，等价于重启后重新加载凭证。
        reloaded = OpenAIAccountPool()
        account = reloaded._accounts[email]
        assert account["access_token"] == "fresh"
        assert account["refresh_token"] == "rotated-refresh"
        assert account["supported_models"] == [{"value": "test-model"}]
        assert reloaded.credentials.ready(account)
    finally:
        stored = database.list_accounts("gpt", email)[0]
        database.delete_account("gpt", email, stored["account_id"], stored["credential_version"])
        owner._health_executor.shutdown(wait=True)
        if reloaded is not None:
            reloaded._health_executor.shutdown(wait=True)


@pytest.mark.skipif(not os.getenv("MODEL_CREDENTIAL_TEST_DATABASE"), reason="需要显式指定隔离测试库")
@pytest.mark.parametrize("platform", ["gemini", "doubao"])
def test_manual_cookie_update_and_proxy_binding_are_durable(platform):
    assert os.environ["MYSQL_PORT"] == "33306"
    assert os.environ["MYSQL_DATABASE"] == os.environ["MODEL_CREDENTIAL_TEST_DATABASE"]
    assert os.environ["MYSQL_DATABASE"].startswith("flexi_model_")
    from core.database import database
    if platform == "gemini":
        from providers.gemini.account import GeminiAccountPool
        owner = GeminiAccountPool()
        original = "__Secure-1PSID=test-psid; __Secure-1PSIDTS=old"
        updated = "__Secure-1PSID=test-psid; __Secure-1PSIDTS=new"
    else:
        from providers.doubao.account import DoubaoAccountPool
        owner = DoubaoAccountPool()
        original, updated = "sessionid=old", "sessionid=new"
    name = "manual-credential-test-" + uuid.uuid4().hex
    proxy_id = None
    try:
        account = owner.add_account(name, original)
        old_version = account["credential_version"]
        owner._accounts[name]["inflight"] = 2
        assert owner.update_cookie(name, updated)
        assert owner._accounts[name]["inflight"] == 2
        stored = database.list_accounts(platform, name)[0]
        assert stored["credential_version"] == old_version + 1
        assert stored["status"] == "active"
        if platform == "gemini":
            assert stored["psidts"] == "new"
        else:
            assert stored["cookies"]["sessionid"] == "new"
        assert not database.update_account_cookie(platform, name, {"test_stale": True}, old_version, stored["account_id"])
        proxy_id = database.create_proxy_node(name, "http://127.0.0.1:19099/" + name)
        assert owner.set_account_proxy(name, proxy_id)
        rebound = database.list_accounts(platform, name)[0]
        assert rebound["proxy_id"] == proxy_id
        assert owner._accounts[name]["credential_version"] == rebound["credential_version"]
        assert owner._accounts[name]["inflight"] == 2
    finally:
        rows = database.list_accounts(platform, name)
        if rows:
            database.delete_account(platform, name, rows[0]["account_id"], rows[0]["credential_version"])
        if proxy_id is not None:
            database.delete_proxy_node(proxy_id)
        owner._health_executor.shutdown(wait=True)
