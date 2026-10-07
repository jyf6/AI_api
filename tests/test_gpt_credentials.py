import asyncio
import time
from threading import Condition, Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from providers.openai.credentials import CredentialUnavailable, GPTCredentials


def pool():
    lock = Lock()
    return SimpleNamespace(
        _lock=lock, _condition=Condition(lock), _management_lock=Lock(),
        _accounts={"one": {"account_id": 11, "email": "one", "access_token": "old-access", "refresh_token": "old-refresh",
            "access_token_expires_at": time.time() + 1000, "credential_version": 7,
            "status": "active", "proxy": "http://bound-proxy"}},
        _decode_jwt=lambda token: {},
        _classify_error=lambda message, code: "fatal" if "invalid_grant" in message else "transient",
    )


def test_known_invalid_credentials_cannot_be_used_for_management_probe(monkeypatch):
    async def scenario():
        owner = pool()
        owner._accounts["one"].update(status="error", error_message="credential_auth_invalid",
                                      access_token_expires_at=time.time() + 10000)
        service = GPTCredentials(owner)
        oauth = AsyncMock(return_value={"access_token": "new", "refresh_token": "rotated"})
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", oauth)
        assert not service.ready(owner._accounts["one"])
        with pytest.raises(CredentialUnavailable, match="requires login"):
            await service.prepare(dict(owner._accounts["one"]))
        oauth.assert_not_awaited()
        # 只有成功取得并保存新凭证，才允许重新进行健康验证。
        monkeypatch.setattr("providers.openai.credentials.database.update_credentials", lambda *args: True)
        refreshed = await service.refresh("one", force=True)
        assert refreshed["error_message"] == ""
        assert service.ready(refreshed)
    asyncio.run(scenario())


def test_same_account_refresh_is_single_flight_and_publishes_after_persist(monkeypatch):
    async def scenario():
        owner = pool()
        service = GPTCredentials(owner)
        oauth = AsyncMock(return_value={"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", oauth)
        records = []
        def save(platform, email, patch, version, account_id):
            assert not owner._lock.locked()
            assert owner._accounts["one"]["access_token"] == "old-access"
            assert not service.ready(owner._accounts["one"])
            records.append((platform, email, dict(patch), version))
            return True
        monkeypatch.setattr("providers.openai.credentials.database.update_credentials", save)
        results = await asyncio.gather(*(service.refresh("one") for _ in range(20)))
        oauth.assert_awaited_once_with("old-refresh", "http://bound-proxy")
        assert len(records) == 1
        assert records[0][2]["refresh_token"] == "new-refresh"
        assert all(result["credential_version"] == 8 for result in results)
        assert owner._accounts["one"]["access_token"] == "new-access"
        assert service.ready(owner._accounts["one"])
    asyncio.run(scenario())


def test_database_failure_retries_same_rotated_credentials_without_oauth(monkeypatch):
    async def scenario():
        owner = pool()
        service = GPTCredentials(owner)
        oauth = AsyncMock(return_value={"access_token": "new-access", "refresh_token": "rotated"})
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", oauth)
        writes = []
        def save(platform, email, patch, version, account_id):
            writes.append(dict(patch))
            if len(writes) == 1:
                raise OSError("database offline")
            return True
        monkeypatch.setattr("providers.openai.credentials.database.update_credentials", save)
        with pytest.raises(CredentialUnavailable, match="pending"):
            await service.refresh("one")
        assert not service.ready(owner._accounts["one"])
        assert service._pending["one"][1]["refresh_token"] == "rotated"
        result = await service.refresh("one", force=True)
        assert result["refresh_token"] == "rotated"
        assert writes[0] == writes[1]
        oauth.assert_awaited_once()
        assert not service._pending
    asyncio.run(scenario())


def test_rotated_result_cannot_overwrite_new_login(monkeypatch):
    async def scenario():
        owner = pool()
        service = GPTCredentials(owner)
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", AsyncMock(return_value={"access_token": "stale-result"}))
        monkeypatch.setattr("providers.openai.credentials.database.update_credentials", lambda *args: False)
        replacement = {**owner._accounts["one"], "access_token": "relogin-access", "refresh_token": "relogin-refresh", "credential_version": 9}
        monkeypatch.setattr("providers.openai.credentials.database.list_accounts", lambda *args: [replacement])
        result = await service.refresh("one")
        assert result["access_token"] == "relogin-access"
        assert result["refresh_token"] == "relogin-refresh"
        assert result["credential_version"] == 9
    asyncio.run(scenario())


def test_cancelled_caller_does_not_abandon_rotated_token_persistence(monkeypatch):
    async def scenario():
        owner = pool()
        service = GPTCredentials(owner)
        started, complete = asyncio.Event(), asyncio.Event()
        async def refresh(*args):
            started.set()
            await complete.wait()
            return {"access_token": "new-access"}
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", refresh)
        monkeypatch.setattr("providers.openai.credentials.database.update_credentials", lambda *args: True)
        caller = asyncio.create_task(service.refresh("one"))
        await started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        complete.set()
        await service.drain()
        assert owner._accounts["one"]["access_token"] == "new-access"
        assert owner._accounts["one"]["refresh_token"] == "old-refresh"
    asyncio.run(scenario())


def test_transient_failure_uses_backoff_and_fatal_failure_disables_account(monkeypatch):
    async def scenario():
        owner = pool()
        service = GPTCredentials(owner)
        oauth = AsyncMock(side_effect=OSError("network"))
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", oauth)
        monkeypatch.setattr("providers.openai.credentials.database.save_account_health", lambda *args: None)
        with pytest.raises(CredentialUnavailable):
            await service.refresh("one")
        with pytest.raises(CredentialUnavailable, match="backoff"):
            await service.refresh("one")
        oauth.assert_awaited_once()
        assert owner._accounts["one"]["status"] == "active"
        oauth.side_effect = RuntimeError("invalid_grant")
        with pytest.raises(CredentialUnavailable, match="login"):
            await service.refresh("one", force=True)
        assert owner._accounts["one"]["status"] == "error"
    asyncio.run(scenario())


def test_different_accounts_refresh_with_limit_five(monkeypatch):
    async def scenario():
        owner = pool()
        template = owner._accounts["one"]
        owner._accounts = {str(index): {**template, "email": str(index)} for index in range(20)}
        service = GPTCredentials(owner, concurrency=5)
        running = peak = 0
        first_batch, complete = asyncio.Event(), asyncio.Event()
        async def refresh(*args):
            nonlocal running, peak
            assert not owner._lock.locked()
            running += 1
            peak = max(peak, running)
            if running == 5:
                first_batch.set()
            await complete.wait()
            running -= 1
            return {"access_token": "updated"}
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", refresh)
        monkeypatch.setattr("providers.openai.credentials.database.update_credentials", lambda *args: True)
        tasks = [asyncio.create_task(service.refresh(str(index))) for index in range(20)]
        await asyncio.wait_for(first_batch.wait(), 1)
        assert peak == 5
        complete.set()
        await asyncio.gather(*tasks)
        assert peak == 5
    asyncio.run(scenario())


def test_deleted_and_recreated_account_with_same_version_rejects_old_refresh(monkeypatch):
    """同名新账号的版本相同，仍须按数据库 ID 拒绝旧刷新结果。"""
    async def scenario():
        owner = pool()
        service = GPTCredentials(owner)
        replacement = {**owner._accounts["one"], "account_id": 22,
                       "access_token": "new-login", "refresh_token": "new-login-refresh"}
        async def oauth(*args):
            owner._accounts["one"] = dict(replacement)
            return {"access_token": "obsolete", "refresh_token": "obsolete-refresh"}
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", oauth)
        writes = []
        def save(platform, email, patch, version, account_id):
            writes.append((version, account_id))
            assert owner._accounts["one"]["account_id"] == 22
            assert not owner._accounts["one"].get("credential_pending")
            return False
        monkeypatch.setattr("providers.openai.credentials.database.update_credentials", save)
        monkeypatch.setattr("providers.openai.credentials.database.list_accounts", lambda *args: [replacement])
        result = await service.refresh("one")
        assert writes == [(7, 11)]
        assert result["account_id"] == 22
        assert result["access_token"] == "new-login"
        assert not service._pending
    asyncio.run(scenario())


def test_old_refresh_failure_does_not_disable_recreated_account(monkeypatch):
    """旧身份认证失败不能暂停同名的新登录身份。"""
    async def scenario():
        owner = pool()
        service = GPTCredentials(owner)
        async def oauth(*args):
            owner._accounts["one"] = {**owner._accounts["one"], "account_id": 22}
            raise RuntimeError("invalid_grant")
        monkeypatch.setattr("providers.openai.credentials.refresh_access_token", oauth)
        def forbidden_save(*args):
            raise AssertionError("stale identity health must not be written")
        monkeypatch.setattr("providers.openai.credentials.database.save_account_health", forbidden_save)
        with pytest.raises(CredentialUnavailable):
            await service.refresh("one")
        assert owner._accounts["one"]["status"] == "active"
        assert "error_message" not in owner._accounts["one"]
    asyncio.run(scenario())
