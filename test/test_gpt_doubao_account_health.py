import asyncio

import pytest

from api.routers import accounts as account_routes
from providers.doubao.account import DoubaoAccountPool
from providers.openai.account import OpenAIAccountPool


def test_gpt_transient_manual_token_refresh_does_not_permanently_disable_account(monkeypatch, tmp_path):
    pool = OpenAIAccountPool()
    pool._platform = ""
    pool._data_file = tmp_path / "accounts.json"
    pool._accounts = {
        "user@example.com": {
            "email": "user@example.com",
            "refresh_token": "refresh-token",
            "access_token": "still-valid",
            "access_token_expires_at": 9999999999,
            "status": "active",
            "cooldown_until": 0,
            "failure_count": 0,
            "error_message": "",
        }
    }
    monkeypatch.setattr(
        "providers.openai.account.refresh_access_token",
        lambda *_args: (_ for _ in ()).throw(ConnectionError("OAuth temporarily unreachable")),
    )

    with pytest.raises(ConnectionError):
        pool.refresh_account("user@example.com")

    account = pool._accounts["user@example.com"]
    assert account["status"] == "active"
    assert account["access_token"] == "still-valid"


def test_gpt_revoked_refresh_token_is_still_marked_invalid(monkeypatch, tmp_path):
    pool = OpenAIAccountPool()
    pool._platform = ""
    pool._data_file = tmp_path / "accounts.json"
    pool._accounts = {"user@example.com": {
        "email": "user@example.com", "refresh_token": "revoked", "status": "active",
        "cooldown_until": 0, "failure_count": 0, "error_message": "",
    }}
    monkeypatch.setattr(
        "providers.openai.account.refresh_access_token",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("invalid_grant: refresh token revoked")),
    )

    with pytest.raises(RuntimeError):
        pool.refresh_account("user@example.com")

    assert pool._accounts["user@example.com"]["status"] == "error"


@pytest.mark.parametrize("message", [
    "Doubao chat failed (401): session expired",
    "\u8c46\u5305\u767b\u5f55\u5df2\u5931\u6548\uff0c\u8bf7\u91cd\u65b0\u767b\u5f55",
    "Cookie \u65e0\u6548\uff0c\u8bf7\u91cd\u65b0\u5f55\u5165",
])
def test_doubao_recognizes_chinese_and_http_auth_failures(message):
    assert DoubaoAccountPool()._classify_error(message) == "fatal"


def test_gpt_account_test_checks_models_endpoint_without_sending_chat(monkeypatch):
    calls = []

    class FakeBackend:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def chat_text(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return "OK."

    monkeypatch.setattr(account_routes, "OpenAIBackendAPI", FakeBackend)
    monkeypatch.setattr(account_routes.account_service, "_accounts", {"user@example.com": {
        "email": "user@example.com", "access_token": "token", "proxy": ""
    }})
    monkeypatch.setattr(account_routes.account_service, "set_account_health", lambda *_args, **_kwargs: None)

    result = asyncio.run(account_routes.test_openai_account("user@example.com"))
    assert result["code"] == 0
    assert calls == [("\u8bf7\u53ea\u56de\u590d OK\u3002", {"model": "auto"})]


@pytest.mark.parametrize("response", ["OK", " ok ", "**OK**", "OK!"])
def test_health_check_requires_a_clean_ok_reply(response):
    account_routes._validate_health_check_response(response)


@pytest.mark.parametrize("response", ["", "ERROR: unauthorized", "OK, but failed", "Error: OK"])
def test_health_check_rejects_error_or_non_ok_replies(response):
    with pytest.raises(RuntimeError, match="expected OK"):
        account_routes._validate_health_check_response(response)


def test_doubao_account_test_transient_generation_error_does_not_cool_account(monkeypatch):
    class FakeBackend:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def chat(self, *_args, **_kwargs):
            raise RuntimeError("Doubao chat returned no text")

    released = []
    monkeypatch.setattr(account_routes, "DoubaoBackendAPI", FakeBackend)
    monkeypatch.setattr(account_routes.doubao_account_service, "_accounts", {"main": {
        "name": "main", "cookies": {"sessionid": "test"}, "proxy": ""
    }})
    monkeypatch.setattr(
        account_routes.doubao_account_service,
        "set_account_health",
        lambda *args, **kwargs: released.append((args, kwargs)),
    )

    with pytest.raises(Exception):
        asyncio.run(account_routes.test_doubao_account("main"))

    assert not released
