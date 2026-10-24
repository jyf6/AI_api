"""分析失败保留真实原因，Gemini 异常处理不得覆盖原始异常。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

with patch("core.database.database.list_accounts", return_value=[]):
    from api.routers import chat
from api.schemas import ChatCompletionRequest
from providers.gemini.webapi.exceptions import APIError, AuthError, TemporarilyBlockedError, UsageLimitExceededError


@pytest.mark.parametrize("error_type", [APIError, AuthError, TemporarilyBlockedError, UsageLimitExceededError])
def test_gemini_error_handler_preserves_original_exception(monkeypatch, error_type):
    original = error_type("Google response stream parsing failed")
    backend = AsyncMock()
    backend.__aenter__.return_value = backend
    backend.chat_with_session.side_effect = original
    monkeypatch.setattr(chat, "GeminiBackendAPI", Mock(return_value=backend))
    monkeypatch.setattr(chat, "acquire_traced_account", AsyncMock(return_value={"name": "test"}))
    release = Mock()
    monkeypatch.setattr(chat.gemini_account_service, "release_account", release)
    with pytest.raises(error_type) as caught:
        asyncio.run(chat._execute_chat_with_in_session_retry(
            SimpleNamespace(platform="gemini", model="auto"), "test", [], "request-test", "INITIAL_TASK_INSTRUCTION_ANALYSIS"))
    assert caught.value is original
    release.assert_called_once()


@pytest.mark.parametrize("platform", ["gemini", "gpt", "doubao"])
def test_analysis_error_contains_stage_cause_and_request_id(monkeypatch, platform):
    monkeypatch.setattr(chat, "resolve_model", lambda *_: SimpleNamespace(platform=platform, model="auto"))
    monkeypatch.setattr(chat, "_execute_chat_with_in_session_retry", AsyncMock(
        side_effect=APIError("Google response stream parsing failed")))
    body = ChatCompletionRequest(model="test", prompt="test", request_id="request-test",
                                 stage="INITIAL_TASK_INSTRUCTION_ANALYSIS")
    with pytest.raises(chat.HTTPException) as caught:
        asyncio.run(chat._chat_completions_prepared(body))
    error = caught.value
    assert error.status_code == 502
    assert "Google response stream parsing failed" in error.detail
    assert "APIError" in error.detail
    assert "用户指令分析" in error.detail
    assert "request-test" in error.detail
    assert error.headers["X-Request-ID"] == "request-test"


def test_validation_failure_explains_why_retry_was_rejected(monkeypatch):
    backend = AsyncMock()
    backend.__aenter__.return_value = backend
    backend.chat_with_session.return_value = ("Oops", object())
    backend.chat_continue.return_value = "Oops"
    monkeypatch.setattr(chat, "GeminiBackendAPI", Mock(return_value=backend))
    monkeypatch.setattr(chat, "acquire_traced_account", AsyncMock(return_value={"name": "test"}))
    monkeypatch.setattr(chat.gemini_account_service, "release_account", Mock())
    monkeypatch.setattr(chat.asyncio, "sleep", AsyncMock())
    with pytest.raises(chat.HTTPException) as caught:
        asyncio.run(chat._execute_chat_with_in_session_retry(
            SimpleNamespace(platform="gemini", model="auto"), "test", [], "request-test", "INITIAL_TASK_INSTRUCTION_ANALYSIS"))
    assert "输出校验失败" in caught.value.detail
    assert "4" in caught.value.detail and "40" in caught.value.detail
    assert "request-test" in caught.value.detail


def test_error_details_redact_credentials_without_hiding_cause():
    detail = chat.analysis_failure_detail("gemini", "INITIAL_TASK_INSTRUCTION_ANALYSIS", "request-test",
                                         "APIError: response parsing failed token=secret Bearer secret2 cookie=secret3")
    assert "APIError: response parsing failed" in detail
    for secret in ("secret", "secret2", "secret3"):
        assert secret not in detail
