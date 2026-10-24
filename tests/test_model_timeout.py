"""三平台、两种业务入口使用相同四分钟预算，超时必须回收本次调用。"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.model_timeout import MODEL_TIMEOUT_SECONDS

with patch("core.database.database.list_accounts", return_value=[]):
    from api.routers import chat, images
from api.schemas import ChatCompletionRequest, ImageGenerationRequest


@pytest.mark.parametrize("platform", ["gpt", "gemini", "doubao"])
@pytest.mark.parametrize("action", ["chat", "image"])
@pytest.mark.parametrize("block", [False, True])
def test_all_platforms_share_timeout_and_cancel_expired_attempts(monkeypatch, platform, action, block):
    router = chat if action == "chat" else images
    assert MODEL_TIMEOUT_SECONDS == router.MODEL_TIMEOUT_SECONDS == 240
    real_wait_for = asyncio.wait_for
    budgets, ended = [], []

    async def measured_wait_for(awaitable, timeout):
        budgets.append(timeout)
        # 用短的真实计时验证取消和 finally，无需让测试等待四分钟。
        return await real_wait_for(awaitable, timeout=0.01 if block else timeout)

    async def upstream(*args):
        assert args[0 if action == "chat" else 1].platform == platform
        try:
            if block:
                await asyncio.Event().wait()
            return '{"analysis":"done"}' if action == "chat" else b'\x89PNG\r\n\x1a\n' + b'\x00' * 32
        finally:
            ended.append(True)

    monkeypatch.setattr(router.asyncio, "wait_for", measured_wait_for)
    monkeypatch.setattr(router, "resolve_model", lambda *_: SimpleNamespace(platform=platform, model="auto"))
    monkeypatch.setattr(router, "_execute_chat_with_in_session_retry" if action == "chat" else "_generate_images_once", upstream)
    if action == "chat":
        body = ChatCompletionRequest(model=f"{platform}-chat", prompt="test")
        call = chat._chat_completions_prepared(body)
    else:
        body = ImageGenerationRequest(model=f"{platform}-image", prompt="test", aspect_ratio="1:1")
        call = images._generate_images(body)
    if block:
        with pytest.raises(router.HTTPException) as error:
            asyncio.run(call)
        assert error.value.status_code == 504
        expected_code = "MODEL_READ_TIMEOUT" if action == "chat" else "IMAGE_READ_TIMEOUT"
        assert f'"code": "{expected_code}"' in str(error.value.detail)
        # 任一模型已提交的生成超时都不能证明上游未出图，禁止自动重放。
        expected = 1
        assert budgets == [480 if action == "chat" else 240] * expected
    else:
        response = asyncio.run(call)
        assert response.status_code == 200
        if action == "chat":
            assert json.loads(response.body)["text"] == '{"analysis":"done"}'
        assert budgets == [480 if action == "chat" else 240]
    assert len(ended) == len(budgets)
