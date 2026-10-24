"""分析与生图将参考图原样交给模型，不在 Python 侧生成拼图。"""
import asyncio
from unittest.mock import patch

with patch("core.database.database.list_accounts", return_value=[]):
    from api.routers import chat
from api.schemas import ChatCompletionRequest


def test_chat_analysis_passes_original_reference_list(monkeypatch):
    images = [f"reference_{index}" for index in range(12)]
    body = ChatCompletionRequest(prompt="p", images=images)

    async def prepared(actual):
        assert actual.images == images
        return "analysis"

    monkeypatch.setattr(chat, "_chat_completions_prepared", prepared)
    assert asyncio.run(chat._chat_completions(body)) == "analysis"
