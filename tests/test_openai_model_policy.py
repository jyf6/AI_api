import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.router import resolve_model
from providers.openai.backend import ChatRequirements, OpenAIBackendAPI


@pytest.mark.parametrize("platform", ["gemini", "doubao"])
@pytest.mark.parametrize("action", ["chat", "image"])
def test_other_platform_routing_is_unchanged(platform, action):
    assert resolve_model(f"{platform}-{action}", action).model == "auto"


@pytest.mark.parametrize("action", ["chat", "image", "health"])
def test_gpt_policy_reaches_actual_upstream_requests(monkeypatch, action):
    """从业务路由走到 HTTP 边界，检查分析、带参考图生图及探活的真实载荷。"""
    async def scenario():
        captured = []
        file_id = "file_00000000bbbbbbbbbbbbbbbbbbbbbbbb"
        if action == "image":
            event = {"conversation_id": "conversation-id", "message": {
                "author": {"role": "tool"},
                "metadata": {"async_task_type": "image_gen"},
                "content": {"parts": [f"file-service://{file_id}"]},
            }}
        else:
            event = {"message": {"author": {"role": "assistant"},
                "content": {"parts": ["final answer"]}, "status": "finished_successfully"}}

        async def chunks():
            yield ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()

        async def request(method, url, **kwargs):
            captured.append((url, kwargs))
            return SimpleNamespace(status_code=200, json=lambda: {"conduit_token": "conduit"},
                aiter_content=chunks, content=b"generated image")

        async with OpenAIBackendAPI("test-token") as backend:
            monkeypatch.setattr(backend, "_request", request)
            monkeypatch.setattr(backend, "_bootstrap", AsyncMock())
            monkeypatch.setattr(backend, "_get_chat_requirements",
                AsyncMock(return_value=ChatRequirements("sentinel")))
            monkeypatch.setattr(backend, "_close_stream", AsyncMock())
            monkeypatch.setattr(backend, "_upload_image_data", AsyncMock(return_value={
                "file_id": "reference-id", "file_name": "reference.png", "mime_type": "image/png",
                "file_size": 24, "width": 10, "height": 10, "content_hash": b"reference-hash",
            }))
            if action == "image":
                model = resolve_model("gpt-image", "image").model
                assert model == "gpt-6-thinking"
                monkeypatch.setattr(backend, "_poll_image_asset_ids",
                    AsyncMock(return_value=([file_id], [])))
                monkeypatch.setattr(backend, "_get_file_download_url",
                    AsyncMock(return_value="https://download.example/image.png"))
                monkeypatch.setattr(backend, "delete_conversation_async", AsyncMock())
                result = await backend.generate_image_bytes("generate", model, references=["oss://reference"])
                assert result == b"generated image"
                backend.delete_conversation_async.assert_not_awaited()
            else:
                model = "auto" if action == "health" else resolve_model("gpt-chat", "chat").model
                assert model == ("auto" if action == "health" else "gpt-5-6-thinking")
                assert await backend.chat_text("analyze", ["oss://reference"], model) == "final answer"

        requests = [(url, options["json"]) for url, options in captured if "json" in options]
        assert [url.split("/backend-api/")[1] for url, _ in requests] == (
            ["f/conversation/prepare", "f/conversation"] if action == "image" else ["conversation"])
        for _, payload in requests:
            assert payload["model"] == model
            if action == "health":
                assert "thinking_effort" not in payload
            else:
                assert payload["thinking_effort"] == ("medium" if action == "chat" else "extended")
        message = requests[-1][1]["messages"][0]
        assert message["content"]["parts"][0]["asset_pointer"] == "file-service://reference-id"
        if action == "image":
            assert requests[-1][1]["system_hints"] == ["picture_v2"]
            assert captured[1][1]["headers"]["X-Conduit-Token"] == "conduit"

    asyncio.run(scenario())
