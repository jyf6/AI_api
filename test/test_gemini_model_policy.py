"""隔离验证 Gemini 统一选型、分析中档/生图高档和真实协议字段，不访问 Google。"""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

from curl_cffi.requests import Cookies
import pytest

from providers.gemini import backend as gemini
from providers.gemini.webapi import GeminiClient
from providers.gemini.webapi.constants import MODEL_HEADER_KEY
from providers.gemini.webapi.exceptions import APIError
from providers.gemini.webapi.types import AvailableModel


def model(name, display, number, *, available=True, model_id=None):
    """使用账号发现后的模型对象，内部 ID 在测试中可变。"""
    return AvailableModel(model_id=model_id or name, model_name=name, display_name=display,
                          description="", capacity=2, model_number=number,
                          is_available=available, aliases=["gemini-" + display.lower().replace(" ", "-")])


class Client:
    def __init__(self, models, failure=None):
        self.models, self.failure, self.calls = models, failure, []
        self.client = None

    def list_models(self):
        return self.models

    def generate_content_stream(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.failure:
            raise self.failure

        async def stream():
            yield SimpleNamespace(candidates=[SimpleNamespace(text="分析结果", generated_images=["image"])],
                                  chosen=0, is_completed=True)
        return stream()


@pytest.mark.parametrize("action", ["chat", "image"])
@pytest.mark.parametrize("incoming_model", ["auto", "gemini-flash", "gemini-pro"])
def test_all_entry_calls_use_required_model_and_task_thinking_level(monkeypatch, action, incoming_model):
    flash = model("gemini-flash", "3.8 Flash", 1)
    pro = model("gemini-pro", "3.1 Pro", 3)
    client = Client([model("gemini-flash-lite", "3.5 Flash-Lite", 6), pro, flash])
    service = gemini.GeminiBackendAPI({})
    service.client = client

    async def download(_image, actual_client, _proxy):
        assert actual_client is client
        return b"image-bytes"
    monkeypatch.setattr(gemini, "_download_generated_image", download)

    if action == "chat":
        assert asyncio.run(service.chat("prompt", model=incoming_model)) == "分析结果"
    else:
        assert asyncio.run(service.image("prompt", incoming_model)) == [b"image-bytes"]
    thinking = {"thinking_level": "medium"} if action == "chat" else {"extended_thinking": True, "chat": None}
    assert client.calls == [("prompt", {"files": None, "model": flash if action == "chat" else pro,
                                        **thinking})]


@pytest.mark.parametrize("action,models", [
    ("chat", [model("gemini-flash", "3.7 Flash", 1)]),
    ("chat", [model("gemini-flash", "3.8 Flash", 1, available=False)]),
    ("image", [model("gemini-flash", "3.8 Flash", 1)]),
    ("image", [model("gemini-pro", "3.1 Pro", 3, available=False)]),
])
def test_missing_or_unavailable_model_never_generates_with_a_substitute(action, models):
    client = Client(models)
    service = gemini.GeminiBackendAPI({})
    service.client = client
    call = service.chat("prompt") if action == "chat" else service.image("prompt", "auto")
    with pytest.raises(ValueError, match="不可用|不使用其他模型替代"):
        asyncio.run(call)
    assert client.calls == []


@pytest.mark.parametrize("action", ["chat", "image"])
def test_reconnect_resolves_current_account_model_again(monkeypatch, action):
    name, display, number = ("gemini-flash", "3.8 Flash", 1) if action == "chat" else ("gemini-pro", "3.1 Pro", 3)
    old = Client([model(name, display, number, model_id="old-id")], RuntimeError("401"))
    new = Client([model(name, display, number, model_id="new-id")])
    service = gemini.GeminiBackendAPI({})
    service.client = old

    async def reconnect():
        service.client = new

    async def download(*_args):
        return b"image-bytes"
    monkeypatch.setattr(service, "_reconnect", reconnect)
    monkeypatch.setattr(gemini, "_download_generated_image", download)
    asyncio.run(service.chat("prompt") if action == "chat" else service.image("prompt", "auto"))
    assert old.calls[0][1]["model"].model_id == "old-id"
    assert new.calls[0][1]["model"].model_id == "new-id"
    for client in (old, new):
        if action == "chat":
            assert client.calls[0][1]["thinking_level"] == "medium"
        else:
            assert client.calls[0][1]["extended_thinking"] is True


@pytest.mark.parametrize("selected", [model("gemini-flash", "3.8 Flash", 1), model("gemini-pro", "3.1 Pro", 3)])
@pytest.mark.parametrize("thinking,wire", [
    ({"extended_thinking": True}, 2),
    ({"extended_thinking": False}, 1),
    ({"thinking_level": "medium"}, 4),
    ({"thinking_level": "medium", "extended_thinking": True}, 4),
    ({"thinking_level": "high", "extended_thinking": False}, 2),
])
def test_real_serializer_sets_model_and_thinking_in_header_and_body(selected, thinking, wire):
    """执行真实网页协议组装，在网络边界截获请求，验证选型不会停留在 Python 参数层。"""
    class RequestCaptured(APIError):
        pass

    class Session:
        cookies = Cookies()
        request = None

        @asynccontextmanager
        async def stream(self, method, url, **kwargs):
            self.request = (method, str(url), kwargs)
            raise RequestCaptured
            yield  # 保持异步上下文管理器形态，禁止发送网络请求。

        async def close(self):
            pass

    async def run():
        client = GeminiClient("fake-psid", "fake-ts")
        session = Session()
        client.client, client._running = session, True
        stream = GeminiClient._generate.__wrapped__(client, prompt="test", model=selected,
                                                    **thinking)
        try:
            with pytest.raises(RequestCaptured):
                await anext(stream)
        finally:
            await stream.aclose()
            await client.close()
        method, url, request = session.request
        assert method == "POST" and url.endswith("/StreamGenerate")
        header = json.loads(request["headers"][MODEL_HEADER_KEY])
        body = json.loads(json.loads(request["data"]["f.req"])[1])
        assert header[4] == selected.model_id
        assert header[-3] == body[79] == selected.model_number
        assert header[-2] == body[80] == wire
        if wire == 4:
            assert len(body) == 97
            assert body[91] == 0
            assert body[96] == 1
        else:
            assert len(body) == 81
    asyncio.run(run())


@pytest.mark.parametrize("streaming", [False, True])
def test_public_generation_forwards_explicit_thinking_level(streaming):
    async def run():
        client = GeminiClient("fake-psid", "fake-ts")
        captured = []

        async def sync_activity():
            pass

        async def generate(**kwargs):
            captured.append(kwargs)
            yield SimpleNamespace(metadata=[], text="ok")

        client._sync_activity = sync_activity
        client._generate = generate
        if streaming:
            async for _ in client.generate_content_stream("test", thinking_level="medium"):
                pass
        else:
            await client.generate_content("test", thinking_level="medium")
        assert captured[0]["thinking_level"] == "medium"
        await client.close()
    asyncio.run(run())


def test_invalid_thinking_level_is_rejected_before_network():
    async def run():
        client = GeminiClient("fake-psid", "fake-ts")
        stream = GeminiClient._generate.__wrapped__(client, prompt="test", thinking_level="invalid")
        try:
            with pytest.raises(ValueError, match="thinking_level"):
                await anext(stream)
        finally:
            await stream.aclose()
            await client.close()
    asyncio.run(run())
