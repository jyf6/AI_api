"""验证平台协议载荷与本轮图片提取，不以缓存命中替代真实续接参数。"""
import asyncio
import contextlib
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from providers.openai.backend import ChatRequirements, OpenAIBackendAPI
from providers.gemini import backend as gemini
from providers.gemini.webapi.client import ChatSession, GeminiClient


def test_gpt_reuse_reaches_prepare_and_send_and_poll_excludes_history(monkeypatch):
    async def scenario():
        sent = []
        old_file = "file_00000000aaaaaaaaaaaaaaaaaaaaaaaa"
        new_file = "file_00000000bbbbbbbbbbbbbbbbbbbbbbbb"
        async with OpenAIBackendAPI("test-token") as backend:
            async def request(method, url, **kwargs):
                sent.append((method, url, kwargs))
                async def chunks():
                    yield ('data: ' + json.dumps({"conversation_id": "same-conversation"}) + '\n\ndata: [DONE]\n\n').encode()
                if method == "GET" and "/backend-api/conversation/" in url:
                    root = getattr(backend, "_image_request_message_id", "current-user")
                    def image(file):
                        return {"author": {"role": "tool"}, "content": {"parts": [f"file-service://{file}"]}}
                    return SimpleNamespace(status_code=200, json=lambda: {
                        "current_node": "previous-node", "mapping": {
                            "previous": {"message": image(old_file)},
                            root: {"parent": "previous-node", "children": ["new-output"],
                                   "message": {"author": {"role": "user"}}},
                            "new-output": {"parent": root, "message": image(new_file)},
                        }})
                return SimpleNamespace(status_code=200, json=lambda: {"conduit_token": "test"},
                                       aiter_content=chunks, content=b"new-image")
            monkeypatch.setattr(backend, "_request", request)
            monkeypatch.setattr(backend, "_bootstrap", AsyncMock())
            monkeypatch.setattr(backend, "_get_chat_requirements", AsyncMock(return_value=ChatRequirements("test")))
            monkeypatch.setattr(backend, "_close_stream", AsyncMock())
            download = AsyncMock(return_value="https://download.example/image")
            monkeypatch.setattr(backend, "_get_file_download_url", download)
            callback = AsyncMock()
            result = await backend.generate_image_bytes("next", conversation_ref={"conversationId": "same-conversation"},
                                                        on_conversation=callback)
            assert result == b"new-image"
            download.assert_awaited_once_with(new_file)
            callback.assert_awaited_once_with({"conversationId": "same-conversation"})
            posts = [options["json"] for method, _, options in sent if method == "POST"]
            assert len(posts) == 2
            assert all(p["conversation_id"] == "same-conversation" and
                       p["parent_message_id"] == "previous-node" for p in posts)
            assert not any(method == "PATCH" for method, _, _ in sent)
    asyncio.run(scenario())


def test_gemini_waits_for_final_state_before_caching_and_keeps_image_candidate(monkeypatch):
    async def scenario():
        saved, incoming, closed = [], [], []
        async def save(ref):
            saved.append(copy.deepcopy(ref))
        async def download(*args):
            return b"image"
        monkeypatch.setattr(gemini, "_download_generated_image", download)
        model = SimpleNamespace(model_name="gemini-pro", display_name="Pro", aliases=[], is_available=True, model_id="pro")
        class Client:
            client = None
            def list_models(self):
                return [model]
            def start_chat(self, **kwargs):
                return ChatSession(self, **kwargs)
            async def _batch_execute(self, *args, **kwargs):
                return SimpleNamespace(text="[]")
            async def generate_content_stream(self, *args, chat, **kwargs):
                incoming.append(copy.deepcopy(chat.metadata))
                # 模拟底层解析器更新完整上下文，但外层收到的output只有cid/rid。
                chat.metadata = ["c_same", f"rid-{len(incoming)}", None, None, None, None, None, None, None, "full-context"]
                try:
                    yield SimpleNamespace(metadata=["c_same", f"rid-{len(incoming)}"], rcid=f"rcid-{len(incoming)}",
                        candidates=[SimpleNamespace(generated_images=["new-image"])], chosen=0)
                    assert len(saved) == len(incoming) - 1, "完成收流前不能提交新会话"
                    chat.metadata = ["c_same", f"rid-{len(incoming)}", None, None, None, None, None, None, None, "final-context"]
                    yield SimpleNamespace(metadata=["c_same", f"rid-{len(incoming)}"], rcid=f"rcid-{len(incoming)}",
                        candidates=[SimpleNamespace(generated_images=[])], chosen=0)
                finally:
                    closed.append(True)
        for _ in range(2):
            backend = gemini.GeminiBackendAPI({"proxy": ""})
            backend.client = Client()
            result = await backend._do_image("prompt", "auto", conversation_ref=saved[-1] if saved else None,
                                             on_conversation=save)
            assert result == [b"image"]
        assert len(closed) == 2
        assert incoming[1][0:3] == ["c_same", "rid-1", "rcid-1"]
        assert incoming[1][9] == "final-context"
        assert saved[-1]["metadata"][0:3] == ["c_same", "rid-2", "rcid-2"]
        assert len(saved) == 2
    asyncio.run(scenario())


def test_gemini_closed_client_cleanup_preserves_original_upstream_error():
    async def scenario():
        from providers.gemini.webapi.exceptions import APIError
        original = APIError("upstream conversation rejected")
        response = SimpleNamespace(astream_task=SimpleNamespace(done=lambda: False), curl=None)
        @contextlib.asynccontextmanager
        async def stream(*args, **kwargs):
            yield response
        remove = Mock(side_effect=TypeError("initializer for ctype 'void *' must be a cdata pointer, not NoneType"))
        session = SimpleNamespace(stream=stream, acurl=SimpleNamespace(remove_handle=remove), _closed=True)
        gemini._install_aspect_ratio_stream_hook(session)
        with pytest.raises(APIError) as caught:
            async with session.stream("POST", "https://example.test/StreamGenerate"):
                raise original
        assert caught.value is original
        remove.assert_not_called()
    asyncio.run(scenario())


def test_gemini_initial_auth_runtime_error_becomes_switchable_auth_error(monkeypatch):
    async def scenario():
        with patch("core.database.database.list_accounts", return_value=[]):
            from providers.gemini.account import gemini_account_service
        monkeypatch.setattr(gemini_account_service, "get_request_client",
                            AsyncMock(side_effect=RuntimeError("Gemini Cookie 未认证，请更新 Cookie")))
        import pytest
        with pytest.raises(gemini.AuthError):
            async with gemini.GeminiBackendAPI({"name": "test"}):
                raise AssertionError("认证失败不能开始生图")
    asyncio.run(scenario())


def test_gemini_initial_network_error_does_not_become_account_auth_failure(monkeypatch):
    async def scenario():
        with patch("core.database.database.list_accounts", return_value=[]):
            from providers.gemini.account import gemini_account_service
        failure = ConnectionError("Cookie initialization proxy connection failed")
        monkeypatch.setattr(gemini_account_service, "get_request_client", AsyncMock(side_effect=failure))
        import pytest
        with pytest.raises(ConnectionError) as caught:
            async with gemini.GeminiBackendAPI({"name": "test"}):
                raise AssertionError("连接失败不能开始生图")
        assert caught.value is failure
    asyncio.run(scenario())


def test_gemini_outer_stream_close_immediately_closes_inner_generator():
    async def scenario():
        closed = []
        async def generate(**kwargs):
            try:
                yield SimpleNamespace(text="image")
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        client = SimpleNamespace(auto_close=False, _sync_activity=AsyncMock(), _generate=generate)
        stream = GeminiClient.generate_content_stream(client, "test")
        await anext(stream)
        await stream.aclose()
        assert closed == [True]
    asyncio.run(scenario())


def test_gemini_real_http_stream_retains_final_frame_and_closes_normally():
    async def scenario():
        from curl_cffi.requests import AsyncSession
        disconnected = asyncio.Event()
        async def upstream(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: result\n\n")
                await writer.drain()
                await asyncio.sleep(.03)
                writer.write(b"data: saved\n\n")
                await writer.drain()
                disconnected.set()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(upstream, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with AsyncSession() as session:
                gemini._install_aspect_ratio_stream_hook(session)
                async def read_once():
                    async with session.stream("GET", f"http://127.0.0.1:{port}/StreamGenerate", timeout=5) as response:
                        body = b"".join([chunk async for chunk in response.aiter_content()])
                        assert b"result" in body and b"saved" in body
                await asyncio.wait_for(read_once(), 1)
                await asyncio.wait_for(disconnected.wait(), 1)
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(scenario())


def test_gemini_missing_history_is_rejected_before_generation(monkeypatch):
    async def scenario():
        from core.slot_sessions import ConversationUnavailable
        from providers.gemini.webapi.constants import GRPC
        backend = gemini.GeminiBackendAPI({"proxy": ""})
        backend.client = SimpleNamespace(
            list_models=lambda: [SimpleNamespace(model_name="gemini-pro", display_name="Pro", aliases=[], is_available=True, model_id="pro")],
            _batch_execute=AsyncMock(return_value=SimpleNamespace(text=json.dumps([
                ["wrb.fr", GRPC.LIST_CONVERSATION_TURNS, None, None, None, [5]]
            ]))),
            start_chat=Mock(), generate_content_stream=Mock())
        with pytest.raises(ConversationUnavailable):
            await backend._do_image("prompt", "auto", conversation_ref={"cid": "missing", "metadata": ["missing", "rid", "rcid"]})
        backend.client.start_chat.assert_not_called()
        backend.client.generate_content_stream.assert_not_called()
    asyncio.run(scenario())


def test_gemini_keeps_image_but_does_not_publish_session_when_tail_fails(monkeypatch):
    async def scenario():
        model = SimpleNamespace(model_name="gemini-pro", display_name="Pro", aliases=[], is_available=True, model_id="pro")
        class Client:
            client = None
            def list_models(self):
                return [model]
            def start_chat(self, **kwargs):
                return ChatSession(self, **kwargs)
            async def generate_content_stream(self, *args, **kwargs):
                yield SimpleNamespace(metadata=["cid", "rid"], rcid="rcid", chosen=0,
                                      candidates=[SimpleNamespace(generated_images=["image"])])
                raise RuntimeError("stream tail interrupted")
        backend = gemini.GeminiBackendAPI({"proxy": ""})
        backend.client = Client()
        saved = AsyncMock()
        monkeypatch.setattr(gemini, "_download_generated_image", AsyncMock(return_value=b"image"))
        assert await backend._do_image("prompt", "auto", on_conversation=saved) == [b"image"]
        saved.assert_not_awaited()
    asyncio.run(scenario())


def test_gemini_cancelled_http_stream_releases_connection_without_hiding_cancel():
    async def scenario():
        from curl_cffi.requests import AsyncSession
        started, disconnected = asyncio.Event(), asyncio.Event()
        async def upstream(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: partial\n\n")
                await writer.drain()
                await reader.read()
                disconnected.set()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(upstream, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with AsyncSession() as session:
                gemini._install_aspect_ratio_stream_hook(session)
                async def consume():
                    async with session.stream("GET", f"http://127.0.0.1:{port}/StreamGenerate", timeout=5) as response:
                        async for chunk in response.aiter_content():
                            started.set()
                            await asyncio.Event().wait()
                task = asyncio.create_task(consume())
                await started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
                await asyncio.wait_for(disconnected.wait(), 1)
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(scenario())
