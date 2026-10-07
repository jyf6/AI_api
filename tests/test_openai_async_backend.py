import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from providers.openai.backend import ChatRequirements, OpenAIBackendAPI
from utils.helper import aiter_sse_payloads


class ChunkResponse:
    def __init__(self, chunks):
        self.chunks = chunks

    async def aiter_content(self):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk


def test_sse_handles_utf8_across_chunks_and_multiline_events():
    async def scenario():
        raw = 'data: {"text":"图片"}\r\ndata: tail\r\n\r\ndata: [DONE]\n\n'.encode()
        response = ChunkResponse([raw[i:i + 1] for i in range(len(raw))])
        result = [event async for event in aiter_sse_payloads(response)]
        assert result == ['{"text":"图片"}\ntail', '[DONE]']
    asyncio.run(scenario())


def test_sse_rejects_truncated_final_event():
    async def scenario():
        with pytest.raises(RuntimeError, match="incomplete"):
            _ = [event async for event in aiter_sse_payloads(ChunkResponse([b'data: {"text":']))]
    asyncio.run(scenario())


def test_credential_refresh_sends_one_authorization_header():
    async def scenario():
        async def credentials():
            return {"access_token": "fresh-token"}

        async with OpenAIBackendAPI("old-token", device_id="stable-device", credential_provider=credentials) as backend:
            captured = {}

            async def get(url, **kwargs):
                captured.update(kwargs["headers"])
                return SimpleNamespace(status_code=200)

            backend.session.get = get
            await backend._request("GET", backend.base_url + "/", headers=backend._headers("/"))
            auth_keys = [key for key in captured if key.lower() == "authorization"]
            assert len(auth_keys) == 1
            assert captured[auth_keys[0]] == "Bearer fresh-token"

    asyncio.run(scenario())


def test_150_gpt_network_waits_are_concurrent_without_worker_threads(monkeypatch):
    async def scenario():
        started = 0
        all_started = asyncio.Event()
        release = asyncio.Event()
        response = ChunkResponse([
            ('data: ' + json.dumps({"message": {"author": {"role": "assistant"},
                "content": {"parts": ["analysis completed"]}}}) + '\n\ndata: [DONE]\n\n').encode()
        ])
        response.status_code = 200

        async def post(*args, **kwargs):
            nonlocal started
            started += 1
            if started == 150:
                all_started.set()
            await release.wait()
            return response

        sessions = []
        def session_factory(**kwargs):
            assert kwargs["impersonate"] == "chrome124"
            session = SimpleNamespace(headers={}, post=post, close=AsyncMock())
            sessions.append(session)
            return session

        monkeypatch.setattr("providers.openai.backend.requests.AsyncSession", session_factory)
        monkeypatch.setattr(OpenAIBackendAPI, "_bootstrap", AsyncMock())
        monkeypatch.setattr(OpenAIBackendAPI, "_get_chat_requirements", AsyncMock(return_value=ChatRequirements("sentinel")))
        monkeypatch.setattr(OpenAIBackendAPI, "_close_stream", AsyncMock())

        async def call():
            async with OpenAIBackendAPI("token", "http://proxy:8080", "stable-device") as backend:
                return await backend.chat_text("test")

        tasks = [asyncio.create_task(call()) for _ in range(150)]
        try:
            await asyncio.wait_for(all_started.wait(), 2)
            assert all(not task.done() for task in tasks)
        finally:
            release.set()
        assert await asyncio.gather(*tasks) == ["analysis completed"] * 150
        for session in sessions:
            assert session.headers["OAI-Device-Id"] == "stable-device"
            session.close.assert_awaited_once()
    asyncio.run(scenario())


def test_real_async_session_can_close_sse_before_upstream_finishes():
    """本地临时端口验证实际 curl_cffi 收尾，不调用模型、OSS 或项目服务。"""
    async def scenario():
        disconnected = asyncio.Event()
        async def upstream(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: result\n\n")
                await writer.drain()
                await reader.read()
                disconnected.set()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(upstream, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with OpenAIBackendAPI() as backend:
                response = await backend.session.get(f"http://127.0.0.1:{port}/sse", stream=True, timeout=10)
                stream = aiter_sse_payloads(response)
                assert await anext(stream) == "result"
                await asyncio.wait_for(backend._close_stream(response), 1)
                await stream.aclose()
                await asyncio.wait_for(disconnected.wait(), 1)
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(scenario())
