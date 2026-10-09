"""分析优先使用完成标记，只有无新增文本时才使用十秒兜底。"""

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest
from providers.gemini import backend as gemini
from providers.gemini.webapi.types import ModelOutput, Candidate


def output(text, done=False):
    return ModelOutput(metadata=[], candidates=[Candidate(rcid="reply", text=text)], is_completed=done)


class Stream:
    """记录读取和关闭，测试无需连接模型或数据库。"""
    def __init__(self, frames, failure=None):
        self.frames = iter(frames)
        self.failure = failure
        self.reads = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.reads += 1
        try:
            return next(self.frames)
        except StopIteration:
            if self.failure:
                raise self.failure
            raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


def setup(monkeypatch, stream):
    @contextlib.asynccontextmanager
    async def prepare(_sources):
        yield []

    monkeypatch.setattr(gemini, "_prepare_reference_images", prepare)
    monkeypatch.setattr(gemini, "mark_model_request_started", AsyncMock())
    backend = object.__new__(gemini.GeminiBackendAPI)
    class Client:
        def generate_content_stream(self, *args, **kwargs):
            return stream
    backend.client = Client()
    return backend


def test_complete_frame_returns_without_waiting_for_stream_end(monkeypatch):
    stream = Stream([output("第一部分"), output("完整说明", True), output("不应读取")])
    backend = setup(monkeypatch, stream)
    assert asyncio.run(backend._do_chat("prompt", [], "auto")) == "完整说明"
    assert stream.reads == 2
    assert stream.closed


def test_empty_completion_frame_retains_last_text(monkeypatch):
    stream = Stream([output("完整说明"), ModelOutput(metadata=[], candidates=[], is_completed=True)])
    backend = setup(monkeypatch, stream)
    assert asyncio.run(backend._do_chat("prompt", [], "auto")) == "完整说明"
    assert stream.closed


def test_idle_fallback_is_ten_seconds_and_duplicate_frames_do_not_reset_it(monkeypatch):
    stream = Stream([output("已有说明"), output("已有说明")])
    backend = setup(monkeypatch, stream)
    timeouts = []

    async def wait_for(awaitable, timeout):
        timeouts.append(timeout)
        if len(timeouts) == 3:
            awaitable.close()
            raise TimeoutError
        result = await awaitable
        await asyncio.sleep(0.01)
        return result

    monkeypatch.setattr(gemini.asyncio, "wait_for", wait_for)
    assert asyncio.run(backend._do_chat("prompt", [], "auto")) == "已有说明"
    assert 119 < timeouts[0] <= 120
    assert 9 < timeouts[1] <= 10
    assert timeouts[2] < timeouts[1]
    assert stream.closed


def test_normal_stream_end_returns_text(monkeypatch):
    stream = Stream([output("正常结束说明")])
    backend = setup(monkeypatch, stream)
    assert asyncio.run(backend._do_chat("prompt", [], "auto")) == "正常结束说明"
    assert stream.closed


def test_stream_error_does_not_claim_partial_text_is_complete(monkeypatch):
    stream = Stream([output("部分说明")], RuntimeError("stream suspended"))
    backend = setup(monkeypatch, stream)
    with pytest.raises(RuntimeError, match="stream suspended"):
        asyncio.run(backend._do_chat("prompt", [], "auto"))
    assert stream.closed


def test_first_response_timeout_does_not_return_empty_result(monkeypatch):
    stream = Stream([])
    backend = setup(monkeypatch, stream)
    async def timeout(awaitable, timeout):
        assert 119 < timeout <= 120
        awaitable.close()
        raise TimeoutError
    monkeypatch.setattr(gemini.asyncio, "wait_for", timeout)
    with pytest.raises(RuntimeError, match="120 seconds"):
        asyncio.run(backend._do_chat("prompt", [], "auto"))
    assert stream.closed
