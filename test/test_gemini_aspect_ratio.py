import json
import asyncio

from curl_cffi.requests import AsyncSession

from providers.gemini.backend import GeminiBackendAPI, _configure_image_aspect_ratio, _inject_aspect_ratio


def _request_data() -> dict[str, str]:
    inner_payload = [["生成小鸡图片", 0, None, None, None, None, 0, None, None], ["zh-CN"]]
    return {"f.req": json.dumps([None, json.dumps(inner_payload)])}


def _read_ratio(request_data: dict[str, str]) -> str:
    outer_payload = json.loads(request_data["f.req"])
    inner_payload = json.loads(outer_payload[1])
    return inner_payload[0][9][6][1][1]


def test_inject_gemini_web_aspect_ratio():
    assert _read_ratio(_inject_aspect_ratio(_request_data(), "1:1")) == "1:1"
    assert _read_ratio(_inject_aspect_ratio(_request_data(), "16:9")) == "16:9"
    assert _read_ratio(_inject_aspect_ratio(_request_data(), "9:16")) == "9:16"


def test_stream_hook_only_rewrites_stream_generate_request():
    class Session(AsyncSession):
        def stream(self, _method, _url, **kwargs):
            return kwargs["data"]

    client = type("Client", (), {"client": Session()})()
    _configure_image_aspect_ratio(client, "16:9")

    generated = client.client.stream("POST", "https://gemini.google.com/StreamGenerate", data=_request_data())
    other = client.client.stream("POST", "https://gemini.google.com/upload", data=_request_data())

    assert _read_ratio(generated) == "16:9"
    assert len(json.loads(json.loads(other["f.req"])[1])[0]) == 9


def test_aspect_ratio_hook_preserves_session_type_for_generated_images():
    """GeneratedImage 对 client 有 AsyncSession 类型校验，不能替换原会话对象。"""
    client = type("Client", (), {"client": AsyncSession()})()
    _configure_image_aspect_ratio(client, "1:1")

    assert isinstance(client.client, AsyncSession)


def test_image_returns_when_image_candidate_arrives_before_text_completion(monkeypatch):
    class Candidate:
        generated_images = ["image"]

    class Output:
        candidates = [Candidate()]
        chosen = 0

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if hasattr(self, "sent"):
                raise StopAsyncIteration
            self.sent = True
            return Output()

        async def aclose(self):
            self.closed = True

    stream = Stream()
    class Client:
        client = AsyncSession()

        def generate_content_stream(self, *_args, **_kwargs):
            return stream

    async def download(_image, _client, _proxy):
        return b"png"

    monkeypatch.setattr("providers.gemini.backend._download_generated_image", download)
    backend = GeminiBackendAPI({})
    backend.client = Client()
    result = asyncio.run(backend.image("prompt", "gemini-flash"))

    assert result == [b"png"]
    assert stream.closed is True


def test_chat_returns_when_text_stream_finishes():
    class Candidate:
        text = "style directive"

    class Output:
        candidates = [Candidate()]
        chosen = 0

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not hasattr(self, "sent"):
                self.sent = True
                return Output()
            raise StopAsyncIteration

        async def aclose(self):
            self.closed = True

    stream = Stream()

    class Client:
        def generate_content_stream(self, *_args, **_kwargs):
            return stream

    backend = GeminiBackendAPI({})
    backend.client = Client()

    assert asyncio.run(backend._do_chat("prompt", None, "gemini-flash")) == "style directive"
    assert stream.closed is True


def test_chat_keeps_received_text_when_stream_breaks_after_response():
    """上游已返回分析文本时，收尾断流不能触发整次分析重试。"""
    class Candidate:
        text = "style directive"

    class Output:
        candidates = [Candidate()]
        chosen = 0

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not hasattr(self, "sent"):
                self.sent = True
                return Output()
            raise RuntimeError("upstream stream reset after text")

        async def aclose(self):
            self.closed = True

    stream = Stream()

    class Client:
        def generate_content_stream(self, *_args, **_kwargs):
            return stream

    backend = GeminiBackendAPI({})
    backend.client = Client()

    assert asyncio.run(backend._do_chat("prompt", None, "gemini-flash")) == "style directive"
    assert stream.closed is True
