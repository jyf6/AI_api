import asyncio
import io
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from api.routers import edit, images
from providers.gemini import backend as gemini_backend
from utils.log import error_http_status


@pytest.mark.parametrize("platform", ["gpt", "doubao", "gemini"])
@pytest.mark.parametrize("result,reason", [(b"not an image", "INVALID_IMAGE"), ([], "IMAGE_COUNT_SHORT")])
def test_invalid_image_does_not_mark_account_success(monkeypatch, platform, result, reason):
    account = {"name": "account", "email": "account", "cookies": {}, "access_token": "token"}
    released = Mock()

    async def acquire(*_args):
        return account

    class Backend:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def image(self, *_args):
            return result if isinstance(result, list) else [result]

        async def generate_image_bytes(self, *_args, **_kwargs):
            return result

        async def generate_image(self, *_args):
            return ["image-url"]

        async def download_images(self, *_args):
            return result if isinstance(result, list) else [result]

    monkeypatch.setattr(images, "acquire_traced_account", acquire)
    monkeypatch.setattr(images, "OpenAIBackendAPI", lambda *_args, **_kwargs: Backend())
    monkeypatch.setattr(images, "DoubaoBackendAPI", lambda *_args, **_kwargs: Backend())
    monkeypatch.setattr(images, "GeminiBackendAPI", lambda *_args, **_kwargs: Backend())
    service = {"gpt": images.account_service, "doubao": images.doubao_account_service,
               "gemini": images.gemini_account_service}[platform]
    monkeypatch.setattr(service, "release_account", released)
    body = SimpleNamespace(images=[], aspect_ratio="1:1")
    resolved = SimpleNamespace(platform=platform, model="auto")

    with pytest.raises(images.ImageResultError) as error:
        asyncio.run(images._generate_images_once(body, resolved, "prompt", "request-1", 1))

    assert error.value.reason_code == reason
    assert released.call_count == 1
    assert released.call_args.args[1] is False
    assert released.call_args.args[2] == ""


@pytest.mark.parametrize("operation", ["remove_bg", "inpaint", "upscale"])
def test_edits_log_completion(monkeypatch, operation):
    source = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(source, format="PNG")
    log_info = Mock()
    monkeypatch.setattr(edit.logger, "info", log_info)
    upload = lambda: UploadFile(file=io.BytesIO(source.getvalue()))

    if operation == "remove_bg":
        monkeypatch.setattr(edit, "u2net_session", None)
        monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(
            new_session=lambda *_args: object(), remove=lambda content, **_kwargs: content))
        response = edit.remove_bg(upload())
    elif operation == "inpaint":
        response = edit.inpaint(upload(), upload(), radius=5)
    else:
        response = edit.upscale(upload(), scale=2)

    assert response.media_type == "image/png"
    assert response.headers["X-Request-ID"]
    assert any(f"event=image_edit_finished operation={operation}" in call.args[0]
               for call in log_info.call_args_list)


def test_gemini_closes_stream_after_image_candidate(monkeypatch):
    class Stream:
        closed = False

        async def __aiter__(self):
            yield SimpleNamespace(candidates=[SimpleNamespace(generated_images=["candidate"])], chosen=0)

        async def aclose(self):
            self.closed = True

    stream = Stream()

    async def mark_started():
        return None

    async def download(*_args):
        return b"\x89PNG\r\n\x1a\nresult"

    monkeypatch.setattr(gemini_backend, "mark_model_request_started", mark_started)
    monkeypatch.setattr(gemini_backend, "_download_generated_image", download)
    backend = gemini_backend.GeminiBackendAPI({"proxy": ""})
    backend.client = SimpleNamespace(client=None, generate_content_stream=lambda *_args, **_kwargs: stream)

    result = asyncio.run(backend._do_image("prompt", "auto"))

    assert stream.closed
    assert len(result) == 1


def test_http_error_status_can_come_from_response():
    error = RuntimeError("upstream rejected")
    error.response = SimpleNamespace(status_code=403)
    assert error_http_status(error) == 403
