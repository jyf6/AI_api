from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import images
from providers.gemini.backend import _create_temp_image_files


class FakeGeminiBackend:
    def __init__(self, _account):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        pass

    async def image(self, *_args):
        return [b'{"error":"not an image"}']


def test_generation_rejects_non_image_response(monkeypatch):
    monkeypatch.setattr(images, "resolve_model", lambda *_args: SimpleNamespace(platform="gemini", model="gemini-pro"))
    monkeypatch.setattr(images.gemini_account_service, "wait_for_available_account", lambda *_args: {"name": "test"})
    monkeypatch.setattr(images.gemini_account_service, "release_account", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(images, "GeminiBackendAPI", FakeGeminiBackend)

    app = FastAPI()
    app.include_router(images.router)
    response = TestClient(app).post("/v1/images/generations", json={
        "model": "gemini-pro", "prompt": "test", "images": [], "aspect_ratio": "1:1",
    })

    assert response.status_code == 502
    assert "valid image" in response.json()["detail"]


def test_gemini_reference_download_failure_aborts_request(monkeypatch):
    monkeypatch.setattr("providers.gemini.backend._read_image_source", lambda _source: (_ for _ in ()).throw(OSError("download failed")))

    with pytest.raises(OSError, match="download failed"):
        with _create_temp_image_files(["http://example.invalid/reference.png"]):
            pass
