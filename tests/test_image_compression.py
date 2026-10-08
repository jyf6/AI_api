"""参考图只下载 OSS 处理结果；超限重请求，活动拼图不被缓存淘汰。"""
import asyncio
import io
import os
from unittest.mock import Mock

import pytest
from PIL import Image
from utils import oss_reference
from utils.image_stitch import stitch_images_to_bytes, _encode_bounded_jpeg

def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(output, format="JPEG")
    return output.getvalue()

@pytest.fixture(autouse=True)
def clean_cache():
    oss_reference._cache_data.clear()
    oss_reference._memory_data.clear()
    yield
    oss_reference._cache_data.clear()
    oss_reference._memory_data.clear()

def test_processed_reads_retry_without_reading_original(monkeypatch):
    valid = image_bytes()
    responses = [io.BytesIO(b"x" * 3_000_000), io.BytesIO(valid)]
    calls = []
    def get_object(key, *, process):
        calls.append((key, process))
        return responses[len(calls) - 1]
    monkeypatch.setattr(oss_reference, "_bucket", lambda: type("Bucket", (), {"get_object": staticmethod(get_object)})())
    assert oss_reference.read_oss_reference("images/a.jpg") == valid
    assert len(calls) == 2
    assert "q_85" in calls[0][1] and "q_75" in calls[1][1]
    assert all(response.closed for response in responses)

def test_specs_have_separate_cache_and_memory_is_pinned(monkeypatch):
    valid = image_bytes()
    calls = []
    def get_object(key, *, process):
        calls.append(process)
        return io.BytesIO(valid)
    monkeypatch.setattr(oss_reference, "_bucket", lambda: type("Bucket", (), {"get_object": staticmethod(get_object)})())
    monkeypatch.setattr(oss_reference, "_CACHE_MAX_SIZE", 1)
    memory = oss_reference.put_memory_reference(valid)
    assert oss_reference.read_oss_reference("images/a.jpg", 768) == valid
    assert oss_reference.read_oss_reference("images/a.jpg", 2048) == valid
    assert len(calls) == 2
    assert oss_reference.read_oss_reference(memory) == valid
    oss_reference.release_memory_references([memory])
    with pytest.raises(KeyError):
        oss_reference.read_oss_reference(memory)

def test_collage_rejects_missing_image_and_caps_output():
    with pytest.raises(Exception):
        stitch_images_to_bytes([image_bytes(), b"invalid"])
    noise = Image.frombytes("RGB", (3000, 2000), os.urandom(18_000_000))
    encoded = _encode_bounded_jpeg(noise)
    assert len(encoded) < 3_000_000
    with Image.open(io.BytesIO(encoded)) as result:
        result.load()
        assert result.width / result.height == pytest.approx(1.5)

@pytest.mark.parametrize("fail", [False, True])
def test_request_releases_memory_references(monkeypatch, fail):
    from api.routers import images
    from api.schemas import ImageGenerationRequest
    from fastapi import HTTPException
    valid = image_bytes()
    monkeypatch.setattr(oss_reference, "read_oss_reference", lambda *_args: valid)
    async def generate(body, *_args):
        assert body.images[0].startswith("memory://")
        if fail:
            raise HTTPException(status_code=502, detail="test")
        return valid
    monkeypatch.setattr(images, "_generate_images_once", generate)
    body = ImageGenerationRequest(prompt="test", aspect_ratio="1:1", images=["images/a.jpg"])
    if fail:
        with pytest.raises(HTTPException):
            asyncio.run(images._generate_images(body))
    else:
        assert asyncio.run(images._generate_images(body)).body == valid
    assert not oss_reference._memory_data
