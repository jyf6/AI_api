"""分析拼图的分组、重试和请求资源生命周期验证。"""
import asyncio
import io
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image
from pydantic import ValidationError

with patch('core.database.database.list_accounts', return_value=[]):
    from api.routers import chat
from api.schemas import ChatCompletionRequest
from utils import oss_reference
from utils.image_stitch import stitch_reference_sheets


def image_bytes():
    out = io.BytesIO()
    with Image.new('RGBA', (1200, 300), (255, 0, 0, 128)) as image:
        image.save(out, format='PNG')
    return out.getvalue()


@pytest.mark.parametrize('count,expected', [(2, [(1536, 768)]), (3, [(1536, 1536)]),
                                            (4, [(1536, 1536)]), (5, [(1536, 1536), (768, 768)])])
def test_reference_sheet_layout(count, expected):
    sheets = stitch_reference_sheets([image_bytes()] * count)
    sizes = []
    for sheet in sheets:
        assert len(sheet) < 3_000_000
        with Image.open(io.BytesIO(sheet)) as image:
            sizes.append(image.size)
            assert image.getpixel((100, 100)) == (255, 255, 255)
            assert image.crop((12, 4, 80, 36)).getextrema()[0][0] < 50
    assert sizes == expected


def test_count_cannot_exceed_images():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(prompt='p', images=['a'], reference_image_count=2)


@pytest.mark.parametrize('count', [None, 0, 1])
def test_legacy_and_single_reference_unchanged(monkeypatch, count):
    body = ChatCompletionRequest(prompt='p', images=['reference', 'products'], reference_image_count=count)
    async def prepared(actual):
        assert actual.images == body.images
        return 'ok'
    monkeypatch.setattr(chat, '_chat_completions_prepared', prepared)
    assert asyncio.run(chat._chat_completions(body)) == 'ok'


def test_retries_reuse_collage_and_release(monkeypatch):
    reads, attempts, keys = [], [], []
    real_read = oss_reference.read_oss_reference
    def read(source, dimension=2048):
        if source.startswith('memory://'):
            return real_read(source, dimension)
        reads.append((source, dimension))
        return image_bytes()
    async def execute(_resolved, _prompt, images, _request_id, _attempt):
        attempts.append(list(images))
        keys.extend(images[:-1])
        assert images[-1] == 'product-sheet'
        for key in images[:-1]:
            assert real_read(key)
        return 'short' if len(attempts) == 1 else '【推荐产品图】：1'
    monkeypatch.setattr(oss_reference, 'read_oss_reference', read)
    monkeypatch.setattr(chat, 'resolve_model', lambda *_: SimpleNamespace(platform='gemini', model='auto'))
    monkeypatch.setattr(chat, '_execute_chat_once', execute)
    body = ChatCompletionRequest(prompt='p', images=['https://example.com/a', 'oss/b', 'product-sheet'],
                                 reference_image_count=2)
    asyncio.run(chat._chat_completions(body))
    assert reads == [('https://example.com/a', 768), ('oss/b', 768)]
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    for key in keys:
        with pytest.raises(KeyError):
            real_read(key)


def test_cancelled_preparation_releases_after_worker(monkeypatch):
    started, release = threading.Event(), threading.Event()
    keys = []
    real_put, real_read = oss_reference.put_memory_reference, oss_reference.read_oss_reference
    def read(*_):
        started.set()
        release.wait(2)
        return image_bytes()
    def put(data):
        key = real_put(data)
        keys.append(key)
        return key
    monkeypatch.setattr(oss_reference, 'read_oss_reference', read)
    monkeypatch.setattr(oss_reference, 'put_memory_reference', put)
    async def scenario():
        body = ChatCompletionRequest(prompt='p', images=['a', 'b'], reference_image_count=2)
        task = asyncio.create_task(chat._chat_completions(body))
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert keys
    for key in keys:
        with pytest.raises(KeyError):
            real_read(key)


def test_preparation_failure_does_not_call_model(monkeypatch):
    def read(*_):
        raise OSError('download failed')
    monkeypatch.setattr(oss_reference, 'read_oss_reference', read)
    body = ChatCompletionRequest(prompt='p', images=['a', 'b'], reference_image_count=2)
    with pytest.raises(chat.HTTPException) as error:
        asyncio.run(chat._chat_completions(body))
    assert error.value.status_code == 502
