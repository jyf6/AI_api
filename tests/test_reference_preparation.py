"""参考图准备不能占用代理事件循环，取消后仍需释放内存。"""

import asyncio
import contextlib
import threading

import pytest

from providers.doubao import backend as doubao
from providers.gemini import backend as gemini


def test_doubao_reference_download_does_not_block_event_loop(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_prepare(_source):
        started.set()
        release.wait(1)
        return b"image", "image/png"

    monkeypatch.setattr(doubao, "_prepare_reference_image", slow_prepare)

    async def scenario():
        client = object.__new__(doubao.DoubaoBackendAPI)
        client.session = object()
        request = asyncio.create_task(client.upload_image("https://example.invalid/image"))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            # 下载线程仍在等待时，事件循环必须能继续执行其他协程。
            await asyncio.sleep(0)
            assert not request.done()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        finally:
            release.set()

    asyncio.run(scenario())


def test_gemini_cancelled_preparation_closes_memory_images(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()
    image_created = threading.Event()
    images = []
    original = gemini._create_reference_images

    def slow_read(_source):
        started.set()
        release.wait(1)
        return b"image", ".png"

    @contextlib.contextmanager
    def track_images(sources):
        with original(sources) as prepared:
            images.extend(prepared)
            image_created.set()
            yield prepared

    monkeypatch.setattr(gemini, "_read_image_source", slow_read)
    monkeypatch.setattr(gemini, "_create_reference_images", track_images)

    async def scenario():
        async def prepare():
            async with gemini._prepare_reference_images(["https://example.invalid/image"]):
                pass

        request = asyncio.create_task(prepare())
        assert await asyncio.to_thread(started.wait, 1)
        assert not request.done()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        release.set()
        assert await asyncio.to_thread(image_created.wait, 1)
        for _ in range(100):
            if images and images[0].closed:
                assert not list(tmp_path.iterdir())
                return
            await asyncio.sleep(0.01)
        assert images[0].closed

    asyncio.run(scenario())
