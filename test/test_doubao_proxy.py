from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from providers.doubao.backend import DoubaoBackendAPI


class FakeResponse:
    status_code = 200

    def __init__(self, body, content=b""):
        self.body = body
        self.content = content

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.closed = False
        self.posts = []

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if url.endswith("/upload_image"):
            return FakeResponse({"code": 0, "data": {"uri": "uploaded-key"}})
        return FakeResponse({"data": {"file_urls": [{"uri": "uploaded-key", "main_url": "https://cdn.example/image"}]}})

    async def close(self):
        self.closed = True


class FakeModelSession(FakeSession):
    async def post(self, url, **kwargs):
        payload = json.loads(kwargs["data"])
        content_type = payload["messages"][0]["content_type"]
        if content_type == 2001:
            content = {"text": "这是一段用于验证豆包文本响应解析的完整内容。"}
        else:
            content = {"data": [{"image_ori": {"url": "https://cdn.example/image.png"}}]}
        event = {"event_data": {"message": {"content_type": 2010, "content": content}}}
        return FakeResponse({}, ("data: " + json.dumps(event) + "\n\n").encode())

    async def get(self, url, **kwargs):
        return FakeResponse({}, b"\x89PNG\r\n\x1a\n")


class DoubaoProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_account_proxy_and_browser_profile_apply_to_session(self) -> None:
        """模型请求、图片上传与下载共用按账号绑定的浏览器会话。"""
        proxy = "socks5h://user:pass@proxy.example:1080"
        backend = DoubaoBackendAPI({"sessionid": "test"}, proxy)
        with patch("providers.doubao.backend.AsyncSession", FakeSession):
            async with backend:
                session = backend.session
                self.assertEqual(session.options["proxy"], proxy)
                self.assertEqual(session.options["impersonate"], "chrome136")
                self.assertEqual(session.options["cookies"]["sessionid"], "test")
                self.assertIn("Chrome/136", session.options["headers"]["User-Agent"])
            self.assertTrue(session.closed)

    async def test_reference_upload_uses_same_browser_session(self) -> None:
        backend = DoubaoBackendAPI({"sessionid": "test"}, "socks5h://proxy.example:1080")
        with patch("providers.doubao.backend.AsyncSession", FakeSession), patch(
            "providers.doubao.backend._prepare_reference_image", return_value=(b"image", "image/png")
        ):
            async with backend:
                session = backend.session
                result = await backend.upload_image("images/reference.png")
        self.assertEqual(result["uri"], "uploaded-key")
        self.assertEqual(len(session.posts), 2)
        self.assertIn("multipart", session.posts[0][1])
        self.assertEqual(session.posts[1][1]["json"]["uris"], ["uploaded-key"])

    async def test_text_and_image_responses_use_chrome_session(self) -> None:
        backend = DoubaoBackendAPI({"sessionid": "test"}, "socks5h://proxy.example:1080")
        with patch("providers.doubao.backend.AsyncSession", FakeModelSession):
            async with backend:
                text = await backend.chat("分析")
                urls = await backend.generate_image("生图")
                images = await backend.download_images(urls)
        self.assertIn("豆包", text)
        self.assertEqual(urls, ["https://cdn.example/image.png"])
        self.assertEqual(images, [b"\x89PNG\r\n\x1a\n"])


if __name__ == "__main__":
    unittest.main()
