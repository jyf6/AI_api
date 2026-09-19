from __future__ import annotations
import urllib.request

import json
import uuid

import aiohttp

from utils.image_binary import image_media_type


class DoubaoBackendAPI:
    BASE_URL = "https://www.doubao.com"
    DEFAULT_DEVICE_ID = "714003710229497"
    DEFAULT_WEB_ID = "7604137868021548590"
    DEFAULT_FP = "verify_mlcfw5f7_TPq0YmFD_NrsC_4RuQ_BJPg_M5W7i58I7wV0"
    DEFAULT_CHROMIUM_VERSION = "135.0.0.0"

    def __init__(self, cookies: dict[str, str], proxy: str = "") -> None:
        if not cookies:
            raise ValueError("Doubao cookies are required")
        self.cookies = cookies
        self.proxy = proxy or None
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        connector = self._build_connector()
        self.session = aiohttp.ClientSession(cookies=self.cookies, connector=connector,
            headers={"User-Agent": "Mozilla/5.0 Chrome/145.0.0.0 Safari/537.36", "Origin": self.BASE_URL,
                     "Referer": self.BASE_URL + "/chat", "Content-Type": "application/json"})
        return self

    def _build_connector(self):
        """账号配置了代理时返回代理连接器，保证所有出站请求走同一代理出口。"""
        if self.proxy:
            from aiohttp_socks import ProxyConnector
            return ProxyConnector.from_url(self.proxy)
        return None

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
        self.session = None

    def _params(self) -> dict[str, str]:
        return {"aid": "582478", "real_aid": "582478", "device_id": self.cookies.get("device_id", self.DEFAULT_DEVICE_ID),
                "tea_uuid": self.cookies.get("device_id", self.DEFAULT_DEVICE_ID), "web_id": self.cookies.get("web_id", self.DEFAULT_WEB_ID),
                "device_platform": "web", "language": "zh", "region": "CN", "sys_region": "CN",
                "pkg_type": "release_version", "version_code": "20800", "pc_version": "2.1.7",
                "chromium_version": self.DEFAULT_CHROMIUM_VERSION, "client_platform": "pc_client", "runtime": "web",
                "runtime_version": "3.5.4", "samantha_web": "1", "use-olympus-account": "1",
                "fp": self.cookies.get("fp", self.cookies.get("s_v_web_id", self.DEFAULT_FP)), "msToken": self.cookies.get("msToken", ""), "web_tab_id": str(uuid.uuid4())}

    async def upload_image(self, image_data: str, filename: str = "image.png") -> dict[str, str]:
        if not self.session:
            raise RuntimeError("Doubao client is not initialized")
        image_bytes = urllib.request.urlopen(image_data, timeout=30).read()
        content_type = image_media_type(image_bytes)
        if content_type is None:
            raise RuntimeError("Doubao reference download did not return a valid image")
        ext = content_type.removeprefix("image/")
        actual_filename = f"image.{ext}"

        form = aiohttp.FormData()
        form.add_field("data", image_bytes, filename=actual_filename, content_type=content_type)
        form.add_field("file_type", ext)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/135.0.0.0 Safari/537.36",
                   "Origin": self.BASE_URL, "Referer": self.BASE_URL + "/chat",
                   "x-tt-passport-csrf-token": self.cookies.get("passport_csrf_token", "")}
        upload_timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(cookies=self.cookies, headers=headers, timeout=upload_timeout,
                                         connector=self._build_connector()) as upload_session:
            async with upload_session.post(self.BASE_URL + "/samantha/pages/upload_image", params=self._params(), data=form) as response:
                if response.status != 200:
                    raise RuntimeError(f"Doubao image upload failed ({response.status})")
                body = await response.json()
                if body.get("code") != 0:
                    raise RuntimeError(f"Doubao image upload error: {body.get('msg', body)}")
                uri = body.get("data", {}).get("uri")
        if not uri:
            raise RuntimeError("Doubao image upload returned no uri")
        async with self.session.post(self.BASE_URL + "/alice/message/get_file_url", params=self._params(),
                                     json={"uris": [uri], "type": "image", "format": ext, "expire_second": 3600}) as response:
            if response.status != 200:
                raise RuntimeError(f"Doubao image URL lookup failed ({response.status})")
            body = await response.json()
            file_urls = (body.get("data") or {}).get("file_urls") or []
        if not file_urls:
            raise RuntimeError("Doubao image URL lookup returned no file")
        item = file_urls[0]
        return {"uri": item.get("uri", uri), "cdn_url": item.get("main_url", ""), "name": filename,
                "format": ext, "width": "0", "height": "0"}

    async def chat(self, text: str, image_attachments: list[dict[str, str]] | None = None) -> str:
        if not self.session:
            raise RuntimeError("Doubao client is not initialized")
        payload = {"messages": [{"content": json.dumps({"text": text}, ensure_ascii=False), "content_type": 2001,
                                  "attachments": [{"type": "image", "key": item["uri"], "extra": {"refer_types": "overall"}}
                                                  for item in (image_attachments or [])], "references": []}],
                   "completion_option": {"is_regen": False, "with_suggest": True, "need_create_conversation": True,
                                          "launch_stage": 1, "is_replace": False, "is_delete": False,
                                          "is_ai_playground": False, "memory_type": 2, "message_from": 0,
                                          "use_deep_think": False, "use_auto_cot": False, "resend_for_regen": False,
                                          "enable_commerce_credit": False}, "evaluate_option": {"web_ab_params": ""},
                   "local_conversation_id": str(uuid.uuid4()), "local_message_id": str(uuid.uuid4())}
        url = self.BASE_URL + "/samantha/chat/completion"
        headers = {"Accept": "text/event-stream", "x-tt-passport-csrf-token": self.cookies.get("passport_csrf_token", "")}
        async with self.session.post(url, params=self._params(), data=json.dumps(payload, ensure_ascii=False), headers=headers) as response:
            if response.status != 200:
                raise RuntimeError(f"Doubao chat failed ({response.status}): {(await response.text())[:300]}")
            raw = (await response.read()).decode("utf-8", errors="replace")
        text_parts: list[str] = []
        for block in raw.split("\n\n"):
            data_line = next((line[5:].strip() for line in block.splitlines() if line.startswith("data:")), "")
            if not data_line:
                continue
            try:
                event = json.loads(data_line)
                event_data = event.get("event_data", {})
                event_data = json.loads(event_data) if isinstance(event_data, str) else event_data
                message = event_data.get("message", {}) if isinstance(event_data, dict) else {}
                content = message.get("content", {})
                content = json.loads(content) if isinstance(content, str) else content
                value = content.get("text", "") if isinstance(content, dict) else ""
                if message.get("content_type") in (2001, 2003, 2008, 2010) and value:
                    text_parts.append(value)
            except (ValueError, TypeError, AttributeError):
                continue
        if not text_parts:
            raise RuntimeError("Doubao chat returned no text")
        return "".join(text_parts)

    async def generate_image(self, prompt: str, ratio: str = "1:1", image_attachments: list[dict[str, str]] | None = None) -> list[str]:
        if not self.session:
            raise RuntimeError("Doubao client is not initialized")
        message = {"content": json.dumps({"text": prompt, "ratio": ratio}, ensure_ascii=False), "content_type": 2009,
                   "attachments": [{"type": "image", "key": item["uri"], "extra": {"refer_types": "overall"}} for item in (image_attachments or [])],
                   "references": [], "skill": {"skill_type": 3, "skill_type_no_default": 3, "skill_id": "3", "skill_id_no_default": "3"}}
        payload = {"messages": [message], "completion_option": {"is_regen": False, "with_suggest": True,
                   "need_create_conversation": True, "launch_stage": 1, "is_ai_playground": False,
                   "memory_type": 2, "message_from": 0, "use_deep_think": False, "use_auto_cot": False,
                   "action_bar_skill_id": 3}, "evaluate_option": {"web_ab_params": ""},
                   "local_conversation_id": str(uuid.uuid4()), "local_message_id": str(uuid.uuid4())}
        url = self.BASE_URL + "/samantha/chat/completion"
        headers = {"Accept": "text/event-stream", "x-tt-passport-csrf-token": self.cookies.get("passport_csrf_token", "")}
        async with self.session.post(url, params=self._params(), data=json.dumps(payload, ensure_ascii=False), headers=headers) as response:
            if response.status != 200:
                raise RuntimeError(f"Doubao image failed ({response.status}): {(await response.text())[:300]}")
            raw = (await response.read()).decode("utf-8", errors="replace")
        urls: list[str] = []
        for block in raw.split("\n\n"):
            data_line = next((line[5:].strip() for line in block.splitlines() if line.startswith("data:")), "")
            try:
                event = json.loads(data_line)
                event_data = event.get("event_data", {})
                event_data = json.loads(event_data) if isinstance(event_data, str) else event_data
                message = event_data.get("message", {})
                content = message.get("content", {})
                content = json.loads(content) if isinstance(content, str) else content
                if message.get("content_type") != 2010:
                    continue
                for item in content.get("data", []):
                    for key in ("image_ori", "image_raw", "image_thumb"):
                        url = (item.get(key) or {}).get("url")
                        if url and url not in urls:
                            urls.append(url)
                            break
            except (ValueError, TypeError, AttributeError):
                continue
        if not urls:
            raise RuntimeError("Doubao image returned no images")
        return urls

    async def download_images(self, urls: list[str]) -> list[bytes]:
        if not self.session:
            raise RuntimeError("Doubao client is not initialized")
        results = []
        for url in urls:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"Doubao image download failed ({response.status})")
                image_bytes = await response.read()
                if image_media_type(image_bytes) is None:
                    raise RuntimeError("Doubao image download did not return a valid image")
                results.append(image_bytes)
        return results

