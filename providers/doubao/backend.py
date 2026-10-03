from __future__ import annotations
import asyncio
import urllib.request

import hashlib
import json
import uuid

import aiohttp

from utils.image_binary import image_media_type


def _prepare_reference_image(image_data: str) -> tuple[bytes, str]:
    """在线程中完成参考图下载和压缩，避免阻塞代理事件循环。"""
    with urllib.request.urlopen(image_data, timeout=30) as response:
        image_bytes = response.read()
    # 大图压缩同样属于同步工作，必须与下载一起移出事件循环。
    if len(image_bytes) > 4 * 1024 * 1024:
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            max_dim = max(img.width, img.height)
            if max_dim > 2048:
                scale = 2048.0 / max_dim
                new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=85, optimize=True)
            image_bytes = buf.getvalue()
        except Exception:
            # 压缩失败沿用原图，保持原有上传行为。
            pass

    content_type = image_media_type(image_bytes)
    if content_type is None:
        raise RuntimeError("Doubao reference download did not return a valid image")
    return image_bytes, content_type


class DoubaoBackendAPI:
    BASE_URL = "https://www.doubao.com"
    DEFAULT_CHROMIUM_VERSION = "135.0.0.0"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"

    def __init__(self, cookies: dict[str, str], proxy: str = "", device_id: str = "", web_id: str = "", fp: str = "") -> None:
        if not cookies:
            raise ValueError("Doubao cookies are required")
        self.cookies = cookies
        self.proxy = proxy or None
        self.session: aiohttp.ClientSession | None = None
        # 账号专有设备指纹初始化（消除全局硬编码，实现多账号设备隔离）
        self.device_id = device_id or self._init_device_id()
        self.web_id = web_id or self._init_web_id()
        self.fp = fp or self._init_fp()

    def _init_device_id(self) -> str:
        """优先使用 Cookie 中的 device_id；若缺失，则基于账号 sessionid/uid 生成专属唯一的 19 位设备 ID，隔离多账号风控。"""
        if self.cookies.get("device_id"):
            return str(self.cookies["device_id"])
        seed = self.cookies.get("sessionid") or self.cookies.get("uid_tt") or self.cookies.get("sid_tt") or "default_device"
        num = int(hashlib.md5(f"doubao_dev_{seed}".encode()).hexdigest()[:15], 16)
        return str(7000000000000000000 + (num % 900000000000000000))

    def _init_web_id(self) -> str:
        """优先使用 Cookie 中的 web_id；若缺失，则基于账号唯一特征生成专属唯一的 19 位 Web 实例 ID。"""
        if self.cookies.get("web_id"):
            return str(self.cookies["web_id"])
        seed = self.cookies.get("sessionid") or self.cookies.get("uid_tt") or self.cookies.get("sid_tt") or "default_web"
        num = int(hashlib.md5(f"doubao_web_{seed}".encode()).hexdigest()[:15], 16)
        return str(7600000000000000000 + (num % 90000000000000000))

    def _init_fp(self) -> str:
        """优先使用 Cookie 中的 fp 或 s_v_web_id；若缺失，则基于账号 Session 生成合规的专属指纹格式。"""
        if self.cookies.get("fp"):
            return str(self.cookies["fp"])
        if self.cookies.get("s_v_web_id"):
            return str(self.cookies["s_v_web_id"])
        seed = self.cookies.get("sessionid") or "default_fp"
        h = hashlib.md5(f"doubao_fp_{seed}".encode()).hexdigest()
        return f"verify_m{h[:7]}_{h[7:15]}_{h[15:19]}_4RuQ_BJPg_M5W7i58I7wV0"

    async def __aenter__(self):
        connector = self._build_connector()
        self.session = aiohttp.ClientSession(cookies=self.cookies, connector=connector,
            headers={"User-Agent": self.USER_AGENT, "Origin": self.BASE_URL,
                     "Referer": self.BASE_URL + "/chat", "Content-Type": "application/json"})
        return self

    def _build_connector(self):
        """账号配置了代理时返回代理连接器，保证所有出站请求走同一代理出口。"""
        if self.proxy:
            from aiohttp_socks import ProxyConnector
            if self.proxy.lower().startswith("socks5h://"):
                # aiohttp-socks uses socks5:// plus rdns=True for remote DNS.
                proxy_url = "socks5://" + self.proxy[len("socks5h://"):]
                return ProxyConnector.from_url(proxy_url, rdns=True)
            return ProxyConnector.from_url(self.proxy)
        return None

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
        self.session = None

    def _params(self) -> dict[str, str]:
        return {
            "aid": "582478",
            "real_aid": "582478",
            "device_id": self.device_id,
            "tea_uuid": self.device_id,
            "web_id": self.web_id,
            "device_platform": "web",
            "language": "zh",
            "region": "CN",
            "sys_region": "CN",
            "pkg_type": "release_version",
            "version_code": "20800",
            "pc_version": "2.1.7",
            "chromium_version": self.DEFAULT_CHROMIUM_VERSION,
            "client_platform": "pc_client",
            "runtime": "web",
            "runtime_version": "3.5.4",
            "samantha_web": "1",
            "use-olympus-account": "1",
            "fp": self.fp,
            "msToken": self.cookies.get("msToken", ""),
            "web_tab_id": str(uuid.uuid4()),
        }

    async def upload_image(self, image_data: str, filename: str = "image.png") -> dict[str, str]:
        if not self.session:
            raise RuntimeError("Doubao client is not initialized")
        image_bytes, content_type = await asyncio.to_thread(_prepare_reference_image, image_data)
        ext = content_type.removeprefix("image/")
        actual_filename = f"image.{ext}"

        form = aiohttp.FormData()
        form.add_field("data", image_bytes, filename=actual_filename, content_type=content_type)
        form.add_field("file_type", ext)
        headers = {"User-Agent": self.USER_AGENT,
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

