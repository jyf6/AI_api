from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from pathlib import Path
import tempfile
from typing import Any
import urllib.request

from curl_cffi.requests import AsyncSession

from utils.log import logger
from utils.image_binary import image_media_type


def _inject_aspect_ratio(request_data: dict[str, Any], aspect_ratio: str) -> dict[str, Any]:
    """将网页端画幅选项写入 Gemini StreamGenerate 的 f.req 请求体。"""
    payload = request_data.get("f.req")
    if not isinstance(payload, str):
        return request_data

    try:
        outer_payload = json.loads(payload)
        inner_payload = json.loads(outer_payload[1])
        message_content = inner_payload[0]
    except (IndexError, TypeError, json.JSONDecodeError):
        logger.warning("Gemini StreamGenerate 请求体格式变化，未写入图片比例")
        return request_data

    # Gemini 网页端将画幅放在消息内容的第 10 个元素，而不是顶层 JSON 字段。
    while len(message_content) <= 9:
        message_content.append(None)
    message_content[9] = [None, None, None, None, None, None, [None, [None, aspect_ratio]]]

    updated_data = dict(request_data)
    updated_data["f.req"] = json.dumps(
        [outer_payload[0], json.dumps(inner_payload, separators=(",", ":"), ensure_ascii=False)],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return updated_data


def _install_aspect_ratio_stream_hook(session: Any) -> None:
    """在原始 AsyncSession 实例上挂钩 stream，保持其类型不变。"""
    if getattr(session, "_gemini_aspect_ratio_hooked", False):
        return

    original_stream = session.stream

    def stream(method: str, url: Any, *args: Any, **kwargs: Any) -> Any:
        aspect_ratio = getattr(session, "_gemini_aspect_ratio", None)
        if aspect_ratio and str(url).endswith("/StreamGenerate"):
            data = kwargs.get("data")
            if isinstance(data, dict):
                kwargs["data"] = _inject_aspect_ratio(data, aspect_ratio)
        return original_stream(method, url, *args, **kwargs)

    # AsyncSession 允许实例属性覆盖；GeneratedImage 仍会收到真实 AsyncSession。
    session.stream = stream
    session._gemini_aspect_ratio_hooked = True
    session._gemini_aspect_ratio = None


def _configure_image_aspect_ratio(client: Any, aspect_ratio: str | None) -> None:
    """为当前账号客户端设置一次性生图画幅，避免修改第三方依赖源码。"""
    session = getattr(client, "client", None)
    if session is None:
        return
    _install_aspect_ratio_stream_hook(session)
    session._gemini_aspect_ratio = aspect_ratio


def _clear_image_aspect_ratio(client: Any) -> None:
    session = getattr(client, "client", None)
    if session is not None and getattr(session, "_gemini_aspect_ratio_hooked", False):
        session._gemini_aspect_ratio = None


def _detect_image_suffix(data: bytes) -> str:
    """Detect image file extension based on magic bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _read_image_source(source: str) -> tuple[bytes, str]:
    """Read image bytes from data-url or http-url and determine file extension."""
    if source.startswith("data:"):
        header_part, data_part = source.split(",", 1)
        raw = base64.b64decode(data_part)
        header_lower = header_part.lower()
        if "image/jpeg" in header_lower or "image/jpg" in header_lower:
            ext = ".jpg"
        elif "image/webp" in header_lower:
            ext = ".webp"
        elif "image/gif" in header_lower:
            ext = ".gif"
        elif "image/png" in header_lower:
            ext = ".png"
        else:
            ext = _detect_image_suffix(raw[:16])
        if image_media_type(raw) is None:
            raise RuntimeError("Gemini reference download did not return a valid image")
        return raw, ext

    req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    if image_media_type(raw) is None:
        raise RuntimeError("Gemini reference download did not return a valid image")
    ext = _detect_image_suffix(raw[:16])
    return raw, ext


@contextlib.contextmanager
def _create_temp_image_files(sources: list[str]):
    """Write input image sources to temp files with valid image extensions so gemini_webapi can upload them properly."""
    temp_paths: list[Path] = []
    try:
        for src in sources:
            if not src:
                continue
            raw_bytes, ext = _read_image_source(str(src))
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tf.write(raw_bytes)
            tf.flush()
            tf.close()
            temp_paths.append(Path(tf.name))
        yield temp_paths
    finally:
        for p in temp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


class GeminiBackendAPI:
    """Gemini Web reverse client; no browser process and no persistent image files."""

    def __init__(self, account: dict[str, Any]) -> None:
        self.account = account
        self.client = None

    async def __aenter__(self):
        from providers.gemini.account import gemini_account_service
        self.client = await gemini_account_service.get_client(self.account)
        return self

    async def __aexit__(self, *_args):
        if self.client:
            from providers.gemini.account import gemini_account_service
            gemini_account_service.merge_cookie(self.account["name"], dict(self.client.cookies))
            # The client is account-scoped; its auto-refresh task must survive this request.
        self.client = None

    async def _reconnect(self) -> None:
        """Discard the potentially invalidated client and re-initialize with fresh/cached cookies."""
        from providers.gemini.account import gemini_account_service
        logger.warning(
            f"[Gemini Self-Healing] 账号 [{self.account.get('name')}] 遇到疑似认证失效，触发重连自愈..."
        )
        await gemini_account_service.discard_client(self.account["name"])
        self.client = await gemini_account_service.get_client(self.account)

    async def _do_chat(self, prompt: str, images: list[str] | None, model: str) -> str:
        image_sources = images or []
        with _create_temp_image_files(image_sources) as temp_files:
            stream = self.client.generate_content_stream(prompt, files=temp_files or None, model=model)
            text = ""
            try:
                while True:
                    # 首段文本最多等待 20 秒；已有文本后，空闲 5 秒即使用现有结果返回。
                    try:
                        output = await asyncio.wait_for(anext(stream), timeout=5 if text else 20)
                    except TimeoutError:
                        if text:
                            return text.strip()
                        raise RuntimeError("Gemini did not return text within 20 seconds")

                    current_text = output.candidates[output.chosen].text if output.candidates else ""
                    text = current_text or getattr(output, "text", "") or text
            except StopAsyncIteration:
                if text.strip():
                    return text.strip()
                raise RuntimeError("Gemini returned no assistant text")
            except Exception as exc:
                # 已获得有效分析文本后，流连接的收尾异常不应覆盖已有结果并触发上游重放。
                if text.strip():
                    logger.warning(f"Gemini analysis stream ended after text was received: {exc}")
                    return text.strip()
                raise
            finally:
                # 文本结果可用后不再等待网页端的最终完成标记，主动结束本次流。
                await stream.aclose()

    async def chat(self, prompt: str, images: list[str] | None = None, model: str = "gemini-1.5-pro") -> str:
        try:
            return await self._do_chat(prompt, images, model)
        except Exception as exc:
            err_str = str(exc).lower()
            if any(w in err_str for w in ("auth", "401", "unauthorized", "cookie", "expired", "stream suspended")):
                await self._reconnect()
                return await self._do_chat(prompt, images, model)
            raise

    async def _do_image(
        self,
        prompt: str,
        model: str,
        references: list[str] | None = None,
        aspect_ratio: str | None = None,
    ) -> list[bytes]:
        _configure_image_aspect_ratio(self.client, aspect_ratio)
        try:
            with _create_temp_image_files(references or []) as temp_files:
                stream = self.client.generate_content_stream(
                    prompt, files=temp_files or None, model=model
                )
                try:
                    async for output in stream:
                        images = list(
                            output.candidates[output.chosen].generated_images
                            if output.candidates
                            else []
                        )
                        if images:
                            return [
                                await _download_generated_image(
                                    image, self.client, self.account.get("proxy") or None
                                )
                                for image in images
                            ]
                finally:
                    # 图片候选已到达即可返回，主动关闭流避免继续等待文本完成标记。
                    await stream.aclose()
                raise RuntimeError("Gemini returned no image")
        finally:
            # 账号客户端会复用，必须在本次调用结束后清除画幅，防止泄漏到聊天请求。
            _clear_image_aspect_ratio(self.client)

    async def image(
        self,
        prompt: str,
        model: str,
        references: list[str] | None = None,
        aspect_ratio: str | None = None,
    ) -> list[bytes]:
        try:
            return await self._do_image(prompt, model, references, aspect_ratio)
        except Exception as exc:
            err_str = str(exc).lower()
            if any(w in err_str for w in ("auth", "401", "unauthorized", "cookie", "expired", "stream suspended")):
                await self._reconnect()
                return await self._do_image(prompt, model, references, aspect_ratio)
            raise


async def _download_generated_image(image: Any, client: Any, proxy: str | None) -> bytes:
    """Download into memory; follows multi-hop redirects and ensures valid binary image."""
    url = getattr(image, "url", "")
    if all(getattr(image, key, "") for key in ("cid", "rid", "rcid", "image_id")):
        try:
            full_size_url = await client._get_full_size_image(
                cid=image.cid, rid=image.rid, rcid=image.rcid, image_id=image.image_id
            )
            if full_size_url:
                url = full_size_url + "=d-I?alr=yes"
        except Exception as exc:
            logger.debug(f"Failed to get full size URL via RPC: {exc}")

    if not url:
        raise RuntimeError("Gemini returned an image without a download URL")

    async with AsyncSession(impersonate="chrome145", cookies=client.cookies, proxy=proxy) as session:
        current_url = url
        for _ in range(5):
            response = await session.get(current_url, headers={"Referer": "https://gemini.google.com/"})
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()

            if content_type.startswith("text/") or content_type.startswith("application/json"):
                next_url = response.text.strip()
                if next_url.startswith("http://") or next_url.startswith("https://"):
                    current_url = next_url
                    continue
                break

            if image_media_type(response.content) is not None:
                return response.content

        fallback_url = getattr(image, "url", "")
        if fallback_url and fallback_url != current_url:
            response = await session.get(fallback_url, headers={"Referer": "https://gemini.google.com/"})
            response.raise_for_status()
            if image_media_type(response.content) is not None:
                return response.content

        raise RuntimeError("Gemini image download did not return a valid image")
