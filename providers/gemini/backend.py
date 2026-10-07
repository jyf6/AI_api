from __future__ import annotations
from core.admission import mark_model_request_started

import asyncio
import contextlib
import io
import json
from typing import Any


from curl_cffi.requests import AsyncSession

from utils.log import logger
from utils.image_binary import image_media_type
from utils.oss_reference import read_oss_reference


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
    """使用统一的 OSS objectKey 读取私有参考图，并对超过 1MB 的大图转 JPEG 压缩以降低跨境上传耗时。"""
    raw = read_oss_reference(source)
    if len(raw) > 1024 * 1024:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(raw))
            max_dim = max(img.width, img.height)
            if max_dim > 2048:
                scale = 2048.0 / max_dim
                img = img.resize(
                    (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            raw = buf.getvalue()
        except Exception:
            pass
    if image_media_type(raw) is None:
        raise RuntimeError("Gemini reference download did not return a valid image")
    ext = _detect_image_suffix(raw[:16])
    return raw, ext


@contextlib.contextmanager
def _create_reference_images(sources: list[str]):
    """Keep reference images in memory with names that preserve their media types."""
    images: list[io.BytesIO] = []
    try:
        valid_sources = [str(src) for src in sources if src]
        # 多张参考图并发从 OSS 拉取并压缩，避免单线程串行拉取耗时过长
        if len(valid_sources) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(8, len(valid_sources))) as pool:
                loaded = list(pool.map(_read_image_source, valid_sources))
        else:
            loaded = [_read_image_source(src) for src in valid_sources]
        for raw_bytes, ext in loaded:
            image = io.BytesIO(raw_bytes)
            image.name = f"reference_{len(images) + 1}{ext}"
            images.append(image)
        yield images
    finally:
        for image in images:
            image.close()


@contextlib.asynccontextmanager
async def _prepare_reference_images(sources: list[str]):
    """在线程中读取参考图；取消等待后仍释放已读取的内存。"""
    files = _create_reference_images(sources)
    preparation = asyncio.create_task(asyncio.to_thread(files.__enter__))
    try:
        images = await asyncio.shield(preparation)
    except asyncio.CancelledError:
        # 线程不会随请求取消而停止，完成后关闭它创建的内存对象。
        def cleanup(done: asyncio.Task) -> None:
            try:
                done.result()
            except Exception:
                return
            asyncio.create_task(asyncio.to_thread(files.__exit__, None, None, None))

        preparation.add_done_callback(cleanup)
        raise
    try:
        yield images
    finally:
        # 即使调用上游期间取消请求，清理线程仍会完成。
        await asyncio.shield(asyncio.to_thread(files.__exit__, None, None, None))


class GeminiBackendAPI:
    """Gemini Web reverse client; no browser process and no persistent image files."""

    def __init__(self, account: dict[str, Any]) -> None:
        self.account = account
        self.client = None
        self._request_generation = 0

    async def __aenter__(self):
        from providers.gemini.account import gemini_account_service
        self.client, self._request_generation = await gemini_account_service.get_request_client(self.account)
        return self

    async def __aexit__(self, exc_type, *_args):
        if self.client:
            from providers.gemini.account import gemini_account_service
            await gemini_account_service.release_request_client(
                self.account["name"], self.client, self._request_generation, exc_type is None,
            )
        self.client = None

    async def _reconnect(self) -> None:
        """Discard the potentially invalidated client and re-initialize with fresh/cached cookies."""
        from providers.gemini.account import gemini_account_service
        logger.warning(
            f"[Gemini Self-Healing] 账号 [{self.account.get('name')}] 遇到疑似认证失效，触发重连自愈..."
        )
        await gemini_account_service.release_request_client(
            self.account["name"], self.client, self._request_generation, False,
        )
        await gemini_account_service.discard_client(self.account["name"])
        self.client, self._request_generation = await gemini_account_service.get_request_client(self.account)

    async def _do_chat(self, prompt: str, images: list[str] | None, model: str) -> str:
        image_sources = images or []
        client_model = None if model == "auto" else model
        async with _prepare_reference_images(image_sources) as images:
            await mark_model_request_started()
            stream = self.client.generate_content_stream(prompt, files=images or None, model=client_model)
            text = ""
            try:
                while True:
                    # 首段文本包含多图上传与模型预填充，最多等待 120 秒；已有文本后，空闲 6 秒即使用现有结果返回。
                    try:
                        output = await asyncio.wait_for(anext(stream), timeout=6 if text else 120)
                    except TimeoutError:
                        if text:
                            return text.strip()
                        raise RuntimeError("Gemini did not return text within 120 seconds") from None

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

    async def chat(self, prompt: str, images: list[str] | None = None, model: str = "auto") -> str:
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
        client_model = None if model == "auto" else model
        try:
            async with _prepare_reference_images(references or []) as images:
                await mark_model_request_started()
                stream = self.client.generate_content_stream(
                    prompt, files=images or None, model=client_model
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
    """下载生成的图片二进制；支持 RPC 高清直链优先，并在遭遇 403 等异常时自动平滑降级至 Google CDN。"""
    headers = {
        "Origin": "https://gemini.google.com",
        "Referer": "https://gemini.google.com/",
    }

    # 1. 优先尝试获取 RPC 高清全尺寸下载直链
    rpc_url = ""
    if all(getattr(image, key, "") for key in ("cid", "rid", "rcid", "image_id")):
        try:
            full_size_url = await client._get_full_size_image(
                cid=image.cid, rid=image.rid, rcid=image.rcid, image_id=image.image_id
            )
            if full_size_url:
                rpc_url = full_size_url + "=d-I?alr=yes"
        except Exception as exc:
            logger.debug(f"Failed to get full size URL via RPC: {exc}")

    # 2. 准备官方 CDN 降级兜底直链（替换/增加 =s2048-rj 获得高清画质，永不 403）
    cdn_url = getattr(image, "url", "")
    if "=s1024-rj" in cdn_url:
        cdn_url = cdn_url.replace("=s1024-rj", "=s2048-rj")
    elif cdn_url and "=s2048-rj" not in cdn_url:
        cdn_url += "=s2048-rj"

    async with AsyncSession(impersonate="chrome145", cookies=client.cookies, proxy=proxy) as session:
        # 3. 优先走 RPC 链接下载
        if rpc_url:
            try:
                current_url = rpc_url
                for _ in range(5):
                    response = await session.get(current_url, headers=headers)
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
            except Exception as exc:
                logger.warning(f"RPC full size image download failed ({exc}), falling back to Google CDN URL.")

        # 4. 降级方案：走 Google CDN URL 下载
        if cdn_url:
            try:
                response = await session.get(cdn_url, headers=headers)
                response.raise_for_status()
                if image_media_type(response.content) is not None:
                    return response.content
            except Exception as exc:
                logger.warning(f"Google CDN image download failed: {exc}")

    raise RuntimeError("Gemini image download did not return a valid image")
