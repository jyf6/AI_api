from __future__ import annotations
from core.admission import mark_model_request_started
from core.model_timeout import MODEL_TIMEOUT_SECONDS
from core.slot_sessions import ConversationUnavailable

import asyncio
import contextlib
import copy
import io
import json
from typing import Any


from curl_cffi.requests import AsyncSession

from utils.log import error_http_status, image_stage, logger, stable_log_ref
from utils.image_binary import image_media_type
from utils.oss_reference import read_oss_reference
from utils.helper import is_transport_error
from providers.gemini.webapi.exceptions import AuthError
from providers.gemini.webapi.constants import GRPC
from providers.gemini.webapi.types import RPCData
from providers.gemini.webapi.utils import extract_json_from_response, get_nested_value


def _required_model(client: Any, name: str) -> Any:
    """按账号真实清单精确选型，禁止版本模糊匹配或回退到默认模型。"""
    for model in client.list_models() or []:
        names = {model.model_name.lower(), model.display_name.lower(),
                 *(alias.lower() for alias in model.aliases)}
        if name.lower() not in names:
            continue
        if not model.is_available:
            raise ValueError(f"Gemini 模型 {name} 当前不可用")
        logger.info("event=gemini_model_selected requested_model=%s model=%s model_id=%s extended_thinking=true",
                    name, model.display_name, model.model_id)
        return model
    raise ValueError(f"Gemini 账号模型清单中没有 {name}，不使用其他模型替代")


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

    @contextlib.asynccontextmanager
    async def stream(method: str, url: Any, *args: Any, **kwargs: Any) -> Any:
        aspect_ratio = getattr(session, "_gemini_aspect_ratio", None)
        if aspect_ratio and str(url).endswith("/StreamGenerate"):
            data = kwargs.get("data")
            if isinstance(data, dict):
                kwargs["data"] = _inject_aspect_ratio(data, aspect_ratio)
        if str(url).endswith("/StreamGenerate"):
            before_send = getattr(session, "_gemini_before_model_send", None)
            if before_send is not None:
                await before_send()
            await mark_model_request_started()
        # 让上游正常完成会话保存；不能在拿到图片时主动中止仍在收尾的传输。
        original_error = None
        try:
            async with original_stream(method, url, *args, **kwargs) as response:
                try:
                    yield response
                except BaseException as exc:
                    original_error = exc
                    # 只在失败/外部取消时停止仍有效的传输，不重复关闭整个客户端。
                    if not isinstance(exc, GeneratorExit) and not getattr(session, "_closed", False):
                        task = getattr(response, "astream_task", None)
                        curl = getattr(response, "curl", None)
                        if task is not None and not task.done() and curl is not None and curl._curl is not None:
                            session.acurl.remove_handle(curl)
                    raise
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling() or original_error is None:
                raise
            # 关闭失败连接引起的底层取消不能覆盖原始上游异常。
            raise original_error

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
    """使用统一的 OSS objectKey 读取私有参考图，读取 OSS 已处理且小于 3MB 的图片。"""
    raw = read_oss_reference(source)
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
        with image_stage("gemini", "reference_prepare"):
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

    def __init__(self, account: dict[str, Any], before_model_send=None) -> None:
        self.account = account
        self.client = None
        self._request_generation = 0
        self.before_model_send = before_model_send
        self.model_request_started = False

    async def _before_actual_send(self) -> None:
        """实际StreamGenerate前串行领取账号发送许可，上传不占发送时间点。"""
        if self.before_model_send is not None:
            await self.before_model_send()
        self.model_request_started = True

    async def __aenter__(self):
        from providers.gemini.account import gemini_account_service
        try:
            self.client, self._request_generation = await gemini_account_service.get_request_client(self.account)
        except Exception as exc:
            # 客户端初始化尚未生图，明确认证失败统一交给调用侧换号。
            if (not isinstance(exc, AuthError) and not is_transport_error(exc)
                    and gemini_account_service.is_auth_error(exc)):
                raise AuthError("Gemini account requires authentication") from exc
            raise
        _install_aspect_ratio_stream_hook(self.client.client)
        self.client.client._gemini_before_model_send = self._before_actual_send
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
        logger.warning("event=account_reconnect_started platform=gemini account_ref=%s",
                       stable_log_ref("gemini-account", self.account.get("name")))
        await gemini_account_service.release_request_client(
            self.account["name"], self.client, self._request_generation, False,
        )
        await gemini_account_service.discard_client(self.account["name"])
        self.client, self._request_generation = await gemini_account_service.get_request_client(self.account)
        _install_aspect_ratio_stream_hook(self.client.client)
        self.client.client._gemini_before_model_send = self._before_actual_send

    async def _do_chat(self, prompt: str, images: list[str] | None, model: str) -> str:
        image_sources = images or []
        # 所有 Gemini 分析固定为 3.8 Flash 高思考；传模型对象绕过客户端的模糊名称解析。
        client_model = _required_model(self.client, "gemini-3.8-flash")
        async with _prepare_reference_images(image_sources) as images:
            stream = self.client.generate_content_stream(
                prompt, files=images or None, model=client_model, extended_thinking=True
            )
            text = ""
            loop = asyncio.get_running_loop()
            deadline = loop.time() + MODEL_TIMEOUT_SECONDS
            try:
                while True:
                    # 正文、思考和重复帧均使用同一四分钟期限，不再十秒停顿就返回部分结果。
                    try:
                        output = await asyncio.wait_for(anext(stream), timeout=max(0, deadline - loop.time()))
                    except TimeoutError:
                        raise RuntimeError(f"Gemini analysis timed out after {MODEL_TIMEOUT_SECONDS} seconds") from None

                    current_text = output.candidates[output.chosen].text if output.candidates else ""
                    if current_text and current_text != text:
                        text = current_text
                    # 明确完成优先于时间兜底；完成帧可以不带新增文本。
                    if output.is_completed:
                        if text.strip():
                            return text.strip()
                        raise RuntimeError("Gemini returned no assistant text")
            except StopAsyncIteration:
                if text.strip():
                    return text.strip()
                raise RuntimeError("Gemini returned no assistant text")
            finally:
                # 完成、超时或失败都释放本次流；异常不再伪装成完整分析结果。
                await stream.aclose()

    async def chat(self, prompt: str, images: list[str] | None = None, model: str = "auto") -> str:
        try:
            return await self._do_chat(prompt, images, model)
        except Exception as exc:
            err_str = str(exc).lower()
            if not self.model_request_started and any(w in err_str for w in ("auth", "401", "unauthorized", "cookie", "expired")):
                await self._reconnect()
                return await self._do_chat(prompt, images, model)
            raise

    async def _do_image(
        self,
        prompt: str,
        model: str,
        references: list[str] | None = None,
        aspect_ratio: str | None = None,
        *, conversation_ref: dict | None = None, on_conversation=None,
    ) -> list[bytes]:
        # 初版、重刷及单张/整套调整均固定使用账号实际提供的 Pro 高思考。
        client_model = _required_model(self.client, "gemini-pro")
        chat = None
        if conversation_ref is not None:
            # 只在正式生图前确认旧会话；明确不存在才允许调用侧在原账号新建。
            response = await self.client._batch_execute([RPCData(
                rpcid=GRPC.LIST_CONVERSATION_TURNS,
                payload=json.dumps([conversation_ref["cid"], 1, None, 1, [1], [4], None, 1]),
            )])
            for part in extract_json_from_response(response.text):
                if (get_nested_value(part, [1]) == GRPC.LIST_CONVERSATION_TURNS
                        and get_nested_value(part, [5, 0]) == 5):
                    raise ConversationUnavailable("Previous Gemini conversation is no longer accessible")
        if conversation_ref is not None or on_conversation is not None:
            chat = self.client.start_chat(model=client_model,
                metadata=copy.deepcopy(conversation_ref["metadata"]) if conversation_ref else None)
        _configure_image_aspect_ratio(self.client, aspect_ratio)
        try:
            async with _prepare_reference_images(references or []) as images:
                generated = []
                with image_stage("gemini", "generate"):
                    stream = self.client.generate_content_stream(
                        prompt, files=images or None, model=client_model, extended_thinking=True, chat=chat
                    )
                    completed = False
                    try:
                        async for output in stream:
                            if chat is not None:
                                # 收集状态但不提前发布绑定；取得图片不代表会话已完成保存。
                                chat.metadata = output.metadata
                                if output.candidates:
                                    chat.rcid = output.rcid
                            found = list(
                                output.candidates[output.chosen].generated_images
                                if output.candidates else []
                            )
                            # 最终文本帧可能不再带图片，保留本轮已取得的图片资源。
                            if found:
                                generated = found
                        completed = True
                    except Exception as exc:
                        if not generated:
                            raise
                        # 已有本次图片资源则继续取图，不为保存会话重发或丢弃图片。
                        logger.warning("event=gemini_session_unconfirmed reason_code=%s", type(exc).__name__)
                    finally:
                        await stream.aclose()
                    if completed and chat is not None and on_conversation is not None:
                        await on_conversation({"cid": chat.cid, "metadata": copy.deepcopy(chat.metadata)})
                    if not generated:
                        raise RuntimeError("Gemini returned no image")
                logger.info("event=image_assets_found platform=gemini count=%d", len(generated))
                with image_stage("gemini", "image_download"):
                    downloaded = [
                        await _download_generated_image(image, self.client, self.account.get("proxy") or None)
                        for image in generated
                    ]
                logger.info("event=image_downloaded platform=gemini count=%d", len(downloaded))
                return downloaded
        finally:
            # 账号客户端会复用，必须在本次调用结束后清除画幅，防止泄漏到聊天请求。
            _clear_image_aspect_ratio(self.client)

    async def image(
        self,
        prompt: str,
        model: str,
        references: list[str] | None = None,
        aspect_ratio: str | None = None,
        *, conversation_ref: dict | None = None, on_conversation=None,
    ) -> list[bytes]:
        try:
            return await self._do_image(prompt, model, references, aspect_ratio,
                                        conversation_ref=conversation_ref, on_conversation=on_conversation)
        except Exception as exc:
            err_str = str(exc).lower()
            if not self.model_request_started and any(w in err_str for w in ("auth", "401", "unauthorized", "cookie", "expired")):
                await self._reconnect()
                return await self._do_image(prompt, model, references, aspect_ratio,
                                            conversation_ref=conversation_ref, on_conversation=on_conversation)
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
            logger.info("event=image_download_fallback platform=gemini from=rpc_url to=cdn reason_code=%s",
                        type(exc).__name__)

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
                        logger.info("event=image_download_source platform=gemini source=rpc bytes=%d", len(response.content))
                        return response.content
            except Exception as exc:
                logger.warning("event=image_download_fallback platform=gemini from=rpc to=cdn reason_code=%s http_status=%s",
                               type(exc).__name__, error_http_status(exc))

        # 4. 降级方案：走 Google CDN URL 下载
        if cdn_url:
            try:
                response = await session.get(cdn_url, headers=headers)
                response.raise_for_status()
                if image_media_type(response.content) is not None:
                    logger.info("event=image_download_source platform=gemini source=cdn bytes=%d", len(response.content))
                    return response.content
            except Exception as exc:
                logger.warning("event=image_download_failed platform=gemini source=cdn reason_code=%s http_status=%s",
                               type(exc).__name__, error_http_status(exc))

    raise RuntimeError("Gemini image download did not return a valid image")
