from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException, Request, Response

from api.schemas import ImageGenerationRequest
from core.router import resolve_model
from core.model_timeout import MODEL_TIMEOUT_SECONDS
from core.admission import CapacityUnavailable
from core.slot_sessions import CURRENT_SESSION, ConversationUnavailable, slot_sessions
from core.account_pool import AccountUnavailable, BoundAccountCooldownError
from api.model_execution import acquire_traced_account, execute_model_request
from providers.doubao.account import doubao_account_service
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.account import gemini_account_service
from providers.gemini.backend import GeminiBackendAPI
from providers.openai.account import account_service
from providers.openai.backend import OpenAIBackendAPI
from providers.openai.credentials import CredentialUnavailable
from providers.gemini.webapi.exceptions import AuthError, ModelInvalidError, TemporarilyBlockedError, UsageLimitExceededError
from utils.helper import (ImageDownloadError, ImageQuotaExceededError, QuotaExhaustedError,
                          UpstreamHTTPError, detect_quota_exhausted,
                          is_connect_error, is_transport_error, is_read_timeout_error)
from utils.image_binary import image_media_type
from utils.image_ratio import build_ratio_prompt
from utils.log import error_http_status, image_stage, logger, proxy_log_ref, python_attempt_log_context, stable_log_ref

router = APIRouter(tags=["images"])


class ImageResultError(RuntimeError):
    def __init__(self, reason_code: str, message: str):
        self.reason_code = reason_code
        super().__init__(message)


def _failure_scope(exc: Exception) -> str:
    if isinstance(exc, ImageResultError):
        return "invalid_output"
    if is_read_timeout_error(exc) or isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "timeout"
    return "transport" if is_transport_error(exc) else "non_transport"


def _checked_image(images, platform: str) -> bytes:
    count = len(images) if isinstance(images, list) else 1
    if not count:
        logger.warning("event=image_result_checked platform=%s outcome=failed expected_count=1 upstream_count=0 returned_count=0 reason_code=IMAGE_COUNT_SHORT", platform)
        raise ImageResultError("IMAGE_COUNT_SHORT", "Upstream returned no image")
    image = images[0] if isinstance(images, list) else images
    if not isinstance(image, (bytes, bytearray)) or image_media_type(image) is None:
        logger.warning("event=image_result_checked platform=%s outcome=failed expected_count=1 upstream_count=%d returned_count=0 reason_code=INVALID_IMAGE", platform, count)
        raise ImageResultError("INVALID_IMAGE", "Upstream did not return a valid image binary")
    logger.info("event=image_result_checked platform=%s outcome=success expected_count=1 upstream_count=%d returned_count=1 bytes=%d", platform, count, len(image))
    return image


def _cooldown_http_error(exc: BoundAccountCooldownError, platform: str) -> HTTPException:
    import json
    err_msg = (exc.error_message or "").lower()
    is_network = err_msg == "network_connect_failed"
    is_quota = exc.remaining_seconds > 300 or any(
        kw in err_msg for kw in ("quota", "upload_protection", "上限", "额度")
    )
    if is_quota:
        wait_sec = exc.remaining_seconds
        wait_min = max(1, wait_sec // 60)
        detail = {
            "code": "ACCOUNT_QUOTA_EXHAUSTED",
            "retry_after": wait_sec,
            "retryAfter": wait_sec,
            "message": f"当前账号额度已耗尽，预计将在 {wait_min} 分钟后恢复。可等待恢复或确认切换账号。"
        }
        code = "ACCOUNT_QUOTA_EXHAUSTED"
    elif is_network:
        wait_sec = exc.remaining_seconds
        detail = {"code": "ACCOUNT_NETWORK_COOLDOWN", "retry_after": wait_sec,
                  "retryAfter": wait_sec, "message": f"账号网络连接失败，已冷却 5 秒，请稍后重试或确认切换账号。"}
        code = "ACCOUNT_NETWORK_COOLDOWN"
    else:
        wait_sec = exc.remaining_seconds
        detail = {
            "code": "ACCOUNT_RATE_LIMITED",
            "retry_after": wait_sec,
            "retryAfter": wait_sec,
            "message": f"当前账号限流，请等待冷却结束或确认切换账号（剩余 {wait_sec} 秒）。"
        }
        code = "ACCOUNT_RATE_LIMITED"
    return HTTPException(
        status_code=429,
        detail=json.dumps({**detail, "code": code, "platform": platform, "reason": code,
                           "userActions": ["WAIT", "SWITCH_ACCOUNT"],
                           "cooldownUntil": int(exc.cooldown_until), "upstreamState": "NOT_SENT"}, ensure_ascii=False),
        headers={"Retry-After": str(wait_sec)}
    )


async def _acquire_image_account(service, platform: str, key_field: str, attempt: int):
    """有槽位时优先指定账号，忙碌只等待，需人工恢复才切换。"""
    session = CURRENT_SESSION.get()
    while True:
        preferred = session.account_id if session else None
        try:
            if session is None:
                return await acquire_traced_account(service, "image", platform, key_field, attempt)
            account = await acquire_traced_account(service, "image", platform, key_field, attempt,
                account_id=preferred, excluded_account_ids=session.excluded_account_ids if session else None)
            session.account_hint = account["account_id"]
            return account
        except BoundAccountCooldownError as exc:
            raise _cooldown_http_error(exc, platform) from exc
        except AccountUnavailable:
            # 原绑定账号需要人工确认后才能切换，Python 不静默删除槽位绑定。
            raise
        except CredentialUnavailable:
            if session is not None and preferred is not None:
                if not service.account_is_recoverable(preferred):
                    raise AccountUnavailable("Bound account requires manual recovery")
                # 自动刷新/凭证写库尚未完成，仍等待同账号，不把它当成人工失效。
                raise CapacityUnavailable() from None
            raise
        except CapacityUnavailable as exc:
            if session is not None and exc.code == "NO_HEALTHY_ACCOUNT":
                raise HTTPException(status_code=502, detail="No automatically recoverable image account") from exc
            raise


def _session_options(account: dict) -> dict:
    """会话参数只在Python内部传递，图片响应不增加任何来源信息。"""
    session = CURRENT_SESSION.get()
    if session is None:
        return {}
    return {"conversation_ref": session.conversation_ref,
            "on_conversation": lambda reference: session.save(account["account_id"], reference)}


async def _retry_session_rejection(exc: Exception, platform: str) -> bool:
    """账号与会话错误交给 Java 展示原因；Python 不自动换号或重建会话。"""
    return False


async def _generate_images_once(body: ImageGenerationRequest, resolved, prompt: str, request_id: str, attempt: int):
    if resolved.platform == "gemini":
        account = await _acquire_image_account(gemini_account_service, "gemini", "name", attempt)
        account_ref = stable_log_ref("gemini-account", account.get("name"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s python_attempt=%d platform=gemini account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        upstream_started = time.monotonic()
        try:
            async with GeminiBackendAPI(account,
                    before_model_send=lambda: gemini_account_service.wait_for_model_send(account),
                    after_model_send=lambda: gemini_account_service.record_model_call(account)) as backend:
                images = await backend.image(prompt, resolved.model, body.images, body.aspect_ratio,
                                              **_session_options(account), thinking_level="low")
            image = _checked_image(images, "gemini")
            logger.info("event=upstream_attempt_finished outcome=success request_id=%s python_attempt=%d platform=gemini account_ref=%s proxy_ref=%s duration_ms=%d",
                        request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            gemini_account_service.release_account(account["name"], True, task_type="image", acquired_account=account)
        except asyncio.CancelledError:
            logger.warning("event=upstream_attempt_finished outcome=cancelled request_id=%s python_attempt=%d platform=gemini account_ref=%s proxy_ref=%s duration_ms=%d",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            gemini_account_service.release_account(account["name"], False, task_type="image", acquired_account=account)
            raise
        except Exception as exc:
            logger.warning("event=upstream_attempt_finished outcome=failed request_id=%s python_attempt=%d platform=gemini account_ref=%s proxy_ref=%s duration_ms=%d reason_code=%s http_status=%s failure_scope=%s transport_code=%s",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000),
                           getattr(exc, "reason_code", type(exc).__name__), error_http_status(exc), _failure_scope(exc), getattr(exc, "code", None))
            st_code = 504 if is_read_timeout_error(exc) or isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else (
                401 if isinstance(exc, AuthError) else 429
                if isinstance(exc, (TemporarilyBlockedError, UsageLimitExceededError))
                else error_http_status(exc)
            )
            gemini_account_service.release_account(
                account["name"], False, "" if isinstance(exc, (ImageResultError, ConversationUnavailable)) else str(exc),
                status_code=st_code,
                retry_after=getattr(exc, "retry_after", None),
                task_type="image", failure_scope=_failure_scope(exc), acquired_account=account,
            )
            raise
    elif resolved.platform == "doubao":
        account = await acquire_traced_account(doubao_account_service, "image", "doubao", "name", attempt)
        account_ref = stable_log_ref("doubao-account", account.get("name"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s python_attempt=%d platform=doubao account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        upstream_started = time.monotonic()
        try:
            async with DoubaoBackendAPI(account["cookies"], account.get("proxy", ""), account.get("device_id"), account.get("web_id"), account.get("fp")) as backend:
                with image_stage("doubao", "reference_upload"):
                    attachments = [
                        await backend.upload_image(url) for url in body.images
                        if "dummyimage" not in url and "placeholder" not in url
                    ]
                with image_stage("doubao", "generate"):
                    urls = await backend.generate_image(prompt, body.aspect_ratio, attachments)
                logger.info("event=image_assets_found platform=doubao count=%d", len(urls))
                with image_stage("doubao", "image_download"):
                    images = await backend.download_images(urls)
                logger.info("event=image_downloaded platform=doubao count=%d", len(images))
            image = _checked_image(images, "doubao")
            logger.info("event=upstream_attempt_finished outcome=success request_id=%s python_attempt=%d platform=doubao account_ref=%s proxy_ref=%s duration_ms=%d",
                        request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            doubao_account_service.release_account(account["name"], True, task_type="image", acquired_account=account)
        except asyncio.CancelledError:
            logger.warning("event=upstream_attempt_finished outcome=cancelled request_id=%s python_attempt=%d platform=doubao account_ref=%s proxy_ref=%s duration_ms=%d",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            doubao_account_service.release_account(account["name"], False, task_type="image", acquired_account=account)
            raise
        except Exception as exc:
            logger.warning("event=upstream_attempt_finished outcome=failed request_id=%s python_attempt=%d platform=doubao account_ref=%s proxy_ref=%s duration_ms=%d reason_code=%s http_status=%s failure_scope=%s transport_code=%s",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000),
                           getattr(exc, "reason_code", type(exc).__name__), error_http_status(exc), _failure_scope(exc), getattr(exc, "code", None))
            st_code = 504 if is_read_timeout_error(exc) or isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else error_http_status(exc)
            doubao_account_service.release_account(
                account["name"], False, "" if isinstance(exc, ImageResultError) else str(exc), status_code=st_code,
                retry_after=getattr(exc, "retry_after", None),
                task_type="image", failure_scope=_failure_scope(exc), acquired_account=account,
            )
            raise
    else:
        account = await _acquire_image_account(account_service, "gpt", "email", attempt)
        account_ref = stable_log_ref("gpt-account", account.get("email"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s python_attempt=%d platform=gpt account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        upstream_started = time.monotonic()
        try:
            async with OpenAIBackendAPI(account["access_token"], account.get("proxy", ""), account.get("device_id", ""),
                    credential_provider=lambda: account_service.prepare_request_account(account),
                    before_model_send=lambda: account_service.wait_for_model_send(account),
                    after_model_send=lambda: account_service.record_model_call(account)) as backend:
                images = await backend.generate_image_bytes(prompt, resolved.model, references=body.images,
                                                            expected_count=1, **_session_options(account),
                                                            thinking_level=body.thinking_level)
            image = _checked_image(images, "gpt")
            account_service.release_account(account["email"], True, task_type="image", acquired_account=account)
            logger.info("event=upstream_attempt_finished outcome=success request_id=%s python_attempt=%d platform=gpt account_ref=%s proxy_ref=%s duration_ms=%d",
                        request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
        except asyncio.CancelledError:
            logger.warning("event=upstream_attempt_finished outcome=cancelled request_id=%s python_attempt=%d platform=gpt account_ref=%s proxy_ref=%s duration_ms=%d",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            account_service.release_account(account["email"], False, task_type="image", acquired_account=account)
            raise
        except Exception as exc:
            logger.warning("event=upstream_attempt_finished outcome=failed request_id=%s python_attempt=%d platform=gpt account_ref=%s proxy_ref=%s duration_ms=%d reason_code=%s http_status=%s failure_scope=%s transport_code=%s",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000),
                           getattr(exc, "reason_code", type(exc).__name__), error_http_status(exc), _failure_scope(exc), getattr(exc, "code", None))
            st_code = 504 if is_read_timeout_error(exc) or isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else error_http_status(exc)
            account_service.release_account(
                account["email"], False, "" if isinstance(exc, ImageResultError) else str(exc), status_code=st_code,
                retry_after=getattr(exc, "retry_after", None), task_type="image",
                failure_scope=_failure_scope(exc), acquired_account=account,
            )
            raise
    return image


@router.post("/v1/images/generations")
async def generate_images(body: ImageGenerationRequest, request: Request):
    resolved = resolve_model(body.model, "image")
    request_id = body.request_id or uuid.uuid4().hex
    body = body.model_copy(update={"request_id": request_id})
    return await execute_model_request(resolved.platform, request_id, lambda: _generate_images(body),
                                       request=request, java_attempt=body.java_attempt)


async def _generate_images(body: ImageGenerationRequest):
    platform = resolve_model(body.model, "image").platform
    async with slot_sessions.claim(body.task_code, body.item_id, platform):
        return await _generate_images_prepared(body)


async def _generate_images_prepared(body: ImageGenerationRequest):
    resolved = resolve_model(body.model, "image")
    prompt = build_ratio_prompt(body.prompt, body.aspect_ratio)
    request_id = body.request_id or uuid.uuid4().hex
    started_at = time.monotonic()
    logger.info(
        "event=image_request_started request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s model=%s images=%d",
        request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
        body.stage, resolved.platform, resolved.model, len(body.images),
    )

    for attempt in range(1):
        try:
            # 账号池内部的健康变化日志也归到本次 Python 上游尝试。
            attempt_token = python_attempt_log_context.set(attempt + 1)
            try:
                # 提交、结果轮询和取回图片共用四分钟，不能在每个阶段重新等待四分钟。
                raw_image = await asyncio.wait_for(
                    _generate_images_once(body, resolved, prompt, request_id, attempt + 1),
                    timeout=MODEL_TIMEOUT_SECONDS,
                )
            finally:
                python_attempt_log_context.reset(attempt_token)
            media_type = image_media_type(raw_image) if isinstance(raw_image, (bytes, bytearray)) else None
            if media_type is None:
                raise RuntimeError("Upstream did not return a valid image binary")
            logger.info(
                "event=image_request_succeeded request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s python_attempt=%d elapsed_ms=%d",
                request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                body.stage, attempt + 1, int((time.monotonic() - started_at) * 1000),
            )
            return Response(content=raw_image, media_type=media_type, headers={"X-Request-ID": request_id})
        except BoundAccountCooldownError as exc:
            error = _cooldown_http_error(exc, resolved.platform)
            error.headers["X-Request-ID"] = request_id
            raise error from exc
        except CapacityUnavailable:
            logger.warning("event=image_request_failed request_id=%s platform=%s python_attempt=%d reason_code=NO_CAPACITY", request_id, resolved.platform, attempt + 1)
            raise
        except CredentialUnavailable as exc:
            logger.warning("event=image_request_failed request_id=%s platform=%s python_attempt=%d reason_code=CredentialUnavailable", request_id, resolved.platform, attempt + 1)
            raise HTTPException(status_code=502, detail="GPT credential preparation failed",
                                headers={"X-Request-ID": request_id}) from exc
        except HTTPException:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "event=image_request_failed request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s python_attempt=%d max_attempts=1 elapsed_ms=%d reason=%s",
                request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                body.stage, resolved.platform, attempt + 1, int((time.monotonic() - started_at) * 1000), type(exc).__name__,
            )

            # 精准识别限流与额度耗尽异常，透传结构化 429 与倒计时，保留槽位供用户选择
            from utils.helper import ImageQuotaExceededError, UpstreamHTTPError
            from providers.gemini.webapi.exceptions import TemporarilyBlockedError, UsageLimitExceededError
            import json

            err_str = str(exc).lower()
            if isinstance(exc, (AccountUnavailable, AuthError)) or (
                isinstance(exc, UpstreamHTTPError) and exc.status_code == 401
            ):
                raise HTTPException(status_code=401, detail=json.dumps({
                    "code": "ACCOUNT_AUTH_INVALID", "reason": "ACCOUNT_AUTH_INVALID",
                    "platform": resolved.platform,
                    "message": "当前绑定账号登录状态失效，需要更新凭证或确认切换账号。",
                    "userActions": ["UPDATE_CREDENTIALS", "SWITCH_ACCOUNT"],
                    "upstreamState": "UNKNOWN",
                }, ensure_ascii=False), headers={"X-Request-ID": request_id}) from exc
            if isinstance(exc, ConversationUnavailable) or (
                isinstance(exc, ModelInvalidError) and "conversation history" in err_str
            ):
                raise HTTPException(status_code=409, detail=json.dumps({
                    "code": "CONVERSATION_UNAVAILABLE", "reason": "CONVERSATION_UNAVAILABLE",
                    "platform": resolved.platform,
                    "message": "当前会话无法继续，可在详情查看原因并选择使用原账号新建会话重试。",
                    "userActions": ["RETRY_NEW_CONVERSATION"], "upstreamState": "SUBMITTED",
                }, ensure_ascii=False), headers={"X-Request-ID": request_id}) from exc
            is_quota = isinstance(exc, (ImageQuotaExceededError, QuotaExhaustedError, UsageLimitExceededError)) or (
                detect_quota_exhausted(str(exc))[0]
            ) or any(
                kw in err_str for kw in ("quota", "upload_protection", "上限", "额度")
            )
            is_rate = isinstance(exc, TemporarilyBlockedError) or (
                isinstance(exc, UpstreamHTTPError) and exc.status_code == 429
            )

            if is_quota or is_rate:
                retry_after = getattr(exc, "retry_after", None)
                if is_quota:
                    # 优先异常自带时间，其次从文本提取恢复时间，都没有则默认 5 小时。
                    _, detected_seconds = detect_quota_exhausted(str(exc))
                    wait_sec = (retry_after if retry_after and retry_after > 0
                                else detected_seconds if detected_seconds and detected_seconds > 0
                                else 18000)
                    wait_min = max(1, wait_sec // 60)
                    detail = {
                        "code": "ACCOUNT_QUOTA_EXHAUSTED",
                        "retry_after": wait_sec,
                        "retryAfter": wait_sec,
                        "cooldownUntil": int(time.time()) + wait_sec,
                        "message": f"当前账号额度已耗尽，预计将在 {wait_min} 分钟后恢复。可等待恢复或确认切换账号。"
                    }
                else:
                    wait_sec = 120
                    detail = {
                        "code": "ACCOUNT_RATE_LIMITED",
                        "retry_after": wait_sec,
                        "retryAfter": wait_sec,
                        "cooldownUntil": int(time.time()) + wait_sec,
                        "message": "当前账号受到限流，请等待冷却结束或确认切换账号。"
                    }
                raise HTTPException(
                    status_code=429,
                    detail=json.dumps({**detail, "platform": resolved.platform}, ensure_ascii=False),
                    headers={"X-Request-ID": request_id, "Retry-After": str(wait_sec)}
                )

            stage = getattr(exc, "image_stage", "model_request")
            if isinstance(exc, ImageDownloadError):
                raise HTTPException(status_code=502, detail=json.dumps({
                    "code": "IMAGE_DOWNLOAD_FAILED", "platform": resolved.platform,
                    "stage": "image_download", "message": str(exc),
                    "upstreamState": "IMAGE_READY", "userActions": ["RETRY_DOWNLOAD"],
                }, ensure_ascii=False), headers={"X-Request-ID": request_id}) from exc

            # 已提交生成后读取超时不能自动重发图片生成；明确返回失败阶段给 Java 展示。
            from utils.helper import is_read_timeout_error
            if is_read_timeout_error(exc) or isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or getattr(exc, "status_code", None) == 504:
                raise HTTPException(
                    status_code=504, detail=json.dumps({
                        "code": "IMAGE_READ_TIMEOUT", "platform": resolved.platform, "stage": stage,
                        "message": f"模型请求读取超时（阶段：{stage}），未自动重新提交生图。",
                        "upstreamState": "SUBMITTED", "userActions": ["RETRY_IMAGE"],
                    }, ensure_ascii=False),
                    headers={"X-Request-ID": request_id},
                ) from exc

            if is_connect_error(exc) or (is_transport_error(exc) and stage in {"model_request", "image_submit"}):
                raise HTTPException(status_code=502, detail=json.dumps({
                    "code": "ACCOUNT_NETWORK_COOLDOWN", "platform": resolved.platform,
                    "retryAfter": 5, "cooldownUntil": int(time.time()) + 5,
                    "message": "账号网络连接重试一次后仍失败，已冷却 5 秒；上游未确认接收本次生图请求。",
                    "upstreamState": "NOT_SENT", "userActions": ["WAIT", "SWITCH_ACCOUNT"],
                }, ensure_ascii=False), headers={"X-Request-ID": request_id}) from exc

            # 其余错误统一对外收敛报错为：模型生成失败
            raise HTTPException(
                status_code=502, detail=json.dumps({
                    "code": "MODEL_GENERATION_FAILED", "platform": resolved.platform, "stage": stage,
                    "message": f"模型生成失败（阶段：{stage}）。",
                    "upstreamState": "SUBMITTED", "userActions": ["VIEW_DETAILS", "RETRY_IMAGE"],
                }, ensure_ascii=False),
                headers={"X-Request-ID": request_id},
            ) from exc

