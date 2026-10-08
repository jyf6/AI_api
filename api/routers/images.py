from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException, Request, Response

from api.schemas import ImageGenerationRequest
from core.router import resolve_model
from core.admission import CapacityUnavailable
from api.model_execution import acquire_traced_account, execute_model_request
from providers.doubao.account import doubao_account_service
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.account import gemini_account_service
from providers.gemini.backend import GeminiBackendAPI
from providers.openai.account import account_service
from providers.openai.backend import OpenAIBackendAPI
from providers.openai.credentials import CredentialUnavailable
from utils.helper import ImageQuotaExceededError, UpstreamHTTPError, is_transport_error
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


async def _generate_images_once(body: ImageGenerationRequest, resolved, prompt: str, request_id: str, attempt: int):
    if resolved.platform == "gemini":
        account = await acquire_traced_account(gemini_account_service, "image", "gemini", "name", attempt)
        account_ref = stable_log_ref("gemini-account", account.get("name"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s python_attempt=%d platform=gemini account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        upstream_started = time.monotonic()
        try:
            async with GeminiBackendAPI(account) as backend:
                images = await backend.image(prompt, resolved.model, body.images, body.aspect_ratio)
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
            gemini_account_service.release_account(
                account["name"], False, "" if isinstance(exc, ImageResultError) else str(exc), status_code=error_http_status(exc),
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
            doubao_account_service.release_account(
                account["name"], False, "" if isinstance(exc, ImageResultError) else str(exc), status_code=error_http_status(exc),
                task_type="image", failure_scope=_failure_scope(exc), acquired_account=account,
            )
            raise
    else:
        account = await acquire_traced_account(account_service, "image", "gpt", "email", attempt)
        account_ref = stable_log_ref("gpt-account", account.get("email"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s python_attempt=%d platform=gpt account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        upstream_started = time.monotonic()
        try:
            async with OpenAIBackendAPI(account["access_token"], account.get("proxy", ""), account.get("device_id", ""),
                    credential_provider=lambda: account_service.prepare_request_account(account)) as backend:
                images = await backend.generate_image_bytes(prompt, resolved.model, references=body.images, expected_count=1)
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
            account_service.release_account(
                account["email"], False, "" if isinstance(exc, ImageResultError) else str(exc), status_code=error_http_status(exc),
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
    # 临时拼图覆盖两次模型尝试，最终一定释放，不被普通缓存淘汰。
    from utils.oss_reference import release_memory_references
    memory_keys = []
    try:
        return await _generate_images_prepared(body, memory_keys)
    finally:
        release_memory_references(memory_keys)


async def _generate_images_prepared(body: ImageGenerationRequest, memory_keys: list[str]):
    resolved = resolve_model(body.model, "image")
    prompt = build_ratio_prompt(body.prompt, body.aspect_ratio)
    request_id = body.request_id or uuid.uuid4().hex
    started_at = time.monotonic()
    logger.info(
        "event=image_request_started request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s model=%s images=%d",
        request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
        body.stage, resolved.platform, resolved.model, len(body.images),
    )

    if len(body.images) > 0:
        preprocess_started = time.monotonic()
        try:
            from concurrent.futures import ThreadPoolExecutor
            from utils.oss_reference import read_oss_reference, put_memory_reference
            from utils.image_stitch import stitch_images_to_bytes
            
            def remember(data):
                key = put_memory_reference(data)
                memory_keys.append(key)
                return key

            def process_images(images, stage):
                with ThreadPoolExecutor(max_workers=min(10, len(images))) as executor:
                    dimensions = [2048] * len(images)
                    if len(images) > 2 and stage == "INITIAL_IMAGE_GENERATION":
                        dimensions = [768] * len(images)
                    elif len(images) > 2 and stage in ("SINGLE_ADJUST_GENERATION", "SUITE_ADJUST_GENERATION"):
                        dimensions = [2048] + [768] * (len(images) - 1)
                    bytes_list = list(executor.map(read_oss_reference, images, dimensions))
                    
                if len(images) > 2:
                    if stage == "INITIAL_IMAGE_GENERATION":
                        stitched = stitch_images_to_bytes(bytes_list)
                        return [remember(stitched)]
                    elif stage in ("SINGLE_ADJUST_GENERATION", "SUITE_ADJUST_GENERATION"):
                        base_img_bytes = bytes_list[0]
                        stitched = stitch_images_to_bytes(bytes_list[1:])
                        return [remember(base_img_bytes), remember(stitched)]
                        
                return [remember(b) for b in bytes_list]
                
            preparation = asyncio.create_task(asyncio.to_thread(process_images, body.images, body.stage))
            try:
                new_images = await asyncio.shield(preparation)
            except asyncio.CancelledError:
                # 线程不能强行取消；等它结束后由外层 finally 释放其临时图片。
                await asyncio.gather(preparation, return_exceptions=True)
                raise
            if new_images != body.images:
                logger.info("event=images_processed request_id=%s old_count=%d new_count=%d duration_ms=%d",
                            request_id, len(body.images), len(new_images), int((time.monotonic() - preprocess_started) * 1000))
                body = body.model_copy(update={"images": new_images})
        except Exception as e:
            logger.warning("event=image_process_failed request_id=%s reason_code=%s duration_ms=%d",
                           request_id, type(e).__name__, int((time.monotonic() - preprocess_started) * 1000))
            raise HTTPException(status_code=502, detail="Reference image preparation failed",
                                headers={"X-Request-ID": request_id}) from e

    for attempt in range(2):
        try:
            # 账号池内部的健康变化日志也归到本次 Python 上游尝试。
            attempt_token = python_attempt_log_context.set(attempt + 1)
            try:
                raw_image = await _generate_images_once(body, resolved, prompt, request_id, attempt + 1)
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
        except CapacityUnavailable:
            logger.warning("event=image_request_failed request_id=%s platform=%s python_attempt=%d reason_code=NO_CAPACITY", request_id, resolved.platform, attempt + 1)
            raise
        except CredentialUnavailable as exc:
            logger.warning("event=image_request_failed request_id=%s platform=%s python_attempt=%d reason_code=CredentialUnavailable", request_id, resolved.platform, attempt + 1)
            raise HTTPException(status_code=502, detail="GPT credential preparation failed",
                                headers={"X-Request-ID": request_id}) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if attempt == 0:
                logger.warning(
                    "event=image_attempt_failed request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s python_attempt=1 max_attempts=2 reason=%s",
                    request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                    body.stage, resolved.platform, type(exc).__name__,
                )
                await asyncio.sleep(1)
                continue
            logger.error(
                "event=image_request_failed request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s python_attempt=2 max_attempts=2 elapsed_ms=%d reason=%s",
                request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                body.stage, resolved.platform, int((time.monotonic() - started_at) * 1000), type(exc).__name__,
            )
            raise HTTPException(
                status_code=502, detail=f"{resolved.platform} image generation failed: {exc}",
                headers={"X-Request-ID": request_id},
            ) from exc

