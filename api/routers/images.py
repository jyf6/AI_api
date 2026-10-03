from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException, Response

from api.schemas import ImageGenerationRequest
from core.account_pool import await_thread_result
from core.router import resolve_model
from providers.doubao.account import doubao_account_service
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.account import gemini_account_service
from providers.gemini.backend import GeminiBackendAPI
from providers.openai.account import account_service
from providers.openai.backend import OpenAIBackendAPI
from utils.helper import ImageQuotaExceededError, UpstreamHTTPError, is_transport_error
from utils.image_binary import image_media_type
from utils.image_ratio import build_ratio_prompt
from utils.log import logger, proxy_log_ref, stable_log_ref

router = APIRouter(tags=["images"])

async def _generate_images_once(body: ImageGenerationRequest, resolved, prompt: str, request_id: str, attempt: int):
    if resolved.platform == "gemini":
        account = await gemini_account_service.acquire_account("image")
        account_ref = stable_log_ref("gemini-account", account.get("name"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s attempt=%d platform=gemini account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        try:
            async with GeminiBackendAPI(account) as backend:
                images = await backend.image(prompt, resolved.model, body.images, body.aspect_ratio)
            logger.info("event=upstream_attempt_succeeded request_id=%s attempt=%d platform=gemini account_ref=%s proxy_ref=%s",
                        request_id, attempt, account_ref, proxy_ref)
            gemini_account_service.release_account(account["name"], True, task_type="image")
        except asyncio.CancelledError:
            logger.warning("event=upstream_attempt_cancelled request_id=%s attempt=%d platform=gemini account_ref=%s proxy_ref=%s",
                           request_id, attempt, account_ref, proxy_ref)
            gemini_account_service.release_account(account["name"], False, task_type="image")
            raise
        except Exception as exc:
            logger.error("event=upstream_attempt_failed request_id=%s attempt=%d platform=gemini account_ref=%s proxy_ref=%s reason=%s status=%s",
                         request_id, attempt, account_ref, proxy_ref, type(exc).__name__, getattr(exc, "status_code", None))
            gemini_account_service.release_account(
                account["name"], False, str(exc), status_code=getattr(exc, "status_code", None),
                task_type="image", failure_scope="transport" if is_transport_error(exc) else "account",
            )
            raise
    elif resolved.platform == "doubao":
        account = await doubao_account_service.acquire_account("image")
        account_ref = stable_log_ref("doubao-account", account.get("name"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s attempt=%d platform=doubao account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        try:
            async with DoubaoBackendAPI(account["cookies"], account.get("proxy", ""), account.get("device_id"), account.get("web_id"), account.get("fp")) as backend:
                attachments = [
                    await backend.upload_image(url) for url in body.images
                    if "dummyimage" not in url and "placeholder" not in url
                ]
                urls = await backend.generate_image(prompt, body.aspect_ratio, attachments)
                images = await backend.download_images(urls)
            logger.info("event=upstream_attempt_succeeded request_id=%s attempt=%d platform=doubao account_ref=%s proxy_ref=%s",
                        request_id, attempt, account_ref, proxy_ref)
            doubao_account_service.release_account(account["name"], True, task_type="image")
        except asyncio.CancelledError:
            logger.warning("event=upstream_attempt_cancelled request_id=%s attempt=%d platform=doubao account_ref=%s proxy_ref=%s",
                           request_id, attempt, account_ref, proxy_ref)
            doubao_account_service.release_account(account["name"], False, task_type="image")
            raise
        except Exception as exc:
            logger.error("event=upstream_attempt_failed request_id=%s attempt=%d platform=doubao account_ref=%s proxy_ref=%s reason=%s status=%s",
                         request_id, attempt, account_ref, proxy_ref, type(exc).__name__, getattr(exc, "status_code", None))
            doubao_account_service.release_account(
                account["name"], False, str(exc), status_code=getattr(exc, "status_code", None),
                task_type="image", failure_scope="transport" if is_transport_error(exc) else "account",
            )
            raise
    else:
        account = await account_service.acquire_account("image")
        account_ref = stable_log_ref("gpt-account", account.get("email"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s attempt=%d platform=gpt account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        try:
            def run() -> bytes | list[bytes]:
                try:
                    with OpenAIBackendAPI(account["access_token"], account.get("proxy", ""), account.get("device_id", "")) as backend:
                        result = backend.generate_image_bytes(prompt, resolved.model, references=body.images, expected_count=1)
                except ImageQuotaExceededError as exc:
                    account_service.release_account(account["email"], False, str(exc),
                                                    retry_after=exc.retry_after, task_type="image")
                    raise
                except Exception as exc:
                    status = exc.status_code if isinstance(exc, UpstreamHTTPError) else None
                    account_service.release_account(account["email"], False, str(exc), status_code=status,
                                                    retry_after=getattr(exc, "retry_after", None), task_type="image",
                                                    failure_scope="transport" if is_transport_error(exc) else "account")
                    raise
                account_service.release_account(account["email"], True, task_type="image")
                return result

            images = await await_thread_result(run)
            logger.info("event=upstream_attempt_succeeded request_id=%s attempt=%d platform=gpt account_ref=%s proxy_ref=%s",
                        request_id, attempt, account_ref, proxy_ref)
        except asyncio.CancelledError:
            logger.warning("event=upstream_attempt_cancelled request_id=%s attempt=%d platform=gpt account_ref=%s proxy_ref=%s",
                           request_id, attempt, account_ref, proxy_ref)
            raise
        except ImageQuotaExceededError as exc:
            logger.error("event=upstream_attempt_failed request_id=%s attempt=%d platform=gpt account_ref=%s proxy_ref=%s reason=%s",
                         request_id, attempt, account_ref, proxy_ref, type(exc).__name__)
            raise
        except Exception as exc:
            status = exc.status_code if isinstance(exc, UpstreamHTTPError) else None
            logger.error("event=upstream_attempt_failed request_id=%s attempt=%d platform=gpt account_ref=%s proxy_ref=%s reason=%s status=%s",
                         request_id, attempt, account_ref, proxy_ref, type(exc).__name__, status)
            raise
    return images[0] if isinstance(images, list) else images


@router.post("/v1/images/generations")
async def generate_images(body: ImageGenerationRequest):
    resolved = resolve_model(body.model, "image")
    prompt = build_ratio_prompt(body.prompt, body.aspect_ratio)
    request_id = body.request_id or uuid.uuid4().hex
    started_at = time.monotonic()
    logger.info(
        "event=image_request_started request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s model=%s images=%d",
        request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
        body.stage, resolved.platform, resolved.model, len(body.images),
    )
    for attempt in range(2):
        try:
            raw_image = await _generate_images_once(body, resolved, prompt, request_id, attempt + 1)
            media_type = image_media_type(raw_image) if isinstance(raw_image, (bytes, bytearray)) else None
            if media_type is None:
                raise RuntimeError("Upstream did not return a valid image binary")
            logger.info(
                "event=image_request_succeeded request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s attempt=%d elapsed_ms=%d",
                request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                body.stage, attempt + 1, int((time.monotonic() - started_at) * 1000),
            )
            return Response(content=raw_image, media_type=media_type, headers={"X-Request-ID": request_id})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if attempt == 0:
                logger.warning(
                    "event=image_attempt_failed request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s attempt=1 max_attempts=2 reason=%s",
                    request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                    body.stage, resolved.platform, type(exc).__name__,
                )
                await asyncio.sleep(1)
                continue
            logger.error(
                "event=image_request_failed request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s attempt=2 max_attempts=2 elapsed_ms=%d reason=%s",
                request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                body.stage, resolved.platform, int((time.monotonic() - started_at) * 1000), type(exc).__name__,
            )
            raise HTTPException(
                status_code=502, detail=f"{resolved.platform} image generation failed: {exc}",
                headers={"X-Request-ID": request_id},
            ) from exc

