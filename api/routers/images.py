from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Response

from api.schemas import ImageGenerationRequest
from core.router import resolve_model
from providers.doubao.account import doubao_account_service
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.account import gemini_account_service
from providers.gemini.backend import GeminiBackendAPI
from providers.openai.account import account_service
from providers.openai.backend import OpenAIBackendAPI
from utils.helper import ImageQuotaExceededError, UpstreamHTTPError
from utils.image_binary import image_media_type
from utils.image_ratio import build_ratio_prompt
from utils.log import logger

router = APIRouter(tags=["images"])

async def _generate_images_once(body: ImageGenerationRequest, resolved, prompt: str):
    if resolved.platform == "gemini":
        account = await asyncio.to_thread(gemini_account_service.wait_for_available_account, "image")
        try:
            async with GeminiBackendAPI(account) as backend:
                images = await backend.image(prompt, resolved.model, body.images, body.aspect_ratio)
            gemini_account_service.release_account(account["name"], True, task_type="image")
        except asyncio.CancelledError:
            gemini_account_service.release_account(account["name"], False, "request cancelled", task_type="image")
            raise
        except Exception as exc:
            gemini_account_service.release_account(
                account["name"], False, str(exc), status_code=getattr(exc, "status_code", None), task_type="image",
            )
            raise
    elif resolved.platform == "doubao":
        account = await asyncio.to_thread(doubao_account_service.wait_for_available_account, "image")
        try:
            async with DoubaoBackendAPI(account["cookies"], account.get("proxy", "")) as backend:
                attachments = [
                    await backend.upload_image(url) for url in body.images
                    if "dummyimage" not in url and "placeholder" not in url
                ]
                urls = await backend.generate_image(prompt, body.aspect_ratio, attachments)
                images = await backend.download_images(urls)
            doubao_account_service.release_account(account["name"], True, task_type="image")
        except asyncio.CancelledError:
            doubao_account_service.release_account(account["name"], False, "request cancelled", task_type="image")
            raise
        except Exception as exc:
            doubao_account_service.release_account(
                account["name"], False, str(exc), status_code=getattr(exc, "status_code", None), task_type="image",
            )
            raise
    else:
        account = await asyncio.to_thread(account_service.wait_for_available_account, "image")
        try:
            def run() -> bytes | list[bytes]:
                with OpenAIBackendAPI(account["access_token"], account.get("proxy", "")) as backend:
                    return backend.generate_image_bytes(prompt, resolved.model, references=body.images, expected_count=1)
            images = await asyncio.to_thread(run)
            account_service.release_account(account["email"], True, task_type="image")
        except asyncio.CancelledError:
            account_service.release_account(account["email"], False, "request cancelled", task_type="image")
            raise
        except ImageQuotaExceededError as exc:
            account_service.release_account(
                account["email"], False, str(exc), retry_after=exc.retry_after, task_type="image",
            )
            raise
        except Exception as exc:
            status = exc.status_code if isinstance(exc, UpstreamHTTPError) else None
            account_service.release_account(
                account["email"], False, str(exc), status_code=status,
                retry_after=getattr(exc, "retry_after", None), task_type="image",
            )
            raise
    return images[0] if isinstance(images, list) else images


@router.post("/v1/images/generations")
async def generate_images(body: ImageGenerationRequest):
    resolved = resolve_model(body.model, "image")
    prompt = build_ratio_prompt(body.prompt, body.aspect_ratio)
    for attempt in range(2):
        try:
            raw_image = await _generate_images_once(body, resolved, prompt)
            media_type = image_media_type(raw_image) if isinstance(raw_image, (bytes, bytearray)) else None
            if media_type is None:
                raise RuntimeError("Upstream did not return a valid image binary")
            return Response(content=raw_image, media_type=media_type)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if attempt == 0:
                logger.warning(f"[Image Retry] {resolved.platform} 图片生成失败: {exc}；1 秒后重试一次")
                await asyncio.sleep(1)
                continue
            raise HTTPException(
                status_code=502, detail=f"{resolved.platform} image generation failed: {exc}"
            ) from exc

