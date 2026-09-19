from __future__ import annotations

import asyncio

from fastapi import APIRouter, Header, HTTPException, Response

from api.schemas import ImageGenerationRequest
from core.router import resolve_model
from core.database import database
from core.operations import begin_operation, replay_image
from providers.doubao.account import doubao_account_service
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.account import gemini_account_service
from providers.gemini.backend import GeminiBackendAPI
from providers.openai.account import account_service
from providers.openai.backend import OpenAIBackendAPI
from utils.helper import ImageQuotaExceededError, UpstreamHTTPError
from utils.image_binary import image_media_type
from utils.image_ratio import build_ratio_prompt

router = APIRouter(tags=["images"])

@router.post("/v1/images/generations")
async def generate_images(body: ImageGenerationRequest, operation_id: str | None = Header(default=None, alias="X-Operation-Id")):
    resolved = resolve_model(body.model, "image")
    # 双重保险：除各平台原生比例控制外，再把画幅要求显式写入提示词，禁止沿用垫图比例。
    prompt = build_ratio_prompt(body.prompt, body.aspect_ratio)
    operation_id, existing = begin_operation(operation_id, "image")
    if existing is not None:
        return replay_image(existing, operation_id)
    try:
        if resolved.platform == "gemini":
            account = await asyncio.to_thread(gemini_account_service.wait_for_available_account, "image")
            try:
                async with GeminiBackendAPI(account) as backend:
                    images = await backend.image(prompt, resolved.model, body.images, body.aspect_ratio)
                gemini_account_service.release_account(account["name"], True, task_type="image")
            except Exception as exc:
                # 临时失败仅重建客户端；账号状态由 Gemini 账号池按错误类别处理。
                await gemini_account_service.discard_client(account["name"])
                gemini_account_service.release_account(account["name"], False, str(exc), task_type="image")
                raise
        elif resolved.platform == "doubao":
            account = await asyncio.to_thread(doubao_account_service.wait_for_available_account, "image")
            try:
                async with DoubaoBackendAPI(account["cookies"], account.get("proxy", "")) as backend:
                    attachments = []
                    for url in body.images:
                        if "dummyimage" in url or "placeholder" in url:
                            continue
                        try:
                            attachments.append(await backend.upload_image(url))
                        except Exception as e:
                            print(f"Failed to upload image {url}: {e}")
                            raise e
                    urls = await backend.generate_image(prompt, body.aspect_ratio, attachments)
                    images = await backend.download_images(urls)
                doubao_account_service.release_account(account["name"], True, task_type="image")
            except Exception as exc:
                doubao_account_service.release_account(account["name"], False, str(exc), task_type="image")
                raise
        else:
            while True:
                account = await asyncio.to_thread(account_service.wait_for_available_account, "image")
                try:
                    def run() -> bytes | list[bytes]:
                        with OpenAIBackendAPI(account["access_token"], account.get("proxy", "")) as backend:
                            return backend.generate_image_bytes(prompt, resolved.model, references=body.images, expected_count=1)
                    images = await asyncio.to_thread(run)
                    account_service.release_account(account["email"], True, task_type="image")
                    break
                except ImageQuotaExceededError as exc:
                    # 网页端额度提示不是调用失败：冷却当前账号后继续等待可用账号补位。
                    account_service.release_account(
                        account["email"], False, str(exc), retry_after=exc.retry_after, task_type="image",
                    )
                except Exception as exc:
                    status = exc.status_code if isinstance(exc, UpstreamHTTPError) else None
                    retry_after = getattr(exc, "retry_after", None)
                    account_service.release_account(
                        account["email"], False, str(exc), status_code=status,
                        retry_after=retry_after, task_type="image",
                    )
                    raise
        raw_image = images[0] if isinstance(images, list) else images
        media_type = image_media_type(raw_image) if isinstance(raw_image, (bytes, bytearray)) else None
        # 仅允许真实图片进入操作记录与 Java 存储链路，禁止将上游 JSON/HTML 错误页伪装成图片。
        if media_type is None:
            raise RuntimeError("Upstream did not return a valid image binary")
        database.complete_operation(operation_id, image=bytes(raw_image), content_type=media_type)
        return Response(content=raw_image, media_type=media_type, headers={"X-Operation-Id": operation_id})
    except HTTPException:
        raise
    except Exception as exc:
        # 上游可能已受理，保留未知状态，禁止调用方以相同操作标识盲目重放。
        database.fail_operation(operation_id, str(exc))
        raise HTTPException(status_code=502, detail=f"{resolved.platform} image generation failed: {exc}") from exc

