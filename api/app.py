import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from api.routers import accounts, capacity, chat, edit, health, images, oauth, proxies
from utils.log import logger
from core.admission import initialize_model_admission

CAPACITY_REDIS_PREFIX = "flexi:ai:capacity:"
CAPACITY_SNAPSHOT_TTL_SECONDS = 4
CAPACITY_REFRESH_INTERVAL_SECONDS = 1


async def publish_capacity_snapshots(redis_client: Redis) -> None:
    """每秒将代理账号池的实时容量快照写入 Redis。"""
    while True:
        try:
            snapshots = capacity.get_platform_capacity_snapshots()
            pipeline = redis_client.pipeline(transaction=True)
            for platform, snapshot in snapshots.items():
                payload = snapshot
                pipeline.set(
                    f"{CAPACITY_REDIS_PREFIX}{platform}",
                    json.dumps(payload, separators=(",", ":")),
                    ex=CAPACITY_SNAPSHOT_TTL_SECONDS,
                )
            await pipeline.execute()
        except Exception:
            logger.exception("发布模型容量快照到 Redis 失败")
        await asyncio.sleep(CAPACITY_REFRESH_INTERVAL_SECONDS)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        from providers.gemini.account import gemini_account_service

        initialize_model_admission()
        from providers.openai.account import account_service
        credential_task = asyncio.create_task(account_service.credentials.maintain())
        redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"), decode_responses=True)
        capacity_task = asyncio.create_task(publish_capacity_snapshots(redis_client))
        # 后台限并发预热，并持续恢复中断的续期客户端。
        warmup_task = asyncio.create_task(gemini_account_service.maintain_clients())
        yield
        credential_task.cancel()
        try:
            await credential_task
        except asyncio.CancelledError:
            pass
        await account_service.credentials.drain()
        warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass
        await gemini_account_service.close_clients()
        capacity_task.cancel()
        try:
            await capacity_task
        except asyncio.CancelledError:
            pass
        await redis_client.aclose()

    app = FastAPI(title="ChatGPT-Image-Service", version="2.0.0", lifespan=lifespan)

    # 浏览器管理页面直连 Python 时，只允许部署方明确配置的前端来源。
    cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5174,http://127.0.0.1:5174").split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Python 服务按需求不校验登录状态，直接注册业务路由。
    app.include_router(images.router)
    app.include_router(capacity.router)
    app.include_router(chat.router)
    app.include_router(accounts.router)
    app.include_router(proxies.router)
    app.include_router(oauth.router)
    app.include_router(health.router)
    app.include_router(edit.router)

    return app
