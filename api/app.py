from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from api.routers import accounts, capacity, chat, edit, health, images, oauth

def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        import asyncio
        from providers.gemini.account import gemini_account_service
        # 服务启动即并发预热活跃账号，由项目内 Gemini Web API 接管后台保活与 Cookie 续期。
        warmup_task = asyncio.create_task(gemini_account_service.warmup_clients())
        yield
        await gemini_account_service.close_clients()
        try:
            warmup_task.cancel()
        except Exception:
            pass

    app = FastAPI(title="ChatGPT-Image-Service", version="2.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册所有领域路由
    app.include_router(images.router)
    app.include_router(capacity.router)
    app.include_router(chat.router)
    app.include_router(accounts.router)
    app.include_router(oauth.router)
    app.include_router(health.router)
    app.include_router(edit.router)

    return app
