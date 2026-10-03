from __future__ import annotations

import time

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "healthy": True,
        "timestamp": int(time.time()),
    }


@router.get("/api/stats")
async def get_stats():
    from providers.openai.account import account_service
    return account_service.stats()
