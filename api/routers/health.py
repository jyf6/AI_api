from __future__ import annotations

import time

from fastapi import APIRouter

from providers.openai.account import account_service

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    stats = account_service.stats()
    healthy = stats["active_accounts"] > 0
    return {
        "status": "ok" if healthy else "degraded",
        "healthy": healthy,
        "timestamp": int(time.time()),
        "accounts": stats,
    }


@router.get("/api/stats")
async def get_stats():
    return account_service.stats()
