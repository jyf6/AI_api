from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from providers.doubao.account import doubao_account_service
from providers.gemini.account import gemini_account_service
from providers.openai.account import account_service

router = APIRouter(tags=["capacity"])


@router.get("/v1/capacity")
async def get_capacity(model: str = Query(..., min_length=1)):
    """Return shared capacity for a model's provider without reserving an account."""
    platform = model.split("-", 1)[0].lower()
    pools = {
        "gpt": account_service,
        "gemini": gemini_account_service,
        "doubao": doubao_account_service,
    }
    pool = pools.get(platform)
    if pool is None:
        raise HTTPException(status_code=400, detail="模型标签必须以 gpt、gemini 或 doubao 开头")
    return {"model": model, "platform": platform, **pool.capacity()}
