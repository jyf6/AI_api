from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from providers.doubao.account import doubao_account_service
from providers.gemini.account import gemini_account_service
from providers.openai.account import account_service
from core.admission import get_model_admission

router = APIRouter(tags=["capacity"])

CAPACITY_POOLS = {
    "gpt": account_service,
    "gemini": gemini_account_service,
    "doubao": doubao_account_service,
}


def get_platform_capacity_snapshots() -> dict[str, dict[str, int | None]]:
    """读取所有平台当前的共享账号容量快照。"""
    admission = get_model_admission()
    return {platform: admission.snapshot(platform, pool.capacity()) for platform, pool in CAPACITY_POOLS.items()}


@router.get("/v1/capacity")
async def get_capacity(model: str = Query(..., min_length=1)):
    """Return shared capacity for a model's provider without reserving an account."""
    platform = model.split("-", 1)[0].lower()
    pool = CAPACITY_POOLS.get(platform)
    if pool is None:
        raise HTTPException(status_code=400, detail="模型标签必须以 gpt、gemini 或 doubao 开头")
    return {"model": model, "platform": platform, **get_model_admission().snapshot(platform, pool.capacity())}
