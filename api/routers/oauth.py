from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import OAuthCallbackRequest
from providers.openai.account import account_service

router = APIRouter(prefix="/api/oauth", tags=["oauth"])


@router.post("/authorize")
async def start_oauth():
    return account_service.start_oauth_session()


@router.post("/callback")
async def finish_oauth(body: OAuthCallbackRequest):
    try:
        account = account_service.finish_oauth_session(body.callback_url, body.proxy, body.session_id)
        return {"code": 0, "message": "Account added successfully", "data": account}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
