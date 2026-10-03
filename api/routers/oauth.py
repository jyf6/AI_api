from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import OAuthCallbackRequest
from providers.openai.account import account_service
from core.database import database
from api.routers.accounts import _proxy_for_account

router = APIRouter(prefix="/api/oauth", tags=["oauth"])


@router.post("/authorize")
async def start_oauth():
    return account_service.start_oauth_session()


@router.post("/callback")
async def finish_oauth(body: OAuthCallbackRequest):
    try:
        proxy, proxy_id = body.proxy.strip(), None
        if body.proxy_id:
            node = database.get_proxy_node(body.proxy_id)
            if not node or node["status"] != "active":
                raise ValueError("Proxy not found or disabled")
            proxy, proxy_id = node["proxy_url"], body.proxy_id
        account = account_service.finish_oauth_session(body.callback_url, proxy, body.session_id, proxy_id)
        return {"code": 0, "message": "Account added successfully", "data": {"email": account["email"], "status": account["status"]}}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
