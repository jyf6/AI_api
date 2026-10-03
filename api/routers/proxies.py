from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import ProxyNodeRequest, ProxyStatusRequest
from core.database import database

router = APIRouter(prefix="/api/proxies", tags=["proxies"])


@router.get("")
async def list_proxies():
    """代理 URL 含认证信息，只返回节点元数据和绑定数量。"""
    return {"proxies": database.list_proxy_nodes()}


@router.post("")
async def create_proxy(body: ProxyNodeRequest):
    try:
        proxy_id = database.create_proxy_node(body.name, body.proxy_url)
        return {"id": proxy_id, "name": body.name, "status": "active"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/{proxy_id}")
async def update_proxy(proxy_id: int, body: ProxyNodeRequest):
    try:
        if not database.update_proxy_node(proxy_id, body.name, body.proxy_url, body.status):
            raise HTTPException(status_code=404, detail="Proxy not found")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # 已加载账号同步节点状态，禁用操作立即从调度池剔除账号。
    from providers.openai.account import account_service
    from providers.doubao.account import doubao_account_service
    from providers.gemini.account import gemini_account_service

    for pool in (account_service, doubao_account_service, gemini_account_service):
        pool.refresh_proxy_node(proxy_id, body.proxy_url, body.status)
    with gemini_account_service._lock:
        gemini_names = [name for name, account in gemini_account_service._accounts.items()
                        if account.get("proxy_id") == proxy_id]
    for name in gemini_names:
        await gemini_account_service.discard_client(name)
    return {"id": proxy_id, "name": body.name, "status": body.status}


@router.put("/{proxy_id}/status")
async def update_proxy_status(proxy_id: int, body: ProxyStatusRequest):
    if not database.update_proxy_status(proxy_id, body.status):
        raise HTTPException(status_code=404, detail="Proxy not found")
    from providers.openai.account import account_service
    from providers.doubao.account import doubao_account_service
    from providers.gemini.account import gemini_account_service

    for pool in (account_service, doubao_account_service, gemini_account_service):
        pool.refresh_proxy_status(proxy_id, body.status)
    return {"id": proxy_id, "status": body.status}


@router.delete("/{proxy_id}")
async def delete_proxy(proxy_id: int):
    try:
        if not database.delete_proxy_node(proxy_id):
            raise HTTPException(status_code=404, detail="Proxy not found")
        return {"id": proxy_id, "status": "deleted"}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        from pymysql.err import IntegrityError
        if isinstance(exc, IntegrityError):
            raise HTTPException(status_code=409, detail="代理仍绑定账号，请先解绑")
        raise
