"""模型入口共用准入与断连收尾，容量错误不进入模型重试。"""
import asyncio
import time

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from core.admission import CapacityUnavailable, current_permit, get_model_admission
from utils.log import logger, proxy_log_ref, request_log_context, stable_log_ref


async def _wait_disconnect(request: Request):
    """请求正文已由 FastAPI 解析；等待 Uvicorn 实际发送的断连事件。"""
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


async def acquire_traced_account(service, task_type: str, platform: str, key_field: str, python_attempt: int):
    """在实际选中账号处记录等待耗时和匿名账号、代理引用。"""
    started = time.monotonic()
    try:
        account = await service.acquire_account(task_type)
    except Exception as exc:
        reason = "NO_ACCOUNT" if isinstance(exc, CapacityUnavailable) else type(exc).__name__
        logger.warning("event=account_acquire_finished outcome=failed platform=%s task_type=%s python_attempt=%d duration_ms=%d reason_code=%s",
                       platform, task_type, python_attempt, int((time.monotonic() - started) * 1000), reason)
        raise
    logger.info("event=account_acquire_finished outcome=success platform=%s task_type=%s python_attempt=%d account_ref=%s proxy_ref=%s duration_ms=%d",
                platform, task_type, python_attempt, stable_log_ref(f"{platform}-account", account[key_field]),
                proxy_log_ref(account), int((time.monotonic() - started) * 1000))
    return account


async def execute_model_request(platform: str, request_id: str, call, request: Request | None = None,
                                java_attempt: int | None = None):
    permit = None

    async def work():
        nonlocal permit
        context_token = request_log_context.set((request_id, java_attempt))
        deadline = asyncio.get_running_loop().time() + 30
        waiting_started = time.monotonic()
        try:
            try:
                permit = await get_model_admission().acquire(platform, deadline)
            except CapacityUnavailable:
                logger.warning("event=admission_wait_finished outcome=denied platform=%s duration_ms=%d reason_code=CAPACITY_TIMEOUT",
                               platform, int((time.monotonic() - waiting_started) * 1000))
                raise
            logger.info("event=admission_wait_finished outcome=success platform=%s duration_ms=%d",
                        platform, int((time.monotonic() - waiting_started) * 1000))
            token = current_permit.set(permit)
            try:
                return await call()
            except CapacityUnavailable as exc:
                if permit.upstream_started:
                    raise HTTPException(status_code=502, detail="Upstream attempt started but no account available for retry",
                                        headers={"X-Request-ID": request_id}) from exc
                raise
            finally:
                current_permit.reset(token)
                await permit.release()
        finally:
            request_log_context.reset(context_token)

    worker = asyncio.create_task(work())

    async def abandon():
        # 同一个事件循环内检查与 cancel 之间不让出执行权，避免取消与上游开始交错。
        if permit is None or not permit.upstream_started:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            return
        # 已发送模型请求继续执行，唯一 finally 在实际结束时归还资源。
        def observe(done):
            if done.cancelled():
                return
            try:
                done.result()
            except Exception as exc:
                logger.warning("Detached model request finished with error request_id=%s reason=%s",
                               request_id, type(exc).__name__)
        worker.add_done_callback(observe)

    disconnected = asyncio.create_task(_wait_disconnect(request)) if request is not None else None
    try:
        if disconnected is not None:
            completed, _ = await asyncio.wait({worker, disconnected}, return_when=asyncio.FIRST_COMPLETED)
            if worker not in completed:
                await disconnected
                await abandon()
                logger.info("Model HTTP client disconnected request_id=%s upstream_started=%s",
                            request_id, permit is not None and permit.upstream_started)
                # 客户端已断开，此响应只结束 ASGI 生命周期，不会改变业务任务为容量错误。
                return Response(status_code=499)
        return await asyncio.shield(worker)
    except CapacityUnavailable as exc:
        return JSONResponse(status_code=503, content=exc.payload(platform, request_id),
                            headers={"X-Request-ID": request_id, "Retry-After": "5"})
    except asyncio.CancelledError:
        await abandon()
        raise
    finally:
        if disconnected is not None:
            disconnected.cancel()
            await asyncio.gather(disconnected, return_exceptions=True)
