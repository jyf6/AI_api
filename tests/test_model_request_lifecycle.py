import asyncio
import io
import json
import logging

import pytest
from fastapi import HTTPException

from api.model_execution import execute_model_request
from core.admission import CapacityUnavailable, ModelAdmission, mark_model_request_started
from utils.log import RequestLogFilter, logger, python_attempt_log_context


def test_request_log_context_isolated_between_concurrent_attempts(monkeypatch):
    """同一 request_id 的 Java 重试在并发日志中仍可按尝试序号区分。"""
    async def scenario():
        owner = ModelAdmission({"gpt": 2, "gemini": 1, "doubao": 1}, 2)
        monkeypatch.setattr("core.admission.model_admission", owner)
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.addFilter(RequestLogFilter())
        logger.addHandler(handler)
        try:
            async def call(n):
                token = python_attempt_log_context.set(n)
                try:
                    logger.info("event=context_probe")
                    await asyncio.sleep(0)
                finally:
                    python_attempt_log_context.reset(token)

            await asyncio.gather(*(execute_model_request("gpt", "same-request", lambda n=n: call(n), java_attempt=n)
                                   for n in (1, 2)))
            logger.info("event=outside_probe")
        finally:
            logger.removeHandler(handler)
        lines = output.getvalue().splitlines()
        probes = [line for line in lines if "event=context_probe" in line]
        assert len(probes) == 2
        assert {line.rsplit("java_attempt=", 1)[-1] for line in probes} == {"1", "2"}
        assert all("request_id=same-request" in line for line in probes)
        assert {line.split("python_attempt=", 1)[-1].split()[0] for line in probes} == {"1", "2"}
        assert "java_attempt=" not in lines[-1]

    asyncio.run(scenario())


def test_busy_error_is_safe_only_before_upstream(monkeypatch):
    async def scenario():
        owner = ModelAdmission({"gpt": 1, "gemini": 1, "doubao": 1}, 1)
        monkeypatch.setattr("core.admission.model_admission", owner)
        async def before():
            raise CapacityUnavailable("NO_HEALTHY_ACCOUNT")
        response = await execute_model_request("gpt", "request-before", before)
        assert response.status_code == 503
        assert json.loads(response.body)["upstream_started"] is False
        async def after():
            await mark_model_request_started()
            raise CapacityUnavailable()
        with pytest.raises(HTTPException) as error:
            await execute_model_request("gpt", "request-after", after)
        assert error.value.status_code == 502
        assert owner._global_reserved == 0
    asyncio.run(scenario())


def test_disconnected_started_request_keeps_permit_until_real_completion(monkeypatch):
    async def scenario():
        owner = ModelAdmission({"gpt": 1, "gemini": 1, "doubao": 1}, 1)
        monkeypatch.setattr("core.admission.model_admission", owner)
        started, finish, ended = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def call():
            await mark_model_request_started()
            started.set()
            await finish.wait()
            ended.set()
        request = asyncio.create_task(execute_model_request("gpt", "detached", call))
        await started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert owner._global_reserved == 1
        assert owner._running["gpt"] == 1
        finish.set()
        await ended.wait()
        await asyncio.sleep(0)
        assert owner._global_reserved == 0
        assert owner._running["gpt"] == 0
    asyncio.run(scenario())


def test_disconnected_preparation_is_cancelled_and_releases_permit(monkeypatch):
    async def scenario():
        owner = ModelAdmission({"gpt": 1, "gemini": 1, "doubao": 1}, 1)
        monkeypatch.setattr("core.admission.model_admission", owner)
        started, ended = asyncio.Event(), asyncio.Event()
        async def call():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()
        request = asyncio.create_task(execute_model_request("gpt", "preparation", call))
        await started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await ended.wait()
        await asyncio.sleep(0)
        assert owner._global_reserved == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("upstream_started", [False, True])
def test_real_uvicorn_tcp_disconnect_obeys_model_start_boundary(monkeypatch, upstream_started):
    """实际 TCP 断开会产生 http.disconnect，不能只测试 task.cancel。"""
    async def scenario():
        import socket
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        owner = ModelAdmission({"gpt":1,"gemini":1,"doubao":1},1)
        monkeypatch.setattr("core.admission.model_admission",owner)
        entered, finish, ended = asyncio.Event(), asyncio.Event(), asyncio.Event()
        app = FastAPI()
        @app.post("/probe")
        async def probe(request: Request, payload: dict):
            async def call():
                if upstream_started:
                    await mark_model_request_started()
                entered.set()
                try:
                    await finish.wait()
                    return JSONResponse({"ok":True})
                finally:
                    ended.set()
            return await execute_model_request("gpt","real-disconnect",call,request=request)
        sock = socket.socket()
        sock.bind(("127.0.0.1",0))
        sock.listen()
        server = uvicorn.Server(uvicorn.Config(app,log_level="critical",lifespan="off",access_log=False))
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        writer = None
        try:
            async with asyncio.timeout(2):
                while not server.started:
                    await asyncio.sleep(.01)
            _,writer = await asyncio.open_connection("127.0.0.1",sock.getsockname()[1])
            writer.write(b"POST /probe HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}")
            await writer.drain()
            await asyncio.wait_for(entered.wait(),2)
            writer.close()
            await writer.wait_closed()
            if upstream_started:
                await asyncio.sleep(.1)
                assert not ended.is_set()
                assert owner._global_reserved == 1
                assert owner._running["gpt"] == 1
                finish.set()
            await asyncio.wait_for(ended.wait(),2)
            async with owner._condition:
                await asyncio.wait_for(owner._condition.wait_for(lambda: owner._global_reserved == 0),2)
            assert owner._running["gpt"] == 0
        finally:
            finish.set()
            if writer is not None:
                writer.close()
            server.should_exit = True
            await asyncio.wait_for(serving,3)
            sock.close()
    asyncio.run(scenario())


def test_waiting_for_gpt_account_does_not_block_gemini_global_capacity(monkeypatch):
    """账号暂忙的请求交还全局许可，另一平台仍能启动。"""
    async def scenario():
        from tests.test_cancellation_and_proxy_transaction import make_pool
        owner = ModelAdmission({"gpt":2,"gemini":1,"doubao":1},2)
        monkeypatch.setattr("core.admission.model_admission",owner)
        pool = make_pool()
        pool._accounts["one"].update(inflight=1,inflight_chat=1)
        occupied = await owner.acquire("gpt",asyncio.get_running_loop().time()+1)
        await occupied.mark_upstream_started()
        async def blocked():
            return await pool.acquire_account("chat")
        waiting = asyncio.create_task(execute_model_request("gpt","waiting-account",blocked))
        try:
            async with owner._condition:
                await asyncio.wait_for(owner._condition.wait_for(
                    lambda: owner._global_reserved == 1 and owner._waiting["gpt"] == 1),1)
            async def gemini():
                await mark_model_request_started()
                assert owner._global_reserved == 2
                return "completed"
            assert await asyncio.wait_for(execute_model_request("gemini","other-platform",gemini),1) == "completed"
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert owner._global_reserved == 1
            assert owner._waiting["gpt"] == 0
        finally:
            waiting.cancel()
            await asyncio.gather(waiting,return_exceptions=True)
            await occupied.release()
        assert owner._global_reserved == 0
    asyncio.run(scenario())
