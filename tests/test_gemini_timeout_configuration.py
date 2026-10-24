"""验证真实SDK初始化后，HTTP分段超时不会污染watchdog数值预算。"""
import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from providers.gemini.webapi.client import GeminiClient
from providers.gemini.webapi.exceptions import APIError


def test_transport_timeout_and_watchdog_keep_separate_types(monkeypatch):
    async def scenario():
        class Transport:
            cookies = {}
            @contextlib.asynccontextmanager
            async def stream(self, *args, **kwargs):
                async def chunks():
                    if False:
                        yield b''
                yield SimpleNamespace(status_code=200, aiter_content=chunks)
            async def close(self):
                pass
        transport = Transport()
        initialized = SimpleNamespace(client=transport, cookie_source='test', access_token='fake',
            build_label=None, session_id=None, language='en', push_id='test')
        monkeypatch.setattr('providers.gemini.webapi.client.get_access_token', AsyncMock(return_value=initialized))
        monkeypatch.setattr(GeminiClient, '_init_rpc', AsyncMock())
        client = GeminiClient()
        await client.init(timeout=240, watchdog_timeout=240, auto_refresh=False)
        try:
            assert client.timeout == 240
            assert transport.timeout == (15, 240)
            # 空响应应报告上游无结果，不能在进入读取前因tuple比较失败。
            with pytest.raises(APIError) as caught:
                await anext(client._generate('test'))
            assert not isinstance(caught.value.__cause__, TypeError)
            assert 'TypeError' not in str(caught.value)
        finally:
            await client.close()
    asyncio.run(scenario())
