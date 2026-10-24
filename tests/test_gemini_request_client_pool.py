"""Gemini 业务请求必须使用账号操作阶段预热好的独立连接。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

with patch("core.database.database.list_accounts", return_value=[]):
    from providers.gemini.account import GeminiAccountPool


def test_account_preparation_warms_eight_reusable_request_clients(monkeypatch):
    async def scenario():
        with patch("core.database.database.list_accounts", return_value=[]):
            pool = GeminiAccountPool()
        pool._accounts["account"] = {
            "name": "account", "account_id": 17, "credential_version": 3,
            "status": "active", "psid": "psid", "psidts": "psidts", "proxy": "",
        }

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self._running = False
                self.cookies = {}
                self.closed = False

            async def init(self, **kwargs):
                self._running = True

            def _check_account_status(self):
                return self._running

            async def close(self):
                self.closed = True
                self._running = False

        monkeypatch.setattr("providers.gemini.webapi.GeminiClient", FakeClient)

        async def ready_master(_account):
            return SimpleNamespace(cookies={}, _running=True, _check_account_status=lambda: True)

        monkeypatch.setattr(pool, "get_client", ready_master)
        await pool.warm_request_clients("account")

        assert len(pool._idle_request_clients["account"]) == 8
        assert pool._request_client_totals["account"][0] == 8

        request_client, generation = await pool.get_request_client(pool.account_snapshot("account"))
        assert generation == 0
        assert request_client in pool._idle_request_clients["account"] or len(pool._idle_request_clients["account"]) == 7
        await pool.release_request_client("account", request_client, generation, success=True)
        assert len(pool._idle_request_clients["account"]) == 8

        await pool.release_request_client("account", request_client, generation, success=False)
        assert request_client.closed
        assert pool._request_client_totals["account"][0] == 7

    asyncio.run(scenario())
