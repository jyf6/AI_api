import asyncio
import sys
import time
import types

from fastapi import HTTPException

from providers.gemini.account import GeminiAccountPool
from providers.gemini.backend import GeminiBackendAPI
from api.routers import accounts
from api.schemas import GeminiAccountRequest


class FakeClient:
    created = 0

    def __init__(self, *_args, **_kwargs):
        type(self).created += 1
        self.cookies = {}
        self.client = self
        self.on_cookie_refreshed = _kwargs.get("on_cookie_refreshed")
        self.closed = False
        self._running = False

    async def init(self, **_kwargs):
        self._running = True

    async def _fetch_user_status(self):
        pass

    def _check_account_status(self, raise_error=False):
        return True

    async def generate_content(self, *_args, **_kwargs):
        return types.SimpleNamespace(text="OK")

    async def close(self):
        self.closed = True
        self._running = False


def test_gemini_client_is_reused_until_pool_shutdown(monkeypatch, tmp_path):
    FakeClient.created = 0
    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=FakeClient))
    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    account = pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")
    monkeypatch.setattr("providers.gemini.account.gemini_account_service", pool)

    async def run():
        async with GeminiBackendAPI(account) as first:
            first_client = first.client
        assert not first_client.closed
        async with GeminiBackendAPI(account) as second:
            assert second.client is first_client
        await pool.close_clients()
        assert first_client.closed

    asyncio.run(run())
    assert FakeClient.created == 2


def test_gemini_parallel_requests_keep_independent_clients(monkeypatch, tmp_path):
    """一个请求失败只关闭自己的连接，不能打断同账号正在运行的另一个请求。"""
    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=FakeClient))
    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    account = pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")
    monkeypatch.setattr("providers.gemini.account.gemini_account_service", pool)

    async def run():
        async with GeminiBackendAPI(account) as other:
            other_client = other.client
            try:
                async with GeminiBackendAPI(account) as failing:
                    assert failing.client is not other.client
                    raise RuntimeError("first request timed out")
            except RuntimeError:
                pass
            assert not other.client.closed
        # 成功请求的独立连接可继续复用。
        async with GeminiBackendAPI(account) as next_request:
            assert next_request.client is other_client
        await pool.close_clients()

    asyncio.run(run())


def test_gemini_warmup_uses_manual_cookie_as_authoritative_source(monkeypatch, tmp_path):
    FakeClient.created = 0
    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=FakeClient))
    monkeypatch.setenv("GEMINI_COOKIE_PATH", str(tmp_path))

    # Write a cached cookies file with rotated psidts
    cache_file = tmp_path / ".cached_cookies_my_psid.json"
    cache_file.write_text(
        '[{"name": "__Secure-1PSID", "value": "my_psid"}, {"name": "__Secure-1PSIDTS", "value": "new_rotated_ts"}]',
        encoding="utf-8",
    )

    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    pool.add_account("test_warmup", "__Secure-1PSID=my_psid; __Secure-1PSIDTS=old_ts")

    async def run():
        # Warmup should eagerly initialize the client
        await pool.warmup_clients()
        assert "test_warmup" in pool._clients
        # 代理只把人工双 Cookie 交给上游；上游自行决定是否使用已验证缓存。
        client = pool._clients["test_warmup"]
        assert client.cookies == {}
        assert pool._accounts["test_warmup"]["psidts"] == "old_ts"

        # 临时错误释放占用但不停止账号。
        pool.release_account("test_warmup", False, error="Google stream error 1100")
        assert pool._accounts["test_warmup"]["status"] == "active"
        assert pool._accounts["test_warmup"]["cooldown_until"] > time.time()

        # 明确未认证才停止账号。
        pool.release_account("test_warmup", False, error="401 Unauthorized")
        assert pool._accounts["test_warmup"]["status"] == "error"

        await pool.close_clients()

    asyncio.run(run())


def test_gemini_api_refresh_callback_verifies_before_persisting(monkeypatch, tmp_path):
    """续期后只由 Gemini-API 回调验证，代理不得再自行轮换 Cookie。"""
    FakeClient.created = 0
    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=FakeClient))

    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    account = pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")

    async def run():
        client = await pool.get_client(account)
        assert client.on_cookie_refreshed is not None
        client.cookies["__Secure-1PSIDTS"] = "renewed_ts"
        await client.on_cookie_refreshed(client)
        assert pool._accounts["main"]["status"] == "active"
        assert pool._accounts["main"]["psidts"] == "renewed_ts"
        await pool.close_clients()

    asyncio.run(run())


def test_discard_client_keeps_latest_cookie_cache(monkeypatch, tmp_path):
    FakeClient.created = 0
    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=FakeClient))
    monkeypatch.setenv("GEMINI_COOKIE_PATH", str(tmp_path))
    cache_file = tmp_path / ".cached_cookies_psid.json"
    cache_file.write_text("[]", encoding="utf-8")

    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    account = pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")
    # 模拟客户端运行后写入的缓存；discard 不应删除这份非人工更新缓存。
    cache_file.write_text("[]", encoding="utf-8")

    async def run():
        await pool.get_client(account)
        await pool.discard_client("main")

    asyncio.run(run())
    assert cache_file.exists()


def test_manual_cookie_update_discards_stale_cache(monkeypatch, tmp_path):
    """人工更新 Cookie 后，旧磁盘缓存不能覆盖新的 PSIDTS。"""
    monkeypatch.setenv("GEMINI_COOKIE_PATH", str(tmp_path))
    cache_file = tmp_path / ".cached_cookies_psid.json"
    cache_file.write_text(
        '[{"name": "__Secure-1PSID", "value": "psid"}, {"name": "__Secure-1PSIDTS", "value": "stale_ts"}]',
        encoding="utf-8",
    )

    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=fresh_ts")

    assert not cache_file.exists()


def test_gemini_client_rejects_unauthenticated_init(monkeypatch, tmp_path):
    """初始化获得访客 token 但账号未认证时，不能保留为可用客户端。"""
    FakeClient.created = 0
    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=FakeClient))
    monkeypatch.setattr(FakeClient, "_check_account_status", lambda _self, _raise_error=False: False)

    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    account = pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")

    async def run():
        try:
            await pool.get_client(account)
        except RuntimeError as exc:
            assert "未认证" in str(exc)
        else:
            assert False, "未认证会话不应进入客户端池"

    asyncio.run(run())
    assert "main" not in pool._clients


def test_gemini_client_retries_with_database_cookie_after_cached_session_is_unauthenticated(monkeypatch, tmp_path):
    class CacheThenDatabaseClient(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cookie_source = "Cache" if type(self).created == 1 else "Base Cookies"
            self.authenticated = type(self).created > 1

        def _check_account_status(self, raise_error=False):
            return self.authenticated

    CacheThenDatabaseClient.created = 0
    monkeypatch.setitem(
        sys.modules,
        "providers.gemini.webapi",
        types.SimpleNamespace(GeminiClient=CacheThenDatabaseClient),
    )
    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    account = pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")

    client = asyncio.run(pool.get_client(account))

    assert CacheThenDatabaseClient.created == 2
    assert client._cookie_source == "Base Cookies"
    assert "main" in pool._clients


def test_expired_refresh_cooldown_is_restored_on_startup_and_account_is_schedulable(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=FakeClient))
    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {
        "main": {
            "name": "main",
            "cookie": "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts",
            "psid": "psid",
            "psidts": "psidts",
            "status": "cooldown",
            "cooldown_until": int(time.time()) + 300,
            "failure_count": 1,
            "error_message": "temporary provider failure",
        }
    }

    assert pool.stats()["cooldown_accounts"] == 1
    pool._accounts["main"]["cooldown_until"] = int(time.time()) - 1
    asyncio.run(pool.warmup_clients())
    scheduled_account = pool.get_available_account("image")

    assert pool._accounts["main"]["status"] == "active"
    assert scheduled_account["name"] == "main"


def test_refresh_verification_uses_auth_status_without_generating_text(monkeypatch, tmp_path):
    class StatusOnlyClient(FakeClient):
        def __init__(self, *_args, **kwargs):
            super().__init__(*_args, **kwargs)
            self.status_checks = 0

        async def _fetch_user_status(self):
            self.status_checks += 1

        async def generate_content(self, *_args, **_kwargs):
            raise AssertionError("Cookie refresh verification must not generate text")

    monkeypatch.setitem(
        sys.modules,
        "providers.gemini.webapi",
        types.SimpleNamespace(GeminiClient=StatusOnlyClient),
    )
    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    account = pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")

    async def run():
        client = await pool.get_client(account)
        client.cookies["__Secure-1PSIDTS"] = "renewed-ts"
        await pool.verify_refreshed_client("main", client)
        assert client.status_checks == 1

    asyncio.run(run())
    assert pool._accounts["main"]["status"] == "active"
    assert pool._accounts["main"]["psidts"] == "renewed-ts"


def test_new_gemini_account_starts_its_refresh_client(monkeypatch):
    class FakePool:
        def __init__(self):
            self.started = None

        async def discard_client(self, name):
            pass

        def add_account(self, name, cookie, proxy):
            return {"name": name, "cookie": cookie, "proxy": proxy, "status": "active"}

        async def get_client(self, account):
            self.started = account["name"]

        def mark_refresh_verification_failed(self, _name, _error):
            raise AssertionError("valid account must not be marked unavailable")

    pool = FakePool()
    monkeypatch.setattr(accounts, "gemini_account_service", pool)

    response = asyncio.run(accounts.add_gemini_account(GeminiAccountRequest(
        name="main", cookie="__Secure-1PSID=psid; __Secure-1PSIDTS=psidts"
    )))

    assert response["status"] == "active"
    assert pool.started == "main"


def test_gemini_account_test_rejects_an_unauthenticated_client(monkeypatch):
    """连通性探测遇到认证错误时停用账号。"""
    class FakePool:
        def __init__(self):
            self._lock = __import__("threading").Lock()
            self._accounts = {"main": {"name": "main"}}
            self.health = None

        def set_account_health(self, *args, **kwargs):
            self.health = (args, kwargs)

        def is_auth_error(self, error):
            return "未认证" in str(error)

        async def discard_client(self, _name):
            pass

    class FakeBackend:
        def __init__(self, _account):
            self.client = types.SimpleNamespace(generate_content=self._generate_content)

        async def _generate_content(self, *_args, **_kwargs):
            raise RuntimeError("Gemini Cookie 未认证")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    pool = FakePool()
    monkeypatch.setattr(accounts, "gemini_account_service", pool)
    monkeypatch.setattr(accounts, "GeminiBackendAPI", FakeBackend)

    try:
        asyncio.run(accounts.test_gemini_account("main"))
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "未认证" in exc.detail
    else:
        assert False, "an unauthenticated Gemini client must fail the health check"

    assert pool.health[1]["healthy"] is False


def test_gemini_account_test_sends_ok_probe_and_validates_response(monkeypatch):
    class FakePool:
        def __init__(self):
            self._lock = __import__("threading").Lock()
            self._accounts = {"main": {"name": "main"}}
            self.health = None

        def set_account_health(self, *args, **kwargs):
            self.health = (args, kwargs)

    class FakeClient:
        def __init__(self):
            self.probe = None

        async def generate_content(self, prompt, **kwargs):
            self.probe = (prompt, kwargs)
            return types.SimpleNamespace(text="OK.")

    class FakeBackend:
        instance = None

        def __init__(self, _account):
            self.client = FakeClient()
            type(self).instance = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    pool = FakePool()
    monkeypatch.setattr(accounts, "gemini_account_service", pool)
    monkeypatch.setattr(accounts, "GeminiBackendAPI", FakeBackend)

    response = asyncio.run(accounts.test_gemini_account("main"))

    assert response["code"] == 0
    assert response["message"] == "Gemini text connectivity test passed"
    assert FakeBackend.instance.client.probe == ("请只回复 OK。", {"temporary": True})
    assert pool.health[1]["healthy"] is True


def test_full_gemini_cookie_header_is_reduced_to_the_auth_pair():
    pool = GeminiAccountPool()
    pool._platform = ""
    pool._accounts = {}

    account = pool.add_account(
        "main",
        "SID=sid; NID=nid; __Secure-1PSID=psid; __Secure-1PSIDTS=psidts; SAPISID=sapisid",
    )

    assert account["cookie"] == "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts"


def test_gemini_accounts_cannot_share_a_psid():
    pool = GeminiAccountPool()
    pool._platform = ""
    pool._accounts = {}
    pool.add_account("first", "__Secure-1PSID=shared; __Secure-1PSIDTS=first-ts")

    try:
        pool.add_account("second", "__Secure-1PSID=shared; __Secure-1PSIDTS=second-ts")
    except ValueError as exc:
        assert "已被账号" in str(exc)
    else:
        assert False, "accounts sharing one 1PSID must be rejected"


def test_gemini_client_bootstraps_with_only_the_auth_pair(monkeypatch):
    class CaptureClient(FakeClient):
        instance = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.auth_args = args[:2]
            self.init_kwargs = None
            type(self).instance = self

        async def init(self, **kwargs):
            self.init_kwargs = kwargs

    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=CaptureClient))
    monkeypatch.setenv("GEMINI_REFRESH_INTERVAL", "720")
    pool = GeminiAccountPool()
    pool._platform = ""
    pool._accounts = {}
    account = pool.add_account(
        "main",
        "SID=sid; __Secure-1PSID=psid; __Secure-1PSIDTS=psidts; SAPISID=sapisid",
    )

    asyncio.run(pool.get_client(account))

    assert CaptureClient.instance.auth_args == ("psid", "psidts")
    assert CaptureClient.instance.cookies == {}
    assert CaptureClient.instance.init_kwargs["refresh_interval"] == 720
