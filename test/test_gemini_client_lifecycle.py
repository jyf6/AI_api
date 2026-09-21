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

    async def init(self, **_kwargs):
        pass

    def _check_account_status(self, raise_error=False):
        return True

    async def generate_content(self, *_args, **_kwargs):
        return types.SimpleNamespace(text="OK")

    async def close(self):
        self.closed = True


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
    assert FakeClient.created == 1


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
        # 初始化不主动轮换，数据库中的人工 Cookie 不能被磁盘缓存覆盖。
        client = pool._clients["test_warmup"]
        assert client.cookies.get("__Secure-1PSIDTS") == "old_ts"
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
    """The management health check must not treat guest text as account health."""
    class FakePool:
        def __init__(self):
            self._lock = __import__("threading").Lock()
            self._accounts = {"main": {"name": "main"}}
            self.released = None

        def release_account(self, *args, **kwargs):
            self.released = (args, kwargs)

    class FakeBackend:
        def __init__(self, _account):
            self.client = types.SimpleNamespace(
                _check_account_status=lambda: False,
                generate_content=lambda *_args, **_kwargs: None,
            )

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

    assert pool.released[1]["success"] is False


def test_gemini_account_test_requires_a_temporary_real_response(monkeypatch):
    class FakePool:
        def __init__(self):
            self._lock = __import__("threading").Lock()
            self._accounts = {"main": {"name": "main"}}
            self.released = None

        def release_account(self, *args, **kwargs):
            self.released = (args, kwargs)

    class FakeClient:
        def __init__(self):
            self.request = None

        def _check_account_status(self):
            return True

        async def generate_content(self, prompt, *, temporary):
            self.request = (prompt, temporary)
            return types.SimpleNamespace(text="OK")

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
    assert FakeBackend.instance.client.request == ("请只回复 OK。", True)
    assert pool.released[1]["success"] is True
