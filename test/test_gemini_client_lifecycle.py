import asyncio
import sys
import time
import types

from curl_cffi.requests import Cookies
from fastapi import HTTPException

from providers.gemini.account import GeminiAccountPool, _restore_cookies
from providers.gemini.backend import GeminiBackendAPI, _create_reference_images
from providers.gemini.webapi.utils.rotate_1psidts import rotate_1psidts
from providers.gemini.webapi.utils.upload_file import parse_file_name
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


def test_warmup_initializes_all_accounts_with_bounded_concurrency(monkeypatch, tmp_path):
    class SlowClient(FakeClient):
        in_progress = 0
        peak = 0

        async def init(self, **_kwargs):
            type(self).in_progress += 1
            type(self).peak = max(type(self).peak, type(self).in_progress)
            try:
                await asyncio.sleep(0.01)
                self._running = True
            finally:
                type(self).in_progress -= 1

    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=SlowClient))
    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "accounts.json"
    pool._platform = ""
    pool._accounts = {}
    for i in range(12):
        pool.add_account(f"account-{i}", f"__Secure-1PSID=psid-{i}; __Secure-1PSIDTS=ts-{i}")

    async def run():
        assert pool.capacity()["available_slots"] == 0
        await pool.warmup_clients()
        assert len(pool._clients) == 12
        assert 1 < SlowClient.peak <= 5
        assert pool.capacity()["available_slots"] == 12 * pool.MAX_INFLIGHT_TOTAL
        await pool.close_clients()

    asyncio.run(run())


def test_verified_cookie_jar_is_stored_in_account_credentials(monkeypatch, tmp_path):
    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "accounts.json"
    pool._platform = ""
    pool._accounts = {}
    pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=old")
    saved = []
    monkeypatch.setattr("core.account_pool.database.import_account", lambda *args: saved.append(args))
    pool._platform = "gemini"
    jar = Cookies()
    jar.set("__Secure-1PSID", "psid", domain=".google.com", secure=True)
    jar.set("__Secure-1PSIDTS", "renewed", domain=".google.com", secure=True)
    jar.set("SIDCC", "extra", domain=".google.com", secure=True)

    pool.merge_cookie("main", dict(jar), jar)

    credentials = saved[-1][2]
    assert credentials["psidts"] == "renewed"
    assert {entry["name"] for entry in credentials["cookie_jar"]} == {
        "__Secure-1PSID", "__Secure-1PSIDTS", "SIDCC"
    }
    restored = _restore_cookies(credentials)
    assert restored is not None and restored.get("SIDCC") == "extra"
    assert "cookie_jar" not in pool.list_accounts()[0]


def test_failed_cookie_database_write_is_retried(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=FakeClient))
    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "accounts.json"
    pool._platform = ""
    pool._accounts = {}
    account = pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=old")

    async def run():
        client = await pool.get_client(account)
        pool._platform = "gemini"
        client.cookies["__Secure-1PSIDTS"] = "renewed"

        def unavailable(*_args):
            raise RuntimeError("database unavailable")

        monkeypatch.setattr("core.account_pool.database.import_account", unavailable)
        await pool.verify_refreshed_client("main", client)
        assert "main" in pool._pending_cookie_saves

        saved = []
        monkeypatch.setattr("core.account_pool.database.import_account", lambda *args: saved.append(args))
        pool._retry_pending_cookie_saves()
        assert "main" not in pool._pending_cookie_saves
        assert saved[-1][2]["psidts"] == "renewed"
        await pool.close_clients()

    asyncio.run(run())


def test_rotation_only_reports_a_new_cookie(monkeypatch, tmp_path):
    class Response:
        status_code = 200
        http_version = 2

        def raise_for_status(self):
            pass

    class Session:
        def __init__(self):
            self.cookies = Cookies()
            self.cookies.set("__Secure-1PSIDTS", "old", domain=".google.com", secure=True)
            self.rotate = False

        async def post(self, **_kwargs):
            if self.rotate:
                self.cookies.set("__Secure-1PSIDTS", "new", domain=".google.com", secure=True)
            return Response()

    monkeypatch.setenv("GEMINI_COOKIE_PATH", str(tmp_path))
    session = Session()
    assert asyncio.run(rotate_1psidts(session)) is None
    session.rotate = True
    assert asyncio.run(rotate_1psidts(session)) == "new"
    assert list(tmp_path.iterdir()) == []


def test_reference_image_stays_in_memory_with_its_file_type(monkeypatch):
    monkeypatch.setattr(
        "providers.gemini.backend._read_image_source",
        lambda _source: (b"image-bytes", ".png"),
    )
    with _create_reference_images(["source"]) as images:
        assert len(images) == 1
        assert images[0].getvalue() == b"image-bytes"
        assert parse_file_name(images[0]).endswith(".png")
    assert images[0].closed


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


def test_gemini_warmup_uses_database_cookie(monkeypatch, tmp_path):
    FakeClient.created = 0
    monkeypatch.setitem(sys.modules, "providers.gemini.webapi", types.SimpleNamespace(GeminiClient=FakeClient))
    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=ts")

    async def run():
        await pool.warmup_clients()
        assert "main" in pool._clients
        assert pool.capacity()["available_slots"] == pool.MAX_INFLIGHT_TOTAL
        await pool.close_clients()
        assert pool.capacity()["available_slots"] == 0

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


def test_manual_cookie_update_replaces_stored_cookie_jar(monkeypatch, tmp_path):
    pool = GeminiAccountPool()
    pool._data_file = tmp_path / "gemini_accounts.json"
    pool._platform = ""
    pool._accounts = {}
    pool.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=old_ts")
    pool._accounts["main"]["cookie_jar"] = [{"name": "__Secure-1PSIDTS", "value": "old_ts"}]

    pool.update_cookie("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=fresh_ts")

    assert pool._accounts["main"]["psidts"] == "fresh_ts"
    assert pool._accounts["main"]["cookie_jar"] == []


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

        def add_account(self, name, cookie, proxy, proxy_id=None):
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
