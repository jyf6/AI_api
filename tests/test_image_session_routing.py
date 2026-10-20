"""在路由与适配器交界验证续用、换号和错误边界，不请求真实模型。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

with patch("core.database.database.list_accounts", return_value=[]):
    from api.routers import images
from api.schemas import ImageGenerationRequest
from core.account_pool import AccountUnavailable
from core.slot_sessions import SlotSessions
from tests.test_slot_sessions import RedisMemory
from tests.test_account_pool_scheduling import make_pool


@pytest.mark.parametrize("platform", ["gpt", "gemini"])
def test_route_reuses_and_replaces_only_current_account_session(monkeypatch, platform):
    async def scenario():
        cache = SlotSessions()
        cache.redis = RedisMemory()
        monkeypatch.setattr(images, "slot_sessions", cache)
        service = images.account_service if platform == "gpt" else images.gemini_account_service
        calls, unavailable = [], set()
        account = lambda aid: {"account_id": aid, "email": str(aid), "name": str(aid), "access_token": "test"}

        async def acquire(*args, account_id=None, excluded_account_ids=None):
            if account_id in unavailable:
                raise AccountUnavailable()
            return account(account_id or (2 if 1 in (excluded_account_ids or set()) else 1))

        class Backend:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def generate(self, *args, conversation_ref=None, on_conversation=None, **kwargs):
                aid = images.CURRENT_SESSION.get().account_id or (2 if 1 in unavailable else 1)
                calls.append((aid, conversation_ref))
                ref = ({"conversationId": f"conversation-{aid}"} if platform == "gpt" else
                       {"cid": f"c_{aid}", "metadata": [f"c_{aid}", "rid", "rcid", None, None, None, None, None, None, "context"]})
                await on_conversation(ref)
                return b"\x89PNG\r\n\x1a\nimage"
            generate_image_bytes = generate
            image = generate

        monkeypatch.setattr(images, "acquire_traced_account", acquire)
        monkeypatch.setattr(images, "OpenAIBackendAPI" if platform == "gpt" else "GeminiBackendAPI", Backend)
        monkeypatch.setattr(service, "release_account", lambda *args, **kwargs: None)
        monkeypatch.setattr(images, "resolve_model", lambda *args: SimpleNamespace(platform=platform, model="auto"))
        body = ImageGenerationRequest(model=f"{platform}-image", prompt="test", aspect_ratio="1:1",
                                      task_code="task", item_id=1)
        for _ in range(2):
            response = await images._generate_images(body)
            assert response.media_type == "image/png"
            assert response.body == b"\x89PNG\r\n\x1a\nimage"
            assert not any("conversation" in key or "account" in key for key in response.headers)
        unavailable.add(1)
        for _ in range(2):
            await images._generate_images(body)
        assert [aid for aid, _ in calls] == [1, 1, 2, 2]
        assert calls[0][1] is None and calls[2][1] is None
        assert calls[1][1] is not None and calls[3][1] is not None
        assert len(cache.redis.values) == 1
    asyncio.run(scenario())


def test_only_definite_session_rejections_reset_preference():
    async def scenario():
        cache = SlotSessions()
        async with cache.claim("task", 1, "gpt") as session:
            await session.save(1, {"conversationId": "old"})
            assert not await images._retry_session_rejection(TimeoutError(), "gpt")
            assert session.account_id == 1
            assert await images._retry_session_rejection(images.UpstreamHTTPError("/backend-api/f/conversation", 429, "limit"), "gpt")
            assert session.conversation_ref == {"conversationId": "old"}
            assert not await images._retry_session_rejection(images.UpstreamHTTPError("download_image_bytes", 429, "limit"), "gpt")
            assert await images._retry_session_rejection(images.ConversationUnavailable(), "gpt")
            assert session.account_id == 1 and session.conversation_ref is None
            await session.save(1, {"conversationId": "new"})
            assert await images._retry_session_rejection(AccountUnavailable(), "gpt")
            assert session.account_id is None and 1 in session.excluded_account_ids
    asyncio.run(scenario())


def test_first_account_manual_refresh_failure_switches_before_model_send(monkeypatch):
    async def scenario():
        pool = make_pool()
        async def prepare(account):
            if account["account_id"] == 1:
                pool._accounts["a"]["status"] = "error"
                raise images.CredentialUnavailable("requires manual login")
            return account
        async def acquire(service, task_type, *args, **kwargs):
            return await service.acquire_account(task_type, **kwargs)
        monkeypatch.setattr(pool, "_prepare_account_async", prepare)
        monkeypatch.setattr(images, "acquire_traced_account", acquire)
        cache = SlotSessions()
        async with cache.claim("task", 1, "gpt") as session:
            account = await images._acquire_image_account(pool, "gpt", "name", 1)
            assert account["account_id"] == session.account_id == 2
            assert session.conversation_ref is None
            assert pool._accounts["a"]["inflight"] == 0
    asyncio.run(scenario())
