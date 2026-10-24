"""
Simulation and regression tests for all 6 core scenarios:
1. 限流后原绑定保留，冷却中不调用上游直接返回结构化429与剩余时间。
2. 额度不足能到达前端，保留绑定，返回 ACCOUNT_QUOTA_EXHAUSTED 与倒计时。
3. 读取超时不会重复提交生图（POST 仅 1 次）。
4. 换号后旧请求不能恢复原绑定（在途写回保护生效）。
5. 错误选图格式能在同会话纠正（INITIAL_ITEM_PRODUCT_SELECTION 严格单行与范围校验）。
6. 初版、单张调整、批量调整都能复用对应槽位会话。
"""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import pytest
from fastapi import HTTPException

with patch("core.database.database.list_accounts", return_value=[]):
    from api.routers import chat, images
    from api.routers.chat import get_anomaly_correction_prompt
from api.schemas import ChatCompletionRequest, ImageGenerationRequest
from core.account_pool import BaseAccountPool, BoundAccountCooldownError
from core.slot_sessions import SlotSessions, CURRENT_SESSION
from tests.test_slot_sessions import RedisMemory
from utils.helper import ImageQuotaExceededError, UpstreamHTTPError, is_read_timeout_error, is_connect_error


@pytest.mark.parametrize("platform", ["gpt", "gemini"])
def test_cooldown_between_reservation_and_send_is_structured_for_java(monkeypatch, platform):
    async def scenario():
        monkeypatch.setattr(images, "resolve_model", lambda *_: SimpleNamespace(platform=platform, model="auto"))
        until = time.time() + 60
        monkeypatch.setattr(images, "_generate_images_once", AsyncMock(side_effect=BoundAccountCooldownError(
            1, 60, "rate limit 429", cooldown_until=until)))
        body = ImageGenerationRequest(model="test", prompt="test", aspect_ratio="1:1")
        with pytest.raises(HTTPException) as caught:
            await images._generate_images(body)
        error = caught.value
        detail = json.loads(error.detail)
        assert error.status_code == 429
        assert detail["reason"] == "ACCOUNT_RATE_LIMITED"
        assert detail["code"] == "ACCOUNT_RATE_LIMITED"
        assert detail["platform"] == platform
        assert detail["cooldownUntil"] == int(until)
        assert detail["retryAfter"] == 60
        assert error.headers["Retry-After"] == "60"
        assert error.headers["X-Request-ID"]
        assert images._generate_images_once.await_count == 1
    asyncio.run(scenario())


def test_scenario_1_rate_limit_retains_binding_and_immediate_return(monkeypatch):
    """场景1：限流后原绑定保留，冷却中再次请求直接返回429与剩余时间，不调用上游"""
    async def scenario():
        redis = RedisMemory()
        cache = SlotSessions()
        cache.redis = redis
        monkeypatch.setattr(images, "slot_sessions", cache)

        # 预设已有槽位绑定：account 1, conv1
        slot_key = "flexi:ai:slot-session:task1:1:gpt"
        await redis.set(slot_key, json.dumps({
            "version": 1, "accountId": 1, "conversationRef": {"conversationId": "conv1"}
        }), ex=3600)

        # 模拟账号池中 account 1 处于限流冷却中 (剩余 100 秒)
        fake_account = {
            "account_id": 1,
            "email": "user1@example.com",
            "status": "cooldown",
            "cooldown_until": time.time() + 100,
            "error_message": "rate limit 429",
            "inflight": 0,
            "inflight_image": 0,
            "inflight_chat": 0,
        }
        monkeypatch.setattr(images.account_service, "_accounts", {"user1@example.com": fake_account})
        monkeypatch.setattr(images, "resolve_model", lambda *_: SimpleNamespace(platform="gpt", model="auto"))

        # 验证直接调用 _acquire_image_account 时：因为账号处于冷却中，立即抛出 429 HTTPException
        with pytest.raises(HTTPException) as exc_info:
            async with cache.claim("task1", 1, "gpt"):
                await images._acquire_image_account(images.account_service, "gpt", "email", 1)

        assert exc_info.value.status_code == 429
        detail = json.loads(exc_info.value.detail)
        assert detail["code"] == "ACCOUNT_RATE_LIMITED"
        assert detail["retry_after"] > 0
        assert abs(detail["cooldownUntil"] - fake_account["cooldown_until"]) < 1
        assert detail["platform"] == "gpt"
        # 验证 Redis 中的槽位会话绑定未被删除！
        assert redis.values.get(slot_key) is not None
        assert "conv1" in redis.values[slot_key]

    asyncio.run(scenario())


def test_scenario_2_quota_exhausted_returns_structured_payload(monkeypatch):
    """场景2：额度耗尽异常能直接向外抛出结构化429与倒计时，且保留槽位会话"""
    async def scenario():
        redis = RedisMemory()
        cache = SlotSessions()
        cache.redis = redis
        monkeypatch.setattr(images, "slot_sessions", cache)

        slot_key = "flexi:ai:slot-session:task2:1:gpt"
        await redis.set(slot_key, json.dumps({
            "version": 1, "accountId": 1, "conversationRef": {"conversationId": "conv2"}
        }), ex=3600)

        monkeypatch.setattr(images.account_service, "release_account", lambda *args, **kwargs: None)
        monkeypatch.setattr(images, "resolve_model", lambda *_: SimpleNamespace(platform="gpt", model="auto"))

        fake_acc = {"account_id": 1, "email": "user2@test.com", "access_token": "token", "inflight": 1}
        monkeypatch.setattr(images, "_acquire_image_account", AsyncMock(return_value=fake_acc))

        class QuotaFailBackend:
            def __init__(self, *args, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def generate_image_bytes(self, *args, **kwargs):
                raise ImageQuotaExceededError("图像生成请求上限，请在 3 小时后重试", retry_after=10800)

        monkeypatch.setattr(images, "OpenAIBackendAPI", QuotaFailBackend)

        body = ImageGenerationRequest(model="gpt-image", prompt="test", aspect_ratio="1:1",
                                      task_code="task2", item_id=1)

        with pytest.raises(HTTPException) as exc_info:
            await images._generate_images(body)

        assert exc_info.value.status_code == 429
        detail = json.loads(exc_info.value.detail)
        assert detail["code"] == "ACCOUNT_QUOTA_EXHAUSTED"
        assert detail["retry_after"] == 10800
        assert detail["retryAfter"] == 10800
        assert detail["platform"] == "gpt"
        assert abs(detail["cooldownUntil"] - time.time() - 10800) < 2
        assert "180 分钟" in detail["message"]
        # 槽位绑定完好保留
        assert redis.values.get(slot_key) is not None
        assert "conv2" in redis.values[slot_key]

    asyncio.run(scenario())


def test_scenario_3_read_timeout_never_reposts_image_generation(monkeypatch):
    """场景3：生成读取超时 (Thinking / Read Timeout) 绝不重复 POST 生图"""
    async def scenario():
        post_count = 0
        from curl_cffi.requests.errors import RequestsError
        
        # 模拟读取超时错误 (errcode=28 为 CURLE_OPERATION_TIMEDOUT)
        read_timeout_exc = RequestsError("Operation timed out after 30000 milliseconds with 0 bytes received", code=28)
        assert is_read_timeout_error(read_timeout_exc)
        assert not is_connect_error(read_timeout_exc)

        redis = RedisMemory()
        cache = SlotSessions()
        cache.redis = redis
        monkeypatch.setattr(images, "slot_sessions", cache)
        monkeypatch.setattr(images, "resolve_model", lambda *_: SimpleNamespace(platform="gpt", model="auto"))

        fake_acc = {"account_id": 1, "email": "user3@test.com", "access_token": "token", "inflight": 1}
        monkeypatch.setattr(images, "_acquire_image_account", AsyncMock(return_value=fake_acc))
        monkeypatch.setattr(images.account_service, "release_account", lambda *args, **kwargs: None)

        class TimeoutBackend:
            def __init__(self, *args, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def generate_image_bytes(self, *args, **kwargs):
                nonlocal post_count
                post_count += 1
                raise read_timeout_exc

        monkeypatch.setattr(images, "OpenAIBackendAPI", TimeoutBackend)

        body = ImageGenerationRequest(model="gpt-image", prompt="test", aspect_ratio="1:1",
                                      task_code="task3", item_id=1)

        with pytest.raises(HTTPException) as exc_info:
            await images._generate_images(body)

        # 验证生图 POST 仅发生了一次，未进行二次 POST！
        assert post_count == 1
        assert exc_info.value.status_code == 504
        detail = json.loads(exc_info.value.detail)
        assert detail["code"] == "IMAGE_READ_TIMEOUT"
        assert detail["upstreamState"] == "SUBMITTED"

    asyncio.run(scenario())


def test_scenario_4_switch_account_protects_against_inflight_writeback():
    """场景4：换号后（Redis键被清除），在途旧请求结束时不能将旧绑定写回"""
    async def scenario():
        redis = RedisMemory()
        cache = SlotSessions()
        cache.redis = redis

        slot_key = "flexi:ai:slot-session:task4:1:gpt"
        # 初始状态：槽位已有绑定
        await redis.set(slot_key, json.dumps({
            "version": 1, "accountId": 1, "conversationRef": {"conversationId": "old_conv"}
        }), ex=3600)

        async with cache.claim("task4", 1, "gpt") as session:
            assert session.initial_existed is True
            assert session.account_id == 1

            # 模拟在途期间用户调用了 switchAccount 接口，从外部删除了该 Redis 键
            await redis.delete(slot_key)
            assert await redis.exists(slot_key) is False

            # 在途旧请求执行完成，尝试 save 旧会话
            await session.save(1, {"conversationId": "old_conv_should_not_restore"})

            # 验证写回被拦截，Redis 键依然不存在，未被复活！
            assert await redis.exists(slot_key) is False
            assert redis.values.get(slot_key) is None

    asyncio.run(scenario())


def test_scenario_5_product_selection_error_correction_in_same_session():
    """场景5：选图阶段严格验证单行或None，拒绝跨行、文字包装与超范围编号"""
    # 1. 包含文字包装的回复：严格触发纠偏
    is_anomaly, prompt = get_anomaly_correction_prompt("推荐产品图如下：\nSelected product images: 1, 2", "INITIAL_ITEM_PRODUCT_SELECTION")
    assert is_anomaly is True
    assert "格式不对" in prompt

    # 2. 超出素材范围的编号 (假设当前素材拼图为2张，最多18张，模型输出了 99)：触发纠偏
    is_anomaly, prompt = get_anomaly_correction_prompt("Selected product images: 99", "INITIAL_ITEM_PRODUCT_SELECTION", image_count=2)
    assert is_anomaly is True
    assert "超出当前素材范围" in prompt

    # 3. 合法单行编号：直接通过
    is_anomaly, _ = get_anomaly_correction_prompt("Selected product images: 1, 2", "INITIAL_ITEM_PRODUCT_SELECTION", image_count=2)
    assert is_anomaly is False

    # 4. 合法 None：直接通过
    is_anomaly, _ = get_anomaly_correction_prompt("Selected product images: None", "INITIAL_ITEM_PRODUCT_SELECTION")
    assert is_anomaly is False


def test_scenario_6_multi_stage_reuse_slot_session(monkeypatch):
    """场景6：初版、单张调整、批量调整都能正确复用对应槽位会话"""
    async def scenario():
        redis = RedisMemory()
        cache = SlotSessions()
        cache.redis = redis
        monkeypatch.setattr(images, "slot_sessions", cache)
        monkeypatch.setattr(images, "resolve_model", lambda *_: SimpleNamespace(platform="gpt", model="auto"))

        saved_calls = []
        fake_acc = {"account_id": 1, "email": "user6@test.com", "access_token": "token", "inflight": 1}
        monkeypatch.setattr(images, "_acquire_image_account", AsyncMock(return_value=fake_acc))
        monkeypatch.setattr(images.account_service, "release_account", lambda *args, **kwargs: None)

        class ReusableBackend:
            def __init__(self, *args, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def generate_image_bytes(self, *args, conversation_ref=None, on_conversation=None, **kwargs):
                saved_calls.append(conversation_ref)
                ref = {"conversationId": "session_persistent_id"}
                if on_conversation:
                    await on_conversation(ref)
                return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

        monkeypatch.setattr(images, "OpenAIBackendAPI", ReusableBackend)

        # 阶段 1: 初版生图 INITIAL_IMAGE_GENERATION
        body1 = ImageGenerationRequest(model="gpt-image", prompt="init", aspect_ratio="1:1",
                                       task_code="task6", item_id=1, stage="INITIAL_IMAGE_GENERATION")
        resp1 = await images._generate_images(body1)
        assert resp1.status_code == 200
        assert saved_calls[0] is None # 首次没有旧会话

        # 阶段 2: 单图微调 SINGLE_ADJUST_GENERATION
        body2 = ImageGenerationRequest(model="gpt-image", prompt="single adjust", aspect_ratio="1:1",
                                       task_code="task6", item_id=1, stage="SINGLE_ADJUST_GENERATION")
        resp2 = await images._generate_images(body2)
        assert resp2.status_code == 200
        # 验证成功复用阶段 1 的 session_persistent_id！
        assert saved_calls[1] == {"conversationId": "session_persistent_id"}

        # 阶段 3: 套图批量重绘 SUITE_ADJUST_GENERATION
        body3 = ImageGenerationRequest(model="gpt-image", prompt="suite adjust", aspect_ratio="1:1",
                                       task_code="task6", item_id=1, stage="SUITE_ADJUST_GENERATION")
        resp3 = await images._generate_images(body3)
        assert resp3.status_code == 200
        # 验证再次成功复用 session_persistent_id！
        assert saved_calls[2] == {"conversationId": "session_persistent_id"}

    asyncio.run(scenario())
