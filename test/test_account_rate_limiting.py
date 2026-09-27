from pathlib import Path
from threading import Thread
import pytest
from core.account_pool import BaseAccountPool


class DummyPool(BaseAccountPool):
    PROVIDER_NAME = "Dummy"
    MIN_DISPATCH_INTERVAL_SECONDS = 0


def create_test_pool():
    data_file = Path("/tmp/dummy_accounts.json")
    p = DummyPool(data_file)
    p._accounts = {
        "acc1": {
            "name": "acc1",
            "status": "active",
            "inflight": 0,
            "inflight_image": 0,
            "inflight_chat": 0,
            "cooldown_until": 0.0,
            "last_used_at": 0,
        }
    }
    return p


def test_dual_channel_concurrency():
    """验证单账号可同时进行 2 个生图 + 2 个分析任务，总并发达 4。"""
    pool = create_test_pool()
    acc_img = pool.get_available_account(task_type="image")
    assert acc_img["name"] == "acc1"
    assert pool._accounts["acc1"]["inflight_image"] == 1
    assert pool._accounts["acc1"]["inflight"] == 1

    acc_chat = pool.get_available_account(task_type="chat")
    assert acc_chat["name"] == "acc1"
    assert pool._accounts["acc1"]["inflight_chat"] == 1
    assert pool._accounts["acc1"]["inflight"] == 2

    pool.get_available_account(task_type="image")
    pool.get_available_account(task_type="chat")
    assert pool._accounts["acc1"]["inflight"] == 4

    # 此时两类槽位均满，再尝试生图或分析均应报错
    failed_img = False
    try:
        pool.get_available_account(task_type="image")
    except RuntimeError:
        failed_img = True
    assert failed_img, "Expected 2nd image task to fail when image inflight is full"

    failed_chat = False
    try:
        pool.get_available_account(task_type="chat")
    except RuntimeError:
        failed_chat = True
    assert failed_chat, "Expected 2nd chat task to fail when chat inflight is full"

    # 释放两个生图槽位
    pool.release_account("acc1", success=True, task_type="image")
    pool.release_account("acc1", success=True, task_type="image")
    assert pool._accounts["acc1"]["inflight_image"] == 0
    assert pool._accounts["acc1"]["inflight"] == 2

    # 释放两个分析槽位
    pool.release_account("acc1", success=True, task_type="chat")
    pool.release_account("acc1", success=True, task_type="chat")
    assert pool._accounts["acc1"]["inflight_chat"] == 0
    assert pool._accounts["acc1"]["inflight"] == 0
    print("✓ test_dual_channel_concurrency passed")


def test_shared_concurrency_limit():
    """验证图片和分析共用账号的四个总额度。"""
    pool = create_test_pool()
    acc1 = pool.get_available_account(task_type="image")
    assert acc1["name"] == "acc1"

    pool.get_available_account(task_type="image")
    pool.get_available_account(task_type="chat")
    pool.get_available_account(task_type="chat")
    try:
        pool.get_available_account(task_type="image")
    except RuntimeError:
        pass
    else:
        raise AssertionError("5th request must be blocked while the shared quota is full")

    for task_type in ("image", "image", "chat", "chat"):
        pool.release_account("acc1", success=True, task_type=task_type)


def test_capacity_reports_shared_slots_and_cooldown():
    pool = create_test_pool()
    pool.get_available_account(task_type="image")
    assert pool.capacity()["available_slots"] == 3
    pool._accounts["acc1"]["cooldown_until"] = 9999999999
    assert pool.capacity()["available_slots"] == 0
    assert pool.capacity()["cooldown_accounts"] == 1


def test_immediate_reuse_and_transient_failure_cooldown():
    """验证成功立即复用，瞬时失败只进入冷却而非永久停用。"""
    pool = create_test_pool()
    pool.get_available_account(task_type="image")
    pool.release_account("acc1", success=True, task_type="image")

    acc_img_again = pool.get_available_account(task_type="image")
    assert acc_img_again["name"] == "acc1"
    pool.release_account("acc1", success=False, error="upstream failed", task_type="image")
    assert pool._accounts["acc1"]["status"] == "active"
    assert pool._accounts["acc1"]["cooldown_until"] > 0
    try:
        pool.get_available_account(task_type="image")
    except RuntimeError:
        print("✓ test_immediate_reuse_and_transient_failure_cooldown passed")
        return
    raise AssertionError("Failed account must not be scheduled")


def test_one_failure_does_not_interrupt_other_inflight_success():
    """同批只要有请求成功，普通超时就不能把整个账号冷却。"""
    pool = create_test_pool()
    pool.get_available_account("chat")
    pool.get_available_account("image")

    pool.release_account("acc1", False, "request timeout", task_type="chat")
    assert pool._accounts["acc1"]["inflight"] == 1
    assert pool._accounts["acc1"]["cooldown_until"] == 0
    assert pool.capacity()["available_slots"] == 0

    pool.release_account("acc1", True, task_type="image")
    assert pool._accounts["acc1"]["cooldown_until"] == 0
    assert pool._accounts["acc1"]["failure_count"] == 0
    assert pool.get_available_account("chat")["name"] == "acc1"


def test_all_inflight_failures_cool_account_once():
    pool = create_test_pool()
    pool.get_available_account("chat")
    pool.get_available_account("image")
    pool.release_account("acc1", False, "request timeout", task_type="chat")
    pool.release_account("acc1", False, "stream closed", task_type="image")

    assert pool._accounts["acc1"]["failure_count"] == 1
    assert pool._accounts["acc1"]["cooldown_until"] > 0


def test_explicit_rate_limit_remains_after_parallel_success():
    pool = create_test_pool()
    pool.get_available_account("chat")
    pool.get_available_account("image")
    pool.release_account("acc1", False, "rate limit", status_code=429, task_type="chat")
    pool.release_account("acc1", True, task_type="image")

    assert pool._accounts["acc1"]["cooldown_until"] > 0


@pytest.mark.parametrize("provider", ["gemini", "gpt", "doubao"])
def test_provider_batch_keeps_account_healthy_after_parallel_success(provider):
    """三个平台共享同一批次判定：一个普通失败和一个成功不冷却账号。"""
    from providers.doubao.account import DoubaoAccountPool
    from providers.gemini.account import GeminiAccountPool
    from providers.openai.account import OpenAIAccountPool

    pool = {"gemini": GeminiAccountPool, "gpt": OpenAIAccountPool, "doubao": DoubaoAccountPool}[provider]()
    pool._platform = ""
    key = "user@example.com" if provider == "gpt" else "acc1"
    pool._accounts = {key: {
        "email" if provider == "gpt" else "name": key,
        "status": "active", "cooldown_until": 0, "failure_count": 0,
        "inflight": 2, "inflight_chat": 1, "inflight_image": 1,
    }}
    pool._batches[key] = {"success": False, "failures": 0, "probing": False, "explicit": False, "error": ""}

    pool.release_account(key, False, "request timeout", task_type="chat")
    pool.release_account(key, True, task_type="image")

    assert pool._accounts[key]["status"] == "active"
    assert pool._accounts[key]["cooldown_until"] == 0
    assert pool._accounts[key]["failure_count"] == 0


def test_successful_release_restores_account_status():
    """成功的任务或人工测试应清除账号此前的错误状态。"""
    pool = create_test_pool()
    account = pool._accounts["acc1"]
    account.update({
        "status": "error",
        "cooldown_until": 9999999999,
        "failure_count": 3,
        "error_message": "previous failure",
    })

    pool.release_account("acc1", success=True, task_type="chat")

    assert account["status"] == "active"
    assert account["cooldown_until"] == 0
    assert account["failure_count"] == 0
    assert account["error_message"] == ""


def test_busy_request_waits_for_release():
    """验证账号忙碌时业务请求等待释放，而不是直接失败。"""
    pool = create_test_pool()
    pool.get_available_account(task_type="image")
    selected = []
    waiter = Thread(target=lambda: selected.append(pool.wait_for_available_account("image")))
    waiter.start()
    pool.release_account("acc1", success=True, task_type="image")
    waiter.join(timeout=1)
    assert selected and selected[0]["name"] == "acc1"
    pool.release_account("acc1", success=True, task_type="image")
    print("✓ test_busy_request_waits_for_release passed")


def test_available_accounts_are_selected_randomly(monkeypatch):
    """验证每次调用仅在可用账号中随机选择，不依赖上次使用时间。"""
    pool = create_test_pool()
    pool._accounts["acc2"] = {**pool._accounts["acc1"], "name": "acc2", "last_used_at": 999}
    monkeypatch.setattr("core.account_pool.random.choice", lambda candidates: candidates[-1])

    account = pool.get_available_account("chat")

    assert account["name"] == "acc2"


def test_account_dispatches_are_spaced_one_second(monkeypatch):
    class PacedPool(DummyPool):
        MIN_DISPATCH_INTERVAL_SECONDS = 1

    pool = PacedPool(Path("/tmp/paced_accounts.json"))
    pool._accounts = create_test_pool()._accounts
    now = [100.0]
    monkeypatch.setattr("core.account_pool.time.time", lambda: now[0])

    pool.get_available_account("chat")
    pool.release_account("acc1", success=True, task_type="chat")

    try:
        pool.get_available_account("image")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected the per-account dispatch interval to block immediate reuse")

    now[0] += 1
    account = pool.get_available_account("image")
    assert account["name"] == "acc1"


if __name__ == "__main__":
    test_dual_channel_concurrency()
    test_shared_concurrency_limit()
    test_immediate_reuse_and_failure_disable()
    test_busy_request_waits_for_release()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
