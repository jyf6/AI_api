import asyncio

import pytest

from core.admission import CapacityUnavailable, ModelAdmission


def test_mixed_platforms_share_atomic_global_budget():
    async def scenario():
        admission = ModelAdmission({"gpt": 2, "gemini": 2, "doubao": 2}, 3)
        deadline = asyncio.get_running_loop().time() + 2
        permits = [await admission.acquire(platform, deadline)
                   for platform in ("gpt", "gpt", "gemini")]
        waiter = asyncio.create_task(admission.acquire("doubao", deadline))
        await asyncio.sleep(0)
        snapshot = admission.snapshot("doubao", {"available_slots": 400})
        assert snapshot["global_free"] == 0
        assert snapshot["available_slots"] == 0
        assert snapshot["waiting_requests"] == 1
        assert admission._reserved["doubao"] == 0
        await permits[0].release()
        replacement = await waiter
        assert admission._global_reserved == 3
        await replacement.mark_upstream_started()
        await replacement.mark_upstream_started()
        assert admission.snapshot("doubao", {"available_slots": 400})["upstream_running"] == 1
        await replacement.release()
        await replacement.release()
        for permit in permits:
            await permit.release()
        assert admission._global_reserved == 0
        assert sum(admission._running.values()) == 0
    asyncio.run(scenario())


def test_cancelled_and_expired_waits_do_not_leak_reservations():
    async def scenario():
        admission = ModelAdmission({"gpt": 1, "gemini": 1, "doubao": 1}, 1)
        deadline = asyncio.get_running_loop().time() + 2
        permit = await admission.acquire("gpt", deadline)
        waiter = asyncio.create_task(admission.acquire("gemini", deadline))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        with pytest.raises(CapacityUnavailable):
            await admission.acquire("doubao", asyncio.get_running_loop().time() + .01)
        assert sum(admission._waiting.values()) == 0
        assert admission._global_reserved == 1
        await permit.release()
        assert admission._global_reserved == 0
    asyncio.run(scenario())


def test_environment_limits_fallback_to_defaults(monkeypatch):
    for key in ("GPT_LIMIT", "GEMINI_LIMIT", "DOUBAO_LIMIT", "MODEL_GLOBAL_LIMIT"):
        monkeypatch.delenv(key, raising=False)
    admission = ModelAdmission.from_environment()
    assert admission.global_limit == 150
    assert admission.limits["gpt"] == 100
    for key in ("GPT_LIMIT", "GEMINI_LIMIT", "DOUBAO_LIMIT", "MODEL_GLOBAL_LIMIT"):
        monkeypatch.setenv(key, "100")
    assert ModelAdmission.from_environment().global_limit == 100


@pytest.mark.parametrize("limit", [75, 100, 125, 150])
def test_staged_execution_budgets_cap_parallel_work_and_release(limit):
    async def scenario():
        owner = ModelAdmission({platform: limit for platform in ("gpt", "gemini", "doubao")}, limit)
        active = peak = 0
        async def execute(index):
            nonlocal active, peak
            permit = await owner.acquire(("gpt", "gemini", "doubao")[index % 3],
                                         asyncio.get_running_loop().time() + 2)
            try:
                await permit.mark_upstream_started()
                active += 1
                peak = max(peak, active)
                assert owner._global_reserved <= limit
                await asyncio.sleep(.005)
                active -= 1
            finally:
                await permit.release()
        await asyncio.gather(*(execute(index) for index in range(300)))
        assert peak == limit
        assert owner._global_reserved == 0
        assert sum(owner._running.values()) == 0
    asyncio.run(scenario())
