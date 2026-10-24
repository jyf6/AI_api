import asyncio

from core.admission import ModelAdmission


def test_platform_and_global_environment_limits_do_not_gate_model_requests(monkeypatch):
    for key in ("GPT_LIMIT", "GEMINI_LIMIT", "DOUBAO_LIMIT", "MODEL_GLOBAL_LIMIT"):
        monkeypatch.setenv(key, "1")

    async def scenario():
        admission = ModelAdmission.from_environment()
        deadline = asyncio.get_running_loop().time() + 1
        permits = await asyncio.gather(*(
            admission.acquire(("gpt", "gemini", "doubao")[index % 3], deadline)
            for index in range(300)
        ))
        assert admission._global_reserved == 300
        assert sum(admission._reserved.values()) == 300
        await asyncio.gather(*(permit.mark_upstream_started() for permit in permits))
        assert sum(admission._running.values()) == 300
        await asyncio.gather(*(permit.release() for permit in permits))
        assert admission._global_reserved == 0
        assert sum(admission._running.values()) == 0

    asyncio.run(scenario())
