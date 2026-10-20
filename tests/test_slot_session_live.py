"""显式选择平台后才执行两次真实生图；只输出续用结论，不保存图片或凭证。"""
import asyncio
import os
from pathlib import Path

import pytest

from utils.image_binary import image_media_type


@pytest.mark.skipif(os.getenv("SLOT_SESSION_LIVE_PLATFORM") not in {"gpt", "gemini"},
                    reason="需要显式选择真实生图平台，会执行两次生成")
def test_two_real_images_reuse_conversation_with_fresh_adapter(monkeypatch):
    # 仅显式联调加载本机环境，普通回归不连接业务数据库。
    for line in (Path(__file__).resolve().parents[1] / ".env").read_text(encoding="utf-8-sig").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    model_text = []
    async def scenario():
        from core.database import database
        from core.router import resolve_model
        from core.slot_sessions import SlotSessions
        from providers.openai.backend import OpenAIBackendAPI
        from providers.openai.account import account_service
        from providers.gemini.backend import GeminiBackendAPI
        from providers.gemini.account import gemini_account_service
        from providers.gemini.webapi.client import GeminiClient

        platform = os.environ["SLOT_SESSION_LIVE_PLATFORM"]
        if platform == "gemini":
            original_stream = GeminiClient.generate_content_stream
            async def observe(client, *args, **kwargs):
                stream = original_stream(client, *args, **kwargs)
                try:
                    async for output in stream:
                        if output.text:
                            model_text[:] = [output.text[:300]]
                        yield output
                finally:
                    await stream.aclose()
            monkeypatch.setattr(GeminiClient, "generate_content_stream", observe)
        accounts = [a for a in database.list_accounts(platform) if a["status"] == "active"
                    and a.get("proxy_status") != "disabled"]
        assert accounts, "没有可用测试账号"
        account = accounts[0]
        # 本机探针可显式覆盖容器专用代理地址，只改本测试进程，不写账号配置。
        proxy_override = os.getenv("SLOT_SESSION_LIVE_PROXY")
        if proxy_override:
            account["proxy"] = proxy_override
            service = account_service if platform == "gpt" else gemini_account_service
            key = account["email"] if platform == "gpt" else account["name"]
            service._accounts[key]["proxy"] = proxy_override
        cache = SlotSessions()
        refs, clients = [], []
        try:
            async with cache.claim("protocol-probe", 1, platform) as session:
                for prompt in ("请使用图片生成工具直接生成一张产品摄影图片：白色桌面上的红色陶瓷杯，无文字。不要代码或SVG。",
                               "请使用图片生成工具调整上一张照片，将红色陶瓷杯改成蓝色，其他保持不变。直接出图，无文字。"):
                    def new_backend():
                        if platform == "gpt":
                            return OpenAIBackendAPI(account["access_token"], account.get("proxy", ""),
                                                    account.get("device_id", ""),
                                                    credential_provider=lambda: account_service.prepare_request_account(account))
                        return GeminiBackendAPI(account)
                    async with new_backend() as backend:
                        options = {"conversation_ref": session.conversation_ref,
                                   "on_conversation": lambda ref: session.save(account["account_id"], ref)}
                        if platform == "gpt":
                            result = await asyncio.wait_for(backend.generate_image_bytes(
                                prompt, resolve_model("gpt-image", "image").model, **options), 240)
                        else:
                            clients.append(backend.client)
                            result = await asyncio.wait_for(backend.image(prompt, "auto", aspect_ratio="1:1", **options), 240)
                        image = result[0] if isinstance(result, list) else result
                        assert image_media_type(image), "没有取得有效图片"
                        assert session.conversation_ref, "图片已生成，但没有取得有效续接状态"
                        refs.append(session.conversation_ref)
                        print(f"platform={platform} round={len(refs)} image_ok=true current_session_ok=true", flush=True)
                        if platform == "gemini":
                            # 强制丢弃独占请求连接，下一轮验证新客户端恢复完整metadata。
                            await backend.client.close()
                    await asyncio.sleep(1)
            key = "conversationId" if platform == "gpt" else "cid"
            assert refs[0][key] == refs[1][key], "第二轮没有续用同一会话"
            if platform == "gemini":
                assert clients[0] is not clients[1]
                assert refs[0]["metadata"][1:3] != refs[1]["metadata"][1:3]
        finally:
            if platform == "gemini":
                await gemini_account_service.close_clients()
    try:
        asyncio.run(scenario())
    except Exception as exc:
        # 上游异常链可能含Cookie对象，联调失败只报告异常类型。
        pytest.fail(f"真实续用联调未通过：{type(exc).__name__}；模型文本：{model_text}", pytrace=False)
