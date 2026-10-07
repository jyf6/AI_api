"""短时同步辅助工作使用专用、有界执行器，不挤占模型事件循环。"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="model-helper")
_slots = asyncio.Semaphore(8)


async def run_blocking(call, *args, **kwargs):
    async with _slots:
        future = asyncio.get_running_loop().run_in_executor(_executor, partial(call, *args, **kwargs))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # 同步工作无法强制停止，结束后才归还辅助额度。
            try:
                await future
            except Exception:
                pass
            raise
