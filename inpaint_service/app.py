"""独立 LaMa 擦除服务：只负责蒙版修复，不修改业务图片记录。"""

import io
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from PIL import Image, ImageFilter


model = None
inference_lock = Lock()
MAX_INFERENCE_SIDE = 1536
MAX_INPUT_PIXELS = 20_000_000


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model
    # 模型只加载一次；容器健康检查在权重加载完成后才通过。
    from simple_lama_inpainting import SimpleLama

    model = SimpleLama()
    yield
    model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": model is not None}


@app.post("/inpaint")
def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    radius: int = Form(5, ge=0, le=20),
):
    original = Image.open(io.BytesIO(image.file.read()))
    if original.width * original.height > MAX_INPUT_PIXELS:
        raise HTTPException(status_code=413, detail="图片尺寸超过擦除服务限制")

    painted = Image.open(io.BytesIO(mask.file.read())).convert("L")
    if painted.size != original.size:
        painted = painted.resize(original.size, Image.Resampling.NEAREST)
    painted = painted.point(lambda value: 255 if value > 128 else 0)
    if not painted.getbbox():
        raise HTTPException(status_code=400, detail="请先涂抹需要擦除的区域")

    # 稍微扩大涂抹区以覆盖物体边缘，推理仍保留周围背景作为上下文。
    repair_mask = painted.filter(ImageFilter.MaxFilter(radius * 2 + 1)) if radius else painted
    source = original.convert("RGB")
    scale = min(1.0, MAX_INFERENCE_SIDE / max(source.size))
    inference_size = tuple(max(1, round(side * scale)) for side in source.size)
    inference_image = source.resize(inference_size, Image.Resampling.LANCZOS) if scale < 1 else source
    inference_mask = repair_mask.resize(inference_size, Image.Resampling.NEAREST) if scale < 1 else repair_mask

    # CPU 推理串行执行，避免并发大图同时占满内存。
    with inference_lock:
        repaired = model(inference_image, inference_mask)
    repaired = repaired.crop((0, 0, *inference_size)).convert("RGB")
    if scale < 1:
        repaired = repaired.resize(source.size, Image.Resampling.LANCZOS)

    # 保留涂抹区外的原始像素，边缘只做轻微羽化以减少接缝。
    result = Image.composite(repaired, source, repair_mask.filter(ImageFilter.GaussianBlur(1)))
    if original.mode == "RGBA":
        result.putalpha(original.getchannel("A"))
    output = io.BytesIO()
    result.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")
