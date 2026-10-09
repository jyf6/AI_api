from __future__ import annotations

import io
import time
import uuid
from threading import Lock
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from PIL import Image, ImageFilter
import cv2
import numpy as np
from utils.log import logger

router = APIRouter(prefix="/api/edit", tags=["edit"])

u2net_session = None
u2net_lock = Lock()


def ensure_u2net_session():
    """获取或初始化 rembg 的 U2NetP 推理会话，保证线程安全单例。"""
    global u2net_session
    if u2net_session is None:
        with u2net_lock:
            if u2net_session is None:
                try:
                    from rembg import new_session
                    u2net_session = new_session("u2netp")
                    logger.info("event=image_edit_warmup operation=rembg status=ready model=u2netp")
                except Exception as rembg_err:
                    logger.warning("event=image_edit_fallback from=rembg to=grabcut reason_code=%s",
                                   type(rembg_err).__name__)
    return u2net_session


def warmup_edit_models():
    """供服务启动生命周期后台静默预热调用，提前加载权重，杜绝首选请求超时。"""
    try:
        ensure_u2net_session()
    except Exception as e:
        logger.warning("event=image_edit_warmup_error error=%s", e)


@router.post("/remove-bg")
def remove_bg(file: UploadFile = File(...)):
    # 复用预热加载好的 U2NetP session；若未预热完成则在此懒加载，并在失败时兜底至 grabcut
    session = ensure_u2net_session()
    request_id = uuid.uuid4().hex
    started = time.monotonic()
    logger.info("event=image_edit_started operation=remove_bg request_id=%s", request_id)
    try:
        content = file.file.read()
        if session is not None:
            try:
                from rembg import remove
                output = remove(content, session=session)
                logger.info("event=image_edit_finished operation=remove_bg request_id=%s outcome=success method=rembg duration_ms=%d input_bytes=%d output_bytes=%d",
                            request_id, int((time.monotonic() - started) * 1000), len(content), len(output))
                return Response(content=output, media_type="image/png", headers={"X-Request-ID": request_id})
            except Exception as rembg_err:
                logger.warning("event=image_edit_fallback operation=remove_bg request_id=%s from=rembg to=grabcut reason_code=%s",
                               request_id, type(rembg_err).__name__)

        img_pil = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(img_pil)
        h, w, _ = img_np.shape

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        margin_x = max(1, int(w * 0.05))
        margin_y = max(1, int(h * 0.05))
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        cv2.grabCut(img_np, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

        r, g, b = cv2.split(img_np)
        rgba = cv2.merge([r, g, b, alpha])
        result_pil = Image.fromarray(rgba)

        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        output = buf.getvalue()
        logger.info("event=image_edit_finished operation=remove_bg request_id=%s outcome=success method=grabcut duration_ms=%d input_bytes=%d output_bytes=%d",
                    request_id, int((time.monotonic() - started) * 1000), len(content), len(output))
        return Response(content=output, media_type="image/png", headers={"X-Request-ID": request_id})
    except Exception as e:
        logger.error("event=image_edit_finished operation=remove_bg request_id=%s outcome=failed duration_ms=%d reason_code=%s",
                     request_id, int((time.monotonic() - started) * 1000), type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Remove background failed: {str(e)}",
                            headers={"X-Request-ID": request_id})


@router.post("/inpaint")
def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    radius: int = Form(5),
):
    request_id = uuid.uuid4().hex
    started = time.monotonic()
    logger.info("event=image_edit_started operation=inpaint request_id=%s", request_id)
    try:
        image_bytes = image.file.read()
        mask_bytes = mask.file.read()

        img_pil = Image.open(io.BytesIO(image_bytes))
        mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")

        has_alpha = img_pil.mode == "RGBA"
        orig_alpha = None
        if has_alpha:
            orig_alpha = img_pil.split()[3]
            img_rgb = img_pil.convert("RGB")
        else:
            img_rgb = img_pil.convert("RGB")

        if mask_pil.size != img_rgb.size:
            mask_pil = mask_pil.resize(img_rgb.size, Image.Resampling.NEAREST)

        img_np = np.array(img_rgb)
        mask_np = np.array(mask_pil)

        _, mask_bin = cv2.threshold(mask_np, 128, 255, cv2.THRESH_BINARY)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(img_bgr, mask_bin, inpaintRadius=max(3, radius), flags=cv2.INPAINT_TELEA)
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

        result_pil = Image.fromarray(inpainted_rgb)
        if has_alpha and orig_alpha:
            result_pil.putalpha(orig_alpha)

        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        output = buf.getvalue()
        logger.info("event=image_edit_finished operation=inpaint request_id=%s outcome=success duration_ms=%d input_bytes=%d mask_bytes=%d output_bytes=%d",
                    request_id, int((time.monotonic() - started) * 1000), len(image_bytes), len(mask_bytes), len(output))
        return Response(content=output, media_type="image/png", headers={"X-Request-ID": request_id})
    except Exception as e:
        logger.error("event=image_edit_finished operation=inpaint request_id=%s outcome=failed duration_ms=%d reason_code=%s",
                     request_id, int((time.monotonic() - started) * 1000), type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {str(e)}",
                            headers={"X-Request-ID": request_id})


MAX_UPSCALE_DIMENSION = 2560


@router.post("/upscale")
def upscale(
    image: UploadFile = File(...),
    scale: int = Form(2),
):
    request_id = uuid.uuid4().hex
    started = time.monotonic()
    logger.info("event=image_edit_started operation=upscale request_id=%s", request_id)
    try:
        content = image.file.read()
        pil_img = Image.open(io.BytesIO(content))

        factor = max(1, min(scale, 4))
        target_w = int(pil_img.width * factor)
        target_h = int(pil_img.height * factor)

        # 限制单边最大分辨率（2560px 满足 4K/2.5K 超清显示与电商大图要求），防止几何级膨胀生成数十兆巨无霸位图
        max_dim = max(target_w, target_h)
        if max_dim > MAX_UPSCALE_DIMENSION:
            ratio = MAX_UPSCALE_DIMENSION / max_dim
            target_w = max(1, int(target_w * ratio))
            target_h = max(1, int(target_h * ratio))

        upscaled = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # 检查是否包含实质性的透明像素
        has_alpha = False
        if upscaled.mode in ("RGBA", "LA") or (upscaled.mode == "P" and "transparency" in upscaled.info):
            rgba = upscaled.convert("RGBA")
            extrema = rgba.getextrema()
            if extrema[3][0] < 255:
                has_alpha = True

        buf = io.BytesIO()
        if has_alpha:
            # 具有透明通道（如去背图）：保留 Alpha 通道并锐化，保存为优化后的 PNG
            r, g, b, a = upscaled.convert("RGBA").split()
            rgb = Image.merge("RGB", (r, g, b))
            sharpened_rgb = rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
            sr, sg, sb = sharpened_rgb.split()
            upscaled = Image.merge("RGBA", (sr, sg, sb, a))
            upscaled.save(buf, format="PNG", optimize=True)
            output = buf.getvalue()
            media_type = "image/png"
        else:
            # 普通不透明商品图/场景图：保存为高质量 JPEG（Quality=93, subsampling=0 高保真色度无损采样）
            rgb = upscaled.convert("RGB")
            upscaled = rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
            upscaled.save(buf, format="JPEG", quality=93, optimize=True, subsampling=0)
            output = buf.getvalue()
            media_type = "image/jpeg"

        logger.info("event=image_edit_finished operation=upscale request_id=%s outcome=success duration_ms=%d input_bytes=%d output_bytes=%d media_type=%s",
                    request_id, int((time.monotonic() - started) * 1000), len(content), len(output), media_type)
        return Response(content=output, media_type=media_type, headers={"X-Request-ID": request_id})
    except Exception as e:
        logger.error("event=image_edit_finished operation=upscale request_id=%s outcome=failed duration_ms=%d reason_code=%s",
                     request_id, int((time.monotonic() - started) * 1000), type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Upscale failed: {str(e)}",
                            headers={"X-Request-ID": request_id})
