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


@router.post("/remove-bg")
def remove_bg(file: UploadFile = File(...)):
    global u2net_session
    request_id = uuid.uuid4().hex
    started = time.monotonic()
    logger.info("event=image_edit_started operation=remove_bg request_id=%s", request_id)
    try:
        content = file.file.read()
        # 首次抠图时再加载模型，避免下载模型阻塞整个代理服务启动。
        if u2net_session is None:
            with u2net_lock:
                if u2net_session is None:
                    try:
                        from rembg import new_session
                        u2net_session = new_session("u2netp")
                    except Exception as rembg_err:
                        logger.warning("event=image_edit_fallback operation=remove_bg request_id=%s from=rembg to=grabcut reason_code=%s",
                                       request_id, type(rembg_err).__name__)
        if u2net_session is not None:
            try:
                from rembg import remove
                output = remove(content, session=u2net_session)
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

        upscaled = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        if upscaled.mode == "RGBA":
            r, g, b, a = upscaled.split()
            rgb = Image.merge("RGB", (r, g, b))
            sharpened_rgb = rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=2))
            sr, sg, sb = sharpened_rgb.split()
            upscaled = Image.merge("RGBA", (sr, sg, sb, a))
        else:
            upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=2))

        buf = io.BytesIO()
        upscaled.save(buf, format="PNG")
        output = buf.getvalue()
        logger.info("event=image_edit_finished operation=upscale request_id=%s outcome=success duration_ms=%d input_bytes=%d output_bytes=%d",
                    request_id, int((time.monotonic() - started) * 1000), len(content), len(output))
        return Response(content=output, media_type="image/png", headers={"X-Request-ID": request_id})
    except Exception as e:
        logger.error("event=image_edit_finished operation=upscale request_id=%s outcome=failed duration_ms=%d reason_code=%s",
                     request_id, int((time.monotonic() - started) * 1000), type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Upscale failed: {str(e)}",
                            headers={"X-Request-ID": request_id})
