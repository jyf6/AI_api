import math
import io
from PIL import Image

def _ensure_rgb(img):
    if img.mode in ("RGBA", "P", "LA"):
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
        if img.mode == "RGBA" or img.mode == "LA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            mask = img.split()[-1]
            background.paste(img, mask=mask)
            return background
        return img.convert("RGB")
    return img

def stitch_images_to_bytes(image_bytes_list: list[bytes], max_width=1024) -> bytes:
    if not image_bytes_list:
        raise ValueError("No images to stitch")
        
    images = []
    for b in image_bytes_list:
        try:
            img = Image.open(io.BytesIO(b))
            img = _ensure_rgb(img)
            images.append(img)
        except Exception as e:
            continue
            
    if not images:
        raise ValueError("Failed to open any images for stitching")

    count = len(images)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    cell_size = 512
    for i in range(len(images)):
        img = images[i]
        img.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
        
    grid_w = cols * cell_size
    grid_h = rows * cell_size
    
    grid_image = Image.new('RGB', (grid_w, grid_h), color='white')
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * cell_size + (cell_size - img.width) // 2
        y = row * cell_size + (cell_size - img.height) // 2
        grid_image.paste(img, (x, y))
        
    if grid_image.width > max_width:
        ratio = max_width / grid_image.width
        new_h = int(grid_image.height * ratio)
        grid_image = grid_image.resize((max_width, new_h), Image.Resampling.LANCZOS)
        
    out = io.BytesIO()
    grid_image.save(out, format="JPEG", quality=85)
    return out.getvalue()

def compress_single_image(image_bytes: bytes, max_width=1024) -> bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = _ensure_rgb(img)
        if img.width > max_width:
            ratio = max_width / img.width
            new_h = int(img.height * ratio)
            img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
        return out.getvalue()
    except Exception:
        return image_bytes
