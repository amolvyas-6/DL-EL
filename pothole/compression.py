"""Shrink uploaded photos before any model sees them.

Each image is rotated upright from its EXIF data, flattened onto white, resized to fit within
MAX_SIDE pixels and re-encoded as JPEG. The models run on the decoded result, so the pixels
they classify are the same pixels that were compressed.
"""

from io import BytesIO

from PIL import Image, ImageOps

MAX_SIDE = 1024
QUALITY = 85


def compress_image(path: str) -> Image.Image:
    with Image.open(path) as source:
        image = _flatten(ImageOps.exif_transpose(source))
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=QUALITY, optimize=True)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _flatten(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")
