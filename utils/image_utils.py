from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import io

def create_outfit_collage(products: List[Dict], data_loader, title: str = "Your Outfit") -> Image.Image:
    """Create a clean outfit collage from a list of products."""
    n = len(products)
    if n == 0:
        return _placeholder_image("No items found")

    CARD_W, CARD_H = 220, 280
    LABEL_H        = 50
    PADDING        = 20
    HEADER_H       = 60
    cols           = min(n, 4)
    rows           = (n + cols - 1) // cols
    canvas_w       = cols * (CARD_W + PADDING) + PADDING
    canvas_h       = HEADER_H + rows * (CARD_H + LABEL_H + PADDING) + PADDING

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(15, 15, 30))
    draw   = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([0,0,canvas_w, HEADER_H], fill=(26,26,46))
    draw.text((PADDING, 18), title, fill=(234,69,96), font=None)

    for idx, product in enumerate(products):
        col   = idx % cols
        row_i = idx // cols
        x     = PADDING + col * (CARD_W + PADDING)
        y     = HEADER_H + PADDING + row_i * (CARD_H + LABEL_H + PADDING)

        # Card background
        draw.rounded_rectangle([x, y, x+CARD_W, y+CARD_H+LABEL_H], radius=12, fill=(22,33,62))

        # Product image
        pid   = product.get("id","")
        img   = data_loader.load_image(pid, size=(CARD_W-20, CARD_H-20))
        if img:
            img_x = x + (CARD_W - img.width)  // 2
            img_y = y + (CARD_H - img.height) // 2
            canvas.paste(img, (img_x, img_y))
        else:
            draw.rectangle([x+10, y+10, x+CARD_W-10, y+CARD_H-10], fill=(40,40,60))
            draw.text((x+CARD_W//2-20, y+CARD_H//2), "No Image", fill=(160,160,176))

        # Label
        label = product.get("name","")[:28] + ("…" if len(product.get("name","")) > 28 else "")
        price = f"₹{int(product.get('price_inr',0)):,}"
        draw.text((x+8, y+CARD_H+6),  label, fill=(234,234,234), font=None)
        draw.text((x+8, y+CARD_H+26), price, fill=(245,166,35),  font=None)

    return canvas

def image_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def _placeholder_image(text="No Image", size=(300,300)) -> Image.Image:
    img  = Image.new("RGB", size, color=(22,33,62))
    draw = ImageDraw.Draw(img)
    draw.text((size[0]//2-30, size[1]//2), text, fill=(160,160,176))
    return img

def resize_for_display(img: Image.Image, max_w=300, max_h=300) -> Image.Image:
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img
