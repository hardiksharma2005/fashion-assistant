import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
IMAGES_DIR      = DATA_DIR / "images"
PRODUCTS_CSV    = DATA_DIR / "products.csv"
OUTFITS_CSV     = DATA_DIR / "outfits.csv"
CHROMA_DIR      = BASE_DIR / "chroma_db"
ASSETS_DIR      = BASE_DIR / "assets"

# ── API Keys ───────────────────────────────────────────────────────────────────
# Works from .env locally AND from Streamlit Cloud secrets
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Model Config ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "clip-ViT-B-32"          # sentence-transformers CLIP model
GROQ_MODEL      = "llama-3.3-70b-versatile"  # Groq LLM

# ── ChromaDB Collections ───────────────────────────────────────────────────────
COLLECTION_TEXT  = "fashion_text"
COLLECTION_IMAGE = "fashion_image"

# ── Recommendation Config ──────────────────────────────────────────────────────
TOP_K           = 10     # candidates retrieved per category
MAX_RESULTS     = 3      # final outfit options shown

# ── Category Mapping ───────────────────────────────────────────────────────────
CATEGORY_ROLES = {
    "hero": [
        "formal-shirts", "casual-shirts", "t-shirts", "shirts",
        "dresses", "tops", "blouses", "kurtas", "co-ord-sets",
        "sweatshirts", "jackets", "blazers", "kurta-sets",
        "sharara-sets", "sarees", "lehenga", "jumpsuits",
        "party-dresses", "maxi-dresses", "mini-dresses",
        "midi-dresses", "ethnic-tops", "tunics", "polo-shirts",
        "hoodies", "vests", "tank-tops", "crop-tops"
    ],
    "bottom": [
        "trousers", "jeans", "chinos", "skirts", "shorts",
        "palazzos", "leggings", "formal-trousers", "track-pants",
        "joggers", "culottes", "dhoti-pants", "salwar",
        "divided-skirts", "cargo-pants"
    ],
    "footwear": [
        "heels", "sneakers", "loafers", "sandals", "boots",
        "flats", "oxfords", "mules", "formal-shoes",
        "ethnic-footwear", "kolhapuris", "juttis", "wedges",
        "slip-ons", "derby-shoes", "monk-straps", "ballerinas"
    ],
    "layer": [
        "blazers", "jackets", "cardigans", "shrugs", "coats",
        "waistcoats", "dupattas", "stoles", "denim-jackets",
        "bomber-jackets", "overcoats", "nehru-jackets"
    ],
    "accessory": [
        "watches", "belts", "bags", "sunglasses", "jewellery",
        "scarves", "caps", "wallets", "ties", "pocket-squares",
        "clutches", "handbags", "necklaces", "earrings",
        "bracelets", "rings", "bangles", "anklets",
        "hair-accessories", "socks", "cufflinks", "brooches",
        "tote-bags", "backpacks", "crossbody-bags"
    ]
}

# ── Occasion Keywords ──────────────────────────────────────────────────────────
OCCASION_MAP = {
    "office":    ["office", "work", "professional", "business", "meeting", "interview", "corporate"],
    "party":     ["party", "night out", "club", "celebration", "cocktail", "evening"],
    "casual":    ["casual", "everyday", "weekend", "relaxed", "coffee", "brunch", "outing"],
    "wedding":   ["wedding", "reception", "ceremony", "sangeet", "engagement", "festive"],
    "beach":     ["beach", "vacation", "holiday", "resort", "summer", "travel", "pool"],
    "ethnic":    ["ethnic", "traditional", "festival", "pooja", "diwali", "holi", "cultural"],
    "date":      ["date", "dinner", "romantic", "anniversary", "special evening"]
}

# ── UI Theme Colors ────────────────────────────────────────────────────────────
THEME = {
    "primary":     "#1A1A2E",   # deep navy
    "secondary":   "#16213E",   # darker navy
    "accent":      "#E94560",   # rose red
    "accent2":     "#F5A623",   # warm gold
    "text_light":  "#EAEAEA",
    "text_muted":  "#A0A0B0",
    "card_bg":     "#0F3460",
    "success":     "#4CAF50",
}
