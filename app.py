"""
Fashion Assistant — Streamlit UI
Run: streamlit run app.py
"""
import streamlit as st
from pathlib import Path
import sys, re
sys.path.insert(0, str(Path(__file__).parent))

# ── Auto-ingest on cold start ──────────────────────────────────────────────────
from startup import ensure_ingested
ensure_ingested()

# ── Page config (must be first) ────────────────────────────────────────────────
st.set_page_config(
    page_title="Dare XAI · Fashion Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

from engines.recommender import FashionRecommender
from utils.data_loader   import DataLoader
from config              import THEME, MAX_RESULTS

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "messages":       [],
    "last_result":    None,
    "processing":     False,
    "pending_prompt": "",
    "input_key":      0,       # increment to clear text input
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_recommender():
    return FashionRecommender()

@st.cache_resource(show_spinner=False)
def load_data_loader():
    return DataLoader()

# ── Constants ─────────────────────────────────────────────────────────────────
ROLE_BADGE = {
    "hero":      ("badge-hero",      "✦ Hero"),
    "bottom":    ("badge-bottom",    "◈ Bottom"),
    "footwear":  ("badge-footwear",  "◉ Footwear"),
    "layer":     ("badge-layer",     "◧ Layer"),
    "accessory": ("badge-accessory", "◆ Accessory"),
}

OCCASION_EMOJIS = {
    "office":  "💼", "party":  "🎉", "casual": "☀️",
    "wedding": "💍", "beach":  "🏖️", "ethnic": "🪔",
    "date":    "💫", "general":"✨",
}

EXAMPLE_PROMPTS = [
    "I need a formal outfit for a job interview, I'm a 24 year old male",
    "Suggest a stylish party outfit for a 22 year old woman",
    "What should I wear to a beach vacation? I'm a girl who loves boho style",
    "I'm attending a wedding next weekend, suggest ethnic wear for women",
    "Smart casual outfit for a dinner date, male, 28 years old",
    "Casual summer outfit for a 20 year old college girl",
]

# ── Utility helpers ───────────────────────────────────────────────────────────
def md_to_html(text: str) -> str:
    """Convert **bold** markdown to HTML <strong> tags for rendering in HTML bubbles."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

def render_product_card(item: dict, dl: DataLoader, col):
    role               = item.get("role", "other")
    badge_cls, badge_label = ROLE_BADGE.get(role, ("badge-hero", "◦ Item"))
    pid                = item.get("id", "")
    name               = item.get("name", "Unknown")
    brand              = item.get("brand", "")
    price              = int(item.get("price_inr", 0))
    rating             = float(item.get("rating", 0))
    stars              = "★" * int(rating) + "☆" * (5 - int(rating))

    with col:
        st.markdown(
            f'<div class="product-card">'
            f'<span class="role-badge {badge_cls}">{badge_label}</span>',
            unsafe_allow_html=True,
        )
        img = dl.load_image(pid, size=(240, 300))
        if img:
            st.image(img, use_container_width=True)
        else:
            st.markdown(
                '<div style="height:160px;background:rgba(255,255,255,0.04);'
                'border-radius:8px;display:flex;align-items:center;'
                'justify-content:center;color:#666;">No Image</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<p style="color:#eaeaea;font-size:0.82rem;font-weight:600;'
            f'margin:6px 0 2px;line-height:1.3;">{name[:45]}</p>'
            f'<p style="color:#a0a0b0;font-size:0.75rem;margin:0 0 4px;">{brand}</p>'
            f'<p style="color:#F5A623;font-size:0.9rem;font-weight:700;margin:0;">&#8377;{price:,}</p>'
            f'<p style="color:#F5A623;font-size:0.72rem;margin:2px 0 0;">'
            f'{stars} {rating:.1f}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

def render_intent_chips(intent: dict):
    chips = []
    if intent.get("occasion"):
        emoji = OCCASION_EMOJIS.get(intent["occasion"], "✨")
        chips.append(f"{emoji} {intent['occasion'].title()}")
    if intent.get("gender"):
        chips.append(f"👤 {intent['gender'].title()}")
    if intent.get("style") and intent["style"] not in ["unknown", ""]:
        chips.append(f"🎨 {intent['style'].title()}")
    if intent.get("age_group") and intent["age_group"] != "unknown":
        chips.append(f"🎂 {intent['age_group']}")
    if intent.get("wear_type") and intent["wear_type"] not in ["any", ""]:
        chips.append(f"👘 {intent['wear_type'].title()}")
    html = "".join(f'<span class="intent-chip">{c}</span>' for c in chips)
    st.markdown(f'<div style="margin:8px 0;">{html}</div>', unsafe_allow_html=True)

def render_outfit(outfit: dict, outfit_num: int, dl: DataLoader):
    items    = outfit.get("items", [])
    score    = outfit.get("score", 0)
    source   = outfit.get("source", "retrieved")
    explain  = outfit.get("explanation", "")
    score_pct= int(score * 100)
    source_label = "✦ Stylist Pick" if source == "curated" else f"Outfit {outfit_num}"

    st.markdown('<div class="outfit-card">', unsafe_allow_html=True)

    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown(
            f'<p class="section-header">{source_label}</p>',
            unsafe_allow_html=True,
        )
    with hcol2:
        st.markdown(
            f'<p style="color:#a0a0b0;font-size:0.8rem;text-align:right;margin-top:14px;">'
            f'Match Score</p>'
            f'<div class="score-bar-bg">'
            f'<div class="score-bar-fill" style="width:{score_pct}%"></div>'
            f'</div>'
            f'<p style="color:#F5A623;font-size:0.78rem;text-align:right;">{score_pct}%</p>',
            unsafe_allow_html=True,
        )

    n    = len(items)
    cols = st.columns(min(n, 4))
    for idx, item in enumerate(items):
        render_product_card(item, dl, cols[idx % len(cols)])

    if explain:
        st.markdown(
            f'<div class="explanation-box">💬 {explain}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

def render_outfits_from_result(result: dict, dl: DataLoader):
    """Render intent chips + all outfit cards from a stored result dict."""
    render_intent_chips(result["intent"])
    for i, outfit in enumerate(result["outfits"], 1):
        render_outfit(outfit, i, dl)
    st.markdown(
        f'<p style="color:#606080;font-size:0.78rem;text-align:right;">'
        f'⚡ Generated in {result.get("elapsed", 0)}s</p>',
        unsafe_allow_html=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h1 style="color:#E94560;font-family:\'Playfair Display\',serif;'
        'font-size:1.6rem;margin-bottom:0;">👗 Dare XAI</h1>'
        '<p style="color:#a0a0b0;font-size:0.85rem;margin-top:4px;">'
        'AI Fashion Assistant</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        '<p style="color:#eaeaea;font-weight:600;font-size:0.9rem;">👤 Your Profile</p>',
        unsafe_allow_html=True,
    )
    gender_pref  = st.selectbox("Gender",
                                ["Women", "Men"],
                                label_visibility="collapsed", key="gender_sel")
    occasion_pref= st.selectbox("Occasion",
                                ["Any","Office","Party","Casual","Wedding","Beach","Ethnic","Date"],
                                label_visibility="collapsed", key="occasion_sel")
    style_pref   = st.selectbox("Style",
                                ["Any","Formal","Smart Casual","Casual","Ethnic","Streetwear"],
                                label_visibility="collapsed", key="style_sel")
    age_inp      = st.text_input("Age (optional)", placeholder="e.g. 24", key="age_inp")

    st.divider()

    st.markdown(
        '<p style="color:#eaeaea;font-weight:600;font-size:0.9rem;">⚡ Quick Prompts</p>',
        unsafe_allow_html=True,
    )
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt[:42] + "…", key=f"qp_{hash(prompt)}", use_container_width=True):
            st.session_state["pending_prompt"] = prompt
            st.session_state["input_key"]     += 1
            st.rerun()

    st.divider()

    st.markdown(
        '<p style="color:#eaeaea;font-weight:600;font-size:0.9rem;">📊 Dataset</p>',
        unsafe_allow_html=True,
    )
    try:
        _dl    = load_data_loader()
        _stats = _dl.get_dataset_stats()
        st.markdown(
            f'<div class="sidebar-stat">🗂 {_stats["total_products"]} Products</div>'
            f'<div class="sidebar-stat">👗 {_stats["total_outfits"]} Curated Outfits</div>'
            f'<div class="sidebar-stat">👩 {_stats["gender_dist"].get("women",0)} Women Items</div>'
            f'<div class="sidebar-stat">👨 {_stats["gender_dist"].get("men",0)} Men Items</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    st.divider()
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.messages    = []
        st.session_state.last_result = None
        st.session_state.processing  = False
        st.session_state.pending_prompt = ""
        st.session_state.input_key  += 1
        if "recommender" in st.session_state:
            try:
                st.session_state.recommender.reset_conversation()
            except Exception:
                pass
        st.rerun()

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="text-align:center;color:#eaeaea;font-family:\'Playfair Display\',serif;'
    'font-size:2.2rem;margin-bottom:0;">AI Fashion Assistant</h1>'
    '<p style="text-align:center;color:#a0a0b0;font-size:1rem;margin-top:6px;">'
    'Describe what you need — get a complete, styled outfit with reasoning.</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Load resources ────────────────────────────────────────────────────────────
rec = load_recommender()
dl  = load_data_loader()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role    = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(
            f'<div class="user-bubble">🙋 {content}</div>',
            unsafe_allow_html=True,
        )
    elif role == "assistant_text":
        st.markdown(
            f'<div class="assistant-bubble">👗 {md_to_html(content)}</div>',
            unsafe_allow_html=True,
        )
    elif role == "assistant_outfits":
        render_outfits_from_result(content, dl)

# ── Input row ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
input_col, btn_col = st.columns([6, 1])

# Consume pending prompt (set by quick-prompt buttons)
prefill = st.session_state.pop("pending_prompt", "")

with input_col:
    user_input = st.text_input(
        "chat_input",
        value=prefill,
        placeholder='e.g. "I need a casual summer outfit for a beach vacation"',
        label_visibility="collapsed",
        key=f"chat_input_{st.session_state['input_key']}",
    )

with btn_col:
    send_btn = st.button("Send ✦", use_container_width=True)

# ── Process — only when Send is explicitly clicked ────────────────────────────
if send_btn and user_input.strip() and not st.session_state["processing"]:
    st.session_state["processing"] = True

    # Build enriched query from sidebar profile
    parts = [user_input.strip()]
    if age_inp.strip():
        parts.append(f"I am {age_inp.strip()} years old")
    if gender_pref != "Women":
        parts.append(f"I am a {gender_pref.lower()}")
    if occasion_pref != "Any":
        parts.append(f"for a {occasion_pref.lower()} occasion")
    if style_pref != "Any":
        parts.append(f"I prefer {style_pref.lower()} style")
    query = ". ".join(parts)

    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    with st.spinner("🎨 Styling your outfit..."):
        try:
            result = rec.recommend(query)
        except Exception as e:
            result = None
            st.error(f"Error: {e}")

    if result and result.get("outfits"):
        n   = len(result["outfits"])
        occ = result["intent"].get("occasion", "your occasion").title()
        emoji = OCCASION_EMOJIS.get(result["intent"].get("occasion", ""), "✨")
        reply = (
            f"{emoji} I found **{n} outfit{'s' if n > 1 else ''}** "
            f"for your **{occ}** look! Here's what I'd recommend:"
        )
        st.session_state.messages.append({"role": "assistant_text",    "content": reply})
        st.session_state.messages.append({"role": "assistant_outfits", "content": result})
    else:
        st.session_state.messages.append({
            "role":    "assistant_text",
            "content": "I couldn't find a perfect match. Try rephrasing — e.g. 'party outfit for women' or 'formal look for men'.",
        })

    # Advance input key so the text box clears on rerun
    st.session_state["input_key"]    += 1
    st.session_state["processing"]    = False
    st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;padding:40px 20px;">'
        '<p style="font-size:3rem;margin-bottom:12px;">👗</p>'
        '<p style="color:#eaeaea;font-size:1.1rem;font-weight:600;'
        'font-family:\'Playfair Display\',serif;">Your Personal Stylist Awaits</p>'
        '<p style="color:#606080;font-size:0.9rem;max-width:420px;margin:8px auto 0;">'
        'Ask me anything — tell me your occasion, age, gender and style '
        'and I\'ll build you a complete outfit with reasoning.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(3)
    examples = [
        ("💼", "Office Look",   "Formal outfit for a business meeting"),
        ("🎉", "Party Ready",   "Stylish evening look for a party"),
        ("🏖️", "Beach Vibes",  "Casual summer outfit for vacation"),
    ]
    for i, (emoji, title, desc) in enumerate(examples):
        with ex_cols[i]:
            st.markdown(
                f'<div class="outfit-card" style="text-align:center;">'
                f'<p style="font-size:2rem;margin:0">{emoji}</p>'
                f'<p style="color:#eaeaea;font-weight:600;margin:6px 0 4px;'
                f'font-family:\'Playfair Display\',serif;">{title}</p>'
                f'<p style="color:#a0a0b0;font-size:0.82rem;margin:0;">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
