"""
Uses Groq LLM to parse natural language fashion requests into structured intent.
"""
import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

# Force reload .env from the project root
load_dotenv(dotenv_path=str(__import__('pathlib').Path(__file__).parent.parent / '.env'), override=True)

from config import GROQ_MODEL, OCCASION_MAP

def _get_groq_key():
    # Try streamlit secrets first, then .env
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")

def _get_client():
    key = _get_groq_key()
    if not key or key == "your_groq_api_key_here":
        print("[Groq] WARNING: No valid API key found")
        return None
    return Groq(api_key=key)

SYSTEM_PROMPT = """You are a fashion intent parser. Extract structured information from user fashion requests.

Always respond with ONLY a valid JSON object — no markdown, no explanation, no backticks.

JSON schema:
{
  "occasion": "office|party|casual|wedding|beach|ethnic|date|general",
  "gender": "men|women|unisex",
  "wear_type": "western|ethnic|any",
  "age_group": "teen|20s|30s|40s|50s+|unknown",
  "style": "formal|smart-casual|casual|ethnic|bohemian|streetwear|unknown",
  "color_preference": "string or empty",
  "specific_item": "if user mentions a specific item they already have, else empty",
  "search_query": "optimized fashion search query for vector similarity search",
  "outfit_context": "one sentence describing the complete outfit needed"
}

Rules:
- gender defaults to "women" unless user says male/man/men/he/his/boy/guy
- wear_type: use "ethnic" for wedding/festive/traditional requests, else "western", else "any"
- search_query: be descriptive, include occasion, style, gender, colors if mentioned
- style: for office/interview use "formal", for party use "smart-casual" or "casual"
"""

def parse_intent(user_message: str, chat_history: list = None) -> dict:
    client = _get_client()

    if client is None:
        print("[Groq] Falling back to keyword parser — check your .env file")
        return _fallback_parse(user_message)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history[-4:]:
            messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        intent = json.loads(raw)
        print(f"[Groq] Intent parsed successfully: {intent}")

        defaults = {
            "occasion": "casual", "gender": "women", "wear_type": "any",
            "age_group": "unknown", "style": "unknown", "color_preference": "",
            "specific_item": "", "search_query": user_message,
            "outfit_context": user_message,
        }
        for k, v in defaults.items():
            intent.setdefault(k, v)
        return intent

    except Exception as e:
        print(f"[Groq] parse_intent error: {e}")
        return _fallback_parse(user_message)


def generate_outfit_explanation(intent: dict, outfit_items: list, stylist_rationale: str = "") -> str:
    client = _get_client()

    if client is None:
        return _fallback_explanation(intent, outfit_items)

    items_desc = "\n".join([
        f"- {item.get('role','').upper()}: {item.get('name','')} "
        f"by {item.get('brand','')} (₹{int(item.get('price_inr',0)):,})"
        for item in outfit_items
    ])
    rationale_hint = f"\nStylist note: {stylist_rationale}" if stylist_rationale else ""

    prompt = f"""You are a professional fashion stylist. A user requested:
"{intent.get('outfit_context', 'an outfit')}"

User profile:
- Gender: {intent.get('gender','not specified')}
- Age group: {intent.get('age_group','not specified')}
- Occasion: {intent.get('occasion','general')}
- Style preference: {intent.get('style','not specified')}
- Color preference: {intent.get('color_preference','none specified')}

Recommended outfit:
{items_desc}
{rationale_hint}

Write a warm, confident 3-4 sentence explanation of why this outfit works for them.
Mention specific items by name. Explain color and style compatibility.
Keep it conversational and encouraging. Do NOT use bullet points."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a friendly, expert fashion stylist."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        explanation = response.choices[0].message.content.strip()
        print(f"[Groq] Explanation generated: {explanation[:80]}...")
        return explanation
    except Exception as e:
        print(f"[Groq] generate_outfit_explanation error: {e}")
        return _fallback_explanation(intent, outfit_items)


def _fallback_explanation(intent: dict, outfit_items: list) -> str:
    occasion = intent.get("occasion", "occasion")
    names    = [i.get("name","") for i in outfit_items if i.get("name")]
    if names:
        return (f"This carefully selected outfit featuring {names[0]} pairs beautifully "
                f"with {', '.join(names[1:3])} — a complete look perfectly suited "
                f"for your {occasion}.")
    return f"A complete, well-coordinated outfit for your {occasion}."


def _fallback_parse(text: str) -> dict:
    text_lower = text.lower()
    gender = "men" if (
        any(w in text_lower for w in ["male","man","men","he","his","boy","guy"])
        and not any(w in text_lower for w in ["woman","female","girl","her","she","womens"])
    ) else "women"

    occasion = "casual"
    for occ, keywords in OCCASION_MAP.items():
        if any(k in text_lower for k in keywords):
            occasion = occ
            break

    wear_type = "ethnic" if occasion in ["wedding","ethnic"] else "western"
    style = "formal" if occasion in ["office","wedding"] else "casual"

    return {
        "occasion": occasion, "gender": gender, "wear_type": wear_type,
        "age_group": "unknown", "style": style, "color_preference": "",
        "specific_item": "", "search_query": text, "outfit_context": text,
    }
