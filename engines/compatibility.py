"""
Outfit compatibility engine.
Assembles complete outfits from candidate items using:
  1. Role-based slot filling (hero → bottom → footwear → layer → accessory)
  2. Cosine similarity scoring between items
  3. Curated outfit lookup for exact matches
  4. Gender + occasion + wear_type hard filters
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from config import CATEGORY_ROLES, TOP_K

# Role priority order for outfit assembly
OUTFIT_SLOTS = ["hero", "bottom", "footwear", "layer", "accessory"]

# Roles that are optional (we try to fill but skip if nothing fits)
OPTIONAL_SLOTS = {"layer", "accessory"}

# Roles that dresses/jumpsuits fill (skip bottom if hero is a full outfit)
FULL_BODY_CATEGORIES = {
    "dresses", "party-dresses", "maxi-dresses", "midi-dresses",
    "mini-dresses", "jumpsuits", "co-ord-sets", "sarees",
    "lehenga", "kurta-sets", "sharara-sets"
}

class CompatibilityEngine:

    def __init__(self, vector_store, data_loader):
        self.vs = vector_store
        self.dl = data_loader
        # Pre-load outfit reference embeddings for curated outfit lookup
        self._curated_index = self._build_curated_index()

    def _build_curated_index(self) -> List[dict]:
        """Index the 25 curated outfits for fast lookup."""
        index = []
        for _, row in self.dl.outfits_df.iterrows():
            index.append({
                "outfit_id": row.get("outfit_id",""),
                "gender":    str(row.get("gender","")).lower(),
                "occasion":  str(row.get("occasion","")).lower(),
                "wear_type": str(row.get("wear_type","")).lower(),
                "theme":     str(row.get("theme","")).lower(),
                "palette":   str(row.get("palette","")).lower(),
                "rationale": str(row.get("stylist_rationale","")),
            })
        return index

    def find_curated_outfit(self, intent: dict) -> Tuple[Optional[dict], str]:
        """Try to find a matching curated outfit. Returns (outfit_products_dict, rationale)."""
        gender   = intent.get("gender","").lower()
        occasion = intent.get("occasion","").lower()
        wear_type= intent.get("wear_type","any").lower()

        best     = None
        best_score = -1

        for curated in self._curated_index:
            score = 0
            # Gender match (hard filter)
            if curated["gender"] and curated["gender"] != gender:
                continue
            # Occasion match
            if occasion and occasion in curated["occasion"]:
                score += 3
            elif occasion and any(
                occasion in t or t in occasion
                for t in curated["theme"].split()
            ):
                score += 1
            # Wear type
            if wear_type != "any" and wear_type in curated["wear_type"]:
                score += 2

            if score > best_score:
                best_score = score
                best = curated

        if best and best_score >= 2:
            products = self.dl.get_outfit_products(best["outfit_id"])
            return products, best["rationale"]

        return None, ""

    def assemble_outfit(
        self,
        intent: dict,
        query_embedding: List[float],
        n_outfits: int = 3,
    ) -> List[dict]:
        gender    = intent.get("gender", "")
        occasion  = intent.get("occasion", "")
        wear_type = intent.get("wear_type", "any")
        style     = intent.get("style", "")
        outfits   = []

        # Build a style-aware search query embedding per slot
        slot_queries = {
            "hero":      f"{style} {occasion} {gender} top shirt dress",
            "bottom":    f"{style} {occasion} {gender} trousers pants skirt",
            "footwear":  f"{style} {occasion} {gender} shoes footwear",
            "layer":     f"{style} {occasion} {gender} jacket blazer layer",
            "accessory": f"{style} {occasion} {gender} accessory watch belt bag",
        }

        # ── Attempt 1: Curated outfit lookup ─────────────────────────────
        curated_products, curated_rationale = self.find_curated_outfit(intent)
        if curated_products:
            items = list(curated_products.values())
            outfits.append({
                "items":     items,
                "score":     1.0,
                "rationale": curated_rationale,
                "source":    "curated",
            })

        # ── Attempts 2-N: Vector search assembly ──────────────────────────
        # Track used product IDs to ensure variety across outfits
        used_ids = set()
        for prev in outfits:
            for item in prev["items"]:
                used_ids.add(item.get("id",""))

        for attempt in range(n_outfits + 2):
            if len(outfits) >= n_outfits:
                break

            outfit_items = []
            slot_scores  = []
            is_full_body = False

            for slot in OUTFIT_SLOTS:
                if slot == "bottom" and is_full_body:
                    continue

                # Use slot-specific query for better relevance
                from engines.embedder import FashionEmbedder
                slot_text  = slot_queries.get(slot, intent.get("search_query",""))
                # Re-use passed embedding for hero; use slot query for others
                if slot == "hero":
                    slot_emb = query_embedding
                else:
                    # Embed slot-specific query inline
                    try:
                        _embedder = FashionEmbedder()
                        slot_emb  = _embedder.embed_text(slot_text)[0].tolist()
                    except Exception:
                        slot_emb  = query_embedding

                wt = wear_type if wear_type != "any" else None

                candidates = self.vs.search(
                    query_embedding=slot_emb,
                    role=slot,
                    gender=gender if gender else None,
                    occasion=occasion if occasion else None,
                    wear_type=wt,
                    top_k=TOP_K,
                )

                # Relax filters if no results
                if not candidates:
                    candidates = self.vs.search(
                        query_embedding=slot_emb,
                        role=slot,
                        gender=gender if gender else None,
                        top_k=TOP_K,
                    )

                if not candidates:
                    if slot not in OPTIONAL_SLOTS:
                        break
                    continue

                # Pick best unused candidate
                chosen = None
                for idx in range(len(candidates)):
                    pick = candidates[(attempt + idx) % len(candidates)]
                    if pick.get("id","") not in used_ids:
                        chosen = pick
                        break
                if chosen is None:
                    chosen = candidates[attempt % len(candidates)]

                # Check full body
                if slot == "hero":
                    cat = chosen.get("category","").lower()
                    if any(fb in cat for fb in FULL_BODY_CATEGORIES):
                        is_full_body = True

                used_ids.add(chosen.get("id",""))
                outfit_items.append({**chosen, "role": slot})
                slot_scores.append(chosen.get("score", 0.5))

            if len(outfit_items) >= 2:
                avg_score = float(np.mean(slot_scores))
                outfits.append({
                    "items":     outfit_items,
                    "score":     avg_score,
                    "rationale": "",
                    "source":    "retrieved",
                })

        return outfits[:n_outfits]

    def score_compatibility(self, item_a: dict, item_b: dict) -> float:
        """
        Simple compatibility score between two items using metadata rules.
        Returns float 0-1.
        """
        score = 0.5  # base

        # Same gender or unisex
        if item_a.get("gender") == item_b.get("gender"):
            score += 0.1

        # Same occasion
        if item_a.get("occasion") == item_b.get("occasion"):
            score += 0.2

        # Same wear_type
        if item_a.get("wear_type") == item_b.get("wear_type"):
            score += 0.15

        # Different roles (good - means they complement)
        if item_a.get("role") != item_b.get("role"):
            score += 0.05

        return min(score, 1.0)
