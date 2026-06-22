"""
End-to-end recommendation pipeline.
Connects: IntentParser → VectorStore → CompatibilityEngine → Explainer
"""
from typing import List, Dict, Optional
import time

from engines.intent_parser   import parse_intent, generate_outfit_explanation
from engines.vector_store    import VectorStore
from engines.compatibility   import CompatibilityEngine
from engines.embedder        import FashionEmbedder
from utils.data_loader       import DataLoader
from config                  import MAX_RESULTS

class FashionRecommender:

    def __init__(self):
        print("[Recommender] Initialising components...")
        self.dl          = DataLoader()
        self.embedder    = FashionEmbedder()
        self.vs          = VectorStore()
        self.compat      = CompatibilityEngine(self.vs, self.dl)
        self.chat_history: List[dict] = []
        print("[Recommender] Ready.")

    def recommend(self, user_message: str) -> Dict:
        """
        Main entry point.
        Returns:
        {
            "intent":   dict,
            "outfits":  [ {items, score, rationale, explanation, source}, ... ],
            "raw_message": str,
            "elapsed":  float,
        }
        """
        t0 = time.time()

        # 1. Parse intent
        intent = parse_intent(user_message, self.chat_history)
        print(f"[Recommender] Intent: {intent}")

        # 2. Embed the search query
        query_emb = self.embedder.embed_text(intent["search_query"])[0].tolist()

        # 3. Assemble outfits
        outfits = self.compat.assemble_outfit(
            intent=intent,
            query_embedding=query_emb,
            n_outfits=MAX_RESULTS,
        )

        # 4. Generate explanations for each outfit
        for outfit in outfits:
            explanation = generate_outfit_explanation(
                intent=intent,
                outfit_items=outfit["items"],
                stylist_rationale=outfit.get("rationale",""),
            )
            outfit["explanation"] = explanation

        # 5. Update chat history
        self.chat_history.append({"role": "user", "content": user_message})
        self.chat_history.append({
            "role":    "assistant",
            "content": f"I found {len(outfits)} outfit(s) for your {intent.get('occasion','request')}."
        })
        # Keep last 10 turns
        self.chat_history = self.chat_history[-10:]

        elapsed = round(time.time() - t0, 2)
        print(f"[Recommender] Done in {elapsed}s. {len(outfits)} outfits generated.")

        return {
            "intent":      intent,
            "outfits":     outfits,
            "raw_message": user_message,
            "elapsed":     elapsed,
        }

    def reset_conversation(self):
        self.chat_history = []
        print("[Recommender] Conversation reset.")
