import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config import EMBEDDING_MODEL

class FashionEmbedder:
    """
    Dual-encoder: text embeddings + image embeddings using CLIP ViT-B/32.
    Both live in the same 512-dim space so we can do cross-modal search.
    """

    def __init__(self):
        print(f"[Embedder] Loading CLIP model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dim   = 512
        print(f"[Embedder] Ready. Embedding dim = {self.dim}")

    # ── Text ──────────────────────────────────────────────────────────────────

    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    # ── Image ─────────────────────────────────────────────────────────────────

    def embed_image(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
        embeddings = self.model.encode(
            images,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    # ── Hybrid (text + image average) ─────────────────────────────────────────

    def embed_hybrid(self, text: str, image: Optional[Image.Image] = None) -> np.ndarray:
        t_emb = self.embed_text(text)[0]
        if image is not None:
            i_emb = self.embed_image(image)[0]
            combined = 0.6 * t_emb + 0.4 * i_emb
            norm = np.linalg.norm(combined)
            return combined / norm if norm > 0 else combined
        return t_emb

    # ── Batch products ────────────────────────────────────────────────────────

    def embed_products_batch(self, products: List[dict], data_loader) -> List[dict]:
        """
        Returns list of dicts with product id + text_embedding + image_embedding + hybrid_embedding.
        """
        results = []
        print(f"[Embedder] Embedding {len(products)} products...")

        for product in tqdm(products, desc="Embedding products"):
            pid  = product.get("id", "")
            text = data_loader.build_product_text(product)

            # Text embedding
            t_emb = self.embed_text(text)[0]

            # Image embedding
            img   = data_loader.load_image(pid)
            i_emb = self.embed_image(img)[0] if img is not None else np.zeros(self.dim)

            # Hybrid
            if img is not None:
                hybrid = 0.6 * t_emb + 0.4 * i_emb
                norm   = np.linalg.norm(hybrid)
                hybrid = hybrid / norm if norm > 0 else hybrid
            else:
                hybrid = t_emb

            results.append({
                "id":               pid,
                "text_embedding":   t_emb.tolist(),
                "image_embedding":  i_emb.tolist(),
                "hybrid_embedding": hybrid.tolist(),
            })

        print(f"[Embedder] Done. {len(results)} products embedded.")
        return results
