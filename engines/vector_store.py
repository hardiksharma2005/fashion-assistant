import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from config import CHROMA_DIR, COLLECTION_TEXT, COLLECTION_IMAGE, TOP_K

class VectorStore:
    """
    ChromaDB wrapper with two collections:
      - fashion_text  : text + hybrid embeddings
      - fashion_image : pure image embeddings
    """

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.text_col  = self.client.get_or_create_collection(
            name=COLLECTION_TEXT,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_col = self.client.get_or_create_collection(
            name=COLLECTION_IMAGE,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VectorStore] Collections ready. "
              f"Text: {self.text_col.count()} | Image: {self.image_col.count()} docs")

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(self, embedded_products: List[dict], products_df):
        """Store all product embeddings with rich metadata."""
        ids_t, embs_t, metas_t = [], [], []
        ids_i, embs_i, metas_i = [], [], []

        for ep in embedded_products:
            pid  = ep["id"]
            row  = products_df[products_df["id"] == pid]
            if row.empty:
                continue
            row  = row.iloc[0]

            meta = {
                "id":             str(pid),
                "name":           str(row.get("name","")),
                "brand":          str(row.get("brand","")),
                "category":       str(row.get("category","")),
                "category_label": str(row.get("category_label","")),
                "role":           str(row.get("role","other")),
                "gender":         str(row.get("gender","")),
                "occasion":       str(row.get("occasion","")),
                "wear_type":      str(row.get("wear_type","")),
                "price_inr":      float(row.get("price_inr", 0)),
                "rating":         float(row.get("rating", 0)),
                "tags":           str(row.get("tags","")),
                "image":          str(row.get("image","")),
            }

            ids_t.append(pid)
            embs_t.append(ep["hybrid_embedding"])
            metas_t.append(meta)

            ids_i.append(pid)
            embs_i.append(ep["image_embedding"])
            metas_i.append(meta)

        # Upsert in batches of 50
        self._batch_upsert(self.text_col,  ids_t, embs_t, metas_t)
        self._batch_upsert(self.image_col, ids_i, embs_i, metas_i)
        print(f"[VectorStore] Ingested {len(ids_t)} products into both collections.")

    def _batch_upsert(self, collection, ids, embeddings, metadatas, batch=50):
        for i in range(0, len(ids), batch):
            collection.upsert(
                ids=ids[i:i+batch],
                embeddings=embeddings[i:i+batch],
                metadatas=metadatas[i:i+batch],
            )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        role: Optional[str]    = None,
        gender: Optional[str]  = None,
        occasion: Optional[str]= None,
        wear_type: Optional[str]= None,
        top_k: int             = TOP_K,
        use_image_col: bool    = False,
    ) -> List[Dict]:
        """Filtered similarity search."""
        where = self._build_where(role, gender, occasion, wear_type)
        collection = self.image_col if use_image_col else self.text_col

        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        try:
            results = collection.query(**kwargs)
        except Exception as e:
            print(f"[VectorStore] Search error (filter dropped): {e}")
            kwargs.pop("where", None)
            results = collection.query(**kwargs)

        hits = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            hits.append({**meta, "score": float(1 - dist)})
        return hits

    def _build_where(self, role, gender, occasion, wear_type) -> Optional[dict]:
        conditions = []
        if role and role != "any":
            conditions.append({"role": {"$eq": role}})
        if gender and gender.lower() not in ["all", "unisex", ""]:
            conditions.append({"gender": {"$eq": gender.lower()}})
        if wear_type and wear_type not in ["any", ""]:
            conditions.append({"wear_type": {"$eq": wear_type.lower()}})
        if not conditions:
            return None
        return {"$and": conditions} if len(conditions) > 1 else conditions[0]

    def is_empty(self) -> bool:
        return self.text_col.count() == 0

    def count(self) -> int:
        return self.text_col.count()
