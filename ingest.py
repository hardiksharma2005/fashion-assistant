"""
Run once to embed all products and populate ChromaDB.
Usage: python ingest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import DataLoader
from engines.embedder   import FashionEmbedder
from engines.vector_store import VectorStore
from config import CHROMA_DIR

def main():
    print("=" * 55)
    print("  Fashion Assistant — Ingestion Pipeline")
    print("=" * 55)

    # 1. Load data
    print("\n[1/3] Loading dataset...")
    dl    = DataLoader()
    stats = dl.get_dataset_stats()
    print(f"  Products : {stats['total_products']}")
    print(f"  Outfits  : {stats['total_outfits']}")
    print(f"  Roles    : {stats['role_dist']}")

    # 2. Embed
    print("\n[2/3] Generating embeddings (CLIP ViT-B/32)...")
    embedder  = FashionEmbedder()
    products  = dl.products_df.to_dict("records")
    embedded  = embedder.embed_products_batch(products, dl)

    # 3. Store
    print("\n[3/3] Storing in ChromaDB...")
    store = VectorStore()
    store.ingest(embedded, dl.products_df)

    print("\n" + "=" * 55)
    print(f"  Ingestion complete! {store.count()} products indexed.")
    print(f"  ChromaDB stored at: {CHROMA_DIR}")
    print("=" * 55)

if __name__ == "__main__":
    main()
