"""
Cloud startup helper — runs ingestion if ChromaDB is not populated.
Called from app.py on cold start.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

def ensure_ingested():
    """Run ingestion pipeline if ChromaDB is empty."""
    try:
        import chromadb
        from config import CHROMA_DIR, COLLECTION_TEXT

        if not CHROMA_DIR.exists():
            print("[Startup] ChromaDB not found — running ingestion...")
            _run_ingest()
            return

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            col   = client.get_collection(COLLECTION_TEXT)
            count = col.count()
            if count == 0:
                print("[Startup] ChromaDB empty — running ingestion...")
                _run_ingest()
            else:
                print(f"[Startup] ChromaDB ready — {count} products indexed.")
        except Exception:
            print("[Startup] Collection missing — running ingestion...")
            _run_ingest()

    except Exception as e:
        print(f"[Startup] Error checking ChromaDB: {e}")
        _run_ingest()

def _run_ingest():
    try:
        from utils.data_loader    import DataLoader
        from engines.embedder     import FashionEmbedder
        from engines.vector_store import VectorStore

        print("[Startup] Loading data...")
        dl       = DataLoader()

        print("[Startup] Generating embeddings — this takes ~60s on first run...")
        embedder = FashionEmbedder()
        products = dl.products_df.to_dict("records")
        embedded = embedder.embed_products_batch(products, dl)

        print("[Startup] Storing in ChromaDB...")
        store    = VectorStore()
        store.ingest(embedded, dl.products_df)

        print(f"[Startup] Ingestion complete — {store.count()} products indexed.")
    except Exception as e:
        print(f"[Startup] Ingestion failed: {e}")
        raise
