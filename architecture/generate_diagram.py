"""
Generates the system architecture diagram as a PNG.
Run: python architecture/generate_diagram.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib"], check=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

def draw_box(ax, x, y, w, h, label, sublabel="", color="#16213E",
             border="#E94560", text_color="#EAEAEA", fontsize=10):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=border, linewidth=1.5)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label,
                ha="center", va="center", fontsize=fontsize,
                color=text_color, fontweight="bold")
        ax.text(x + w/2, y + h*0.28, sublabel,
                ha="center", va="center", fontsize=fontsize-2,
                color="#A0A0B0")
    else:
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", fontsize=fontsize,
                color=text_color, fontweight="bold")

def draw_arrow(ax, x1, y1, x2, y2, color="#E94560"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.8, connectionstyle="arc3,rad=0"))

def main():
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor("#0A0A1A")
    ax.set_facecolor("#0A0A1A")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(9, 10.5, "AI Fashion Outfit Recommendation System — Architecture",
            ha="center", va="center", fontsize=15, color="#E94560",
            fontweight="bold", fontfamily="serif")
    ax.text(9, 10.1, "Dare XAI · ML & AI Engineer Intern Assignment",
            ha="center", va="center", fontsize=10, color="#A0A0B0")

    # ── Layer labels ───────────────────────────────────────────────────────
    for y_pos, label, color in [
        (8.8,  "① USER INTERFACE",       "#F5A623"),
        (7.2,  "② NLP & INTENT LAYER",   "#E94560"),
        (5.5,  "③ EMBEDDING LAYER",      "#2196F3"),
        (3.8,  "④ VECTOR STORE",         "#4CAF50"),
        (2.1,  "⑤ COMPATIBILITY ENGINE", "#9C27B0"),
        (0.5,  "⑥ OUTPUT LAYER",         "#F5A623"),
    ]:
        ax.text(0.3, y_pos, label, fontsize=8, color=color,
                fontweight="bold", va="center")

    # ── Row 1: User Interface ──────────────────────────────────────────────
    draw_box(ax, 2.5, 8.55, 3.2, 0.75,
             "Streamlit Chat UI", "app.py",
             color="#1A1A2E", border="#F5A623")
    draw_box(ax, 6.5, 8.55, 2.5, 0.75,
             "Sidebar Profile", "gender·age·occasion",
             color="#1A1A2E", border="#F5A623")
    draw_box(ax, 9.8, 8.55, 2.5, 0.75,
             "Quick Prompts", "6 example queries",
             color="#1A1A2E", border="#F5A623")
    draw_box(ax, 13.0, 8.55, 2.5, 0.75,
             "Chat History", "multi-turn memory",
             color="#1A1A2E", border="#F5A623")

    # ── Row 2: Intent Layer ────────────────────────────────────────────────
    draw_box(ax, 3.5, 6.85, 4.0, 0.85,
             "Groq LLM (Llama 3.3 70B)", "Intent Parser · intent_parser.py",
             color="#1A1A2E", border="#E94560")
    draw_box(ax, 8.5, 6.85, 3.5, 0.85,
             "Structured Intent JSON", "occasion·gender·style·age",
             color="#0F3460", border="#E94560")
    draw_box(ax, 12.8, 6.85, 2.8, 0.85,
             "Fallback Parser", "keyword matching",
             color="#1A1A2E", border="#606080")

    # ── Row 3: Embedding Layer ─────────────────────────────────────────────
    draw_box(ax, 2.0, 5.2, 3.2, 0.85,
             "CLIP ViT-B/32", "sentence-transformers",
             color="#1A1A2E", border="#2196F3")
    draw_box(ax, 6.0, 5.2, 3.0, 0.85,
             "Text Embeddings", "product descriptions",
             color="#0F3460", border="#2196F3")
    draw_box(ax, 9.8, 5.2, 3.0, 0.85,
             "Image Embeddings", "product photos",
             color="#0F3460", border="#2196F3")
    draw_box(ax, 13.5, 5.2, 2.5, 0.85,
             "Hybrid Embeddings", "0.6 text + 0.4 image",
             color="#0F3460", border="#2196F3")

    # ── Row 4: Vector Store ────────────────────────────────────────────────
    draw_box(ax, 2.5, 3.55, 3.5, 0.85,
             "ChromaDB", "Persistent Vector Store",
             color="#1A1A2E", border="#4CAF50")
    draw_box(ax, 7.0, 3.55, 3.0, 0.85,
             "fashion_text", "hybrid embeddings · 68 docs",
             color="#0F3460", border="#4CAF50")
    draw_box(ax, 11.0, 3.55, 3.0, 0.85,
             "fashion_image", "image embeddings · 68 docs",
             color="#0F3460", border="#4CAF50")
    draw_box(ax, 14.8, 3.55, 2.5, 0.85,
             "Cosine Similarity", "HNSW index search",
             color="#0F3460", border="#4CAF50")

    # ── Row 5: Compatibility Engine ────────────────────────────────────────
    draw_box(ax, 2.0, 1.85, 3.0, 0.85,
             "Curated Lookup", "25 expert outfits",
             color="#1A1A2E", border="#9C27B0")
    draw_box(ax, 5.8, 1.85, 3.2, 0.85,
             "Slot Filling", "hero→bottom→footwear→layer→acc",
             color="#1A1A2E", border="#9C27B0")
    draw_box(ax, 9.8, 1.85, 3.2, 0.85,
             "Full-Body Detection", "dresses·jumpsuits·sarees",
             color="#1A1A2E", border="#9C27B0")
    draw_box(ax, 13.8, 1.85, 3.0, 0.85,
             "Variety Engine", "used_id dedup · 3 outfits",
             color="#1A1A2E", border="#9C27B0")

    # ── Row 6: Output ──────────────────────────────────────────────────────
    draw_box(ax, 2.5, 0.2, 3.2, 0.85,
             "Outfit Cards", "image · price · rating",
             color="#1A1A2E", border="#F5A623")
    draw_box(ax, 6.5, 0.2, 3.0, 0.85,
             "Role Badges", "HERO·BOTTOM·FOOTWEAR·LAYER·ACC",
             color="#1A1A2E", border="#F5A623")
    draw_box(ax, 10.3, 0.2, 3.2, 0.85,
             "Groq Explanation", "LLM-generated rationale",
             color="#1A1A2E", border="#F5A623")
    draw_box(ax, 14.3, 0.2, 2.8, 0.85,
             "Match Score", "cosine similarity %",
             color="#1A1A2E", border="#F5A623")

    # ── Arrows: UI → Intent ────────────────────────────────────────────────
    draw_arrow(ax, 4.1,  8.55, 5.5,  7.70)
    draw_arrow(ax, 14.25, 7.70, 13.2, 7.70)

    # ── Arrows: Intent → Embedding ────────────────────────────────────────
    draw_arrow(ax, 7.0, 6.85, 5.5, 6.05)

    # ── Arrows: Embedding → VectorStore ───────────────────────────────────
    draw_arrow(ax, 5.5,  5.2, 5.5,  4.40)
    draw_arrow(ax, 10.3, 5.2, 10.3, 4.40)
    draw_arrow(ax, 14.75, 5.2, 14.75, 4.40)

    # ── Arrows: VectorStore → Compatibility ───────────────────────────────
    draw_arrow(ax, 5.5,  3.55, 5.5,  2.70)
    draw_arrow(ax, 10.3, 3.55, 10.3, 2.70)

    # ── Arrows: Compatibility → Output ────────────────────────────────────
    draw_arrow(ax, 5.5,  1.85, 5.5,  1.05)
    draw_arrow(ax, 10.3, 1.85, 10.3, 1.05)

    # ── Dataset annotation ─────────────────────────────────────────────────
    draw_box(ax, 15.5, 5.2, 2.2, 2.2,
             "Dataset", "68 products\n25 outfits\n3 sources\n(Ajio·Myntra·Nykaa)",
             color="#0A1628", border="#606080", fontsize=8)
    ax.text(16.6, 4.9, "products.csv\noutfits.csv\nimages/",
            ha="center", va="top", fontsize=7, color="#606080")

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#F5A623", label="UI Layer"),
        mpatches.Patch(color="#E94560", label="LLM / Intent"),
        mpatches.Patch(color="#2196F3", label="Embedding (CLIP)"),
        mpatches.Patch(color="#4CAF50", label="Vector Store (Chroma)"),
        mpatches.Patch(color="#9C27B0", label="Compatibility Engine"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              facecolor="#0F0F23", edgecolor="#E94560",
              labelcolor="#EAEAEA", fontsize=8, framealpha=0.9)

    out = Path(__file__).parent / "architecture_diagram.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0A0A1A")
    print(f"[Diagram] Saved to {out}")
    plt.close()

if __name__ == "__main__":
    main()
