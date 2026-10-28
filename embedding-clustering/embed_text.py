"""Text embeddings over Gemini's core_concept_summary and multimodal signals.

Supports:
- Model A/B: all-MiniLM-L6-v2 (384-d, fast) or all-mpnet-base-v2 (768-d, higher fidelity)
- Incremental processing and embedding caching
- Optional OCR signal fusion
- Local CPU torch inference (zero API cost)

Usage:
    python embedding-clustering/embed_text.py
    python embedding-clustering/embed_text.py --model mpnet
    python embedding-clustering/embed_text.py --force
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import config as cfg
from common.logging import get_logger

logger = get_logger("embed_text", stage="embedding")

MODEL_ALIASES = {
    "minilm": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2",
}


def main():
    parser = argparse.ArgumentParser(description="Generate text embeddings for concept summaries")
    parser.add_argument("--model", default=getattr(cfg, "TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                        help="Model name or alias ('minilm', 'mpnet')")
    parser.add_argument("--force", action="store_true", help="Force recomputing embeddings")
    parser.add_argument("--include-ocr", action="store_true", help="Append OCR extracted text to input")
    args = parser.parse_args()

    model_name = MODEL_ALIASES.get(args.model.lower(), args.model)
    path = cfg.ASSETS_ENRICHED_CSV

    if not os.path.exists(path):
        logger.error(f"Missing enriched assets file: {path}")
        return

    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for r in rows if (r.get("core_concept_summary") or "").strip()]

    out = os.path.join(cfg.DATA_DIR, "text_embeddings.npz")
    cached_vecs = {}
    cached_model = None

    if os.path.exists(out) and not args.force:
        try:
            cached = np.load(out, allow_pickle=True)
            cached_model = str(cached.get("model_name", ""))
            # If the cached model matches the requested model, reuse embeddings
            if not cached_model or cached_model == model_name:
                c_ids = [str(x) for x in cached["asset_ids"]]
                for aid, vec in zip(c_ids, cached["vec_text"]):
                    cached_vecs[aid] = vec
                logger.info(f"Loaded {len(cached_vecs)} cached text embeddings ({model_name}) from {out}")
            else:
                logger.info(f"Cached model ({cached_model}) differs from requested ({model_name}); re-embedding.")
        except Exception as ex:
            logger.warning(f"Could not load cached embeddings ({ex}); proceeding fresh")

    pending_rows = [r for r in rows if r["asset_id"] not in cached_vecs]

    if not pending_rows and cached_vecs:
        logger.info(f"All {len(rows)} text embeddings already cached with model {model_name}. (Use --force to re-embed)")
        return

    logger.info(f"Embedding {len(pending_rows)} concept summaries with {model_name} ({len(rows)} total)")

    texts = []
    for r in pending_rows:
        parts = [r["core_concept_summary"]]
        if r.get("messaging_angle"):
            parts.append(r["messaging_angle"])
        if r.get("visual_style_tags"):
            parts.append(r["visual_style_tags"].replace("|", ", "))
        if args.include_ocr and r.get("ocr_text"):
            parts.append(f"On-screen text: {r['ocr_text']}")
        texts.append(". ".join(parts))

    t0 = time.time()
    new_vecs = None

    # Priority 1: SentenceTransformers if installed and requested
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        new_vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=32)
        logger.info(f"Encoded {len(pending_rows)} items with SentenceTransformer in {time.time()-t0:.1f}s")
    except ImportError:
        logger.info("SentenceTransformer not installed; using Google GenAI text embedding adapter.")

    # Priority 2: Google GenAI text embedding (gemini-embedding-001)
    if new_vecs is None:
        try:
            from google import genai
            from google.genai import types
            api_key = os.getenv("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key) if api_key else genai.Client(vertexai=True, project=cfg.GCP_PROJECT_ID, location=cfg.GCP_REGION)
            embeddings_list = []
            for i in range(0, len(texts), 50):
                chunk = texts[i:i+50]
                resp = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=chunk,
                    config=types.EmbedContentConfig(output_dimensionality=384),
                )
                embeddings_list.extend([emb.values for emb in resp.embeddings])
            new_vecs = np.array(embeddings_list, dtype=np.float32)
            norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
            new_vecs = np.where(norms == 0, new_vecs, new_vecs / norms)
            model_name = "gemini-embedding-001"
            logger.info(f"Encoded {len(pending_rows)} items with gemini-embedding-001 in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.warning(f"Google GenAI embedding failed ({e}); falling back to TF-IDF TruncatedSVD.")

    # Priority 3: Offline deterministic TF-IDF + TruncatedSVD fallback
    if new_vecs is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        vec = TfidfVectorizer(max_features=1000)
        X_tfidf = vec.fit_transform(texts)
        dim = min(384, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
        if dim >= 2:
            svd = TruncatedSVD(n_components=dim, random_state=42)
            reduced = svd.fit_transform(X_tfidf)
        else:
            reduced = X_tfidf.toarray()
        if reduced.shape[1] < 384:
            reduced = np.pad(reduced, ((0, 0), (0, 384 - reduced.shape[1])))
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        new_vecs = np.where(norms == 0, reduced, reduced / norms).astype(np.float32)
        model_name = "tfidf-svd-fallback"
        logger.info(f"Encoded {len(pending_rows)} items with TF-IDF fallback in {time.time()-t0:.1f}s")

    for r, v in zip(pending_rows, new_vecs):
        cached_vecs[r["asset_id"]] = v

    # Preserve stable order matching rows
    all_ids = [r["asset_id"] for r in rows]
    all_vecs = np.array([cached_vecs[aid] for aid in all_ids], dtype=np.float32)

    np.savez_compressed(
        out,
        vec_text=all_vecs,
        asset_ids=np.array(all_ids),
        model_name=np.array(model_name),
    )
    logger.info(f"Wrote {out} shape={all_vecs.shape}")

    # Nearest-neighbor similarity check
    sim = all_vecs @ all_vecs.T
    np.fill_diagonal(sim, -1)
    top = sim.max(axis=1)
    logger.info(f"Nearest-neighbour cosine: mean={top.mean():.3f} median={np.median(top):.3f} min={top.min():.3f}")
    logger.info(f"Pairs above 0.95: {(sim > 0.95).sum() // 2}")


if __name__ == "__main__":
    main()
