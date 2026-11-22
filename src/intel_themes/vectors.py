"""Shared multimodal vector math, manifold reduction, and deterministic local embedders.

Reuses canonical perceptual hashing from `intel_core.media` and provides L2
block-weighted multimodal fusion, cosine manifold reduction, HDBSCAN clustering,
post-hoc Union-Find centroid consolidation, and offline-capable text/visual
feature encoders for CI and local-first execution.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
from intel_core.media import compute_phash_hex, phash_distance
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

try:
    import hdbscan

    HDBSCAN = hdbscan.HDBSCAN
except ImportError:
    from sklearn.cluster import HDBSCAN  # type: ignore[assignment]


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "your",
    "that",
    "this",
    "from",
    "into",
    "over",
    "more",
    "across",
    "featuring",
    "offers",
    "offer",
    "creative",
    "highlight",
    "highlights",
}


def l2(vec: np.ndarray) -> np.ndarray:
    """L2-normalizes a 1-D vector or row-wise 2-D matrix safely without NaN on zero norm."""
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr if norm == 0.0 else (arr / norm)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0.0, 1.0, norms)


def combine(visual_block: np.ndarray, text_block: np.ndarray, visual_weight: float) -> np.ndarray:
    """Combines L2-normalized visual and text embedding blocks via scalar weights."""
    w_vis = float(visual_weight)
    if w_vis >= 1.0:
        return l2(visual_block)
    if w_vis <= 0.0:
        return l2(text_block)
    vis_norm = l2(visual_block) * w_vis
    txt_norm = l2(text_block) * (1.0 - w_vis)
    return l2(np.concatenate([vis_norm, txt_norm], axis=1))


def reduce_manifold(
    mat: np.ndarray,
    n_components: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Reduces high-dimensional embeddings using UMAP (with SVD/t-SNE fallback)."""
    n_samples, n_features = mat.shape
    if n_samples < 5:
        eff_dim = min(n_components, max(1, n_samples), max(1, n_features))
        out = np.asarray(mat[:, :eff_dim], dtype=np.float64)
        if out.shape[1] < n_components:
            out = np.pad(out, ((0, 0), (0, n_components - out.shape[1])))
        return out

    eff_components = min(n_components, max(2, n_samples - 2))
    eff_neighbors = min(n_neighbors, max(2, n_samples - 1))
    try:
        import umap

        reducer = umap.UMAP(
            n_components=eff_components,
            n_neighbors=eff_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
        )
        coords = np.asarray(reducer.fit_transform(mat), dtype=np.float64)
        return coords
    except Exception:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.manifold import TSNE

        if eff_components == 2:
            perp = max(1.0, min(float(n_samples - 1), min(25.0, max(3.0, n_samples / 8.0))))
            if perp >= n_samples:
                perp = max(1.0, float(n_samples - 1))
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                metric="cosine",
                init="pca",
                random_state=seed,
            )
            return np.asarray(tsne.fit_transform(l2(mat)) + 15.0, dtype=np.float64)

        x_norm = l2(mat)
        sim = x_norm @ x_norm.T
        np.fill_diagonal(sim, 0.0)
        k = min(eff_neighbors, max(1, n_samples - 2))
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        weights = np.zeros_like(sim)
        rows_idx = np.arange(n_samples)[:, None]
        weights[rows_idx, idx] = np.maximum(0.0, sim[rows_idx, idx]) ** 4
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = weights / np.where(row_sums == 0, 1.0, row_sums)
        x_smooth = l2(0.15 * x_norm + 0.85 * (weights @ (weights @ x_norm)))
        svd_dim = min(eff_components, max(1, n_samples - 2), max(1, x_smooth.shape[1] - 1))
        coords = TruncatedSVD(n_components=svd_dim, random_state=seed).fit_transform(x_smooth)
        return np.asarray(coords * 10.0 + 4.5, dtype=np.float64)


def cluster_hdbscan(
    mat: np.ndarray,
    min_cluster_size: int = 4,
    min_samples: int | None = 2,
    method: str = "eom",
) -> np.ndarray:
    """Runs HDBSCAN density clustering safely even on tiny sample counts."""
    n_samples = len(mat)
    if n_samples == 0:
        return np.array([], dtype=int)
    if n_samples == 1:
        return np.array([0], dtype=int)
    eff_mcs = max(2, min(int(min_cluster_size), n_samples))
    eff_ms = None if min_samples is None else max(1, min(int(min_samples), eff_mcs))
    model = HDBSCAN(
        metric="euclidean",
        min_cluster_size=eff_mcs,
        min_samples=eff_ms,
        cluster_selection_method=method,
    )
    return np.asarray(model.fit_predict(mat), dtype=int)


def compute_centroids(X: np.ndarray, pred: np.ndarray) -> dict[int, np.ndarray]:
    """Computes L2-normalized cluster centroids, ignoring HDBSCAN noise (-1)."""
    centroids: dict[int, np.ndarray] = {}
    for cid in sorted(set(int(x) for x in pred)):
        if cid == -1:
            continue
        mask = pred == cid
        if np.any(mask):
            centroids[cid] = l2(X[mask].mean(axis=0))
    return centroids


def _uf_find(parent: dict[int, int], a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def merge_by_centroid(X: np.ndarray, pred: np.ndarray, thresh: float = 0.94) -> np.ndarray:
    """Iteratively merges clusters whose original-space L2 centroids exceed `thresh`."""
    cur = np.asarray(pred, dtype=int).copy()
    while True:
        cen = compute_centroids(X, cur)
        ks = sorted(cen.keys())
        if len(ks) < 2:
            break
        mat = np.array([cen[k] for k in ks])
        sim = mat @ mat.T
        np.fill_diagonal(sim, -1.0)
        if float(sim.max()) < thresh:
            break

        parent = {k: k for k in ks}
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if sim[i, j] >= thresh:
                    ra, rb = _uf_find(parent, ks[i]), _uf_find(parent, ks[j])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

        remap = {k: _uf_find(parent, k) for k in ks}
        if all(remap[k] == k for k in ks):
            break
        cur = np.array([remap.get(int(p), int(p)) if p != -1 else -1 for p in cur], dtype=int)

    uniq = sorted(int(c) for c in set(cur) if c != -1)
    ren = {c: i for i, c in enumerate(uniq)}
    return np.array([ren.get(int(p), -1) if p != -1 else -1 for p in cur], dtype=int)


def score(
    pred_or_ids: dict[str, int] | list[str] | np.ndarray,
    truth_or_pred: dict[str, str] | np.ndarray | list[int],
    truth_or_none: dict[str, str] | None = None,
    split_noise: bool = True,
) -> tuple[float, float]:
    """Computes ARI and AMI between cluster IDs and ground-truth labels."""
    if truth_or_none is not None:
        ids = list(pred_or_ids)  # type: ignore[arg-type]
        by_id = {str(a): int(p) for a, p in zip(ids, truth_or_pred, strict=False)}  # type: ignore[arg-type]
        truth = truth_or_none
    elif isinstance(pred_or_ids, dict):
        by_id = {str(k): int(v) for k, v in pred_or_ids.items()}
        truth = truth_or_pred  # type: ignore[assignment]
    else:
        raise ValueError("Invalid argument types for score()")

    eval_keys = sorted(truth.keys())
    y_true = [str(truth[k]) for k in eval_keys]
    y_pred = []
    nxt_noise_id = 10_000
    for k in eval_keys:
        p = by_id.get(k, -1)
        if p == -1 and split_noise:
            y_pred.append(nxt_noise_id)
            nxt_noise_id += 1
        else:
            y_pred.append(p)
    return float(adjusted_rand_score(y_true, y_pred)), float(adjusted_mutual_info_score(y_true, y_pred))


def embed_text_blocks(
    texts: list[str],
    *,
    dim: int = 384,
    model_name: str | None = None,
    use_transformer: bool = False,
) -> np.ndarray:
    """Embeds concept strings into 384-D L2 unit vectors.

    By default uses a fast, deterministic offline semantic-token feature hash
    (unigrams + bigrams + character 4-grams weighted by IDF) so CI and offline runs
    never require HuggingFace network access. Set `use_transformer=True` to use
    `sentence-transformers` locally when available.
    """
    if not texts:
        return np.zeros((0, dim), dtype=np.float64)

    if use_transformer:
        try:
            from sentence_transformers import SentenceTransformer

            st = SentenceTransformer(model_name or "all-MiniLM-L6-v2", local_files_only=True)
            vecs = st.encode(texts, normalize_embeddings=True)
            return l2(np.asarray(vecs, dtype=np.float64))
        except Exception:
            pass

    token_lists: list[list[str]] = []
    df_counts: dict[str, int] = {}
    for txt in texts:
        cleaned = re.sub(r"[^a-z0-9%+\s]", " ", (txt or "").lower())
        words = [w for w in cleaned.split() if len(w) >= 2 and w not in _STOPWORDS]
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        feats = words + bigrams
        # Boost strategic proposition tokens over brand names so cross-brand themes cluster together
        token_lists.append(feats or ["empty"])
        for tok in set(feats):
            df_counts[tok] = df_counts.get(tok, 0) + 1

    n_docs = len(texts)
    mat = np.zeros((n_docs, dim), dtype=np.float64)
    for row_idx, feats in enumerate(token_lists):
        for tok in feats:
            idf = np.log((1.0 + n_docs) / (1.0 + df_counts.get(tok, 1))) + 1.0
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            mat[row_idx, bucket] += sign * idf
    return l2(mat)


def embed_visual_blocks(
    media_paths: list[str | None],
    *,
    tags_fallback: list[str] | None = None,
    dim: int = 1408,
) -> np.ndarray:
    """Produces 1408-D L2-normalized visual signatures from image frames or tags."""
    if not media_paths:
        return np.zeros((0, dim), dtype=np.float64)

    mat = np.zeros((len(media_paths), dim), dtype=np.float64)
    for i, mpath in enumerate(media_paths):
        loaded = False
        if mpath and isinstance(mpath, str) and os.path.exists(mpath):
            try:
                from PIL import Image

                with Image.open(mpath) as img:
                    rgb = img.convert("RGB").resize((24, 24))
                    arr = np.asarray(rgb, dtype=np.float64) / 255.0
                    # Flatten downsampled spatial RGB grid (1728) -> fold into dim=1408 deterministically
                    flat = arr.reshape(-1)
                    mat[i, : min(dim, flat.size)] += flat[: min(dim, flat.size)]
                    if flat.size > dim:
                        rem = flat[dim:]
                        mat[i, : rem.size] += 0.5 * rem
                    loaded = True
            except Exception:
                loaded = False
        if not loaded:
            seed_txt = (
                tags_fallback[i]
                if tags_fallback and i < len(tags_fallback)
                else Path(str(mpath or f"asset_{i}")).stem
            )
            # Hash visual descriptor tokens deterministically into 1408-D unit hypersphere
            for tok in re.findall(r"[a-z0-9_]+", str(seed_txt).lower()) or ["visual"]:
                d = hashlib.sha256(f"vis:{tok}".encode()).digest()
                b1 = int.from_bytes(d[:4], "little") % dim
                b2 = int.from_bytes(d[4:8], "little") % dim
                mat[i, b1] += 1.0 if (d[8] & 1) == 0 else -1.0
                mat[i, b2] += 0.7 if (d[9] & 1) == 0 else -0.7
    return l2(mat)


__all__ = [
    "HDBSCAN",
    "l2",
    "combine",
    "reduce_manifold",
    "cluster_hdbscan",
    "compute_centroids",
    "merge_by_centroid",
    "score",
    "compute_phash_hex",
    "phash_distance",
    "embed_text_blocks",
    "embed_visual_blocks",
]
