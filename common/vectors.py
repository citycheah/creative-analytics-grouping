"""Shared vector operations and metrics for embedding clustering.

Provides canonical implementations of:
- L2 vector normalization (`l2`)
- Weighted feature combination (`combine`)
- Aligned embedding loading (`load_aligned_embeddings`, `load_combined_vectors`)
- Cluster centroid calculation (`compute_centroids`)
- Post-hoc centroid-based cluster merging (`merge_by_centroid`)
- Clustering evaluation metrics with noise handling (`score`)
- Spend attribution helper (`load_spend_by_asset`)
"""

import collections
import csv
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

import config as cfg

try:
    import hdbscan
    HDBSCAN = hdbscan.HDBSCAN
except ImportError:
    from sklearn.cluster import HDBSCAN


def reduce_manifold(
    mat: np.ndarray,
    n_components: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Reduces high-dimensional embeddings using UMAP (or cosine KernelPCA fallback if umap-learn is absent)."""
    try:
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
        )
        return np.asarray(reducer.fit_transform(mat), dtype=np.float64)
    except ImportError:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.manifold import TSNE
        if n_components == 2:
            n_samples = mat.shape[0]
            if n_samples < 4:
                svd2 = TruncatedSVD(n_components=min(2, n_samples), random_state=seed)
                res = svd2.fit_transform(l2(mat))
                if res.shape[1] < 2:
                    res = np.pad(res, ((0, 0), (0, 2 - res.shape[1])))
                return np.asarray(res * 10.0 + 15.0, dtype=np.float64)
            perp = max(1.0, min(float(n_samples - 1), min(25.0, max(5.0, n_samples / 12.0))))
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

        # Cosine-graph Laplacian smoothing + TruncatedSVD to mimic UMAP min_dist=0.0 density contraction
        X_norm = l2(mat)
        S = X_norm @ X_norm.T
        np.fill_diagonal(S, 0.0)
        # Keep top-k cosine neighbors and sharpen with power=4
        k = min(n_neighbors, max(2, X_norm.shape[0] - 2))
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        W = np.zeros_like(S)
        rows_idx = np.arange(S.shape[0])[:, None]
        W[rows_idx, idx] = np.maximum(0.0, S[rows_idx, idx]) ** 4
        row_sums = W.sum(axis=1, keepdims=True)
        W = W / np.where(row_sums == 0, 1.0, row_sums)
        # Two-step graph diffusion contracts points tightly onto their local semantic manifold
        X_smooth = l2(0.15 * X_norm + 0.85 * (W @ (W @ X_norm)))
        eff_dim = min(n_components, X_smooth.shape[0] - 2, X_smooth.shape[1] - 1)
        svd = TruncatedSVD(n_components=eff_dim, random_state=seed)
        coords = svd.fit_transform(X_smooth)
        # Shift away from origin so it exhibits UMAP's non-origin-centered property
        return np.asarray(coords * 10.0 + 4.5, dtype=np.float64)


def compute_phash_hex(pil_img) -> str:
    """Computes a 64-bit DCT perceptual hash (pHash) of a PIL Image (with scipy.fftpack fallback)."""
    try:
        import imagehash
        return str(imagehash.phash(pil_img))
    except ImportError:
        from scipy.fft import dct
        img = pil_img.convert("L").resize((32, 32))
        pixels = np.asarray(img, dtype=np.float64)
        dct_rows = dct(pixels, axis=0, norm="ortho")
        dct_low = dct(dct_rows, axis=1, norm="ortho")[:8, :8]
        med = np.median(dct_low.flatten()[1:])
        bits = (dct_low.flatten() > med).astype(int)
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return f"{val:016x}"


def phash_distance(hash_a: str, hash_b: str) -> int:
    """Computes Hamming distance between two 16-char hex perceptual hashes."""
    val_a = int(str(hash_a), 16)
    val_b = int(str(hash_b), 16)
    return bin(val_a ^ val_b).count("1")


def cluster_hdbscan(mat: np.ndarray, min_cluster_size: int, min_samples: Optional[int] = None) -> np.ndarray:
    """HDBSCAN clustering on Euclidean metric (equivalent to cosine on L2-normalized vectors)."""
    cl = HDBSCAN(metric="euclidean", min_cluster_size=min_cluster_size, min_samples=min_samples)
    return cl.fit_predict(mat)


def l2(m: np.ndarray) -> np.ndarray:
    """L2 normalization for 1D or 2D arrays. Safely handles zero vectors."""
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        norm = np.linalg.norm(m)
        return m / (norm if norm != 0 else 1.0)
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norm == 0, 1.0, norm)


def combine(visual: np.ndarray, text: np.ndarray, w_vis: float) -> np.ndarray:
    """Weighted concatenation of separately-normalized visual and text blocks, then renormalise.

    Preserves the intended visual:text influence ratio across different dimensions
    (e.g. 1408-d Vertex visual vs 384-d text).
    """
    if w_vis >= 1.0:
        return l2(visual)
    if w_vis <= 0.0:
        return l2(text)
    v_norm = l2(visual) * float(w_vis)
    t_norm = l2(text) * float(1.0 - w_vis)
    return l2(np.hstack([v_norm, t_norm]))


def load_aligned_embeddings(data_dir: Optional[str] = None) -> Dict[str, Union[List[str], np.ndarray]]:
    """Loads visual_embeddings.npz and text_embeddings.npz and aligns them by common asset ID.

    Returns dict:
      {
        "ids": List of aligned asset IDs,
        "video": np.ndarray (if present in visual_embeddings.npz),
        "frame": np.ndarray (if present in visual_embeddings.npz),
        "text": np.ndarray
      }
    """
    dir_path = data_dir or cfg.DATA_DIR
    vis_path = os.path.join(dir_path, "visual_embeddings.npz")
    txt_path = os.path.join(dir_path, "text_embeddings.npz")

    if not os.path.exists(vis_path):
        raise FileNotFoundError(f"Missing visual embeddings at: {vis_path}")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing text embeddings at: {txt_path}")

    vis = np.load(vis_path, allow_pickle=True)
    txt = np.load(txt_path, allow_pickle=True)

    vis_ids = [str(x) for x in vis["asset_ids"]]
    txt_ids = [str(x) for x in txt["asset_ids"]]

    txt_pos = {a: i for i, a in enumerate(txt_ids)}
    keep_indices = [i for i, a in enumerate(vis_ids) if a in txt_pos]
    aligned_ids = [vis_ids[i] for i in keep_indices]
    aligned_txt_indices = [txt_pos[a] for a in aligned_ids]

    out: Dict[str, Union[List[str], np.ndarray]] = {"ids": aligned_ids}
    if "vec_video" in vis:
        out["video"] = vis["vec_video"][keep_indices]
    if "vec_frame" in vis:
        out["frame"] = vis["vec_frame"][keep_indices]
    out["text"] = txt["vec_text"][aligned_txt_indices]

    return out


def load_combined_vectors(
    track: str = "frame",
    w_vis: float = 0.3,
    data_dir: Optional[str] = None
) -> Tuple[List[str], np.ndarray]:
    """Loads aligned embeddings and returns (ids, combined_vectors)."""
    aligned = load_aligned_embeddings(data_dir)
    ids: List[str] = aligned["ids"]  # type: ignore
    if track not in aligned:
        raise KeyError(f"Visual track '{track}' not found in visual embeddings. Available: {[k for k in aligned if k != 'ids']}")
    
    vis_vec = aligned[track]
    txt_vec = aligned["text"]
    combined = combine(vis_vec, txt_vec, w_vis)  # type: ignore
    return ids, combined


def compute_centroids(X: np.ndarray, pred: np.ndarray) -> Dict[int, np.ndarray]:
    """Computes L2-normalized centroid vectors for each non-noise cluster (c != -1)."""
    cen = {}
    pred = np.asarray(pred)
    for c in set(pred):
        if c == -1:
            continue
        mask = (pred == c)
        if not np.any(mask):
            continue
        v = X[mask].mean(axis=0)
        norm = np.linalg.norm(v)
        cen[int(c)] = (v / norm) if norm > 0 else v
    return cen


def merge_by_centroid(
    X: np.ndarray,
    pred: np.ndarray,
    thresh: float = 0.94,
    max_iter: int = 20
) -> np.ndarray:
    """Merge clusters whose centroids exceed `thresh` cosine similarity in embedding space X.

    Note: X MUST be the L2-normalized embedding space (not UMAP projection).
    """
    pred = np.asarray(pred).copy()
    for _ in range(max_iter):
        cen = compute_centroids(X, pred)
        ks = sorted(cen.keys())
        if len(ks) < 2:
            break

        M = np.array([cen[k] for k in ks])
        S = M @ M.T
        np.fill_diagonal(S, -1.0)
        if S.max() < thresh:
            break

        parent = {k: k for k in ks}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if S[i, j] >= thresh:
                    ra, rb = find(ks[i]), find(ks[j])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

        remap = {k: find(k) for k in ks}
        if all(remap[k] == k for k in ks):
            break
        pred = np.array([remap.get(p, p) if p != -1 else -1 for p in pred])

    # Re-index contiguous cluster labels 0, 1, 2...
    uniq = sorted(c for c in set(pred) if c != -1)
    ren = {c: i for i, c in enumerate(uniq)}
    return np.array([ren.get(p, -1) if p != -1 else -1 for p in pred])


def score(
    pred_or_ids: Union[Dict[str, int], List[str], np.ndarray],
    truth_or_pred: Union[Dict[str, str], np.ndarray, List[int]],
    truth_or_none: Optional[Dict[str, str]] = None,
    split_noise: bool = True
) -> Tuple[float, float]:
    """Computes Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI).

    Supported calling signatures:
      1. score(pred_by_id: dict, truth: dict, split_noise=True)
      2. score(ids: list, pred: array/list, truth: dict, split_noise=True)
    """
    if truth_or_none is not None:
        # Signature: score(ids, pred, truth, split_noise)
        ids = list(pred_or_ids)  # type: ignore
        pred = truth_or_pred
        truth = truth_or_none
        by_id = {str(a): int(p) for a, p in zip(ids, pred)}
    elif isinstance(pred_or_ids, dict):
        # Signature: score(pred_by_id, truth, split_noise)
        by_id = {str(k): int(v) for k, v in pred_or_ids.items()}
        truth = truth_or_pred  # type: ignore
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

    ari = float(adjusted_rand_score(y_true, y_pred))
    ami = float(adjusted_mutual_info_score(y_true, y_pred))
    return ari, ami


def load_spend_by_asset(
    ids: List[str],
    staged_csv: Optional[str] = None
) -> Tuple[np.ndarray, float]:
    """Computes spend attributable to each representative asset ID via its phash group."""
    csv_path = staged_csv or cfg.ASSETS_STAGED_CSV
    if not os.path.exists(csv_path):
        return np.zeros(len(ids), dtype=float), 0.0

    with open(csv_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("stage_status") == "OK"]

    ph_spend = collections.defaultdict(float)
    for r in rows:
        phash = r.get("phash", "")
        spend = float(r.get("spend") or 0.0)
        ph_spend[phash] += spend

    ph_of = {r["asset_id"]: r.get("phash", "") for r in rows}
    asset_spends = np.array([ph_spend.get(ph_of.get(a, ""), 0.0) for a in ids], dtype=float)
    total_spend = sum(ph_spend.values())
    return asset_spends, total_spend
