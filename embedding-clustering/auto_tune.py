"""Automated hyperparameter tuning for creative asset clustering.

Can evaluate against known evaluation labels (ARI/AMI) or perform unsupervised
optimization using Silhouette score and spend coverage.
Generates or updates per-advertiser YAML configuration.

Usage:
    python embedding-clustering/auto_tune.py --advertiser maxis_my
    python embedding-clustering/auto_tune.py --output configs/my_brand.yaml
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import silhouette_score
import yaml

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import config as cfg
from common.logging import get_logger
from common.vectors import (
    HDBSCAN,
    compute_centroids,
    load_aligned_embeddings,
    load_spend_by_asset,
    merge_by_centroid,
    score,
)

logger = get_logger("auto_tune", stage="tuning")


def run_auto_tune(
    data_dir: Optional[str] = None,
    eval_labels_path: Optional[str] = None,
    advertiser_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs a hyperparameter sweep and returns recommended parameters."""
    d_dir = data_dir or cfg.DATA_DIR
    aligned = load_aligned_embeddings(d_dir)
    ids: List[str] = aligned["ids"]  # type: ignore
    spend_arr, total_spend = load_spend_by_asset(ids)

    # Check for ground truth labels
    truth = None
    lbl_path = eval_labels_path or os.path.join(cfg.EVAL_DIR, "eval_labels.json")
    if os.path.exists(lbl_path):
        try:
            with open(lbl_path, "r", encoding="utf-8") as fh:
                truth = {str(k): str(v) for k, v in json.load(fh).items()}
            logger.info(f"Loaded {len(truth)} ground-truth evaluation labels.")
        except Exception as e:
            logger.warning(f"Could not load eval labels ({e}); using unsupervised scoring.")

    from common.vectors import combine, l2, reduce_manifold

    tracks_to_try = [t for t in ("frame", "video") if t in aligned]
    if not tracks_to_try:
        tracks_to_try = ["frame"]

    best_config: Optional[Dict[str, Any]] = None
    best_score = -1.0

    logger.info("Running parameter sweep...")
    for track in tracks_to_try:
        vis_vec = aligned.get(track)
        txt_vec = aligned.get("text")
        if vis_vec is None or txt_vec is None:
            continue

        for w_vis in (0.2, 0.3, 0.5):
            X = combine(vis_vec, txt_vec, w_vis)  # type: ignore

            # Manifold reduction
            try:
                red_X = reduce_manifold(X, n_components=30, n_neighbors=15, min_dist=0.0, seed=42)
            except Exception:
                red_X = X

            for mcs in (3, 4, 5):
                for ms in (1, 2):
                    try:
                        pred = HDBSCAN(
                            metric="euclidean",
                            min_cluster_size=mcs,
                            min_samples=ms,
                            cluster_selection_method="eom",
                        ).fit_predict(red_X)
                    except Exception:
                        continue

                    # Post-hoc centroid merge
                    for m_thresh in (0.92, 0.94, 0.96):
                        merged_pred = merge_by_centroid(X, pred, thresh=m_thresh)
                        n_clusters = len(set(p for p in merged_pred if p != -1))
                        if n_clusters < 2:
                            continue

                        noise_mask = merged_pred == -1
                        noise_spend = float(spend_arr[noise_mask].sum())
                        noise_spend_pct = (100.0 * noise_spend / total_spend) if total_spend > 0 else 0.0

                        if truth:
                            ari, ami = score(ids, merged_pred, truth)
                            # Composite objective: high ARI and penalize excessive noise spend
                            metric_val = ari - (0.005 * noise_spend_pct)
                        else:
                            # Unsupervised: silhouette score on clustered items + coverage
                            non_noise = ~noise_mask
                            if np.sum(non_noise) > n_clusters:
                                try:
                                    sil = float(silhouette_score(X[non_noise], merged_pred[non_noise], metric="cosine"))
                                except Exception:
                                    sil = 0.0
                            else:
                                sil = 0.0
                            coverage = (100.0 - noise_spend_pct) / 100.0
                            metric_val = (0.7 * sil) + (0.3 * coverage)

                        if metric_val > best_score:
                            best_score = metric_val
                            best_config = {
                                "track": track,
                                "visual_weight": w_vis,
                                "min_cluster_size": mcs,
                                "min_samples": ms,
                                "umap_components": 30,
                                "umap_seed": 42,
                                "merge_threshold": m_thresh,
                                "text_embedding_model": "all-MiniLM-L6-v2",
                                "clusters": n_clusters,
                                "score": round(metric_val, 4),
                            }

    if not best_config:
        # Default fallback
        best_config = {
            "track": "frame",
            "visual_weight": 0.3,
            "min_cluster_size": 4,
            "min_samples": 2,
            "umap_components": 30,
            "umap_seed": 42,
            "merge_threshold": 0.94,
            "text_embedding_model": "all-MiniLM-L6-v2",
        }

    logger.info(f"Best configuration found: {best_config}")
    return best_config


def main():
    parser = argparse.ArgumentParser(description="Auto-tune clustering hyperparameters")
    parser.add_argument("--advertiser", type=str, help="Existing advertiser config name")
    parser.add_argument("--output", type=str, help="Path to save recommended YAML config")
    args = parser.parse_args()

    if args.advertiser:
        cfg.load_advertiser_yaml(args.advertiser)

    recommended = run_auto_tune(advertiser_name=cfg.ADVERTISER_NAME)
    print("\n" + "=" * 50)
    print("RECOMMENDED CLUSTERING HYPERPARAMETERS:")
    print("=" * 50)
    for k, v in recommended.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    if args.output:
        out_data = {
            "advertiser_name": cfg.ADVERTISER_NAME,
            "advertiser_names": cfg.ADVERTISER_NAMES,
            "country": cfg.COUNTRY,
            "date_range": [cfg.START_DATE, cfg.END_DATE],
            "clustering": recommended,
            "gemini": {
                "model": cfg.GEMINI_MODEL,
                "thinking_budget": cfg.GEMINI_THINKING_BUDGET,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            yaml.safe_dump(out_data, fh, sort_keys=False)
        print(f"\nSaved configuration to {args.output}")


if __name__ == "__main__":
    main()
