"""Test two fixes for HDBSCAN aspect-ratio over-splitting across hero campaigns.

Diagnosis: Multi-format cuts (9:16 Reels, 16:9 YouTube, 1:1 Feed) of the same hero
campaign often get split into separate clusters whose centroid cosine similarity exceeds
0.95+, while their internal cohesion is ~0.92. HDBSCAN applies uneven granularity across
aspect ratios, splitting one campaign across formats.

Option 2 - cluster_selection_epsilon, HDBSCAN's own merge mechanism in UMAP space.
Option 1 - post-hoc centroid merge: union-find over cluster pairs whose
  centroid cosine in the original L2-normalized embedding space exceeds a threshold.

Reports, for each setting: cluster count, noise, ARI/AMI, and the highest remaining
centroid similarity.

Usage:
    python embedding-clustering/fix_oversplit.py
"""

import collections
import csv
import json
import os
import sys
import warnings

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

warnings.filterwarnings("ignore")

import hdbscan
import umap
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import ingestion.config as cfg
from common.vectors import (
    l2,
    load_combined_vectors,
    load_spend_by_asset,
    score,
    compute_centroids,
    merge_by_centroid,
)
from common.logging import get_logger

logger = get_logger("fix_oversplit", stage="clustering")

TRACK, W_VIS = "frame", 0.3
MCS, MS, NC, SEED = 4, 2, 30, 42
EVAL_DIR = os.path.join(repo_root, "evaluation")


def load():
    return load_combined_vectors(track=TRACK, w_vis=W_VIS)


def spend_map(ids):
    return load_spend_by_asset(ids)


def centroids(X, pred):
    return compute_centroids(X, pred)


def worst_pair(X, pred):
    cen = centroids(X, pred)
    ks = sorted(cen)
    if len(ks) < 2:
        return 0.0, None
    M = np.array([cen[k] for k in ks])
    S = M @ M.T
    np.fill_diagonal(S, -1)
    i, j = np.unravel_index(np.argmax(S), S.shape)
    return float(S[i, j]), (ks[i], ks[j])


def family(pred, ids, targets):
    """How many distinct clusters the given representative ids now span."""
    idx = {a: i for i, a in enumerate(ids)}
    got = {int(pred[idx[a]]) for a in targets if a in idx}
    return got


def main():
    labels_file = os.path.join(EVAL_DIR, "eval_labels.json")
    with open(labels_file, "r", encoding="utf-8") as fh:
        truth = {str(k): str(v) for k, v in json.load(fh).items()}
    ids, X = load()
    spend, total = spend_map(ids)

    red = np.asarray(umap.UMAP(n_components=NC, n_neighbors=15, min_dist=0.0,
                               metric="cosine", random_state=SEED).fit_transform(X),
                     dtype=np.float64)
    # Epsilon is in UMAP coordinate units, so establish the actual scale first
    # rather than guessing - this is what the earlier 0.0-0.10 sweep got wrong.
    from scipy.spatial.distance import pdist
    d = pdist(red)
    print(f"UMAP space: {NC}-d, pairwise distance "
          f"p5={np.percentile(d,5):.2f} median={np.median(d):.2f} "
          f"p95={np.percentile(d,95):.2f} max={d.max():.2f}\n")

    hdr = (f"{'method':<26}{'clus':>6}{'noise%':>8}{'sp%':>7}"
           f"{'ARI':>8}{'AMI':>8}{'max_cos':>9}")
    print(hdr); print("-" * len(hdr))

    def report(name, pred):
        n_cl = len(set(p for p in pred if p != -1))
        nm = pred == -1
        ari, ami = score(ids, pred, truth)
        # Measured in the original embedding space; cosine is meaningless on
        # the UMAP projection (see merge_by_centroid docstring).
        w, _ = worst_pair(X, pred)
        print(f"{name:<26}{n_cl:>6}{100*nm.sum()/len(ids):>8.1f}"
              f"{100*spend[nm].sum()/total:>7.1f}{ari:>8.4f}{ami:>8.4f}"
              f"{w:>9.4f}")
        return ari, n_cl

    base = hdbscan.HDBSCAN(metric="euclidean", min_cluster_size=MCS, min_samples=MS,
                           cluster_selection_method="eom").fit_predict(red)
    report("baseline (pre-merge)", base)

    print()
    for eps in (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00):
        p = hdbscan.HDBSCAN(metric="euclidean", min_cluster_size=MCS, min_samples=MS,
                            cluster_selection_epsilon=eps,
                            cluster_selection_method="eom").fit_predict(red)
        report(f"opt2 epsilon={eps}", p)

    print()
    for th in (0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.90):
        p = merge_by_centroid(X, base, th)
        report(f"opt1 merge>={th}", p)

    print("\nmax_cos = highest remaining centroid cosine between any two clusters in 1792-D space.")


if __name__ == "__main__":
    main()
