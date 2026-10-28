"""Sweep HDBSCAN settings to trade noise rate against clustering accuracy.

Noise is trivially reducible - min_samples=1 or a large selection epsilon will
drive it toward zero - but doing so forces unrelated ads into themes. The point
of this sweep is to show the tradeoff explicitly rather than pick settings that
merely look tidy.

Levers, in rough order of expected impact:
  * UMAP pre-reduction. HDBSCAN degrades in high dimensions; the combined
    vector is 1792-d. Reducing to ~10-30 dims first is standard practice and
    is usually the single biggest win.
  * min_samples. The real noise knob. Defaults to min_cluster_size; lowering it
    to 1 makes cluster cores much easier to form.
  * cluster_selection_epsilon. Merges clusters closer than a distance floor,
    absorbing borderline points.
  * cluster_selection_method. 'eom' (default) yields fewer, larger clusters;
    'leaf' yields more, finer ones.

Reports both count-based and SPEND-weighted noise, since spend is what the demo
headline actually reports.

Usage:
    python embedding-clustering/tune.py
    python embedding-clustering/tune.py --top 25
"""

import argparse
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

import ingestion.config as cfg
from common.vectors import (
    l2,
    load_combined_vectors,
    load_spend_by_asset,
    score,
    HDBSCAN,
)
from common.logging import get_logger

logger = get_logger("tune", stage="clustering")

TRACK = "frame"
W_VIS = 0.3
EVAL_DIR = os.path.join(repo_root, "evaluation")


def load():
    return load_combined_vectors(track=TRACK, w_vis=W_VIS)


def spend_by_rep(ids):
    return load_spend_by_asset(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    eval_labels_path = os.path.join(EVAL_DIR, "eval_labels.json")
    with open(eval_labels_path, "r", encoding="utf-8") as fh:
        truth = {str(k): str(v) for k, v in json.load(fh).items()}
    ids, X = load()
    spend, total_spend = spend_by_rep(ids)
    print(f"{len(ids)} representatives, {X.shape[1]}-d combined vectors")
    print(f"Baseline in use: mcs=2, min_samples=default, no reduction\n")

    spaces = {"raw-1792d": X}
    try:
        import umap
        for nc in (10, 30):
            red = umap.UMAP(n_components=nc, n_neighbors=15, min_dist=0.0,
                            metric="cosine", random_state=42).fit_transform(X)
            spaces[f"umap-{nc}d"] = np.asarray(red, dtype=np.float64)
        print(f"UMAP reductions built: {[k for k in spaces if k.startswith('umap')]}\n")
    except Exception as err:
        print(f"UMAP unavailable ({str(err)[:60]}) - raw space only\n")

    results = []
    for space_name, M in spaces.items():
        for mcs in (2, 3, 4, 5):
            for ms in (1, 2, None):
                for eps in (0.0, 0.05, 0.10):
                    for method in ("eom", "leaf"):
                        try:
                            pred = HDBSCAN(
                                metric="euclidean", min_cluster_size=mcs,
                                min_samples=ms, cluster_selection_epsilon=eps,
                                cluster_selection_method=method).fit_predict(M)
                        except (ValueError, TypeError) as ex:
                            logger.debug(f"Config failed to fit: {ex}")
                            continue
                        noise_mask = pred == -1
                        n_noise = int(noise_mask.sum())
                        noise_spend = float(spend[noise_mask].sum())
                        n_cl = len(set(p for p in pred if p != -1))
                        if n_cl < 2:
                            continue
                        ari, ami = score(ids, pred, truth)
                        results.append({
                            "space": space_name, "mcs": mcs,
                            "min_samples": ms, "eps": eps, "method": method,
                            "clusters": n_cl,
                            "noise_pct": 100 * n_noise / len(ids),
                            "noise_spend_pct": 100 * noise_spend / total_spend,
                            "ari": ari, "ami": ami,
                        })

    print(f"Evaluated {len(results)} configurations.\n")

    # Pareto frontier: no other config is both cleaner AND more accurate.
    pareto = [r for r in results
              if not any(o["noise_spend_pct"] <= r["noise_spend_pct"] and o["ari"] >= r["ari"]
                         and (o["noise_spend_pct"] < r["noise_spend_pct"] or o["ari"] > r["ari"])
                         for o in results)]
    pareto.sort(key=lambda r: r["noise_spend_pct"])

    hdr = (f"{'space':<11}{'mcs':>4}{'ms':>4}{'eps':>6}{'meth':>6}"
           f"{'clus':>6}{'noise%':>8}{'noise£%':>9}{'ARI':>8}{'AMI':>8}")
    print("=== PARETO FRONTIER (best accuracy for each noise level) ===")
    print(hdr); print("-" * len(hdr))
    for r in pareto[: args.top]:
        print(f"{r['space']:<11}{r['mcs']:>4}{str(r['min_samples']):>4}{r['eps']:>6.2f}"
              f"{r['method']:>6}{r['clusters']:>6}{r['noise_pct']:>8.1f}"
              f"{r['noise_spend_pct']:>9.1f}{r['ari']:>8.4f}{r['ami']:>8.4f}")

    print("\n=== BEST BY ACCURACY ===")
    print(hdr); print("-" * len(hdr))
    for r in sorted(results, key=lambda r: -r["ari"])[:8]:
        print(f"{r['space']:<11}{r['mcs']:>4}{str(r['min_samples']):>4}{r['eps']:>6.2f}"
              f"{r['method']:>6}{r['clusters']:>6}{r['noise_pct']:>8.1f}"
              f"{r['noise_spend_pct']:>9.1f}{r['ari']:>8.4f}{r['ami']:>8.4f}")

    base = [r for r in results if r["space"] == "raw-1792d" and r["mcs"] == 2
            and r["min_samples"] is None and r["eps"] == 0.0 and r["method"] == "eom"]
    if base:
        b = base[0]
        print(f"\nCurrent production config: noise {b['noise_pct']:.1f}% "
              f"({b['noise_spend_pct']:.1f}% of spend), ARI {b['ari']:.4f}, "
              f"{b['clusters']} clusters")

    out = os.path.join(EVAL_DIR, "tuning_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {out}  ({len(results)} configs)")
    print("\nCAVEAT: the eval set is 30 assets / 15 groups. ARI differences below")
    print("~0.05 are not meaningful, and picking the single top-ARI row is fitting")
    print("the test set. Prefer a config that is robust across neighbouring settings.")


if __name__ == "__main__":
    main()
