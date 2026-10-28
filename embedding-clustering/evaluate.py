"""Score HDBSCAN clustering against the blind eval set.

Rewritten 2026-08-13. The previous version set `predicted_combined` equal to
`true_labels` and scored a list against a copy of itself, reporting ARI 1.0000.
That number was meaningless. This version clusters real embeddings and scores
them against evaluation/eval_labels.json, which was hand-tagged before any
clustering output existed.

Sweeps embedding configurations so the spec's required comparison - visual-only
vs combined, and video mode vs thumbnail mode - is a measurement rather than an
assertion.

Usage:
    python embedding-clustering/evaluate.py
    python embedding-clustering/evaluate.py --min-cluster-size 3
"""

import argparse
import json
import os
import sys

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ingestion.config as cfg
from common.vectors import l2, combine, score, load_aligned_embeddings as load_vectors, cluster_hdbscan as cluster
from common.logging import get_logger

logger = get_logger("evaluate", stage="evaluation")
EVAL_DIR = os.path.join(repo_root, "evaluation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cluster-size", type=int, default=2)
    ap.add_argument("--min-samples", type=int, default=None)
    args = ap.parse_args()

    labels_path = os.path.join(EVAL_DIR, "eval_labels.json")
    if not os.path.exists(labels_path):
        print(f"Missing {labels_path} - tag evaluation/eval_sheet.html first.")
        return
    with open(labels_path, "r", encoding="utf-8") as fh:
        truth = {str(k): str(v) for k, v in json.load(fh).items()}

    V = load_vectors()
    ids = V["ids"]
    print(f"Clustering {len(ids)} assets  (min_cluster_size={args.min_cluster_size})")
    tagged = {a for a in truth if a in set(ids)}
    print(f"Eval set: {len(truth)} tagged, {len(tagged)} present in the clustered set")
    n_groups = len(set(truth.values()))
    multi = sum(1 for g in set(truth.values())
                if sum(1 for v in truth.values() if v == g) > 1)
    print(f"  human groups: {n_groups} ({multi} with >1 member)\n")

    configs = []
    for track in ("video", "frame"):
        configs.append((f"{track}-only", track, 1.0))
        for w in (0.8, 0.7, 0.5, 0.3, 0.2):
            configs.append((f"{track}+text w={w}", track, w))
    # track=None: uses no visual block at all, so it must be excluded from the
    # video-vs-frame comparison below.
    configs.append(("text-only", None, 0.0))

    print(f"{'config':<22}{'clusters':>9}{'noise':>7}{'ARI':>9}{'AMI':>9}"
          f"{'ARI*':>9}")
    print("-" * 65)
    results = []
    for name, track, w in configs:
        mat = combine(V[track] if track else V["text"], V["text"], w)
        pred = cluster(mat, args.min_cluster_size, args.min_samples)
        by_id = {a: int(p) for a, p in zip(ids, pred)}
        ari, ami = score(by_id, truth, split_noise=True)
        ari_lump, _ = score(by_id, truth, split_noise=False)
        n_cl = len(set(p for p in pred if p != -1))
        n_noise = int((pred == -1).sum())
        results.append((name, track, w, ari, ami, n_cl, n_noise))
        print(f"{name:<22}{n_cl:>9}{n_noise:>7}{ari:>9.4f}{ami:>9.4f}{ari_lump:>9.4f}")

    print("-" * 65)
    print("ARI  = noise points treated as separate singletons (fairer)")
    print("ARI* = all noise lumped into one cluster (harsher)")

    best = max(results, key=lambda r: r[3])
    print(f"\nBest config: {best[0]}  ARI={best[3]:.4f}  AMI={best[4]:.4f}")
    print(f"  {best[5]} clusters, {best[6]} noise assets")

    # Compare only configs that actually use a visual block. text-only has
    # track=None and would otherwise be credited to whichever track it was
    # arbitrarily filed under.
    vid = max((r for r in results if r[1] == "video"), key=lambda r: r[3])
    frm = max((r for r in results if r[1] == "frame"), key=lambda r: r[3])
    txt = next(r for r in results if r[1] is None)
    print(f"\nVideo mode best : {vid[0]:<22} ARI={vid[3]:.4f}")
    print(f"Frame mode best : {frm[0]:<22} ARI={frm[3]:.4f}")
    delta = vid[3] - frm[3]
    verdict = ("video mode justifies its ~13x cost"
               if delta > 0.02 else
               "frame mode is competitive - video mode's extra cost is NOT justified"
               if delta > -0.02 else "frame mode OUTPERFORMS video mode")
    print(f"  delta = {delta:+.4f}  ->  {verdict}")
    print(f"\nText-only       : {txt[0]:<22} ARI={txt[3]:.4f}")
    if txt[3] >= max(vid[3], frm[3]):
        print("  NOTE: text-only matches or beats every multimodal config. The visual")
        print("  vectors are not adding discriminating signal on this dataset.")

    out = os.path.join(EVAL_DIR, "eval_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump([{"config": r[0], "track": r[1], "visual_weight": r[2],
                    "ari": r[3], "ami": r[4], "clusters": r[5], "noise": r[6]}
                   for r in results], fh, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
