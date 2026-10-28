"""Generate real Vertex multimodal visual embeddings for the staged assets.

Replaces the `np.random.rand(1408)` placeholder in embed.py.

Produces two parallel embedding tracks over the same assets so the spec's
video-vs-thumbnail question can be answered with a measurement:

  track "video"  - video mode (mp4, Essential) for videos; image mode for stills
  track "frame"  - image mode on the extracted cover frame; image mode for stills

Both tracks are 1408-d, so they are directly comparable downstream.

Every call is billing-labelled and written to the cost ledger. Calls go through
PredictionServiceClient directly because the high-level SDK wrapper exposes no
labels parameter.

Usage:
    python embedding-clustering/embed_visual.py                 # 340 reps, both tracks
    python embedding-clustering/embed_visual.py --limit 20      # smoke test
    python embedding-clustering/embed_visual.py --tracks frame  # frame only
"""

import argparse
import collections
import concurrent.futures
import csv
import os
import sys
import threading
import time

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from google.cloud import aiplatform_v1
from google.protobuf import json_format, struct_pb2

import ingestion.config as cfg
from common import costs

DIM = 1408
INTERVAL_SEC = 16   # >=15 => Essential tier. Pinned so an SDK default change
                    # cannot silently move us to a pricier tier.
MAX_RETRIES = 3

_local = threading.local()


def client():
    if not hasattr(_local, "c"):
        _local.c = aiplatform_v1.PredictionServiceClient(
            client_options={"api_endpoint": f"{cfg.GCP_REGION}-aiplatform.googleapis.com"})
    return _local.c


ENDPOINT = (f"projects/{cfg.GCP_PROJECT_ID}/locations/{cfg.GCP_REGION}"
            f"/publishers/google/models/multimodalembedding@001")


def _predict(instance: dict, label_stage: str):
    req = aiplatform_v1.PredictRequest(endpoint=ENDPOINT, labels=costs.labels(label_stage))
    req.instances.append(json_format.ParseDict(instance, struct_pb2.Value()))
    req.parameters = json_format.ParseDict({"dimension": DIM}, struct_pb2.Value())
    resp = client().predict(request=req)
    pred = json_format.MessageToDict(resp._pb)["predictions"][0]
    if "videoEmbeddings" in pred:
        return pred["videoEmbeddings"][0]["embedding"]
    return pred["imageEmbedding"]


def embed_one(row: dict, track: str):
    """Return (asset_id, track, vector|None, error|None)."""
    aid = row["asset_id"]
    is_video = row["media_type"] == "video"
    stage = f"embed-{track}"

    if track == "video" and is_video:
        uri = f"gs://{cfg.GCS_BUCKET}/{cfg.GCS_RAW_PREFIX}/{os.path.basename(row['local_path'])}"
        instance = {"video": {"gcsUri": uri,
                              "videoSegmentConfig": {"intervalSec": INTERVAL_SEC}}}
        sku, units = "mm_video_essential", float(row["duration_sec"] or 0)
    else:
        # frame track for videos, or any still asset in either track
        src = row["frame_path"] if (is_video and row["frame_path"]) else row["local_path"]
        prefix = cfg.GCS_FRAME_PREFIX if (is_video and row["frame_path"]) else cfg.GCS_RAW_PREFIX
        uri = f"gs://{cfg.GCS_BUCKET}/{prefix}/{os.path.basename(src)}"
        instance = {"image": {"gcsUri": uri}}
        sku, units = "mm_image", 1.0

    last = None
    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        try:
            vec = _predict(instance, stage)
            costs.record(stage, "vertex-mm-embedding", sku, units, aid, time.time() - t0)
            return aid, track, vec, None
        except Exception as err:
            last = f"{type(err).__name__}: {str(err)[:120]}"
            # A failed call is generally not billed; record it as an error row so
            # the ledger reflects attempts without inflating cost.
            costs.record(stage, "vertex-mm-embedding", sku, 0.0, aid,
                         time.time() - t0, status="ERROR")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return aid, track, None, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tracks", default="video,frame")
    ap.add_argument("--all", action="store_true",
                    help="embed all 787 assets instead of phash-distinct representatives")
    ap.add_argument("--force", action="store_true",
                    help="force re-embedding of already cached assets")
    args = ap.parse_args()

    tracks = [t.strip() for t in args.tracks.split(",")]
    for t in tracks:
        if t not in ("video", "frame"):
            sys.exit(f"Unknown track '{t}'. Use 'video', 'frame', or 'video,frame'.")

    path = os.path.join(cfg.DATA_DIR, "assets_staged.csv")
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["stage_status"] == "OK"]

    if args.all:
        targets = rows
    else:
        by_hash = collections.defaultdict(list)
        for r in rows:
            by_hash[r["phash"]].append(r)
        targets = [max(v, key=lambda r: float(r["spend"] or 0)) for v in by_hash.values()]
        targets.sort(key=lambda r: r["asset_id"])
    if args.limit:
        targets = targets[: args.limit]

    out = os.path.join(cfg.DATA_DIR, "visual_embeddings.npz")
    vecs = {t: {} for t in tracks}

    # Load existing cached embeddings if available
    if os.path.exists(out) and not args.force:
        try:
            cached = np.load(out, allow_pickle=True)
            c_ids = [str(x) for x in cached.get("asset_ids", [])]
            id_to_idx = {aid: idx for idx, aid in enumerate(c_ids)}
            for t in tracks:
                key = f"vec_{t}"
                if key in cached:
                    mat = cached[key]
                    for r in targets:
                        aid = r["asset_id"]
                        if aid in id_to_idx:
                            row_vec = mat[id_to_idx[aid]]
                            if not np.all(np.isnan(row_vec)):
                                vecs[t][aid] = row_vec
            cached_count = sum(len(vecs[t]) for t in tracks)
            if cached_count:
                print(f"Loaded {cached_count} cached visual embeddings from {out}")
        except Exception as ex:
            print(f"Warning: could not read cache ({ex}); proceeding fresh")

    jobs = [(r, t) for t in tracks for r in targets if r["asset_id"] not in vecs[t]]
    if not jobs:
        print(f"All {len(targets)} assets for tracks {tracks} are already cached. (Use --force to re-embed)")
        return

    n_vid = sum(1 for r, t in jobs if r["media_type"] == "video" and t == "video")
    vid_secs = sum(float(r["duration_sec"] or 0) for r, t in jobs if r["media_type"] == "video" and t == "video")
    print(f"Embedding {len(jobs)} pending asset-track pairs ({len(targets)} total targets)")
    print(f"Pending tracks jobs: {len(jobs)}")
    est = 0.0
    for r, t in jobs:
        if t == "video" and r["media_type"] == "video":
            est += float(r["duration_sec"] or 0) * costs.RATES["mm_video_essential"][0]
        else:
            est += costs.RATES["mm_image"][0]
    print(f"Projected API cost for pending jobs: ${est:.2f}\n")

    errors = []
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(embed_one, r, t) for r, t in jobs]
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            aid, track, vec, err = f.result()
            if vec is not None:
                vecs[track][aid] = vec
            else:
                errors.append((aid, track, err))
            if i % 50 == 0 or i == len(futs):
                el = time.time() - t_start
                print(f"  {i}/{len(futs)}  {el/60:.1f} min elapsed  {len(errors)} errors")

    out = os.path.join(cfg.DATA_DIR, "visual_embeddings.npz")
    payload = {}
    ids = [r["asset_id"] for r in targets]
    for t in tracks:
        mat = np.array([vecs[t].get(a, [np.nan] * DIM) for a in ids], dtype=np.float32)
        payload[f"vec_{t}"] = mat
    payload["asset_ids"] = np.array(ids)
    np.savez_compressed(out, **payload)

    print(f"\nWrote {out}")
    for t in tracks:
        got = len(vecs[t])
        print(f"  track {t:6}: {got}/{len(ids)} embedded")
    if errors:
        print(f"\n{len(errors)} failures:")
        for aid, t, e in errors[:8]:
            print(f"  {aid} [{t}] {e}")
    print(f"\nElapsed: {(time.time()-t_start)/60:.1f} min")
    print("\n--- ledger ---")
    costs.summary()


if __name__ == "__main__":
    main()
