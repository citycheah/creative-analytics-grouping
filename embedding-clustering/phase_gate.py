"""Phase Gate: measure Vertex Multimodal Embedding cost/latency before scaling.

The spec marks this "do not skip". It answers one question: is video-mode
embedding acceptable at target dataset size, or do we fall back to
thumbnail/image mode as a *stated* scoping decision?

Runs three modes over the same assets so the comparison is like-for-like:
  video  - full mp4 via Vertex video mode
  frame  - the extracted cover frame via Vertex image mode
  image  - native still assets via Vertex image mode

Usage:
    python embedding-clustering/phase_gate.py --n 10
"""

import argparse
import csv
import os
import statistics
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ingestion.config as cfg

DIM = 1408


def gcs_uri(local_path: str, prefix: str) -> str:
    return f"gs://{cfg.GCS_BUCKET}/{prefix}/{os.path.basename(local_path)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="assets per mode")
    args = ap.parse_args()

    import vertexai
    from vertexai.vision_models import Image, MultiModalEmbeddingModel, Video

    vertexai.init(project=cfg.GCP_PROJECT_ID, location=cfg.GCP_REGION)
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    staged = os.path.join(cfg.DATA_DIR, "assets_staged.csv")
    with open(staged, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["stage_status"] == "OK"]

    videos = [r for r in rows if r["media_type"] == "video"][: args.n]
    images = [r for r in rows if r["media_type"] == "image"][: args.n]

    results = {}

    def timed(label, items, call):
        lats, dims, errs = [], [], []
        print(f"\n=== {label}  (n={len(items)}) ===")
        for r in items:
            t0 = time.time()
            try:
                vec = call(r)
                dt = time.time() - t0
                lats.append(dt)
                dims.append(len(vec) if vec is not None else 0)
                print(f"  ok  {dt:6.2f}s  dim={len(vec)}  {r['asset_id']}")
            except Exception as err:
                dt = time.time() - t0
                errs.append(f"{type(err).__name__}: {str(err)[:110]}")
                print(f"  ERR {dt:6.2f}s  {r['asset_id']}  {errs[-1]}")
        results[label] = {"lat": lats, "dims": dims, "errs": errs}

    def embed_video(r):
        v = Video.load_from_file(gcs_uri(r["local_path"], cfg.GCS_RAW_PREFIX))
        emb = model.get_embeddings(video=v, dimension=DIM)
        return emb.video_embeddings[0].embedding

    def embed_frame(r):
        im = Image.load_from_file(gcs_uri(r["frame_path"], cfg.GCS_FRAME_PREFIX))
        return model.get_embeddings(image=im, dimension=DIM).image_embedding

    def embed_image(r):
        im = Image.load_from_file(gcs_uri(r["local_path"], cfg.GCS_RAW_PREFIX))
        return model.get_embeddings(image=im, dimension=DIM).image_embedding

    timed("video mode (mp4)", videos, embed_video)
    timed("frame mode (cover jpg)", videos, embed_frame)
    timed("image mode (native still)", images, embed_image)

    print("\n" + "=" * 62)
    print("PHASE GATE SUMMARY")
    print("=" * 62)
    total_assets = len(rows)
    for label, d in results.items():
        if d["lat"]:
            mean = statistics.mean(d["lat"])
            med = statistics.median(d["lat"])
            print(f"\n{label}")
            print(f"  ok={len(d['lat'])} err={len(d['errs'])}  dim={set(d['dims'])}")
            print(f"  mean={mean:.2f}s  median={med:.2f}s  "
                  f"min={min(d['lat']):.2f}s  max={max(d['lat']):.2f}s")
            print(f"  projected serial time for {total_assets} assets: "
                  f"{mean * total_assets / 60:.1f} min "
                  f"({mean * total_assets / 60 / 8:.1f} min at 8 workers)")
        else:
            print(f"\n{label}\n  ALL FAILED ({len(d['errs'])} errors)")
        for e in dict.fromkeys(d["errs"]):
            print(f"    ! {e}")


if __name__ == "__main__":
    main()
