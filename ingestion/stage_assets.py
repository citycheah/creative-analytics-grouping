"""Stage advertising creative assets locally: download, extract cover frames, compute pHash.

Handles common real-world ad telemetry data-quality issues:
  * `creative_mime_type` is often unreliable (e.g., rows claiming `application/json`
    when their URLs are really `.mp4` or `.jpeg`). Media type is derived from the URL.
  * `creative_video_duration` may be absent from the source table and is recovered
    via `ffprobe`.
  * `creative_width`/`creative_height` are parsed and bucketed from `creative_size`.

Usage:
    python ingestion/stage_assets.py            # all assets
    python ingestion/stage_assets.py --limit 20 # Phase Gate subset
"""

import argparse
import concurrent.futures
import csv
import os
import subprocess
import sys
import urllib.request
from fractions import Fraction

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from PIL import Image

import ingestion.config as cfg
from common.vectors import compute_phash_hex

STAGE_DIR = cfg.STAGE_DIR
RAW_DIR = cfg.RAW_DIR
FRAME_DIR = cfg.FRAME_DIR

VIDEO_EXTS = {"mp4", "mov", "webm"}
IMAGE_EXTS = {"jpg", "jpeg", "png", "gif"}

UA = {"User-Agent": "Mozilla/5.0"}


def url_ext(url: str) -> str:
    """Real media extension. `creative_mime_type` cannot be trusted."""
    path = url.split("?")[0]
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


def aspect_bucket(size: str) -> tuple[str, str]:
    """'360x640' -> ('9:16', '0.562'). Returns ('unknown', '') if unparseable."""
    try:
        w, h = (int(x) for x in size.lower().split("x", 1))
        if w <= 0 or h <= 0:
            return "unknown", ""
        ratio = w / h
        # Snap to the nearest common delivery ratio; anything else keeps its
        # reduced exact form so odd display sizes stay distinguishable.
        known = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 0.8, "2:3": 2 / 3}
        for name, target in known.items():
            if abs(ratio - target) / target < 0.06:
                return name, f"{ratio:.3f}"
        frac = Fraction(w, h).limit_denominator(20)
        return f"{frac.numerator}:{frac.denominator}", f"{ratio:.3f}"
    except Exception:
        return "unknown", ""


def ffprobe_duration(path: str) -> str:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=60,
        )
        val = out.stdout.strip()
        return f"{float(val):.2f}" if val else ""
    except Exception:
        return ""


def extract_frame(src: str, dest: str) -> bool:
    """Grab a representative frame. Seeks to 1s, falls back to frame 0 for
    very short clips where the 1s seek lands past the end."""
    for seek in ("1", "0"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", seek, "-i", src,
                 "-frames:v", "1", "-q:v", "3", dest],
                capture_output=True, timeout=120,
            )
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                return True
        except Exception:
            continue
    return False


def download(url: str, dest: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True  # resume: already staged
    
    # Support local asset paths for offline/benchmark suites
    candidate = url if os.path.isabs(url) else os.path.join(repo_root, url)
    if os.path.exists(candidate) and os.path.isfile(candidate):
        import shutil
        shutil.copyfile(candidate, dest)
        return True

    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=90) as resp, open(dest, "wb") as fh:
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
        return os.path.getsize(dest) > 0
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        return False


def stage_one(row: dict) -> dict:
    asset_id = row["asset_id"]
    url = row["source_url"]
    ext = url_ext(url)
    media_type = "video" if ext in VIDEO_EXTS else "image" if ext in IMAGE_EXTS else "other"

    ratio, ratio_num = aspect_bucket(row.get("creative_size", ""))
    placement = (row.get("social_ad_placement") or "").strip()
    channel = (row.get("channel_name") or "").strip()

    out = dict(row)
    out.update({
        "media_type": media_type,
        "file_ext": ext,
        "aspect_ratio": ratio,
        "aspect_ratio_num": ratio_num,
        "platform": f"{channel} / {placement}" if placement else channel,
        "local_path": "",
        "frame_path": "",
        "duration_sec": "",
        "phash": "",
        "stage_status": "",
    })

    raw_path = os.path.join(RAW_DIR, f"{asset_id}.{ext}")
    if not download(url, raw_path):
        out["stage_status"] = "DOWNLOAD_FAILED"
        return out
    out["local_path"] = raw_path

    # phash needs a still image; for video that means a cover frame first.
    if media_type == "video":
        frame_path = os.path.join(FRAME_DIR, f"{asset_id}.jpg")
        if not (os.path.exists(frame_path) and os.path.getsize(frame_path) > 0):
            if not extract_frame(raw_path, frame_path):
                out["stage_status"] = "FRAME_FAILED"
                return out
        out["frame_path"] = frame_path
        out["duration_sec"] = ffprobe_duration(raw_path)
        hash_src = frame_path
    else:
        hash_src = raw_path

    try:
        with Image.open(hash_src) as im:
            out["phash"] = compute_phash_hex(im.convert("RGB"))
        out["stage_status"] = "OK"
    except Exception as err:
        out["stage_status"] = f"PHASH_FAILED: {str(err)[:60]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stage only the first N assets")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    with open(cfg.ASSETS_RAW_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Staging {len(rows)} assets -> {STAGE_DIR}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(stage_one, r): r for r in rows}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(fut.result())
            if i % 25 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}")

    results.sort(key=lambda r: r["asset_id"])
    out_path = os.path.join(cfg.DATA_DIR, "assets_staged.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    ok = [r for r in results if r["stage_status"] == "OK"]
    print(f"\nStaged OK: {len(ok)}/{len(results)}  -> {out_path}")

    from collections import Counter
    bad = Counter(r["stage_status"] for r in results if r["stage_status"] != "OK")
    if bad:
        print("Failures:", dict(bad))
    print("media_type:", dict(Counter(r["media_type"] for r in results)))
    print("aspect_ratio:", dict(Counter(r["aspect_ratio"] for r in ok).most_common(8)))

    # Near-duplicate pre-filter signal (spec tier 2). Exact dedup (tier 1) is
    # already satisfied: every source_url carries a distinct MD5.
    ph = Counter(r["phash"] for r in ok if r["phash"])
    dup_groups = {h: c for h, c in ph.items() if c > 1}
    print(f"phash groups with >1 member: {len(dup_groups)} "
          f"covering {sum(dup_groups.values())} assets")


if __name__ == "__main__":
    main()
