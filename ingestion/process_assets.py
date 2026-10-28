"""Gemini concept extraction over the staged assets -> assets_enriched.

Perceptual hashing lives in ingestion/stage_assets.py. This module runs
structured Gemini 2.5 Flash extraction (with thinking_budget=0 for cost control)
over the deduplicated pHash representatives and writes `assets_enriched.csv`.

Assets are referenced by their gs:// URI rather than external CDN URLs, so
Gemini reads from GCS directly instead of re-fetching media over HTTP.

Usage:
    python ingestion/process_assets.py --limit 10   # cost-controlled trial
    python ingestion/process_assets.py              # all representatives
"""

import argparse
import collections
import csv
import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd

from common import costs
from ingestion.ad_analyser.analyser import AdAnalyser
from ingestion.ad_analyser.config import AdAnalyserConfig
from ingestion.ad_analyser.models import CreativeConceptSchema
import ingestion.config as cfg

# Thinking budget and model from centralized config
MODEL = getattr(cfg, "GEMINI_MODEL", "gemini-2.5-flash")
THINKING_BUDGET = getattr(cfg, "GEMINI_THINKING_BUDGET", 0)

CONTEXT_COLS = ["advertiser_name", "creative_landingpage_url", "creative_campaign_name",
                "social_headline_text", "social_description", "social_campaign_text"]


def load_representatives(limit=None, use_all=False):
    """Staged assets collapsed to phash-distinct representatives.

    Exact dedup is already satisfied upstream: every creative_url_supplier ends
    in a distinct MD5, so GROUP BY creative_id removed byte-identical copies.
    This is the spec's tier-2 near-dup filter - one Gemini call per distinct
    render instead of one per raw row (787 -> 340).
    """
    path = os.path.join(cfg.DATA_DIR, "assets_staged.csv")
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["stage_status"] == "OK"]
    if use_all:
        out = rows
    else:
        by_hash = collections.defaultdict(list)
        for r in rows:
            by_hash[r["phash"]].append(r)
        out = [max(v, key=lambda r: float(r["spend"] or 0)) for v in by_hash.values()]
    out.sort(key=lambda r: r["asset_id"])
    return out[:limit] if limit else out


def gcs_uri(row):
    """gs:// URI for the asset. Videos point at the mp4; stills at the image."""
    name = os.path.basename(row["local_path"])
    return f"gs://{cfg.GCS_BUCKET}/{cfg.GCS_RAW_PREFIX}/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--all", action="store_true",
                    help="process all 787 rows instead of phash-distinct representatives")
    ap.add_argument("--force", action="store_true",
                    help="re-enrich already processed assets")
    args = ap.parse_args()

    reps = load_representatives(args.limit, args.all)

    existing_enriched = {}
    out_path = cfg.ASSETS_ENRICHED_CSV
    if os.path.exists(out_path) and not args.force:
        try:
            with open(out_path, "r", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("core_concept_summary"):
                        existing_enriched[row["asset_id"]] = row
            if existing_enriched:
                print(f"Loaded {len(existing_enriched)} existing enriched assets from {out_path}")
        except Exception as ex:
            print(f"Warning: could not read existing enriched assets ({ex})")

    pending_reps = [r for r in reps if r["asset_id"] not in existing_enriched]
    if not pending_reps:
        print(f"All {len(reps)} assets are already enriched in {out_path}. (Use --force to re-enrich)")
        return

    n_vid = sum(1 for r in pending_reps if r["media_type"] == "video")
    print(f"Gemini concept extraction over {len(pending_reps)} pending assets ({len(reps)} total) "
          f"({n_vid} video, {len(pending_reps)-n_vid} image)")
    print(f"Model: {MODEL}  thinking_budget={THINKING_BUDGET}")

    df = pd.DataFrame([{
        "creative_id": r["asset_id"],
        "creative_url_supplier": gcs_uri(r),
        **{c: (r.get(c) or "") for c in CONTEXT_COLS},
    } for r in pending_reps])

    config = AdAnalyserConfig(
        df=df,
        gcp_project_id=cfg.GCP_PROJECT_ID,
        gcp_region=cfg.GCP_REGION,
        gcs_bucket=cfg.GCS_BUCKET,
        gcs_prefix=cfg.GCS_PREFIX,
        response_schema=CreativeConceptSchema,
        model=MODEL,
        labels=costs.labels("enrichment"),
        include_logs=True,
        max_workers=args.workers,
    )
    # Applied post-construction so the analyser's own config assembly is untouched.
    analyser = AdAnalyser(config)
    if THINKING_BUDGET > 0:
        from google.genai import types
        analyser.ai_manager.generation_config.thinking_config = types.ThinkingConfig(thinking_budget=THINKING_BUDGET)

    t0 = time.time()
    result = analyser.run_job()
    elapsed = time.time() - t0

    if result is None or result.empty:
        print("No results returned.")
        return

    ok = result[result.get("status", "SUCCESS") == "SUCCESS"] if "status" in result else result
    print(f"\nProcessed {len(result)} rows, {len(ok)} succeeded, "
          f"{elapsed/60:.1f} min ({elapsed/max(len(result),1):.1f}s per asset)")

    # Join Gemini output back onto the staged metadata.
    staged = {r["asset_id"]: r for r in reps}
    enriched = []
    for _, row in result.iterrows():
        base = staged.get(row["creative_id"], {})
        tags = row.get("visual_style_tags")
        enriched.append({
            "asset_id": row["creative_id"],
            "core_concept_summary": row.get("core_concept_summary") or "",
            "messaging_angle": row.get("messaging_angle") or "",
            "visual_style_tags": "|".join(tags) if isinstance(tags, list) else (tags or ""),
            "phash": base.get("phash", ""),
            "media_type": base.get("media_type", ""),
            "aspect_ratio": base.get("aspect_ratio", ""),
            "platform": base.get("platform", ""),
            "duration_sec": base.get("duration_sec", ""),
            "spend": base.get("spend", ""),
            "impressions": base.get("impressions", ""),
            "status": row.get("status", ""),
            "error_message": row.get("error_message", ""),
        })

    out_path = cfg.ASSETS_ENRICHED_CSV
    all_enriched = list(existing_enriched.values()) + enriched
    by_id = {r["asset_id"]: r for r in all_enriched}
    final_enriched = list(by_id.values())
    final_enriched.sort(key=lambda r: r["asset_id"])

    fieldnames = list(final_enriched[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(final_enriched)
    print(f"Wrote {out_path} ({len(final_enriched)} total rows)")

    filled = sum(1 for e in final_enriched if str(e.get("core_concept_summary") or "").strip())
    print(f"core_concept_summary populated: {filled}/{len(enriched)}")
    uniq = len({e["core_concept_summary"] for e in enriched if e["core_concept_summary"]})
    warn = "  <-- LOW: prompt may not be discriminating" if filled and uniq < filled * 0.8 else ""
    print(f"distinct summaries: {uniq}/{filled}{warn}")

    for e in enriched[:3]:
        if e["core_concept_summary"]:
            print(f"\n  [{e['asset_id']}] {e['core_concept_summary'][:150]}")
            print(f"      angle: {e['messaging_angle'][:90]}")
            print(f"      tags : {e['visual_style_tags'][:90]}")


if __name__ == "__main__":
    main()
