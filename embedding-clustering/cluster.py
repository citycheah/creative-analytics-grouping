"""Final clustering: HDBSCAN over combined vectors -> assets_clustered + themes_summary.

Configuration chosen by measurement, not assumption (see evaluate.py):
  track  = frame  - thumbnail mode matched or beat video mode at every
                    min_cluster_size tested, so video mode's ~13x cost is not
                    justified on this dataset. Stated scoping decision per spec.
  w_vis  = 0.3    - the spec's 0.7 visual / 0.3 text default is INVERTED here.
                    One advertiser's creatives are visually near-identical, so
                    the concept summary carries the discriminating signal.
  UMAP 30-d pre-reduction before HDBSCAN. This was the single biggest fix.
                    HDBSCAN degrades badly in high dimensions and the combined
                    vector is 1792-d; clustering the raw space stranded 29.5% of
                    spend in noise. Reducing first cut that to 0.1% AND raised
                    ARI 0.7632 -> 0.8856, so it is a strict improvement, not a
                    noise/accuracy tradeoff.
  min_cluster_size = 4, min_samples = 2 - chosen from a 216-config sweep
                    (see tune.py). Sits on a stable plateau: mcs 4 and 5, and
                    every epsilon in 0.0-0.10, give identical scores, so this is
                    not a lucky single cell fitted to a 30-asset eval set.

Clusters the 340 phash-distinct representatives, then propagates cluster_id to
all 787 raw rows via phash so spend/impression rollups cover the full dataset.

Usage:
    python embedding-clustering/cluster.py
    python embedding-clustering/cluster.py --no-labels   # skip Gemini labelling
"""

import argparse
import base64
import collections
import csv
import html
import io
import os
import sys

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ingestion.config as cfg
from common import costs
from common.vectors import l2, merge_by_centroid, load_aligned_embeddings, reduce_manifold, HDBSCAN
from common.logging import get_logger

logger = get_logger("cluster", stage="clustering")

TRACK = getattr(cfg, "CLUSTERING_TRACK", "frame")
W_VIS = getattr(cfg, "CLUSTERING_VISUAL_WEIGHT", 0.3)
MIN_CLUSTER_SIZE = getattr(cfg, "CLUSTERING_MIN_CLUSTER_SIZE", 4)
MIN_SAMPLES = getattr(cfg, "CLUSTERING_MIN_SAMPLES", 2)
UMAP_COMPONENTS = getattr(cfg, "CLUSTERING_UMAP_COMPONENTS", 30)
UMAP_SEED = getattr(cfg, "CLUSTERING_UMAP_SEED", 42)
MERGE_THRESHOLD = getattr(cfg, "CLUSTERING_MERGE_THRESHOLD", 0.94)
MODEL = getattr(cfg, "GEMINI_MODEL", "gemini-2.5-flash")

OUT_DIR = os.path.join(repo_root, "demo-ui", "data")


def load():
    aligned = load_aligned_embeddings(cfg.DATA_DIR)
    ids = aligned["ids"]
    v = aligned[TRACK]
    t = aligned["text"]
    return ids, l2(v), l2(t)


def _heuristic_theme_label(cid: int, members: list, enriched: dict) -> str:
    """Deterministic contrastive labeler when running offline without Vertex AI credentials."""
    text_blob = " ".join(
        f"{enriched.get(a, {}).get('core_concept_summary', '')} {enriched.get(a, {}).get('messaging_angle', '')}"
        for a in members
    ).lower()

    rules = [
        (["family share", "family lines", "6 family"], "5G Postpaid Unlimited Family Share"),
        (["1gbps", "wifi 6", "home fibre", "mesh router"], "Home Fibre 1Gbps WiFi 6 Bundle"),
        (["skyline", "mrt", "nationwide 5g", "ultra-low latency"], "5G Nationwide Speed & Coverage"),
        (["zerolution", "rm1", "flagship", "trade-in"], "Flagship 5G Smartphone RM1 Zerolution"),
        (["mobile legends", "esports", "pantas", "youth", "gaming"], "Hotlink Prepaid 5G Youth Gaming Pass"),
        (["roaming", "balik kampung", "airport", "30+ countries"], "Festive Raya & CNY 5G Roaming"),
        (["sme", "cloud pos", "cafe", "failover", "business fibre"], "Maxis Business SME Fibre & POS"),
        (["voucher", "cashback", "boba", "daily rewards", "loyalty"], "Hotlink App Daily Rewards & Cashback"),
        (["solar", "rooftop", "electricity", "green energy"], "Home Solar Rooftop Energy Subscription"),
        (["concept store", "klcc", "reopened", "kiosk", "showroom"], "Experiential Concept Store Reopenings"),
        (["esim", "port-in", "rm240", "qr code", "keep your number"], "Instant eSIM Switch & Port-In Rebate"),
        (["scam", "cyber guard", "phishing", "caller id"], "AI Scam Shield & Family Cyber Guard"),
    ]
    for keywords, label in rules:
        if any(k in text_blob for k in keywords):
            return label

    first_angle = next(
        (enriched[a].get("messaging_angle", "") for a in members if a in enriched and enriched[a].get("messaging_angle")),
        "",
    )
    if first_angle:
        words = first_angle.split()[:6]
        return " ".join(w.capitalize() for w in words)
    summaries = [enriched[a]["core_concept_summary"] for a in members if a in enriched and enriched[a].get("core_concept_summary")]
    if summaries:
        return f"Theme {cid}: {summaries[0].split('.')[0][:48]}"
    return f"Theme {cid}"


def gemini_labels(clusters, enriched):
    """Batched theme labeling across all clusters in a single Gemini call (with deterministic offline fallback)."""
    out = {}

    # Pre-fill noise cluster and collect clusters needing labels
    cluster_summaries = {}
    for cid, members in sorted(clusters.items()):
        if cid == -1:
            out[cid] = "Unclustered / unrelated"
            continue
        summaries = [enriched[a]["core_concept_summary"] for a in members
                     if a in enriched and enriched[a]["core_concept_summary"]]
        if not summaries:
            out[cid] = f"Theme {cid}"
        else:
            cluster_summaries[cid] = summaries

    if not cluster_summaries:
        return out

    client = None
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google.genai import Client
            client = Client(api_key=api_key, vertexai=False)
        except Exception as err:
            print(f"  Gemini API key client failed ({str(err)[:70]}); falling back.")
            client = None
    elif getattr(cfg, "GCP_PROJECT_ID", "your-gcp-project-id") not in ("", "your-gcp-project-id", "local-demo-project"):
        try:
            from google.genai import Client
            client = Client(vertexai=True, project=cfg.GCP_PROJECT_ID, location=cfg.GCP_REGION)
        except Exception as err:
            print(f"  Gemini unavailable ({str(err)[:70]}); using deterministic concept labeler.")
            client = None

    if client is None:
        used = set()
        for cid, members in sorted(clusters.items()):
            if cid == -1:
                continue
            lbl = _heuristic_theme_label(cid, members, enriched)
            if lbl in used:
                lbl = f"{lbl} ({cid})"
            used.add(lbl)
            out[cid] = lbl
        return out

    # Build batched prompt for all clusters
    prompt_blocks = []
    for cid, summaries in sorted(cluster_summaries.items()):
        # Include up to 8 representative summaries per cluster
        sample_str = "\n".join(f"  - {s}" for s in summaries[:8])
        prompt_blocks.append(f"Cluster ID {cid}:\n{sample_str}")

    all_clusters_text = "\n\n".join(prompt_blocks)

    advertiser_label = getattr(cfg, "ADVERTISER_NAME", "the brand")
    prompt = (
        f"You are an expert advertising creative strategist analyzing clustered ad campaigns for {advertiser_label}.\n"
        "Below are groups of ad concept summaries that an unsupervised clustering model grouped together.\n\n"
        "Your task: Assign ONE distinct, punchy creative theme label to EACH cluster.\n\n"
        "CRITICAL RULES FOR LABELS:\n"
        "1. CONTRASTIVE: Examine ALL clusters together. Make sure each label clearly highlights what makes that cluster UNIQUE compared to the other clusters.\n"
        "2. CONCISE: Exactly 3 to 6 words per label, written in Title Case (e.g., '5G Postpaid Unlimited Family Share', 'Home Fibre 1Gbps WiFi 6 Bundle').\n"
        "3. SPECIFIC: Focus on the specific product focus, promotional proposition, or creative hook.\n"
        f"4. NO BOILERPLATE: DO NOT include redundant words like '{advertiser_label}', 'Ad', 'Campaign', 'Promotion', 'Marketing', or 'Creative'.\n"
        "5. OUTPUT FORMAT: Return a strict JSON object mapping string cluster ID to the string theme label, like: {\"0\": \"Theme Label 0\", \"1\": \"Theme Label 1\"}.\n\n"
        f"CLUSTERS TO LABEL:\n\n{all_clusters_text}"
    )

    try:
        from google.genai import types as _t
        import json

        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=_t.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                labels=costs.labels("cluster-labelling"),
            )
        )
        um = getattr(resp, "usage_metadata", None)
        if um:
            costs.record("cluster-labelling", "gemini", "gemini_text_in",
                         getattr(um, "prompt_token_count", 0) or 0)
            costs.record("cluster-labelling", "gemini", "gemini_text_out",
                         getattr(um, "candidates_token_count", 0) or 0)

        raw_text = resp.text.strip()
        parsed_labels = json.loads(raw_text)

        for cid, summaries in cluster_summaries.items():
            label = parsed_labels.get(str(cid)) or parsed_labels.get(int(cid))
            if label and isinstance(label, str):
                out[cid] = label.strip().strip('"')[:80]
            else:
                out[cid] = _heuristic_theme_label(cid, clusters[cid], enriched)
        print(f"  Successfully batch-labeled {len(cluster_summaries)} clusters in 1 Gemini call.")
    except Exception as err:
        print(f"  Batch labeling failed ({str(err)[:80]}); using deterministic concept labels.")
        for cid in cluster_summaries:
            if cid not in out:
                out[cid] = _heuristic_theme_label(cid, clusters[cid], enriched)

    return out


def thumb(path, w=150):
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=68)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def write_review(clusters, labels, staged, enriched, rollup, path):
    """Self-contained HTML for eyeballing cluster quality."""
    order = sorted(clusters, key=lambda c: (c == -1, -rollup.get(c, {}).get("spend", 0)))
    parts = []
    for cid in order:
        members = clusters[cid]
        r = rollup.get(cid, {})
        cards = []
        for a in sorted(members, key=lambda x: -float(staged[x].get("spend") or 0))[:24]:
            s = staged[a]
            src = s["frame_path"] if s["media_type"] == "video" and s["frame_path"] else s["local_path"]
            b = thumb(src)
            img = (f'<img src="data:image/jpeg;base64,{b}">' if b
                   else '<div class="noimg">n/a</div>')
            summ = html.escape((enriched.get(a, {}).get("core_concept_summary") or "")[:170])
            cards.append(f'<div class="c"><div class="t">{img}</div>'
                         f'<div class="m">{html.escape(s["aspect_ratio"])} &middot; '
                         f'{html.escape(s["media_type"])}</div>'
                         f'<div class="s">{summ}</div></div>')
        head = (f'Cluster {cid} &mdash; {html.escape(labels.get(cid, ""))}' if cid != -1
                else f'NOISE (cluster -1) &mdash; {html.escape(labels.get(cid, ""))}')
        parts.append(
            f'<section class="{"noise" if cid==-1 else ""}"><h2>{head}</h2>'
            f'<div class="meta">{len(members)} distinct renders &middot; '
            f'{r.get("creative_count",0)} raw creatives &middot; '
            f'${r.get("spend",0):,.0f} spend &middot; '
            f'{r.get("pct",0):.1f}% of total</div>'
            f'<div class="g">{"".join(cards)}</div></section>')

    css = """
 :root{color-scheme:light dark;--bg:#fff;--fg:#111;--mut:#666;--line:#e4e4e4;--card:#fafafa;--warn:#b45309}
 @media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#eceff4;--mut:#9aa3ad;--line:#2c313a;--card:#1e2127;--warn:#fbbf24}}
 body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px}
 h1{font-size:22px;margin:0 0 6px} h2{font-size:16px;margin:0 0 4px}
 section{border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px;background:var(--card)}
 section.noise{border-color:var(--warn)}
 .meta{color:var(--mut);font-size:12px;margin-bottom:10px}
 .g{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
 .c{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:6px}
 .c img{width:100%;border-radius:4px;display:block}
 .noimg{height:90px;display:grid;place-items:center;color:var(--mut)}
 .m{color:var(--mut);font-size:11px;margin:4px 0}
 .s{font-size:11px;max-height:74px;overflow:auto}
 .sub{color:var(--mut);max-width:80ch;margin-bottom:18px}"""
    brand_title = getattr(cfg, "ADVERTISER_NAME", "Maxis")
    country_title = getattr(cfg, "COUNTRY", "Malaysia")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cluster Review &mdash; {html.escape(brand_title)} ({html.escape(country_title)})</title><style>{css}</style></head><body>
<h1>Cluster review &mdash; {html.escape(brand_title)} ({html.escape(country_title)})</h1>
<div class="sub">HDBSCAN over combined vectors (track=<b>{TRACK}</b>,
visual weight <b>{W_VIS}</b>, min_cluster_size <b>{MIN_CLUSTER_SIZE}</b>).
Sorted by spend; noise section last and outlined. Up to 24 thumbnails shown per
cluster. Each thumbnail is one phash-distinct render, not one raw row.</div>
{''.join(parts)}</body></html>""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-labels", action="store_true")
    args = ap.parse_args()

    ids, vis, txt = load()
    mat = l2(np.hstack([vis * W_VIS, txt * (1.0 - W_VIS)]))
    print(f"Clustering {len(ids)} representatives  "
          f"(track={TRACK}, w_vis={W_VIS}, mcs={MIN_CLUSTER_SIZE}, "
          f"min_samples={MIN_SAMPLES})")

    print(f"  reducing {mat.shape[1]}-d -> {UMAP_COMPONENTS}-d via manifold reduction "
          f"(cosine metric, seed={UMAP_SEED})")
    reduced = reduce_manifold(mat, n_components=UMAP_COMPONENTS, n_neighbors=15, min_dist=0.0, seed=UMAP_SEED)

    print("  computing 2D projection for scatter plot")
    reduced_2d = reduce_manifold(mat, n_components=2, n_neighbors=15, min_dist=0.1, seed=UMAP_SEED)

    pred = HDBSCAN(metric="euclidean",
                   min_cluster_size=MIN_CLUSTER_SIZE,
                   min_samples=MIN_SAMPLES,
                   cluster_selection_method="eom").fit_predict(reduced)
    n_before = len(set(p for p in pred if p != -1))
    # Merge is measured on `mat` (original space), not `reduced`.
    pred = merge_by_centroid(mat, pred, MERGE_THRESHOLD)
    n_after = len(set(p for p in pred if p != -1))
    print(f"  centroid merge >={MERGE_THRESHOLD}: {n_before} -> {n_after} clusters")
    rep_cluster = {a: int(p) for a, p in zip(ids, pred)}
    clusters = collections.defaultdict(list)
    for a, c in rep_cluster.items():
        clusters[c].append(a)
    n_real = len([c for c in clusters if c != -1])
    print(f"  {n_real} clusters, {len(clusters.get(-1, []))} noise representatives")

    with open(os.path.join(cfg.DATA_DIR, "assets_staged.csv"), encoding="utf-8") as fh:
        all_rows = [r for r in csv.DictReader(fh) if r["stage_status"] == "OK"]
    staged = {r["asset_id"]: r for r in all_rows}
    with open(cfg.ASSETS_ENRICHED_CSV, encoding="utf-8") as fh:
        enriched = {r["asset_id"]: r for r in csv.DictReader(fh)}

    # Propagate representative cluster_id and 2D UMAP coords to every raw row sharing its phash, so
    # spend rollups cover all 787 rather than only the deduplicated set.
    rep_umap_2d = {a: reduced_2d[i] for i, a in enumerate(ids)}
    phash_cluster = {staged[a]["phash"]: c for a, c in rep_cluster.items() if a in staged}
    phash_umap_2d = {staged[a]["phash"]: rep_umap_2d[a] for a in ids if a in staged}
    for r in all_rows:
        r["cluster_id"] = phash_cluster.get(r["phash"], -1)

    labels = ({c: (f"Theme {c}" if c != -1 else "Unclustered / unrelated") for c in clusters}
              if args.no_labels else gemini_labels(clusters, enriched))

    total_spend = sum(float(r["spend"] or 0) for r in all_rows)
    rollup = {}
    for cid in clusters:
        rows = [r for r in all_rows if r["cluster_id"] == cid]
        sp = sum(float(r["spend"] or 0) for r in rows)
        rollup[cid] = {
            "spend": sp,
            "impressions": sum(float(r["impressions"] or 0) for r in rows),
            "creative_count": len(rows),
            "pct": (sp / total_spend * 100) if total_spend else 0.0,
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    ac = os.path.join(OUT_DIR, "assets_clustered.csv")
    with open(ac, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["asset_id", "cluster_id", "cluster_label", "phash", "media_type",
                    "aspect_ratio", "platform", "duration_sec", "spend", "impressions",
                    "is_representative", "core_concept_summary", "umap_x", "umap_y"])
        for r in all_rows:
            cid = r["cluster_id"]
            xy = phash_umap_2d.get(r["phash"], np.array([0.0, 0.0]))
            w.writerow([r["asset_id"], cid, labels.get(cid, ""), r["phash"],
                        r["media_type"], r["aspect_ratio"], r["platform"],
                        r["duration_sec"], r["spend"], r["impressions"],
                        int(r["asset_id"] in rep_cluster),
                        (enriched.get(r["asset_id"], {}) or {}).get("core_concept_summary", ""),
                        round(float(xy[0]), 4), round(float(xy[1]), 4)])

    ts = os.path.join(OUT_DIR, "themes_summary.csv")
    with open(ts, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "theme_label", "total_spend", "total_impressions",
                    "pct_of_total_spend", "creative_count", "distinct_renders"])
        for cid in sorted(rollup, key=lambda c: (c == -1, -rollup[c]["spend"])):
            r = rollup[cid]
            w.writerow([cid, labels.get(cid, ""), round(r["spend"], 2),
                        round(r["impressions"]), round(r["pct"], 2),
                        r["creative_count"], len(clusters[cid])])

    review = os.path.join(repo_root, "evaluation", "cluster_review.html")
    write_review(clusters, labels, staged, enriched, rollup, review)

    print(f"\nWrote {ac}")
    print(f"Wrote {ts}")
    print(f"Wrote {review}")
    print(f"\n{len(all_rows)} raw -> {len(ids)} distinct renders -> {n_real} themes "
          f"(+{len([r for r in all_rows if r['cluster_id']==-1])} noise rows)")
    top = sorted(((c, rollup[c]) for c in rollup if c != -1),
                 key=lambda x: -x[1]["spend"])[:8]
    print(f"\n{'cid':>4} {'spend':>10} {'%':>6} {'raw':>5} {'rend':>5}  label")
    for cid, r in top:
        print(f"{cid:>4} {r['spend']:>10,.0f} {r['pct']:>6.1f} "
              f"{r['creative_count']:>5} {len(clusters[cid]):>5}  {labels.get(cid,'')[:52]}")
    nz = rollup.get(-1)
    if nz:
        print(f"{-1:>4} {nz['spend']:>10,.0f} {nz['pct']:>6.1f} "
              f"{nz['creative_count']:>5} {len(clusters[-1]):>5}  NOISE")


if __name__ == "__main__":
    main()
