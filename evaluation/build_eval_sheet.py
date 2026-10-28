"""Generate a blind tagging sheet for the clustering evaluation set.

The spec requires a manually tagged eval set built BEFORE any clustering output
is seen, so the accuracy figure is independent of the thing it measures. This
script produces a self-contained HTML page: thumbnails + ad copy, no cluster
IDs, no phash, no embedding output.

Sampling: 30 assets drawn from the 340 phash-distinct representatives,
stratified across `social_campaign_text` families so the sample plausibly
contains both genuine multi-format variants and unrelated ads. Family identity
is NOT shown in the sheet and the display order is shuffled, so the tagger sees
only the creative itself.

Usage:
    python evaluation/build_eval_sheet.py
    -> evaluation/eval_sheet.html   (open in a browser, tag, export JSON)
    -> evaluation/eval_sample.csv   (the sampled asset list)
"""

import base64
import collections
import csv
import io
import json
import os
import random
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from PIL import Image

import ingestion.config as cfg

SEED = 20260813
N_ASSETS = 30
N_FAMILIES = 9
THUMB_W = 240

HERE = os.path.dirname(os.path.abspath(__file__))


def thumb_b64(path: str) -> str:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w = THUMB_W
            h = max(1, int(im.height * (w / im.width)))
            im = im.resize((w, h), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=72)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def main():
    rng = random.Random(SEED)

    with open(os.path.join(cfg.DATA_DIR, "assets_staged.csv"), encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["stage_status"] == "OK"]

    # Collapse to phash-distinct representatives (highest spend wins).
    by_hash = collections.defaultdict(list)
    for r in rows:
        by_hash[r["phash"]].append(r)
    reps = [max(v, key=lambda r: float(r["spend"] or 0)) for v in by_hash.values()]

    # Stratify by ad-copy family so the sample has structure worth finding.
    fams = collections.defaultdict(list)
    for r in reps:
        key = (r.get("social_campaign_text") or r.get("creative_campaign_name") or "").strip()
        fams[key].append(r)
    ranked = sorted(fams.values(), key=len, reverse=True)
    candidates = [f for f in ranked if len(f) >= 2][:N_FAMILIES]

    sample, seen = [], set()
    per = max(2, N_ASSETS // max(len(candidates), 1))
    for fam in candidates:
        for r in rng.sample(fam, min(per, len(fam))):
            if r["asset_id"] not in seen:
                seen.add(r["asset_id"])
                sample.append(r)
    # Top up with singletons so unrelated "noise" is represented too.
    singles = [r for f in ranked if len(f) == 1 for r in f]
    rng.shuffle(singles)
    for r in singles:
        if len(sample) >= N_ASSETS:
            break
        if r["asset_id"] not in seen:
            seen.add(r["asset_id"])
            sample.append(r)
    sample = sample[:N_ASSETS]
    rng.shuffle(sample)  # order must not encode family

    with open(os.path.join(HERE, "eval_sample.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sample[0].keys()))
        w.writeheader()
        w.writerows(sample)

    cards = []
    for i, r in enumerate(sample, 1):
        src = r["frame_path"] if r["media_type"] == "video" and r["frame_path"] else r["local_path"]
        b64 = thumb_b64(src)
        img = (f'<img src="data:image/jpeg;base64,{b64}" alt="creative {i}">'
               if b64 else '<div class="noimg">no preview</div>')
        copy_bits = [r.get("creative_campaign_name"), r.get("social_headline_text"),
                     r.get("social_description"), r.get("social_campaign_text")]
        seen_bits, copy_lines = set(), []
        for b in copy_bits:
            b = (b or "").strip()
            if b and b not in seen_bits:
                seen_bits.add(b)
                copy_lines.append(b)
        copy_html = "<br>".join(
            l.replace("&", "&amp;").replace("<", "&lt;")[:220] for l in copy_lines
        ) or "<i>no ad copy</i>"
        dur = f" &middot; {float(r['duration_sec']):.0f}s" if r.get("duration_sec") else ""
        cards.append(f"""
    <div class="card">
      <div class="idx">#{i}</div>
      <div class="thumb">{img}</div>
      <div class="meta">{r['media_type']} &middot; {r['aspect_ratio']}{dur}</div>
      <div class="copy">{copy_html}</div>
      <label>group
        <input class="grp" data-asset="{r['asset_id']}" placeholder="A / B / C&hellip; or 'noise'">
      </label>
    </div>""")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Blind Eval Tagging Sheet</title>
<style>
 :root {{ color-scheme: light dark; --bg:#fff; --fg:#111; --mut:#666; --line:#e3e3e3; --card:#fafafa; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --bg:#16181c; --fg:#eceff4; --mut:#9aa3ad; --line:#2c313a; --card:#1e2127; }}
 }}
 body {{ background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,sans-serif; margin:0; padding:24px; }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 .sub {{ color:var(--mut); margin-bottom:20px; max-width:70ch; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px; }}
 .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }}
 .idx {{ font-weight:700; color:var(--mut); font-size:12px; }}
 .thumb img {{ width:100%; border-radius:6px; display:block; }}
 .noimg {{ height:120px; display:grid; place-items:center; color:var(--mut); border:1px dashed var(--line); border-radius:6px; }}
 .meta {{ color:var(--mut); font-size:12px; margin:8px 0 6px; }}
 .copy {{ font-size:12px; max-height:96px; overflow:auto; margin-bottom:10px; }}
 label {{ font-size:12px; color:var(--mut); display:block; }}
 input {{ width:100%; padding:6px 8px; margin-top:4px; border:1px solid var(--line);
          border-radius:6px; background:var(--bg); color:var(--fg); font-size:14px; }}
 .bar {{ position:sticky; top:0; background:var(--bg); padding:12px 0; border-bottom:1px solid var(--line);
         margin-bottom:16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
 button {{ padding:8px 14px; border-radius:6px; border:1px solid var(--line); cursor:pointer;
           background:var(--card); color:var(--fg); font-size:14px; }}
 textarea {{ width:100%; height:140px; margin-top:12px; font-family:ui-monospace,monospace; font-size:12px;
             background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:8px; }}
</style></head><body>
<h1>Blind eval tagging sheet</h1>
<div class="sub">Give every creative that shows <b>the same core ad concept</b> the same group
letter &mdash; regardless of aspect ratio, crop or platform. Give genuinely unrelated
one-offs their own letter, or type <b>noise</b>. Ignore differences in size and framing.
No clustering output has been computed yet, so nothing here reflects a model prediction.</div>
<div class="bar">
  <button onclick="exportJSON()">Export JSON</button>
  <button onclick="navigator.clipboard.writeText(document.getElementById('out').value)">Copy</button>
  <span id="count" class="sub" style="margin:0"></span>
</div>
<div class="grid">{''.join(cards)}</div>
<textarea id="out" placeholder="Export JSON, then save it as evaluation/eval_labels.json"></textarea>
<script>
 const ins = () => [...document.querySelectorAll('.grp')];
 function tally() {{
   const n = ins().filter(i => i.value.trim()).length;
   document.getElementById('count').textContent = n + ' / ' + ins().length + ' tagged';
 }}
 ins().forEach(i => i.addEventListener('input', tally));
 tally();
 function exportJSON() {{
   const out = {{}};
   ins().forEach(i => {{ const v = i.value.trim(); if (v) out[i.dataset.asset] = v; }});
   document.getElementById('out').value = JSON.stringify(out, null, 2);
 }}
</script></body></html>"""

    out_path = os.path.join(HERE, "eval_sheet.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"Sampled {len(sample)} assets from {len(reps)} representatives")
    print(f"  families used: {len(candidates)}  (>=2 members)")
    print(f"  media mix: {dict(collections.Counter(r['media_type'] for r in sample))}")
    print(f"  ratios   : {dict(collections.Counter(r['aspect_ratio'] for r in sample))}")
    print(f"\nOpen: {out_path}")
    print("Tag it, click Export JSON, save as evaluation/eval_labels.json")


if __name__ == "__main__":
    main()
