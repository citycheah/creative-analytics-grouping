"""Semi-automated ground truth labeling for creative clustering evaluation.

Expands the evaluation set to 100+ assets (up from 30) using AI-assisted pre-grouping.
Generates an interactive HTML review tool where human reviewers can accept, edit, or reject
suggested labels with 1 click.

Usage:
    python evaluation/auto_label.py --sample-size 120
    python evaluation/auto_label.py --output-draft evaluation/eval_labels_draft.json
"""

import argparse
import base64
import collections
import csv
import io
import json
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import config as cfg
from common.logging import get_logger

logger = get_logger("auto_label", stage="evaluation")

THUMB_W = 220


def thumb_b64(path: str) -> str:
    """Encodes image to thumbnail base64 data URI."""
    if not path or not os.path.exists(path):
        return ""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w = THUMB_W
            h = max(1, int(im.height * (w / im.width)))
            im = im.resize((w, h), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def propose_groups_from_enriched(reps: List[dict]) -> Dict[str, str]:
    """Generates initial group letter suggestions based on campaign identity and concepts."""
    # Group by normalized campaign name or concept similarities
    groups = collections.defaultdict(list)
    for r in reps:
        camp = (r.get("creative_campaign_name") or "").strip().lower()
        if not camp:
            # Fallback to first 4 words of concept summary
            summary = (r.get("core_concept_summary") or "").strip().lower()
            camp = " ".join(summary.split()[:4]) if summary else "unrelated"
        groups[camp].append(r["asset_id"])

    # Map groups to letters A, B, C... (or AA, BB if >26)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    draft_labels = {}
    g_idx = 0
    for camp, aids in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(aids) > 1:
            lbl = alphabet[g_idx % 26] if g_idx < 26 else f"G{g_idx}"
            g_idx += 1
            for aid in aids:
                draft_labels[aid] = lbl
        else:
            # Singleton / noise candidate
            for aid in aids:
                draft_labels[aid] = "noise"

    return draft_labels


def build_expanded_eval_sheet(sample_size: int = 120, output_html: Optional[str] = None):
    staged_path = cfg.ASSETS_STAGED_CSV
    if not os.path.exists(staged_path):
        logger.error(f"Missing staged CSV: {staged_path}")
        return

    with open(staged_path, "r", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("stage_status") == "OK"]

    # Phash representatives
    by_hash = collections.defaultdict(list)
    for r in rows:
        by_hash[r["phash"]].append(r)
    reps = [max(v, key=lambda r: float(r.get("spend") or 0)) for v in by_hash.values()]

    # Join enriched data for concept context
    enriched_by_id = {}
    if os.path.exists(cfg.ASSETS_ENRICHED_CSV):
        with open(cfg.ASSETS_ENRICHED_CSV, "r", encoding="utf-8") as fh:
            for er in csv.DictReader(fh):
                enriched_by_id[er["asset_id"]] = er

    for r in reps:
        if r["asset_id"] in enriched_by_id:
            r["core_concept_summary"] = enriched_by_id[r["asset_id"]].get("core_concept_summary", "")

    # Sample stratified
    rng = random.Random(42)
    rng.shuffle(reps)
    sample = reps[:sample_size]

    # Propose AI suggestions
    draft_labels = propose_groups_from_enriched(sample)

    # Save draft JSON
    draft_json_path = os.path.join(cfg.EVAL_DIR, "eval_labels_draft.json")
    with open(draft_json_path, "w", encoding="utf-8") as fh:
        json.dump(draft_labels, fh, indent=2)
    logger.info(f"Saved {len(draft_labels)} AI proposed labels to {draft_json_path}")

    # Build HTML Cards
    cards = []
    for i, r in enumerate(sample, 1):
        src = r.get("frame_path") if (r.get("media_type") == "video" and r.get("frame_path")) else r.get("local_path")
        b64 = thumb_b64(src)
        img = f'<img src="data:image/jpeg;base64,{b64}" alt="creative {i}">' if b64 else '<div class="noimg">no preview</div>'
        
        copy_lines = [r.get("creative_campaign_name") or "", r.get("social_headline_text") or "", r.get("core_concept_summary") or ""]
        copy_html = "<br>".join(l[:180].replace("<", "&lt;").replace(">", "&gt;") for l in copy_lines if l.strip()) or "<i>no text</i>"
        
        suggested = draft_labels.get(r["asset_id"], "")
        cards.append(f"""
        <div class="card" id="card_{r['asset_id']}">
          <div class="header">
            <span class="idx">#{i}</span>
            <span class="badge">AI Suggestion: <b>{suggested}</b></span>
          </div>
          <div class="thumb">{img}</div>
          <div class="meta">{r.get('media_type')} &middot; {r.get('aspect_ratio')}</div>
          <div class="copy">{copy_html}</div>
          <div class="input-row">
            <input class="grp" data-asset="{r['asset_id']}" value="{suggested}" placeholder="Group letter">
            <button type="button" class="btn-sm" onclick="setNoise('{r['asset_id']}')">Noise</button>
          </div>
        </div>""")

    html_file = output_html or os.path.join(cfg.EVAL_DIR, "eval_sheet_expanded.html")
    html_content = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Semi-Automated Evaluation Labeling (Expanded {len(sample)} Set)</title>
<style>
  :root {{ color-scheme: light dark; --bg:#0f172a; --fg:#f8fafc; --card:#1e293b; --line:#334155; --pri:#38bdf8; }}
  body {{ background:var(--bg); color:var(--fg); font-family:system-ui,-apple-system,sans-serif; margin:0; padding:24px; }}
  h1 {{ font-size:22px; margin:0 0 6px; }}
  .sub {{ color:#94a3b8; margin-bottom:16px; font-size:14px; max-width:80ch; }}
  .bar {{ position:sticky; top:0; z-index:10; background:var(--bg); padding:12px 0; border-bottom:1px solid var(--line);
          margin-bottom:20px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
  button {{ background:#2563eb; color:#fff; border:none; padding:8px 16px; border-radius:6px; font-weight:600; cursor:pointer; }}
  button:hover {{ background:#1d4ed8; }}
  .btn-sm {{ background:#475569; padding:4px 8px; font-size:12px; }}
  .btn-sm:hover {{ background:#64748b; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:16px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; display:flex; flex-direction:column; }}
  .header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
  .idx {{ font-weight:700; color:#94a3b8; font-size:12px; }}
  .badge {{ background:#0284c7; color:#fff; font-size:11px; padding:2px 6px; border-radius:4px; }}
  .thumb img {{ width:100%; height:180px; object-fit:contain; background:#020617; border-radius:6px; }}
  .noimg {{ height:180px; display:grid; place-items:center; color:#64748b; border:1px dashed var(--line); border-radius:6px; }}
  .meta {{ color:#94a3b8; font-size:12px; margin:8px 0 4px; }}
  .copy {{ font-size:12px; color:#cbd5e1; max-height:72px; overflow-y:auto; margin-bottom:10px; line-height:1.4; }}
  .input-row {{ display:flex; gap:6px; margin-top:auto; }}
  input {{ flex:1; padding:6px 10px; border:1px solid var(--line); border-radius:6px; background:#0f172a; color:#fff; font-weight:bold; }}
  textarea {{ width:100%; height:140px; margin-top:16px; background:#020617; color:#38bdf8; border:1px solid var(--line); border-radius:6px; padding:10px; font-family:monospace; font-size:12px; }}
</style>
</head>
<body>
<h1>Semi-Automated Evaluation Labeling ({len(sample)} Assets)</h1>
<div class="sub">
  Review the proposed AI group suggestions. Creative assets sharing the exact same creative concept/offer should have matching group letters (e.g. A, B, C). Unrelated singletons should be marked <b>noise</b>.
</div>
<div class="bar">
  <button onclick="exportJSON()">Export Finalized JSON</button>
  <button onclick="navigator.clipboard.writeText(document.getElementById('out').value)">Copy JSON</button>
  <span id="counter" style="color:#94a3b8; font-size:13px; font-weight:bold;"></span>
</div>
<div class="grid">{''.join(cards)}</div>
<textarea id="out" placeholder="Click 'Export Finalized JSON' above to view output."></textarea>
<script>
  function setNoise(aid) {{
    const el = document.querySelector(`input[data-asset="${{aid}}"]`);
    if (el) {{ el.value = 'noise'; updateCount(); }}
  }}
  function updateCount() {{
    const inputs = document.querySelectorAll('.grp');
    const filled = [...inputs].filter(i => i.value.trim()).length;
    document.getElementById('counter').innerText = `${{filled}} / ${{inputs.length}} Reviewed`;
  }}
  function exportJSON() {{
    const out = {{}};
    document.querySelectorAll('.grp').forEach(i => {{
      const val = i.value.trim();
      if (val) out[i.getAttribute('data-asset')] = val;
    }});
    document.getElementById('out').value = JSON.stringify(out, null, 2);
  }}
  document.querySelectorAll('.grp').forEach(i => i.addEventListener('input', updateCount));
  updateCount();
</script>
</body>
</html>"""

    with open(html_file, "w", encoding="utf-8") as fh:
        fh.write(html_content)
    logger.info(f"Generated interactive review sheet: {html_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate AI-assisted expanded evaluation set")
    parser.add_argument("--sample-size", type=int, default=120, help="Number of assets to sample")
    parser.add_argument("--output-html", type=str, default=None, help="HTML review tool path")
    args = parser.parse_args()

    build_expanded_eval_sheet(sample_size=args.sample_size, output_html=args.output_html)


if __name__ == "__main__":
    main()
