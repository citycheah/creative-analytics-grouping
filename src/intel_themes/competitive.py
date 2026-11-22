"""Dynamic joint multi-brand theme discovery, centroid-matched stability, and creative fatigue."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from intel_core.schemas import DimCreativeTheme, canonicalize_brand_id
from intel_core.storage import (
    read_parquet_table,
    register_live_dissection_view,
    write_parquet_table,
)

from intel_themes.vectors import (
    cluster_hdbscan,
    combine,
    compute_centroids,
    embed_text_blocks,
    embed_visual_blocks,
    l2,
    merge_by_centroid,
    reduce_manifold,
)


def compute_hhi(spend_shares: list[float]) -> float:
    """Computes normalized Herfindahl-Hirschman Index (HHI = sum(s_i^2)) in [0, 1]."""
    arr = np.asarray(spend_shares, dtype=float)
    if arr.size == 0 or float(arr.sum()) <= 0.0:
        return 0.0
    proportions = arr / arr.sum()
    return float(np.sum(proportions**2))


def compute_jaccard_overlap(themes_a: set[str], themes_b: set[str]) -> float:
    """Computes Jaccard theme overlap between two brands."""
    if not themes_a and not themes_b:
        return 0.0
    union = themes_a | themes_b
    return float(len(themes_a & themes_b) / len(union)) if union else 0.0


def compute_theme_sov(brand_spends: dict[str, float]) -> dict[str, float]:
    """Computes Share of Voice (SOV %) per brand within a dynamic theme."""
    total = sum(max(0.0, float(v)) for v in brand_spends.values())
    if total <= 0.0:
        return {k: 0.0 for k in brand_spends}
    return {k: round(100.0 * max(0.0, float(v)) / total, 2) for k, v in brand_spends.items()}


def _derive_dynamic_theme_label(concepts: list[str], angles: list[str]) -> str:
    """Synthesizes a concise proposition label dynamically from cluster concepts & angles."""
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "bmw",
        "mercedes",
        "benz",
        "audi",
        "porsche",
        "creative",
        "highlight",
        "highlights",
        "automotive",
        "seasonal",
        "showcase",
    }
    angle_counts = Counter(a.strip() for a in angles if a and str(a).strip())
    top_angle = angle_counts.most_common(1)[0][0] if angle_counts else ""

    tok_counts: Counter[str] = Counter()
    for txt in concepts:
        for tok in re.findall(r"[A-Za-z0-9%]+", str(txt)):
            low = tok.lower()
            if len(low) >= 2 and low not in stopwords:
                tok_counts[tok] += 1
    top_tokens = [w for w, _ in tok_counts.most_common(3)]
    phrase = " ".join(top_tokens).strip()
    if top_angle and phrase:
        return f"{top_angle}: {phrase}"[:75]
    return (top_angle or phrase or "Dynamic Automotive Proposition")[:75]


def match_prior_themes(
    new_centroids: dict[int, np.ndarray],
    prior_centroids: dict[str, dict[str, Any]] | None,
    *,
    cluster_concepts: dict[int, list[str]],
    cluster_angles: dict[int, list[str]],
    sim_threshold: float = 0.85,
) -> dict[int, dict[str, Any]]:
    """Matches new cluster centroids to prior runs by cosine >= `sim_threshold` to preserve `theme_id`."""
    prior_items: list[tuple[str, str, np.ndarray]] = []
    for tid, meta in (prior_centroids or {}).items():
        vec = np.asarray(meta.get("centroid", []), dtype=np.float64)
        if vec.size > 0:
            prior_items.append((str(tid), str(meta.get("theme_label", tid)), l2(vec)))

    # Greedy descending similarity assignment enforces 1-to-1 stable ID matching across weekly runs
    candidates: list[tuple[float, int, str, str]] = []
    for cid, cvec in new_centroids.items():
        c_norm = l2(cvec)
        for tid, tlabel, pvec in prior_items:
            if pvec.shape == c_norm.shape:
                sim = float(np.dot(c_norm, pvec))
                if sim >= sim_threshold:
                    candidates.append((sim, cid, tid, tlabel))
    candidates.sort(key=lambda x: x[0], reverse=True)

    assigned: dict[int, dict[str, Any]] = {}
    used_tids: set[str] = set()
    for sim, cid, tid, tlabel in candidates:
        if cid in assigned or tid in used_tids:
            continue
        assigned[cid] = {
            "theme_id": tid,
            "theme_label": tlabel,
            "matched_prior_sim": round(sim, 4),
            "centroid": l2(new_centroids[cid]).tolist(),
        }
        used_tids.add(tid)

    for cid, cvec in sorted(new_centroids.items()):
        if cid in assigned:
            continue
        label = _derive_dynamic_theme_label(
            cluster_concepts.get(cid, []),
            cluster_angles.get(cid, []),
        )
        slug = hashlib.sha1(f"{label}|{np.round(cvec[:16], 3).tolist()}".encode()).hexdigest()[:8]
        tid = f"THM_{slug.upper()}"
        assigned[cid] = {
            "theme_id": tid,
            "theme_label": label,
            "matched_prior_sim": 1.0,
            "centroid": l2(cvec).tolist(),
        }
    return assigned


def compute_creative_fatigue(
    delivery_df: pd.DataFrame,
    themes_df: pd.DataFrame,
    *,
    fatigue_age_weeks: int = 8,
    decay_ratio_threshold: float = 0.55,
) -> pd.DataFrame:
    """Computes asset & theme creative lifespan, impression decay, and fatigue flags."""
    if delivery_df.empty:
        return pd.DataFrame(
            columns=[
                "brand_id",
                "theme_id",
                "media_hash_key",
                "active_weeks",
                "avg_creative_age_weeks",
                "peak_weekly_impressions",
                "recent_weekly_impressions",
                "impression_decay_ratio",
                "is_fatigued",
            ]
        )

    work = delivery_df.copy()
    work["week_start"] = pd.to_datetime(work["week_start"])
    if "brand_id" not in work.columns:
        work["brand_id"] = work["candidate_brand_ids"].map(
            lambda x: str(x[0]) if (isinstance(x, (list, tuple)) or hasattr(x, "tolist")) and len(x) > 0 else "UNKNOWN"
        )
    impr_col = "impressions" if "impressions" in work.columns else "raw_impressions"

    if not themes_df.empty:
        tmap = themes_df[["media_hash_key", "theme_id"]].drop_duplicates("media_hash_key")
        work = work.merge(tmap, on="media_hash_key", how="left")
    else:
        work["theme_id"] = "UNCLUSTERED"
    work["theme_id"] = work["theme_id"].fillna("UNCLUSTERED")

    weekly_asset = (
        work.groupby(["brand_id", "theme_id", "media_hash_key", "week_start"], as_index=False)[
            impr_col
        ]
        .sum()
        .sort_values(["brand_id", "media_hash_key", "week_start"])
    )

    rows: list[dict[str, Any]] = []
    for (brand_id, theme_id, mkey), grp in weekly_asset.groupby(
        ["brand_id", "theme_id", "media_hash_key"]
    ):
        n_weeks = int(grp["week_start"].nunique())
        first_wk = grp["week_start"].min()
        last_wk = grp["week_start"].max()
        span_weeks = max(1, int((last_wk - first_wk).days // 7) + 1)
        peak_impr = float(grp[impr_col].max())
        recent_impr = float(grp.tail(2)[impr_col].mean())
        decay_ratio = round(recent_impr / peak_impr, 4) if peak_impr > 0.0 else 1.0
        is_fatigued = bool(span_weeks >= fatigue_age_weeks and decay_ratio <= decay_ratio_threshold)
        rows.append(
            {
                "brand_id": str(brand_id),
                "theme_id": str(theme_id),
                "media_hash_key": str(mkey),
                "active_weeks": n_weeks,
                "avg_creative_age_weeks": float(span_weeks),
                "peak_weekly_impressions": round(peak_impr, 2),
                "recent_weekly_impressions": round(recent_impr, 2),
                "impression_decay_ratio": decay_ratio,
                "is_fatigued": is_fatigued,
            }
        )
    return pd.DataFrame(rows)


def run_dynamic_competitive_pipeline(
    lakehouse_dir: str | Path,
    *,
    run_id: str = "run_latest",
    visual_weight: float = 0.30,
    umap_components: int = 30,
    merge_threshold: float = 0.94,
    prior_sim_threshold: float = 0.85,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    prior_centroids_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Runs dynamic joint multi-brand multimodal clustering & competitive telemetry."""
    root = Path(lakehouse_dir)
    silver_dir = root / "silver"
    bronze_dir = root / "bronze"
    gold_dir = root / "gold"
    overrides_dir = root / "overrides"
    out_dir = Path(output_dir) if output_dir else (gold_dir / "themes_analytics")

    assets_path = silver_dir / "dim_creative_asset"
    diss_path = silver_dir / "dim_creative_dissection"
    if not diss_path.exists():
        diss_path = bronze_dir / "seed_creative_dissection"

    con = duckdb.connect(":memory:")
    try:
        register_live_dissection_view(
            con,
            assets_path=assets_path,
            dissection_path=diss_path,
            overrides_dir=overrides_dir if overrides_dir.exists() else None,
        )
        live_df = con.execute("SELECT * FROM vw_live_creative_dissection").df()
    finally:
        con.close()

    # Exclude rejected off-brand assets before joint theme discovery
    valid_df = live_df[live_df["belongs_to_brand"] != False].copy()  # noqa: E712
    valid_df = valid_df.drop_duplicates(subset=["media_hash_key"]).reset_index(drop=True)
    if valid_df.empty:
        raise ValueError("No verified/non-rejected creatives available for theme clustering.")

    # Resolve primary brand for spend/SOV attribution
    def _resolve_brand(row: pd.Series) -> str:
        mb = row.get("matched_brand_id")
        if pd.notna(mb) and str(mb).strip():
            first = str(mb).split(",")[0].strip()
            return canonicalize_brand_id(first) or first
        raw = str(row.get("brand_id_raw") or "")
        return canonicalize_brand_id(raw) or raw

    valid_df["canonical_brand"] = valid_df.apply(_resolve_brand, axis=1)

    def _to_str_list(val: Any) -> list[str]:
        if isinstance(val, (list, tuple)):
            return [str(v) for v in val]
        if hasattr(val, "tolist"):
            return [str(v) for v in val.tolist()]
        return []

    concept_texts = [
        f"{r.get('core_concept_summary') or ''} | {r.get('messaging_angle') or ''} | "
        f"{r.get('hook_type') or ''} | {' '.join(_to_str_list(r.get('visual_style_tags')))}"
        for _, r in valid_df.iterrows()
    ]
    tags_fallback = [
        " ".join(_to_str_list(r.get("visual_style_tags")) or [str(r.get("aspect_ratio") or "creative")])
        for _, r in valid_df.iterrows()
    ]
    media_paths = [str(r.get("media_uri") or "") for _, r in valid_df.iterrows()]

    txt_mat = embed_text_blocks(concept_texts, dim=384)
    vis_mat = embed_visual_blocks(media_paths, tags_fallback=tags_fallback, dim=1408)
    fused = combine(vis_mat, txt_mat, visual_weight=visual_weight)

    reduced = reduce_manifold(
        fused,
        n_components=umap_components,
        n_neighbors=min(15, max(2, len(valid_df) - 1)),
        min_dist=0.0,
        seed=42,
    )
    umap_2d = reduce_manifold(
        fused,
        n_components=2,
        n_neighbors=min(10, max(2, len(valid_df) - 1)),
        min_dist=0.1,
        seed=42,
    )
    raw_pred = cluster_hdbscan(
        reduced,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        method="eom",
    )
    merged_pred = merge_by_centroid(fused, raw_pred, thresh=merge_threshold)

    # Re-assign any singleton noise (-1) by nearest cluster centroid or keep isolated singleton cluster
    centroids = compute_centroids(fused, merged_pred)
    final_pred = merged_pred.copy()
    next_cid = (max(centroids.keys()) + 1) if centroids else 0
    for idx, p in enumerate(final_pred):
        if p == -1:
            if centroids:
                best_c, best_s = max(
                    ((c, float(np.dot(fused[idx], cvec))) for c, cvec in centroids.items()),
                    key=lambda kv: kv[1],
                )
                if best_s >= 0.55:
                    final_pred[idx] = best_c
                    continue
            final_pred[idx] = next_cid
            next_cid += 1
    centroids = compute_centroids(fused, final_pred)

    # Load prior centroids for theme_id stability across runs (>= 0.85 cosine)
    centroids_file = Path(prior_centroids_path) if prior_centroids_path else (out_dir / "theme_centroids.json")
    prior_centroids: dict[str, Any] = {}
    if centroids_file.exists():
        prior_centroids = json.loads(centroids_file.read_text(encoding="utf-8"))

    cluster_concepts: dict[int, list[str]] = defaultdict(list)
    cluster_angles: dict[int, list[str]] = defaultdict(list)
    for idx, cid in enumerate(final_pred):
        cluster_concepts[int(cid)].append(str(valid_df.iloc[idx].get("core_concept_summary") or ""))
        cluster_angles[int(cid)].append(str(valid_df.iloc[idx].get("messaging_angle") or ""))

    theme_meta = match_prior_themes(
        centroids,
        prior_centroids,
        cluster_concepts=cluster_concepts,
        cluster_angles=cluster_angles,
        sim_threshold=prior_sim_threshold,
    )

    # Compute spend per media_hash_key & brand from fct_ad_delivery_weekly or silver_ad_delivery_raw
    fct_path = gold_dir / "fct_ad_delivery_weekly"
    raw_path = silver_dir / "silver_ad_delivery_raw"
    delivery_df = read_parquet_table(fct_path) if fct_path.exists() else read_parquet_table(raw_path)
    spend_col = "spend_usd" if "spend_usd" in delivery_df.columns else "raw_spend_usd"
    brand_col = "brand_id" if "brand_id" in delivery_df.columns else None

    if brand_col is None:
        deliv_work = delivery_df.copy()
        deliv_work["brand_id"] = deliv_work["candidate_brand_ids"].map(
            lambda x: str(x[0]) if (isinstance(x, (list, tuple)) or hasattr(x, "tolist")) and len(x) > 0 else "UNKNOWN"
        )
    else:
        deliv_work = delivery_df

    mkey_brand_spend: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in deliv_work.groupby(["media_hash_key", "brand_id"], as_index=False)[spend_col].sum().iterrows():
        mkey_brand_spend[str(row["media_hash_key"])][str(row["brand_id"])] += float(row[spend_col])

    # Classify Battleground vs Whitespace using active spend share >= 5%
    theme_brand_spend: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    mkey_to_cid: dict[str, int] = {}
    for idx, r in valid_df.iterrows():
        cid = int(final_pred[idx])
        mkey = str(r["media_hash_key"])
        mkey_to_cid[mkey] = cid
        tid = theme_meta[cid]["theme_id"]
        sp_map = mkey_brand_spend.get(mkey)
        if sp_map:
            for b, sp in sp_map.items():
                theme_brand_spend[tid][b] += sp
        else:
            theme_brand_spend[tid][str(r["canonical_brand"])] += 1.0

    theme_scope_map: dict[str, str] = {}
    theme_sov_rows: list[dict[str, Any]] = []
    brand_theme_spends: dict[str, list[float]] = defaultdict(list)
    brand_active_themes: dict[str, set[str]] = defaultdict(set)

    for _cid, meta in sorted(theme_meta.items(), key=lambda kv: kv[1]["theme_id"]):
        tid = meta["theme_id"]
        sp_dict = dict(theme_brand_spend.get(tid, {}))
        total_t = sum(sp_dict.values())
        sov = compute_theme_sov(sp_dict)
        active_brands = [b for b, pct in sov.items() if pct >= 5.0]
        scope = "shared_battleground" if len(active_brands) >= 2 else "brand_exclusive_whitespace"
        theme_scope_map[tid] = scope
        for b, sp in sp_dict.items():
            if sp > 0:
                brand_theme_spends[b].append(sp)
            if sov.get(b, 0.0) >= 5.0:
                brand_active_themes[b].add(tid)
        theme_sov_rows.append(
            {
                "theme_id": tid,
                "theme_label": meta["theme_label"],
                "theme_scope": scope,
                "total_spend_usd": round(total_t, 2),
                "active_brands": sorted(active_brands),
                "brand_sov_pct": sov,
                "matched_prior_sim": meta["matched_prior_sim"],
            }
        )

    dim_theme_rows: list[dict[str, Any]] = []
    for idx, r in valid_df.iterrows():
        cid = int(final_pred[idx])
        meta = theme_meta[cid]
        tid = meta["theme_id"]
        cvec = centroids[cid]
        sim = float(np.dot(fused[idx], cvec))
        sim_clamped = max(-1.0, min(1.0, sim))
        rec = DimCreativeTheme(
            media_hash_key=str(r["media_hash_key"]),
            run_id=run_id,
            is_active_run=True,
            theme_id=tid,
            theme_label=meta["theme_label"],
            theme_scope=theme_scope_map[tid],  # type: ignore[arg-type]
            umap_x=round(float(umap_2d[idx, 0]), 5),
            umap_y=round(float(umap_2d[idx, 1]), 5),
            centroid_similarity=round(sim_clamped, 5),
        )
        dim_theme_rows.append(rec.model_dump(mode="python"))

    dim_theme_df = pd.DataFrame(dim_theme_rows)
    fatigue_df = compute_creative_fatigue(deliv_work, dim_theme_df)

    brand_hhi = {b: round(compute_hhi(spends), 4) for b, spends in sorted(brand_theme_spends.items())}
    brands_sorted = sorted(brand_active_themes.keys())
    jaccard_matrix = {
        b1: {b2: round(compute_jaccard_overlap(brand_active_themes[b1], brand_active_themes[b2]), 4) for b2 in brands_sorted}
        for b1 in brands_sorted
    }

    centroids_payload = {
        meta["theme_id"]: {"theme_label": meta["theme_label"], "centroid": meta["centroid"]}
        for meta in theme_meta.values()
    }

    metrics_payload = {
        "run_id": run_id,
        "assets_clustered": len(dim_theme_df),
        "themes_discovered": len(theme_meta),
        "battleground_count": sum(1 for v in theme_scope_map.values() if v == "shared_battleground"),
        "whitespace_count": sum(1 for v in theme_scope_map.values() if v == "brand_exclusive_whitespace"),
        "brand_hhi": brand_hhi,
        "jaccard_overlap": jaccard_matrix,
        "themes": theme_sov_rows,
        "fatigued_assets": int(fatigue_df["is_fatigued"].sum()) if not fatigue_df.empty else 0,
    }

    if persist:
        write_parquet_table(
            dim_theme_df,
            silver_dir / "dim_creative_themes",
            table_name="dim_creative_themes",
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "theme_centroids.json").write_text(
            json.dumps(centroids_payload, indent=2), encoding="utf-8"
        )
        (out_dir / "competitive_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
        write_parquet_table(fatigue_df, out_dir / "creative_fatigue")

    return {
        **metrics_payload,
        "dim_themes_df": dim_theme_df,
        "fatigue_df": fatigue_df,
        "centroids_path": str(out_dir / "theme_centroids.json"),
        "dim_themes_path": str(silver_dir / "dim_creative_themes"),
    }
