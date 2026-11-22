"""Incremental Lakehouse Creative Dissection using `intel_core` Vision + FinOpsGuard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from intel_core.gemini import FinOpsGuard, GeminiMediaClient
from intel_core.media import MediaStager, compute_format_fit_score
from intel_core.schemas import DimCreativeAsset, DimCreativeDissection, canonicalize_brand_id
from intel_core.storage import read_parquet_table, write_parquet_table


def select_unseen_assets(
    silver_dir: str | Path,
    *,
    prompt_version: str = "creative_dissection_v2",
) -> pd.DataFrame:
    """Runs a DuckDB SQL anti-join to return assets not yet dissected for `prompt_version`."""
    silver_root = Path(silver_dir)
    assets_path = silver_root / "dim_creative_asset"
    diss_path = silver_root / "dim_creative_dissection"
    if not assets_path.exists():
        raise FileNotFoundError(f"Missing dim_creative_asset table at {assets_path}")

    con = duckdb.connect(":memory:")
    try:
        assets_df = read_parquet_table(assets_path)
        con.register("assets", assets_df)
        if not diss_path.exists():
            return assets_df
        diss_df = read_parquet_table(diss_path)
        con.register("diss", diss_df)
        query = """
            SELECT a.*
            FROM assets a
            LEFT JOIN (
                SELECT DISTINCT media_hash_key
                FROM diss
                WHERE is_active_version = TRUE AND prompt_version = ?
            ) d
              ON a.media_hash_key = d.media_hash_key
            WHERE d.media_hash_key IS NULL
        """
        return con.execute(query, [prompt_version]).df()
    finally:
        con.close()


def _heuristic_offline_dissection(
    asset: DimCreativeAsset,
    *,
    evaluated_brand_id: str,
    channel_group: str = "meta_paid",
) -> DimCreativeDissection:
    """Deterministic zero-API fallback when running in offline/fixture mode."""
    uri_low = asset.media_uri.lower()
    raw_brand = canonicalize_brand_id(asset.brand_id_raw) or evaluated_brand_id
    is_offbrand = "reseller" in uri_low or "unauthorized" in uri_low or "counterfeit" in uri_low
    is_comparative = "vs_" in uri_low or "compare" in uri_low
    matched = "BMW,Porsche" if is_comparative else (raw_brand if not is_offbrand else "OFFBRAND")

    if "charging" in uri_low or "800v" in uri_low or "ev" in uri_low:
        concept = f"{raw_brand} 800V ultra-fast EV charging range highlight"
        angle = "Innovation & EV Range"
        hook = "product_demo"
    elif "lease" in uri_low or "apr" in uri_low or "offer" in uri_low:
        concept = f"{raw_brand} 0% APR financing & flexible seasonal lease offer"
        angle = "Financial Affordability & Offer"
        hook = "offer_first"
    elif "track" in uri_low or "performance" in uri_low or "hybrid" in uri_low:
        concept = f"{raw_brand} precision track telemetry & plug-in hybrid dynamics"
        angle = "Performance Engineering"
        hook = "brand_story"
    elif "cockpit" in uri_low or "safety" in uri_low or "assist" in uri_low:
        concept = f"{raw_brand} panoramic cockpit & highway driving assistant"
        angle = "Safety & Digital UX"
        hook = "problem_agitation"
    else:
        concept = f"{raw_brand} seasonal automotive showcase creative"
        angle = "Brand Showcase"
        hook = "ugc_testimonial"

    return DimCreativeDissection(
        media_hash_key=asset.media_hash_key,
        evaluated_brand_id=evaluated_brand_id,
        prompt_version="creative_dissection_v2",
        is_active_version=True,
        model_name="offline-heuristic-v1",
        format_fit_score=compute_format_fit_score(
            aspect_ratio=asset.aspect_ratio,
            duration_sec=asset.duration_sec,
            channel_group=channel_group,
            media_type=asset.media_type,
        ),
        belongs_to_brand=not is_offbrand,
        matched_brand_id=matched if not is_offbrand else None,
        confidence=0.92 if not is_offbrand else 0.95,
        category="automotive",
        core_concept_summary=concept,
        messaging_angle=angle,
        visual_style_tags=["studio", "automotive", str(asset.media_type)],
        hook_type=hook,  # type: ignore[arg-type]
        hook_transcript=concept[:80],
        time_to_brand_sec=1.2 if asset.media_type == "video" else 0.0,
        has_voiceover=asset.media_type == "video",
        has_captions=True,
        cta_text="Book Test Drive" if not is_offbrand else "Shop Used Cars",
        cta_clarity_score=4 if not is_offbrand else 2,
    )


def _clean_row(raw_row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for k, v in raw_row.items():
        if not isinstance(v, (list, tuple, dict)) and not hasattr(v, "tolist") and pd.isna(v):
            cleaned[k] = None
        elif hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
            cleaned[k] = v.tolist()
        else:
            cleaned[k] = v
    return cleaned


def run_lakehouse_dissection(
    lakehouse_dir: str | Path,
    *,
    gemini_client: GeminiMediaClient | None = None,
    budget_usd: float = 5.0,
    prompt_version: str = "creative_dissection_v2",
    staging_dir: str | Path | None = None,
    fallback_dissection_path: str | Path | None = None,
    allow_network: bool = False,
) -> dict[str, Any]:
    """Dissects unseen `dim_creative_asset` rows and merges into `silver/dim_creative_dissection`."""
    root = Path(lakehouse_dir)
    silver_dir = root / "silver"
    diss_out = silver_dir / "dim_creative_dissection"
    seed_path = (
        Path(fallback_dissection_path)
        if fallback_dissection_path
        else (root / "bronze" / "seed_creative_dissection")
    )

    unseen_df = select_unseen_assets(silver_dir, prompt_version=prompt_version)
    existing_df = read_parquet_table(diss_out) if diss_out.exists() else pd.DataFrame()
    if unseen_df.empty:
        return {
            "unseen_assets": 0,
            "newly_dissected": 0,
            "total_dissections": len(existing_df),
            "dissection_path": str(diss_out),
        }

    seed_lookup: dict[str, dict[str, Any]] = {}
    if seed_path.exists():
        seed_df = read_parquet_table(seed_path)
        for row in seed_df.to_dict(orient="records"):
            cleaned_seed = _clean_row(row)
            seed_lookup[str(cleaned_seed["media_hash_key"])] = cleaned_seed

    stager = MediaStager(staging_dir or (root / "assets_staging"))
    new_records: list[DimCreativeDissection] = []
    client_items: list[
        tuple[DimCreativeAsset, str, str, list[str | Path] | None, dict[str, Any] | None]
    ] = []

    for raw_row in unseen_df.to_dict(orient="records"):
        row = _clean_row(raw_row)
        asset = DimCreativeAsset.model_validate(row)
        eval_brand = canonicalize_brand_id(asset.brand_id_raw) or asset.brand_id_raw
        mkey = asset.media_hash_key

        if gemini_client is not None:
            staged = stager.stage(
                asset.media_uri,
                media_type=asset.media_type,
                aspect_ratio=asset.aspect_ratio,
                duration_sec=asset.duration_sec,
                allow_network=allow_network,
            )
            media_paths: list[str | Path] | None = (
                [staged.storyboard_paths[0]]
                if staged.storyboard_paths
                else ([staged.local_path] if staged.local_path else None)
            )
            client_items.append(
                (
                    asset,
                    eval_brand,
                    "meta_paid",
                    media_paths,
                    {
                        "brand_id_raw": asset.brand_id_raw,
                        "media_uri": asset.media_uri,
                        "media_type": asset.media_type,
                        "aspect_ratio": asset.aspect_ratio,
                    },
                )
            )
        elif mkey in seed_lookup:
            seed_row = dict(seed_lookup[mkey])
            seed_row["prompt_version"] = prompt_version
            new_records.append(DimCreativeDissection.model_validate(seed_row))
        else:
            new_records.append(
                _heuristic_offline_dissection(asset, evaluated_brand_id=eval_brand)
            )

    if client_items and gemini_client is not None:
        if gemini_client.finops.max_budget_usd is None:
            gemini_client.finops = FinOpsGuard(
                max_budget_usd=budget_usd,
                ledger_path=gemini_client.finops.ledger_path,
            )
        seen_keys: set[tuple[str, str, str]] = set()
        if not existing_df.empty:
            for r in existing_df.to_dict(orient="records"):
                seen_keys.add(
                    (
                        str(r["media_hash_key"]),
                        str(r["evaluated_brand_id"]),
                        str(r["prompt_version"]),
                    )
                )
        new_records = gemini_client.dissect_batch(client_items, seen_keys=seen_keys)

    new_df = pd.DataFrame([r.model_dump(mode="python") for r in new_records])
    combined = (
        pd.concat([existing_df, new_df], ignore_index=True)
        if not existing_df.empty
        else new_df
    )
    combined = combined.drop_duplicates(
        subset=["media_hash_key", "evaluated_brand_id", "prompt_version"], keep="last"
    )
    write_parquet_table(combined, diss_out, table_name="dim_creative_dissection")

    return {
        "unseen_assets": len(unseen_df),
        "newly_dissected": len(new_records),
        "total_dissections": len(combined),
        "dissection_path": str(diss_out),
    }
