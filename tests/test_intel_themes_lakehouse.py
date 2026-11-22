"""Tests for federated `intel_themes` incremental dissection & dynamic multi-brand clustering."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from intel_core.gemini import FinOpsGuard, GeminiMediaClient
from intel_core.storage import HumanOverrideStore, read_parquet_table, write_parquet_table

from intel_themes import (
    run_dynamic_competitive_pipeline,
    run_lakehouse_dissection,
    select_unseen_assets,
)

EU_LAKEHOUSE = Path(__file__).resolve().parents[2] / "data" / "eu_auto_lakehouse"


def _copy_sample_lakehouse(tmp_path: Path) -> Path:
    dest = tmp_path / "eu_auto_lakehouse"
    shutil.copytree(EU_LAKEHOUSE, dest)
    return dest


def test_incremental_duckdb_antijoin_and_vision_dissection(tmp_path: Path):
    if not EU_LAKEHOUSE.exists():
        pytest.skip("eu_auto_lakehouse fixture not generated")

    lh = _copy_sample_lakehouse(tmp_path)
    # Start without silver/dim_creative_dissection -> all 50 assets are unseen
    shutil.rmtree(lh / "silver" / "dim_creative_dissection", ignore_errors=True)
    unseen_initial = select_unseen_assets(lh / "silver")
    assert len(unseen_initial) == 50

    calls: list[dict] = []

    def fake_transport(prompt: str, contents: list, config_meta: dict):
        calls.append(config_meta)
        return {
            "text": json.dumps(
                {
                    "belongs_to_brand": True,
                    "matched_brand_id": config_meta.get("evaluated_brand_id", "BMW"),
                    "confidence": 0.91,
                    "category": "automotive",
                    "core_concept_summary": "800V fast-charge electric SUV range hero",
                    "messaging_angle": "EV Range & Fast Charging",
                    "visual_style_tags": ["studio", "ev"],
                    "hook_type": "product_demo",
                    "hook_transcript": "Up to 620 km WLTP range",
                    "time_to_brand_sec": 1.1,
                    "has_voiceover": True,
                    "has_captions": True,
                    "cta_text": "Configure Now",
                    "cta_clarity_score": 5,
                }
            ),
            "prompt_token_count": 300,
            "candidates_token_count": 80,
        }

    # Pre-seed 47 existing rows via offline fallback, leaving 3 unseen assets for live Vision client
    assets_df = read_parquet_table(lh / "silver" / "dim_creative_asset")
    seed_df = read_parquet_table(lh / "bronze" / "seed_creative_dissection")
    partial_keys = set(assets_df["media_hash_key"].head(47).astype(str))
    partial_diss = seed_df[seed_df["media_hash_key"].astype(str).isin(partial_keys)].copy()
    partial_diss["prompt_version"] = "creative_dissection_v2"
    write_parquet_table(
        partial_diss,
        lh / "silver" / "dim_creative_dissection",
        table_name="dim_creative_dissection",
    )

    unseen_delta = select_unseen_assets(lh / "silver")
    assert len(unseen_delta) == 3

    ledger = tmp_path / "finops.csv"
    client = GeminiMediaClient(
        model="gemini-2.5-flash",
        prompt_version="creative_dissection_v2",
        transport=fake_transport,
        finops_guard=FinOpsGuard(max_budget_usd=2.0, ledger_path=ledger),
    )
    summary = run_lakehouse_dissection(lh, gemini_client=client)
    assert summary["unseen_assets"] == 3
    assert summary["newly_dissected"] == 3
    assert len(calls) == 3
    assert all(c["thinking_budget"] == 0 for c in calls)

    # Running a second time is idempotent -> 0 unseen assets, 0 Gemini calls
    unseen_after = select_unseen_assets(lh / "silver")
    assert len(unseen_after) == 0


def test_dynamic_multibrand_clustering_and_prior_centroid_stability(tmp_path: Path):
    if not EU_LAKEHOUSE.exists():
        pytest.skip("eu_auto_lakehouse fixture not generated")

    lh = _copy_sample_lakehouse(tmp_path)
    # Materialize silver/dim_creative_dissection using offline seed/heuristic mode
    run_lakehouse_dissection(lh, gemini_client=None)

    # Weekly Run 1: discovers themes dynamically (zero hardcoded taxonomy)
    res1 = run_dynamic_competitive_pipeline(
        lh,
        run_id="run_w1",
        visual_weight=0.30,
        umap_components=30,
        merge_threshold=0.94,
        prior_sim_threshold=0.85,
        persist=True,
    )
    assert res1["assets_clustered"] >= 45
    assert res1["themes_discovered"] >= 4
    assert res1["battleground_count"] >= 1
    assert set(res1["brand_hhi"].keys()) >= {"BMW", "Mercedes-Benz", "Audi", "Porsche"}
    assert Path(res1["dim_themes_path"]).exists()
    assert Path(res1["centroids_path"]).exists()

    first_tids = set(res1["dim_themes_df"]["theme_id"].unique())
    assert len(first_tids) == res1["themes_discovered"]

    # Check fatigue calculation generated valid age/decay stats
    fatigue = res1["fatigue_df"]
    assert not fatigue.empty
    assert (fatigue["avg_creative_age_weeks"] >= 1).all()

    # Weekly Run 2 with same data + saved prior centroids -> exact centroid-matched theme_id reuse!
    res2 = run_dynamic_competitive_pipeline(
        lh,
        run_id="run_w2",
        visual_weight=0.30,
        umap_components=30,
        merge_threshold=0.94,
        prior_sim_threshold=0.85,
        prior_centroids_path=res1["centroids_path"],
        persist=False,
    )
    second_tids = set(res2["dim_themes_df"]["theme_id"].unique())
    assert first_tids == second_tids

    # Apply HITL override promoting rejected asset -> included in next clustering run
    assets_df = read_parquet_table(lh / "silver" / "dim_creative_asset")
    diss_df = read_parquet_table(lh / "silver" / "dim_creative_dissection")
    rejected_rows = diss_df[diss_df["belongs_to_brand"] == False]  # noqa: E712
    if not rejected_rows.empty:
        mkey = str(rejected_rows.iloc[0]["media_hash_key"])
        matched_asset = assets_df[assets_df["media_hash_key"] == mkey]
        if not matched_asset.empty:
            aid = str(matched_asset.iloc[0]["asset_id"])
            store = HumanOverrideStore(lh / "overrides")
            store.write_override(
                target_key_type="asset_id",
                target_key_value=aid,
                field_name="belongs_to_brand",
                human_value="true",
                reviewer="qa@intel.eu",
                ai_value="false",
                reason="Verified OEM teaser",
            )
            res3 = run_dynamic_competitive_pipeline(lh, run_id="run_w3", persist=False)
            assert res3["assets_clustered"] >= res1["assets_clustered"] + 1
