"""Integration tests for Methodology Ablation, Pipeline Checkpoints, and Executive Reporting."""

import json
import os
import pandas as pd
import pytest

import config as cfg
from embedding_clustering_helpers import run_ablation_suite, run_competitive_intelligence


def test_generated_portfolio_integrity():
    assert os.path.exists(cfg.ASSETS_RAW_CSV)
    assert os.path.exists(cfg.ASSETS_STAGED_CSV)
    assert os.path.exists(cfg.ASSETS_ENRICHED_CSV)

    raw_df = pd.read_csv(cfg.ASSETS_RAW_CSV)
    enriched_df = pd.read_csv(cfg.ASSETS_ENRICHED_CSV)
    clustered_df = pd.read_csv(os.path.join(cfg.UI_DATA_DIR, "assets_clustered.csv"))
    themes_df = pd.read_csv(os.path.join(cfg.UI_DATA_DIR, "themes_summary.csv"))

    # 780 raw rows compressed to 340 perceptually distinct renders
    assert len(raw_df) == 780
    assert len(enriched_df) == 340
    assert len(clustered_df) == 780

    # Ensure themes are human-readable Malaysia Telco themes, not fallback "Theme 1"
    valid_themes = themes_df[themes_df["cluster_id"] != -1]
    assert len(valid_themes) >= 10
    for lbl in valid_themes["theme_label"]:
        assert not str(lbl).startswith("Theme "), f"Unexpected fallback theme label: {lbl}"


def test_ablation_report_discoveries():
    ablation_path = os.path.join(cfg.EVAL_DIR, "ablation_report.json")
    assert os.path.exists(ablation_path)
    with open(ablation_path, "r", encoding="utf-8") as fh:
        rep = json.load(fh)

    # Finding 1: Inverted modality weighting (w_vis=0.30 beats visual-only w_vis=1.0)
    frame_sweep = {r["visual_weight"]: r["ari"] for r in rep["modality_sweep"] if r["track"] == "frame"}
    assert frame_sweep[0.3] > frame_sweep[1.0]

    # Finding 2: Manifold pre-reduction cuts unclustered spend noise drastically vs raw 1792-D
    ladder = {s["stage_num"]: s for s in rep["architecture_ladder"]}
    assert ladder[1]["noise_spend_pct"] > 10.0
    assert ladder[4]["noise_spend_pct"] < 1.0
    assert ladder[4]["ari"] >= 0.80

    # Finding 3: Centroid merge consolidates format splits cleanly
    geom = rep["centroid_geometry"]
    assert geom["pre_merge_clusters"] > geom["post_merge_clusters"]


def test_competitive_intelligence_outputs():
    comp_themes_path = os.path.join(cfg.UI_DATA_DIR, "competitive_themes.csv")
    comp_metrics_path = os.path.join(cfg.UI_DATA_DIR, "competitive_brand_metrics.csv")
    assert os.path.exists(comp_themes_path)
    assert os.path.exists(comp_metrics_path)

    bm = pd.read_csv(comp_metrics_path)
    assert set(bm["brand"]) == {"Maxis", "CelcomDigi", "U Mobile", "Unifi"}
    assert bm["market_sos_pct"].sum() == pytest.approx(100.0, abs=0.2)
    assert (bm["hhi_index"] > 0.10).all() and (bm["hhi_index"] < 0.50).all()
