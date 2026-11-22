"""Multimodal Creative Dissection & Dynamic Multi-Brand Theme Engine."""

from intel_themes.competitive import (
    compute_creative_fatigue,
    compute_hhi,
    compute_jaccard_overlap,
    compute_theme_sov,
    match_prior_themes,
    run_dynamic_competitive_pipeline,
)
from intel_themes.dissection import run_lakehouse_dissection, select_unseen_assets
from intel_themes.vectors import (
    cluster_hdbscan,
    combine,
    compute_centroids,
    compute_phash_hex,
    embed_text_blocks,
    embed_visual_blocks,
    l2,
    merge_by_centroid,
    phash_distance,
    reduce_manifold,
    score,
)

__all__ = [
    "l2",
    "combine",
    "reduce_manifold",
    "cluster_hdbscan",
    "compute_centroids",
    "merge_by_centroid",
    "score",
    "compute_phash_hex",
    "phash_distance",
    "embed_text_blocks",
    "embed_visual_blocks",
    "select_unseen_assets",
    "run_lakehouse_dissection",
    "compute_hhi",
    "compute_jaccard_overlap",
    "compute_theme_sov",
    "compute_creative_fatigue",
    "match_prior_themes",
    "run_dynamic_competitive_pipeline",
]
