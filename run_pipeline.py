"""Single CLI entry point for AI-Driven Creative Clustering pipeline.

Chains and coordinates all pipeline stages:
1. `stage`        - Media download, frame extraction & perceptual hash dedup
2. `enrich`       - Structured concept extraction via Gemini 2.5 Flash
3. `embed_text`   - SentenceTransformer text embeddings (MiniLM / MPNet)
4. `embed_visual` - Vertex AI multimodal visual embeddings
5. `cluster`      - Dimensionality reduction (UMAP) + HDBSCAN + centroid merge + theme labeling
6. `aggregate`    - Rollup metrics generation for dashboard consumption

Features:
- Per-advertiser configuration loading (--advertiser <name>)
- Stage checkpointing and skipping of existing outputs (override with --force)
- Selective stage execution (--steps stage,enrich,...)
- Dry-run validation (--dry-run)
- Real-time progress and duration tracking

Usage:
    python run_pipeline.py --advertiser maxis_my
    python run_pipeline.py --dry-run
    python run_pipeline.py --steps cluster,aggregate,competitive,ablation
    python run_pipeline.py --force
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import config as cfg
from common.logging import get_logger

logger = get_logger("pipeline", stage="pipeline")

STAGES = [
    "stage",
    "enrich",
    "embed_text",
    "embed_visual",
    "cluster",
    "aggregate",
    "competitive",
    "ablation",
]

STAGE_COMMANDS = {
    "stage": [sys.executable, os.path.join(repo_root, "ingestion", "stage_assets.py")],
    "enrich": [sys.executable, os.path.join(repo_root, "ingestion", "process_assets.py")],
    "embed_text": [sys.executable, os.path.join(repo_root, "embedding-clustering", "embed_text.py")],
    "embed_visual": [sys.executable, os.path.join(repo_root, "embedding-clustering", "embed_visual.py"), "--tracks", "frame"],
    "cluster": [sys.executable, os.path.join(repo_root, "embedding-clustering", "cluster.py")],
    "aggregate": [sys.executable, os.path.join(repo_root, "demo-ui", "aggregate.py")],
    "competitive": [sys.executable, os.path.join(repo_root, "embedding-clustering", "competitive.py")],
    "ablation": [sys.executable, os.path.join(repo_root, "embedding-clustering", "benchmark_ablation.py")],
}


def get_checkpoint_files() -> Dict[str, List[str]]:
    """Returns the expected output artifacts for each stage."""
    return {
        "stage": [cfg.ASSETS_STAGED_CSV],
        "enrich": [cfg.ASSETS_ENRICHED_CSV],
        "embed_text": [os.path.join(cfg.DATA_DIR, "text_embeddings.npz")],
        "embed_visual": [os.path.join(cfg.DATA_DIR, "visual_embeddings.npz")],
        "cluster": [
            os.path.join(cfg.UI_DATA_DIR, "assets_clustered.csv"),
            os.path.join(cfg.UI_DATA_DIR, "themes_summary.csv"),
        ],
        "aggregate": [os.path.join(cfg.UI_DATA_DIR, "themes_summary_computed.csv")],
        "competitive": [
            os.path.join(cfg.UI_DATA_DIR, "competitive_themes.csv"),
            os.path.join(cfg.UI_DATA_DIR, "competitive_brand_metrics.csv"),
        ],
        "ablation": [os.path.join(cfg.EVAL_DIR, "ablation_report.json")],
    }


def is_stage_complete(stage: str) -> bool:
    """Checks whether all output checkpoints for a stage exist and are non-empty."""
    checkpoints = get_checkpoint_files().get(stage, [])
    if not checkpoints:
        return False
    for path in checkpoints:
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            return False
    return True


def run_stage(stage: str, extra_args: Optional[List[str]] = None) -> bool:
    """Executes a single pipeline stage as a subprocess with real-time output."""
    cmd = list(STAGE_COMMANDS[stage])
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"===> [STAGE: {stage.upper()}] Running command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=repo_root, check=True)
        elapsed = time.time() - t0
        logger.info(f"===> [STAGE: {stage.upper()}] Succeeded in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        logger.error(f"===> [STAGE: {stage.upper()}] Failed with return code {e.returncode} after {elapsed:.1f}s")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the AI Creative Clustering Pipeline")
    parser.add_argument("--advertiser", type=str, default=None,
                        help="Name or path of advertiser YAML config (e.g. maxis_my)")
    parser.add_argument("--steps", type=str, default=None,
                        help=f"Comma-separated list of steps to run. Options: {','.join(STAGES)}")
    parser.add_argument("--force", action="store_true",
                        help="Force re-running stages even if output checkpoints already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration and checkpoints without running execution commands")
    parser.add_argument("--no-labels", action="store_true",
                        help="Skip Gemini theme labeling during clustering (uses fallback Theme N)")
    args = parser.parse_args()

    # Load advertiser configuration if requested
    if args.advertiser:
        cfg.load_advertiser_yaml(args.advertiser)
        logger.info(f"Loaded configuration for advertiser: {cfg.ADVERTISER_NAME}")
    else:
        logger.info(f"Using default configuration for advertiser: {cfg.ADVERTISER_NAME}")

    cfg.ensure_dirs()

    steps = [s.strip() for s in args.steps.split(",")] if args.steps else STAGES
    for s in steps:
        if s not in STAGES:
            logger.error(f"Unknown stage '{s}'. Available stages: {STAGES}")
            sys.exit(1)

    print("\n" + "=" * 65)
    print(f" PIPELINE EXECUTION PLAN: {cfg.ADVERTISER_NAME.upper()}")
    print("=" * 65)
    print(f"  Advertiser : {cfg.ADVERTISER_NAME} ({cfg.COUNTRY})")
    print(f"  Date Range : {cfg.START_DATE} -> {cfg.END_DATE}")
    print(f"  Data Dir   : {cfg.DATA_DIR}")
    print(f"  Steps      : {', '.join(steps)}")
    print(f"  Force Run  : {args.force}")
    print(f"  Dry Run    : {args.dry_run}")
    print("-" * 65)

    checkpoints = get_checkpoint_files()
    stages_to_run = []
    for stage in steps:
        complete = is_stage_complete(stage)
        status_str = "CACHED (skipping)" if (complete and not args.force) else "PENDING"
        if complete and not args.force:
            print(f"  [{status_str:18}] {stage:<14} -> Checkpoint exists")
        else:
            print(f"  [{status_str:18}] {stage:<14} -> Outputs will be generated")
            stages_to_run.append(stage)
    print("=" * 65 + "\n")

    if args.dry_run:
        print("[DRY-RUN] Verification complete. No API calls or compute executed.")
        return

    if not stages_to_run:
        print("All requested stages are already complete! Use --force to re-run from scratch.")
        return

    pipeline_start = time.time()
    for idx, stage in enumerate(stages_to_run, 1):
        extra = []
        if stage == "cluster" and args.no_labels:
            extra.append("--no-labels")
        if args.force and stage in ("enrich", "embed_text", "embed_visual"):
            extra.append("--force")

        print(f"\n[{idx}/{len(stages_to_run)}] Starting stage: {stage}...")
        success = run_stage(stage, extra_args=extra)
        if not success:
            logger.error(f"Pipeline stopped due to failure in stage '{stage}'.")
            sys.exit(1)

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 65)
    print(f" PIPELINE FINISHED SUCCESSFULLY IN {total_time:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
