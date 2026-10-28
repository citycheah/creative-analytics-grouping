"""Pipeline configuration.

Delegates configuration loading to the root `config.py` module while preserving
backwards compatibility for existing imports of `ingestion.config`.
"""

import os
import sys

# Explicitly load repository root config.py without circular collision
import importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_root_config_path = os.path.join(_REPO_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("root_config", _root_config_path)
_root_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_cfg)

# Re-export all configuration values from root config
GCP_PROJECT_ID = _root_cfg.GCP_PROJECT_ID
GCP_REGION = _root_cfg.GCP_REGION
BQ_LOCATION = _root_cfg.BQ_LOCATION

SOURCE_TABLE = _root_cfg.SOURCE_TABLE
BQ_DATASET = _root_cfg.BQ_DATASET

ADVERTISER_NAMES = _root_cfg.ADVERTISER_NAMES
ADVERTISER_NAME = _root_cfg.ADVERTISER_NAME
COUNTRY = _root_cfg.COUNTRY
CURRENCY_SYMBOL = _root_cfg.CURRENCY_SYMBOL
START_DATE = _root_cfg.START_DATE
END_DATE = _root_cfg.END_DATE
COMPETITORS = _root_cfg.COMPETITORS

DATA_DIR = _root_cfg.DATA_DIR
ASSETS_RAW_CSV = _root_cfg.ASSETS_RAW_CSV
ASSETS_ENRICHED_CSV = _root_cfg.ASSETS_ENRICHED_CSV
ASSETS_STAGED_CSV = _root_cfg.ASSETS_STAGED_CSV

GCS_BUCKET = _root_cfg.GCS_BUCKET
GCS_PREFIX = _root_cfg.GCS_PREFIX
GCS_RAW_PREFIX = _root_cfg.GCS_RAW_PREFIX
GCS_FRAME_PREFIX = _root_cfg.GCS_FRAME_PREFIX

STAGE_DIR = _root_cfg.STAGE_DIR
RAW_DIR = _root_cfg.RAW_DIR
FRAME_DIR = _root_cfg.FRAME_DIR
UI_DATA_DIR = _root_cfg.UI_DATA_DIR
EVAL_DIR = _root_cfg.EVAL_DIR

# --- Clustering Defaults ---
CLUSTERING_TRACK = _root_cfg.CLUSTERING_TRACK
CLUSTERING_VISUAL_WEIGHT = _root_cfg.CLUSTERING_VISUAL_WEIGHT
CLUSTERING_MIN_CLUSTER_SIZE = _root_cfg.CLUSTERING_MIN_CLUSTER_SIZE
CLUSTERING_MIN_SAMPLES = _root_cfg.CLUSTERING_MIN_SAMPLES
CLUSTERING_UMAP_COMPONENTS = _root_cfg.CLUSTERING_UMAP_COMPONENTS
CLUSTERING_UMAP_SEED = _root_cfg.CLUSTERING_UMAP_SEED
CLUSTERING_MERGE_THRESHOLD = _root_cfg.CLUSTERING_MERGE_THRESHOLD

# --- AI & Embeddings ---
GEMINI_MODEL = _root_cfg.GEMINI_MODEL
GEMINI_THINKING_BUDGET = _root_cfg.GEMINI_THINKING_BUDGET
TEXT_EMBEDDING_MODEL = _root_cfg.TEXT_EMBEDDING_MODEL
