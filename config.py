"""Centralized configuration system for AI-Driven Creative Clustering & Category Intelligence.

Loads settings from .env file (if present), environment variables,
and optional per-advertiser or category YAML configuration files.
"""

import os
from typing import Any, Dict, List
from dotenv import load_dotenv

# Base paths
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

# Load .env from project root if it exists
dotenv_path = os.path.join(REPO_ROOT, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# --- GCP Settings ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")  # Vertex AI multimodalembedding region
BQ_LOCATION = os.getenv("BQ_LOCATION", "US")

# --- Source Data & BigQuery ---
SOURCE_TABLE = os.getenv("SOURCE_TABLE", "ad_intelligence.creative_occurrences_all_markets")
BQ_DATASET = os.getenv("BQ_DATASET", "creative_theme_analytics")

# --- Pull / Advertiser Parameters (Default Showcase: Malaysia Telco — Maxis) ---
_default_advertisers = "MAXIS,MAXIS BROADBAND,HOTLINK,MAXIS BUSINESS"
ADVERTISER_NAMES_ENV = os.getenv("ADVERTISER_NAMES", _default_advertisers)
ADVERTISER_NAMES: List[str] = [x.strip() for x in ADVERTISER_NAMES_ENV.split(",") if x.strip()]
ADVERTISER_NAME = os.getenv("ADVERTISER_NAME", "Maxis")
COUNTRY = os.getenv("COUNTRY", "Malaysia")
CURRENCY_SYMBOL = os.getenv("CURRENCY_SYMBOL", "$")
START_DATE = os.getenv("START_DATE", "2026-01-01")
END_DATE = os.getenv("END_DATE", "2026-03-31")

# --- Competitive Category Cohort (Malaysia Telco Big 4) ---
COMPETITORS: List[str] = [
    "Maxis",
    "CelcomDigi",
    "U Mobile",
    "Unifi",
]

# --- Local Storage & Artifacts (anchored to REPO_ROOT if relative) ---
def _resolve_repo_path(val: str, default_rel: str) -> str:
    if not val:
        return os.path.join(REPO_ROOT, default_rel)
    return val if os.path.isabs(val) else os.path.join(REPO_ROOT, val)

DATA_DIR = _resolve_repo_path(os.getenv("DATA_DIR", ""), os.path.join("ingestion", "data"))
ASSETS_RAW_CSV = _resolve_repo_path(os.getenv("ASSETS_RAW_CSV", ""), os.path.join(DATA_DIR, "assets_raw.csv"))
ASSETS_ENRICHED_CSV = _resolve_repo_path(os.getenv("ASSETS_ENRICHED_CSV", ""), os.path.join(DATA_DIR, "assets_enriched.csv"))
ASSETS_STAGED_CSV = _resolve_repo_path(os.getenv("ASSETS_STAGED_CSV", ""), os.path.join(DATA_DIR, "assets_staged.csv"))

# Staging directories for raw media and extracted frames
STAGE_DIR = _resolve_repo_path(os.getenv("STAGE_DIR", ""), "assets")
RAW_DIR = _resolve_repo_path(os.getenv("RAW_DIR", ""), os.path.join(STAGE_DIR, "raw"))
FRAME_DIR = _resolve_repo_path(os.getenv("FRAME_DIR", ""), os.path.join(STAGE_DIR, "frames"))

# UI & Evaluation paths
UI_DATA_DIR = _resolve_repo_path(os.getenv("UI_DATA_DIR", ""), os.path.join("demo-ui", "data"))
EVAL_DIR = _resolve_repo_path(os.getenv("EVAL_DIR", ""), "evaluation")

# --- GCS Storage ---
GCS_BUCKET = os.getenv("GCS_BUCKET", "creative-analytics-staging-bucket")
GCS_PREFIX = os.getenv("GCS_PREFIX", "malaysia-telco-clustering")
GCS_RAW_PREFIX = f"{GCS_PREFIX}/raw"
GCS_FRAME_PREFIX = f"{GCS_PREFIX}/frames"

# --- Clustering Defaults ---
CLUSTERING_TRACK = os.getenv("CLUSTERING_TRACK", "frame")
CLUSTERING_VISUAL_WEIGHT = float(os.getenv("CLUSTERING_VISUAL_WEIGHT", "0.3"))
CLUSTERING_MIN_CLUSTER_SIZE = int(os.getenv("CLUSTERING_MIN_CLUSTER_SIZE", "4"))
CLUSTERING_MIN_SAMPLES = int(os.getenv("CLUSTERING_MIN_SAMPLES", "2"))
CLUSTERING_UMAP_COMPONENTS = int(os.getenv("CLUSTERING_UMAP_COMPONENTS", "30"))
CLUSTERING_UMAP_SEED = int(os.getenv("CLUSTERING_UMAP_SEED", "42"))
CLUSTERING_MERGE_THRESHOLD = float(os.getenv("CLUSTERING_MERGE_THRESHOLD", "0.94"))

# --- AI & Embeddings ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def ensure_dirs():
    """Ensure required local directories exist."""
    for d in [DATA_DIR, STAGE_DIR, RAW_DIR, FRAME_DIR, UI_DATA_DIR, EVAL_DIR]:
        os.makedirs(d, exist_ok=True)


def load_advertiser_yaml(config_path_or_name: str) -> Dict[str, Any]:
    """Load per-advertiser YAML configuration and override module attributes.

    Accepts either an absolute/relative filepath or a config name (e.g. 'maxis_my').
    """
    import yaml

    if os.path.isabs(config_path_or_name) or os.path.exists(config_path_or_name):
        path = config_path_or_name
    else:
        cfg_name = (
            config_path_or_name
            if config_path_or_name.endswith((".yaml", ".yml"))
            else f"{config_path_or_name}.yaml"
        )
        path = os.path.join(REPO_ROOT, "configs", cfg_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Advertiser config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    global ADVERTISER_NAME, ADVERTISER_NAMES, COUNTRY, START_DATE, END_DATE, COMPETITORS
    global CLUSTERING_TRACK, CLUSTERING_VISUAL_WEIGHT, CLUSTERING_MIN_CLUSTER_SIZE
    global CLUSTERING_MIN_SAMPLES, CLUSTERING_UMAP_COMPONENTS, CLUSTERING_MERGE_THRESHOLD
    global GEMINI_MODEL, GEMINI_THINKING_BUDGET, TEXT_EMBEDDING_MODEL

    if "advertiser_name" in data:
        ADVERTISER_NAME = data["advertiser_name"]
    if "advertiser_names" in data:
        ADVERTISER_NAMES = data["advertiser_names"]
    elif "advertiser_name" in data:
        ADVERTISER_NAMES = [data["advertiser_name"]]
    if "country" in data:
        COUNTRY = data["country"]
    if "competitors" in data and isinstance(data["competitors"], list):
        COMPETITORS = data["competitors"]
    if "date_range" in data and len(data["date_range"]) == 2:
        START_DATE, END_DATE = data["date_range"][0], data["date_range"][1]

    clustering = data.get("clustering", {})
    if "track" in clustering:
        CLUSTERING_TRACK = clustering["track"]
    if "visual_weight" in clustering:
        CLUSTERING_VISUAL_WEIGHT = float(clustering["visual_weight"])
    if "min_cluster_size" in clustering:
        CLUSTERING_MIN_CLUSTER_SIZE = int(clustering["min_cluster_size"])
    if "min_samples" in clustering:
        CLUSTERING_MIN_SAMPLES = int(clustering["min_samples"])
    if "umap_components" in clustering:
        CLUSTERING_UMAP_COMPONENTS = int(clustering["umap_components"])
    if "merge_threshold" in clustering:
        CLUSTERING_MERGE_THRESHOLD = float(clustering["merge_threshold"])
    if "text_embedding_model" in clustering:
        TEXT_EMBEDDING_MODEL = clustering["text_embedding_model"]

    gemini = data.get("gemini", {})
    if "model" in gemini:
        GEMINI_MODEL = gemini["model"]
    if "thinking_budget" in gemini:
        GEMINI_THINKING_BUDGET = int(gemini["thinking_budget"])

    return data
