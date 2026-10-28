"""Helper exports for embedding-clustering modules (hyphenated directory bridge for tests)."""

import importlib.util
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_COMP_PATH = os.path.join(_REPO_ROOT, "embedding-clustering", "competitive.py")
_ABL_PATH = os.path.join(_REPO_ROOT, "embedding-clustering", "benchmark_ablation.py")

_spec_comp = importlib.util.spec_from_file_location("competitive", _COMP_PATH)
competitive = importlib.util.module_from_spec(_spec_comp)
_spec_comp.loader.exec_module(competitive)

_spec_abl = importlib.util.spec_from_file_location("benchmark_ablation", _ABL_PATH)
benchmark_ablation = importlib.util.module_from_spec(_spec_abl)
_spec_abl.loader.exec_module(benchmark_ablation)

compute_hhi = competitive.compute_hhi
compute_jaccard_overlap = competitive.compute_jaccard_overlap
compute_theme_sov = competitive.compute_theme_sov
run_competitive_intelligence = competitive.run_competitive_intelligence
run_ablation_suite = benchmark_ablation.run_ablation_suite
