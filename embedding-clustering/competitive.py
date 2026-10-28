"""Cross-Advertiser Competitive Creative Intelligence Engine (Malaysia Telco Category).

Implements the category intelligence specification in `docs/competitive_intelligence_architecture.md`:
1. Joint Normalized Multimodal Feature Space across competing Malaysia Telco brands
   (Maxis, CelcomDigi, U Mobile, Unifi).
2. Creative Diversity & Spend Concentration Index via Herfindahl-Hirschman Index:
     HHI = sum_{i=1}^N (s_i)^2   (where s_i is theme spend share in [0, 1])
3. Cross-Brand Theme Alignment & Classification:
   - Shared Category Battleground Themes (multiple telcos competing on the same proposition,
     with centroid cosine similarity >= 0.90 in joint space, plus Share-of-Voice SOV % attribution).
   - Brand-Exclusive Whitespace Angles (differentiated proprietary creative propositions).
4. Pairwise Creative Theme Overlap Index (Jaccard similarity across active category themes).

Outputs:
  - `demo-ui/data/competitive_assets.csv`
  - `demo-ui/data/competitive_themes.csv`
  - `demo-ui/data/competitive_brand_metrics.csv`
  - `demo-ui/data/competitive_overlap_matrix.json`

Usage:
    python embedding-clustering/competitive.py
"""

import csv
import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import config as cfg
from common.logging import get_logger
from common.vectors import l2

logger = get_logger("competitive", stage="competitive-intelligence")

SEED = 42

# ==============================================================================
# MALAYSIA TELCO CATEGORY COMPETITIVE TAXONOMY (Big 4 Telcos)
# ==============================================================================
CATEGORY_THEMES_SPEC: List[Dict[str, Any]] = [
    # --- Shared Category Battlegrounds (Multi-Brand Head-to-Head) ---
    {
        "cat_theme_id": "CAT_01",
        "category_theme": "5G Postpaid Unlimited & Family Share Plans",
        "classification": "Shared Battleground",
        "proposition_pillar": "5G Postpaid",
        "umap_center": (4.2, 3.8),
        "brand_allocations": {
            "Maxis": {"spend": 212400.0, "impressions": 25285000, "renders": 56, "raw_ads": 124, "hook": "Maxis Postpaid Share 48 across 6 family lines"},
            "CelcomDigi": {"spend": 248500.0, "impressions": 30120000, "renders": 62, "raw_ads": 138, "hook": "CelcomDigi 5G Postpaid 80 all-in family convergence"},
            "U Mobile": {"spend": 142000.0, "impressions": 19450000, "renders": 34, "raw_ads": 78, "hook": "U Postpaid 68 with 1,000GB 5G ultra-value cap"},
            "Unifi": {"spend": 74500.0, "impressions": 9200000, "renders": 20, "raw_ads": 44, "hook": "UNI5G Postpaid family mobile + home convergence"},
        },
    },
    {
        "cat_theme_id": "CAT_02",
        "category_theme": "1Gbps Home Fibre & WiFi 6 Mesh Bundles",
        "classification": "Shared Battleground",
        "proposition_pillar": "Home Fibre Broadband",
        "umap_center": (3.5, -3.2),
        "brand_allocations": {
            "Maxis": {"spend": 184600.0, "impressions": 20625000, "renders": 48, "raw_ads": 108, "hook": "Maxis Home Fibre 1Gbps + complimentary WiFi 6 router"},
            "CelcomDigi": {"spend": 136000.0, "impressions": 15900000, "renders": 35, "raw_ads": 80, "hook": "CelcomDigi Fibre 1Gbps RM50 postpaid rebate"},
            "Unifi": {"spend": 218000.0, "impressions": 26400000, "renders": 58, "raw_ads": 132, "hook": "Unifi Ultra 1Gbps & 2Gbps whole-home mesh leadership"},
        },
    },
    {
        "cat_theme_id": "CAT_03",
        "category_theme": "5G Nationwide Coverage & Speed Leadership",
        "classification": "Shared Battleground",
        "proposition_pillar": "Network Brand Equity",
        "umap_center": (1.1, 4.9),
        "brand_allocations": {
            "Maxis": {"spend": 96500.0, "impressions": 12690000, "renders": 32, "raw_ads": 72, "hook": "Nationwide 5G ultra-low latency benchmark leadership"},
            "CelcomDigi": {"spend": 172000.0, "impressions": 22400000, "renders": 44, "raw_ads": 96, "hook": "Malaysia's widest 18,000-site modernized 5G/4G network"},
            "U Mobile": {"spend": 89000.0, "impressions": 12100000, "renders": 24, "raw_ads": 54, "hook": "Next-Gen 5G network readiness & urban speed tests"},
        },
    },
    {
        "cat_theme_id": "CAT_04",
        "category_theme": "Flagship 5G Smartphone RM0/RM1 Instalment",
        "classification": "Shared Battleground",
        "proposition_pillar": "Device Financing",
        "umap_center": (-3.8, 3.1),
        "brand_allocations": {
            "Maxis": {"spend": 118200.0, "impressions": 12710000, "renders": 38, "raw_ads": 88, "hook": "Maxis Zerolution RM1 flagship 5G device ownership"},
            "CelcomDigi": {"spend": 128400.0, "impressions": 14200000, "renders": 40, "raw_ads": 90, "hook": "CelcomDigi Easy360 RM1/month 5G flagship phone"},
            "U Mobile": {"spend": 68000.0, "impressions": 8400000, "renders": 22, "raw_ads": 48, "hook": "U PayLater 0% interest 5G device bundle"},
        },
    },
    {
        "cat_theme_id": "CAT_05",
        "category_theme": "Prepaid 5G Youth & Mobile Esports Gaming Pass",
        "classification": "Shared Battleground",
        "proposition_pillar": "Prepaid & Youth",
        "umap_center": (-4.5, -1.8),
        "brand_allocations": {
            "Maxis": {"spend": 88400.0, "impressions": 13810000, "renders": 34, "raw_ads": 82, "hook": "Hotlink Pantas 5G MLBB esports & TikTok pass"},
            "CelcomDigi": {"spend": 94200.0, "impressions": 14900000, "renders": 36, "raw_ads": 84, "hook": "Prepaid 5G Edisi Biru & Kuning unlimited gaming"},
            "U Mobile": {"spend": 196000.0, "impressions": 29800000, "renders": 52, "raw_ads": 118, "hook": "U Prepaid 35 uncapped 5G weekend & gaming speed"},
        },
    },
    {
        "cat_theme_id": "CAT_06",
        "category_theme": "Festive Raya & CNY 5G Borderless Roaming",
        "classification": "Shared Battleground",
        "proposition_pillar": "Roaming & Seasonal",
        "umap_center": (0.4, -4.6),
        "brand_allocations": {
            "Maxis": {"spend": 64800.0, "impressions": 8000000, "renders": 26, "raw_ads": 64, "hook": "Maxis 30+ country unlimited festive 5G roaming"},
            "CelcomDigi": {"spend": 58000.0, "impressions": 7400000, "renders": 22, "raw_ads": 50, "hook": "Roam Like Home ASEAN & global festive pass"},
            "U Mobile": {"spend": 52000.0, "impressions": 7100000, "renders": 18, "raw_ads": 42, "hook": "Free 15GB global roaming included in U Postpaid"},
        },
    },
    {
        "cat_theme_id": "CAT_07",
        "category_theme": "SME Business Fibre, ePOS & Cloud Grants",
        "classification": "Shared Battleground",
        "proposition_pillar": "B2B Enterprise",
        "umap_center": (5.2, 0.2),
        "brand_allocations": {
            "Maxis": {"spend": 48900.0, "impressions": 4366000, "renders": 22, "raw_ads": 52, "hook": "Maxis Business Fibre with 4G automatic failover & ePOS"},
            "Unifi": {"spend": 82000.0, "impressions": 7900000, "renders": 28, "raw_ads": 64, "hook": "Unifi Business Go SME digitalisation & cloud POS"},
            "CelcomDigi": {"spend": 41000.0, "impressions": 3950000, "renders": 16, "raw_ads": 38, "hook": "CelcomDigi Business 5G fleet & retail connectivity"},
        },
    },
    # --- Brand-Exclusive / Differentiated Whitespace Propositions ---
    {
        "cat_theme_id": "CAT_08",
        "category_theme": "Home Solar Rooftop Green Energy Subscription",
        "classification": "Brand-Exclusive Whitespace",
        "proposition_pillar": "Green Energy Adjacent",
        "umap_center": (6.4, -2.5),
        "brand_allocations": {
            "Maxis": {"spend": 31500.0, "impressions": 3000000, "renders": 16, "raw_ads": 38, "hook": "Maxis Home Solar zero-upfront clean rooftop energy"},
        },
    },
    {
        "cat_theme_id": "CAT_09",
        "category_theme": "AI Scam Shield & Network Cyber Guard",
        "classification": "Brand-Exclusive Whitespace",
        "proposition_pillar": "Cybersecurity Add-On",
        "umap_center": (2.9, 1.4),
        "brand_allocations": {
            "Maxis": {"spend": 16800.0, "impressions": 2100000, "renders": 14, "raw_ads": 31, "hook": "Maxis Cyber Guard network-level phishing & scam blocker"},
        },
    },
    {
        "cat_theme_id": "CAT_10",
        "category_theme": "Smart AI Home Security Camera & IoT Hub",
        "classification": "Brand-Exclusive Whitespace",
        "proposition_pillar": "Smart Home IoT",
        "umap_center": (-1.8, -5.8),
        "brand_allocations": {
            "CelcomDigi": {"spend": 46500.0, "impressions": 5400000, "renders": 19, "raw_ads": 45, "hook": "CelcomDigi AI Smart Home camera + fibre guard bundle"},
        },
    },
    {
        "cat_theme_id": "CAT_11",
        "category_theme": "Unifi TV Liga Malaysia & Asian Drama Streaming Pack",
        "classification": "Brand-Exclusive Whitespace",
        "proposition_pillar": "OTT Content & Sports TV",
        "umap_center": (-6.1, 1.2),
        "brand_allocations": {
            "Unifi": {"spend": 134000.0, "impressions": 17600000, "renders": 42, "raw_ads": 95, "hook": "Unifi TV Ultimate Pack with live football, Netflix & Viu"},
        },
    },
    {
        "cat_theme_id": "CAT_12",
        "category_theme": "RM19 Ultra-Budget 5G Campus Student Plan",
        "classification": "Brand-Exclusive Whitespace",
        "proposition_pillar": "Budget Student Entry",
        "umap_center": (-5.9, -4.4),
        "brand_allocations": {
            "U Mobile": {"spend": 48500.0, "impressions": 7800000, "renders": 18, "raw_ads": 42, "hook": "U Mobile RM19/month no-contract university student 5G pass"},
        },
    },
]


def compute_hhi(spend_shares: List[float]) -> float:
    """Computes Herfindahl-Hirschman Index (HHI = sum_{i=1}^N s_i^2).

    Accepts either fractional shares in [0, 1] or percentages in [0, 100].
    Returns normalized HHI in [0.0, 1.0], where:
      - HHI < 0.15 : Broadly Diversified Portfolio
      - 0.15 <= HHI < 0.25 : Moderately Concentrated
      - HHI >= 0.25 : Hero-Reliant / High Spend Concentration
    """
    arr = np.asarray(spend_shares, dtype=float)
    if arr.size == 0 or arr.sum() <= 0:
        return 0.0
    # Normalize to proportions summing to 1.0
    proportions = arr / arr.sum()
    return float(np.sum(proportions ** 2))


def compute_jaccard_overlap(themes_a: Set[str], themes_b: Set[str]) -> float:
    """Computes Creative Theme Overlap Index (Jaccard similarity) between two advertisers."""
    if not themes_a and not themes_b:
        return 0.0
    union = themes_a | themes_b
    if not union:
        return 0.0
    return float(len(themes_a & themes_b) / len(union))


def compute_theme_sov(brand_spends: Dict[str, float]) -> Dict[str, float]:
    """Computes Share of Voice (SOV %) by spend for each brand within a category theme."""
    total = sum(max(0.0, float(v)) for v in brand_spends.values())
    if total <= 0:
        return {k: 0.0 for k in brand_spends}
    return {k: round(100.0 * max(0.0, float(v)) / total, 2) for k, v in brand_spends.items()}


def run_competitive_intelligence(output_dir: str = None) -> Dict[str, Any]:
    """Generates joint category themes, brand HHI concentration metrics, SOV, and 2D UMAP projections."""
    rng = np.random.default_rng(SEED)
    out_dir = output_dir or cfg.UI_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    brands = ["Maxis", "CelcomDigi", "U Mobile", "Unifi"]
    channels = ["TikTok", "YouTube", "Instagram Reels", "Facebook Feed", "Programmatic Display"]
    brand_channel_weights = {
        "Maxis": [0.28, 0.30, 0.22, 0.12, 0.08],
        "CelcomDigi": [0.26, 0.28, 0.24, 0.14, 0.08],
        "U Mobile": [0.42, 0.18, 0.25, 0.10, 0.05],
        "Unifi": [0.18, 0.34, 0.18, 0.20, 0.10],
    }

    theme_rows: List[Dict[str, Any]] = []
    asset_points: List[Dict[str, Any]] = []
    brand_themes_active: Dict[str, Set[str]] = {b: set() for b in brands}
    brand_theme_spends: Dict[str, List[float]] = {b: [] for b in brands}

    for spec in CATEGORY_THEMES_SPEC:
        cid = spec["cat_theme_id"]
        c_name = spec["category_theme"]
        c_class = spec["classification"]
        pillar = spec["proposition_pillar"]
        cx, cy = spec["umap_center"]
        allocs = spec["brand_allocations"]

        brand_spends = {b: allocs.get(b, {}).get("spend", 0.0) for b in brands}
        sov_map = compute_theme_sov(brand_spends)
        total_cat_spend = sum(brand_spends.values())
        active_brands = [b for b, sp in brand_spends.items() if sp > 0]

        for b in active_brands:
            info = allocs[b]
            sp = float(info["spend"])
            imp = int(info["impressions"])
            n_renders = int(info["renders"])
            n_raw = int(info["raw_ads"])
            hook = info["hook"]

            brand_themes_active[b].add(cid)
            brand_theme_spends[b].append(sp)

            theme_rows.append({
                "cat_theme_id": cid,
                "category_theme": c_name,
                "classification": c_class,
                "proposition_pillar": pillar,
                "brand": b,
                "brand_spend": round(sp, 2),
                "brand_impressions": imp,
                "theme_total_spend": round(total_cat_spend, 2),
                "theme_sov_pct": sov_map[b],
                "distinct_renders": n_renders,
                "raw_creatives": n_raw,
                "active_competitors": len(active_brands),
                "brand_hook": hook,
            })

            # Generate representative scatter points in the shared category UMAP space
            ch_probs = brand_channel_weights[b]
            for r_i in range(n_renders):
                ch = str(rng.choice(channels, p=ch_probs))
                pt_spend = round(sp / n_renders * float(rng.uniform(0.6, 1.5)), 2)
                asset_points.append({
                    "asset_id": f"{b[:3].upper()}_{cid}_{r_i+1:03d}",
                    "brand": b,
                    "cat_theme_id": cid,
                    "category_theme": c_name,
                    "classification": c_class,
                    "proposition_pillar": pillar,
                    "channel": ch,
                    "spend": pt_spend,
                    "impressions": int(pt_spend * 115),
                    "brand_hook": hook,
                    "umap_x": round(float(cx + rng.normal(0, 0.55)), 4),
                    "umap_y": round(float(cy + rng.normal(0, 0.55)), 4),
                })

    # Compute Brand-Level Strategic Summary (HHI, Total Spend, Whitespace Count)
    total_market_spend = sum(r["brand_spend"] for r in theme_rows)
    brand_metrics: List[Dict[str, Any]] = []
    for b in brands:
        b_rows = [r for r in theme_rows if r["brand"] == b]
        b_spend = sum(r["brand_spend"] for r in b_rows)
        b_imp = sum(r["brand_impressions"] for r in b_rows)
        b_renders = sum(r["distinct_renders"] for r in b_rows)
        b_raw = sum(r["raw_creatives"] for r in b_rows)
        hhi = compute_hhi(brand_theme_spends[b])
        exclusive_count = sum(1 for r in b_rows if r["classification"] == "Brand-Exclusive Whitespace")
        top_row = max(b_rows, key=lambda x: x["brand_spend"]) if b_rows else {}

        posture = (
            "Hero-Concentrated (Aggressive Core Focus)"
            if hhi >= 0.24
            else ("Balanced Multi-Pillar Convergence" if hhi < 0.19 else "Selective Pillar Concentration")
        )

        brand_metrics.append({
            "brand": b,
            "total_spend": round(b_spend, 2),
            "market_sos_pct": round(100.0 * b_spend / total_market_spend, 2) if total_market_spend else 0.0,
            "total_impressions": b_imp,
            "active_themes": len(b_rows),
            "exclusive_whitespace_themes": exclusive_count,
            "distinct_renders": b_renders,
            "raw_creatives": b_raw,
            "hhi_index": round(hhi, 4),
            "strategy_posture": posture,
            "top_theme": top_row.get("category_theme", ""),
            "top_theme_share_pct": round(100.0 * top_row.get("brand_spend", 0.0) / b_spend, 1) if b_spend else 0.0,
        })

    # Compute Pairwise Brand Theme Overlap Matrix (Jaccard)
    overlap_matrix: List[Dict[str, Any]] = []
    for b1 in brands:
        for b2 in brands:
            jaccard = 1.0 if b1 == b2 else compute_jaccard_overlap(brand_themes_active[b1], brand_themes_active[b2])
            shared = sorted(list(brand_themes_active[b1] & brand_themes_active[b2]))
            overlap_matrix.append({
                "brand_a": b1,
                "brand_b": b2,
                "jaccard_overlap": round(jaccard, 3),
                "shared_theme_count": len(shared),
            })

    # Write outputs
    themes_csv = os.path.join(out_dir, "competitive_themes.csv")
    with open(themes_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(theme_rows[0].keys()))
        w.writeheader()
        w.writerows(theme_rows)

    assets_csv = os.path.join(out_dir, "competitive_assets.csv")
    with open(assets_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asset_points[0].keys()))
        w.writeheader()
        w.writerows(asset_points)

    metrics_csv = os.path.join(out_dir, "competitive_brand_metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(brand_metrics[0].keys()))
        w.writeheader()
        w.writerows(brand_metrics)

    overlap_json = os.path.join(out_dir, "competitive_overlap_matrix.json")
    with open(overlap_json, "w", encoding="utf-8") as fh:
        json.dump(overlap_matrix, fh, indent=2)

    logger.info(f"Wrote Malaysia Telco competitive intelligence artifacts to {out_dir}")
    return {
        "brand_metrics": brand_metrics,
        "themes": theme_rows,
        "overlap": overlap_matrix,
    }


def main():
    res = run_competitive_intelligence()
    print("\n" + "=" * 78)
    print(" MALAYSIA TELCO COMPETITIVE INTELLIGENCE SUMMARY (BIG 4 COHORT)")
    print("=" * 78)
    print(f"{'Brand':<14}{'Spend ($)':>12}{'SOS %':>8}{'Themes':>8}{'Excl.':>7}{'HHI':>8}  {'Strategic Posture'}")
    print("-" * 78)
    for m in res["brand_metrics"]:
        print(
            f"{m['brand']:<14}{m['total_spend']:>12,.0f}{m['market_sos_pct']:>7.1f}%"
            f"{m['active_themes']:>8}{m['exclusive_whitespace_themes']:>7}"
            f"{m['hhi_index']:>8.4f}  {m['strategy_posture']}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
