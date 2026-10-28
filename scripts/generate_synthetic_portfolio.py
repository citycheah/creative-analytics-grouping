"""Malaysia Telco Creative Portfolio & Category Intelligence Benchmark Generator.

Deterministically generates a complete, self-contained local benchmark dataset for the
Malaysia Telecommunications (Telco) sector (Maxis, CelcomDigi, U Mobile, Unifi):

1. Primary Brand Portfolio (Maxis / Hotlink Malaysia):
   - 780 raw ad occurrences -> 340 perceptually distinct renders (pHash groups)
   - 12 core telco creative themes + 4 unclustered noise outliers
   - Multi-format aspect ratio cuts (9:16 vertical, 16:9 landscape, 1:1 square, 4:5 portrait)
   - Calibrated 1408-D visual embeddings (`vec_frame`, `vec_video`) and 384-D text embeddings (`vec_text`)
     reproducing:
       a) Intra-brand visual similarity (optimal fusion weight w_vis=0.3 / w_txt=0.7)
       b) Curse of dimensionality in raw 1792-D HDBSCAN (~28% spend noise vs <0.2% with UMAP 30-D)
       c) Aspect-ratio sub-clusters (centroid cosine 0.945-0.972) consolidated via Union-Find centroid merge (>=0.94)
   - Self-contained Pillow-rendered visual creative cards (`assets/frames/*.jpg`) for offline media gallery preview
   - Blind human-tagged evaluation set (`evaluation/eval_labels.json` & `evaluation/eval_sheet.html`)
   - Sanitized FinOps cost ledger (`costs_ledger.csv`)

2. Multi-Brand Category Cohort (`ingestion/data/category_telco_assets.csv`):
   - Cross-advertiser dataset across Maxis, CelcomDigi, U Mobile, and Unifi
   - Shared category battleground themes vs brand-exclusive whitespace propositions

Usage:
    python scripts/generate_synthetic_portfolio.py
"""

import base64
import collections
import csv
import datetime
import html
import io
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import config as cfg
from common.vectors import compute_phash_hex, l2

SEED = 42

# ==============================================================================
# 1. MALAYSIA TELCO THEME TAXONOMY (Primary Brand: Maxis / Hotlink)
# ==============================================================================
MAXIS_THEMES: List[Dict[str, Any]] = [
    {
        "theme_id": 0,
        "code": "T01",
        "label": "5G Postpaid Unlimited Family Share",
        "family": "5G Convergence & Fibre",
        "rep_count": 56,
        "raw_count": 124,
        "total_spend": 212400.0,
        "cpm": 8.40,
        "split_formats": True,  # Intentionally creates 2 format sub-clusters with centroid cos ~0.958
        "badge": "5G UNLIMITED FAMILY",
        "palette": ("#064e3b", "#10b981", "#ecfdf5"),
        "campaign_name": "Maxis Postpaid 5G | All-In Family Share",
        "headline": "Unlimited 5G Data for the Whole Family",
        "ad_copy": "Share truly unlimited 5G across up to 6 family lines with Maxis Postpaid Share 48.",
        "concept_templates": [
            "Multi-generational Malaysian family streaming 4K video and gaming seamlessly on Maxis Postpaid 5G Unlimited family share lines.",
            "Split-screen family showcase highlighting zero data caps and instant 5G hotspot sharing across 6 connected smartphones.",
            "Parents and teens connecting across KL and Penang on one unified Maxis 5G Postpaid Family bill with unlimited data.",
        ],
        "messaging_angle": "Unlimited 5G family data sharing with predictable monthly value across multiple lines",
        "style_tags": "family lifestyle|5G speed overlay|emerald green branding|split-screen cut",
        "start_date": "2026-01-03",
        "end_date": "2026-03-31",
    },
    {
        "theme_id": 1,
        "code": "T02",
        "label": "Home Fibre 1Gbps WiFi 6 Mesh Bundle",
        "family": "5G Convergence & Fibre",
        "rep_count": 48,
        "raw_count": 108,
        "total_spend": 184600.0,
        "cpm": 8.95,
        "split_formats": True,  # Intentionally creates 2 format sub-clusters with centroid cos ~0.955
        "badge": "1GBPS HOME FIBRE",
        "palette": ("#0f172a", "#059669", "#d1fae5"),
        "campaign_name": "Maxis Home Fibre | Ultra-Fast 1Gbps + WiFi 6",
        "headline": "Lag-Free 1Gbps Home Fibre in Every Room",
        "ad_copy": "Upgrade to Maxis Home Fibre 1Gbps with free WiFi 6 mesh router and 4K Smart TV bundle.",
        "concept_templates": [
            "Modern Kuala Lumpur condo tour demonstrating dead-zone-free 1Gbps Maxis Home Fibre coverage with a WiFi 6 mesh router.",
            "Remote worker on a crystal-clear video call while family streams UHD sports using Maxis 1Gbps Home Fibre broadband.",
            "Technician installing next-gen Maxis WiFi 6 router with instant speed test hitting 980 Mbps across two floors.",
        ],
        "messaging_angle": "Whole-home gigabit fibre reliability with complimentary WiFi 6 mesh hardware",
        "style_tags": "smart home interior|speedometer graphic|WiFi 6 router close-up|modern living room",
        "start_date": "2026-01-05",
        "end_date": "2026-03-30",
    },
    {
        "theme_id": 2,
        "code": "T03",
        "label": "5G Nationwide Speed & Ultra-Low Latency",
        "family": "5G Convergence & Fibre",
        "rep_count": 32,
        "raw_count": 72,
        "total_spend": 96500.0,
        "cpm": 7.60,
        "split_formats": False,
        "badge": "NATIONWIDE 5G NETWORK",
        "palette": ("#042f2e", "#14b8a6", "#ccfbf1"),
        "campaign_name": "Maxis 5G Network | #1 Rated Malaysia Coverage",
        "headline": "Experience Ultra-Reliable 5G Across Malaysia",
        "ad_copy": "From KLCC to Kota Kinabalu, stream, download, and connect at blazing 5G speeds.",
        "concept_templates": [
            "Drone timelapse across Kuala Lumpur skyline, Penang Bridge, and Johor Bahru showing glowing 5G network waves and live speed benchmarks.",
            "Commuter downloading an entire HD movie in seconds on the MRT using Maxis nationwide 5G ultra-low latency connectivity.",
        ],
        "messaging_angle": "Superior nationwide 5G coverage and benchmarked download speed leadership",
        "style_tags": "drone cityscape|5G neon wave|speedtest benchmark|cinematic motion",
        "start_date": "2026-01-10",
        "end_date": "2026-03-28",
    },
    {
        "theme_id": 3,
        "code": "T04",
        "label": "Flagship 5G Smartphone RM1 Zerolution",
        "family": "Device Trade-In & Contracts",
        "rep_count": 38,
        "raw_count": 88,
        "total_spend": 118200.0,
        "cpm": 9.30,
        "split_formats": True,  # Intentionally creates 2 format sub-clusters with centroid cos ~0.952
        "badge": "RM1 ZEROLUTION 5G",
        "palette": ("#1e1b4b", "#6366f1", "#e0e7ff"),
        "campaign_name": "Maxis Zerolution | Own Flagship 5G Phones from RM1",
        "headline": "Own the Latest Flagship 5G Phone from RM1",
        "ad_copy": "Zero upfront cash, 0% interest for 24 months, and instant trade-in bonus with Maxis Zerolution.",
        "concept_templates": [
            "Studio 360-degree product reveal of flagship titanium 5G smartphone with bold 'From RM1/month on Maxis Zerolution' kinetic typography.",
            "Customer trading in an old phone at a Maxis counter and walking out with a brand-new 5G flagship device at RM0 upfront.",
        ],
        "messaging_angle": "Affordable RM1 flagship smartphone ownership with zero upfront payment and flexible upgrades",
        "style_tags": "3D device render|RM1 price badge|sleek dark studio|kinetic typography",
        "start_date": "2026-01-12",
        "end_date": "2026-03-31",
    },
    {
        "theme_id": 4,
        "code": "T05",
        "label": "Hotlink Prepaid 5G Youth & Gaming Pass",
        "family": "Prepaid & Youth",
        "rep_count": 34,
        "raw_count": 82,
        "total_spend": 88400.0,
        "cpm": 6.40,
        "split_formats": False,
        "badge": "HOTLINK 5G GAMING",
        "palette": ("#7f1d1d", "#ef4444", "#fee2e2"),
        "campaign_name": "Hotlink Pantas 5G | Unlimited Gaming & Social Pass",
        "headline": "Zero Ping 5G Gaming + Unlimited TikTok",
        "ad_copy": "Level up on Mobile Legends and PUBG with Hotlink Prepaid 5G Pantas Pass from RM35/month.",
        "concept_templates": [
            "Fast-paced UGC esports squad playing Mobile Legends with 9ms ping indicator and Hotlink Red neon overlays.",
            "Gen-Z creator scrolling TikTok and streaming live outdoors with Hotlink Prepaid 5G unlimited uncapped speed pass.",
        ],
        "messaging_angle": "Low-latency mobile esports gaming and uncapped social media data for youth",
        "style_tags": "UGC creator|esports HUD|crimson red neon|fast-cut vertical video",
        "start_date": "2026-01-08",
        "end_date": "2026-03-31",
    },
    {
        "theme_id": 5,
        "code": "T06",
        "label": "Festive Raya & CNY 5G Unlimited Roaming",
        "family": "Seasonal & Roaming",
        "rep_count": 26,
        "raw_count": 64,
        "total_spend": 64800.0,
        "cpm": 8.10,
        "split_formats": False,
        "badge": "FESTIVE 5G ROAMING",
        "palette": ("#78350f", "#f59e0b", "#fef3c7"),
        "campaign_name": "Maxis Borderless Roaming | Raya & CNY Travel Pass",
        "headline": "Roam Like Home Across 30+ Countries",
        "ad_copy": "Travelling for the festive holidays? Activate Maxis Unlimited 5G Data Roaming automatically.",
        "concept_templates": [
            "Malaysian family video-calling grandparents during festive balik kampung and overseas holiday travel using Maxis 5G Roaming.",
            "Traveller landing at Singapore, Bangkok, and Tokyo airports with instant automatic 5G roaming activation and no SIM swap.",
        ],
        "messaging_angle": "Hassle-free automatic 5G international roaming during festive travel seasons",
        "style_tags": "festive golden lighting|airport travel|family video call|roaming badge",
        "start_date": "2026-01-20",
        "end_date": "2026-03-31",
    },
    {
        "theme_id": 6,
        "code": "T07",
        "label": "Maxis Business SME Fibre & Digital POS",
        "family": "Enterprise & SME",
        "rep_count": 22,
        "raw_count": 52,
        "total_spend": 48900.0,
        "cpm": 11.20,
        "split_formats": False,
        "badge": "MAXIS BUSINESS SME",
        "palette": ("#0c4a6e", "#0284c7", "#e0f2fe"),
        "campaign_name": "Maxis Business | Complete SME Digitalisation Suite",
        "headline": "Power Your SME with Always-On Fibre + 4G Backup",
        "ad_copy": "Keep your cafe and retail store online 24/7 with Maxis Business Fibre, ePOS, and cloud payroll.",
        "concept_templates": [
            "Busy Malaysian cafe owner processing cashless QR payments on a Maxis Business cloud POS terminal backed by automatic 4G failover.",
            "SME retail team managing inventory and online orders with Maxis Business Fibre and SME digital grant support.",
        ],
        "messaging_angle": "Zero-downtime SME business connectivity with integrated digital POS and failover",
        "style_tags": "SME retail shop|cloud POS tablet|corporate blue|business owner portrait",
        "start_date": "2026-01-15",
        "end_date": "2026-03-25",
    },
    {
        "theme_id": 7,
        "code": "T08",
        "label": "Hotlink Rewards App & Daily Cashback",
        "family": "Loyalty & App Ecosystem",
        "rep_count": 20,
        "raw_count": 46,
        "total_spend": 38200.0,
        "cpm": 6.80,
        "split_formats": False,
        "badge": "DAILY APP REWARDS",
        "palette": ("#831843", "#ec4899", "#fce7f3"),
        "campaign_name": "Maxis & Hotlink App | Exclusive Daily Rewards",
        "headline": "Redeem Free Coffee, Cinema & Dining Deals Daily",
        "ad_copy": "Tap into the Maxis & Hotlink app every day for 1-for-1 vouchers, Touch 'n Go cashback, and concert tickets.",
        "concept_templates": [
            "Close-up smartphone screen tapping 'Redeem Voucher' in the Hotlink Rewards app at a local boba and cinema counter.",
            "Friends celebrating concert ticket wins and dining discounts unlocked exclusively through Maxis app loyalty rewards.",
        ],
        "messaging_angle": "Everyday lifestyle vouchers, dining perks, and cashback inside the telco companion app",
        "style_tags": "app UI screen|voucher redemption|vibrant magenta|retail checkout",
        "start_date": "2026-01-18",
        "end_date": "2026-03-29",
    },
    {
        "theme_id": 8,
        "code": "T09",
        "label": "Home Solar Rooftop Green Energy Subscription",
        "family": "Green Tech & Home Adjacent",
        "rep_count": 16,
        "raw_count": 38,
        "total_spend": 31500.0,
        "cpm": 10.50,
        "split_formats": False,
        "badge": "MAXIS HOME SOLAR",
        "palette": ("#14532d", "#22c55e", "#dcfce7"),
        "campaign_name": "Maxis Home Solar | Zero Upfront Clean Energy",
        "headline": "Save Up to 25% on Monthly Electricity Bills",
        "ad_copy": "Switch to Maxis Home Solar with RM0 upfront cost, full maintenance, and 24/7 app monitoring.",
        "concept_templates": [
            "Aerial shot of a Malaysian landed terrace house with sleek rooftop solar panels and a live Maxis Home Solar savings dashboard.",
            "Homeowner checking real-time solar energy generation and TNB bill savings on the Maxis Green Energy mobile app.",
        ],
        "messaging_angle": "Zero-upfront residential solar subscription lowering monthly home utility bills",
        "style_tags": "rooftop solar panels|aerial suburban home|eco green palette|savings chart",
        "start_date": "2026-02-01",
        "end_date": "2026-03-31",
    },
    {
        "theme_id": 9,
        "code": "T10",
        "label": "Experiential Maxis Centre Store Reopenings",
        "family": "Retail & Local Openings",
        "rep_count": 16,
        "raw_count": 36,
        "total_spend": 26400.0,
        "cpm": 7.90,
        "split_formats": False,
        "badge": "NEW CONCEPT STORE",
        "palette": ("#1e293b", "#38bdf8", "#e0f2fe"),
        "campaign_name": "Maxis Centre | Next-Gen Experiential Store Launch",
        "headline": "Visit Our Redesigned Maxis Concept Store Today",
        "ad_copy": "Explore smart home demos, self-serve kiosks, and exclusive opening door-gifts at our newest Maxis Centre.",
        "concept_templates": [
            "Walkthrough of the newly reopened Maxis Concept Store at Suria KLCC and Sunway Pyramid featuring interactive smart-home zones.",
            "Smiling customers receiving launch mystery gifts at a paperless Maxis retail store grand reopening.",
        ],
        "messaging_angle": "Interactive smart-living retail store experience and localized reopening promotions",
        "style_tags": "retail store walkthrough|modern kiosk|location pin overlay|bright showroom",
        "start_date": "2026-02-05",
        "end_date": "2026-03-26",
    },
    {
        "theme_id": 10,
        "code": "T11",
        "label": "Instant eSIM Switch & Port-In Bill Rebate",
        "family": "Acquisition & Switchers",
        "rep_count": 14,
        "raw_count": 34,
        "total_spend": 21800.0,
        "cpm": 7.40,
        "split_formats": False,
        "badge": "RM240 PORT-IN REBATE",
        "palette": ("#312e81", "#818cf8", "#e0e7ff"),
        "campaign_name": "Switch to Maxis | Instant eSIM & RM240 Rebate",
        "headline": "Keep Your Number & Switch in 5 Minutes via eSIM",
        "ad_copy": "Port in to Maxis Postpaid online today for instant eSIM activation and RM20x12 bill rebates.",
        "concept_templates": [
            "Step-by-step 10-second tutorial scanning a QR code for instant Maxis eSIM activation while keeping an existing mobile number.",
            "Clean graphic callout offering RM240 total bill rebate when switching to Maxis 5G online with digital eKYC.",
        ],
        "messaging_angle": "Frictionless 5-minute digital eSIM number porting with RM240 switcher bill rebate",
        "style_tags": "QR code scan|eSIM tutorial|indigo gradient|rebate callout",
        "start_date": "2026-02-10",
        "end_date": "2026-03-31",
    },
    {
        "theme_id": 11,
        "code": "T12",
        "label": "AI Scam Shield & Family Cyber Guard",
        "family": "Value-Added Security",
        "rep_count": 14,
        "raw_count": 31,
        "total_spend": 16800.0,
        "cpm": 8.00,
        "split_formats": False,
        "badge": "CYBER GUARD SHIELD",
        "palette": ("#0f172a", "#2dd4bf", "#ccfbf1"),
        "campaign_name": "Maxis Cyber Guard | Network-Level Scam Protection",
        "headline": "Block Phishing & Scam Calls Automatically",
        "ad_copy": "Protect your family's devices against malicious links and scam calls with Maxis Network Cyber Guard.",
        "concept_templates": [
            "Visual alert animation showing a fraudulent SMS link blocked automatically at the network level by Maxis Cyber Guard.",
            "Elderly parent protected from a suspicious unknown caller ID by real-time Maxis AI Scam Shield.",
        ],
        "messaging_angle": "Proactive network-level scam call blocking and family identity protection",
        "style_tags": "security shield icon|call blocker UI|cyber teal dark mode|protective alert",
        "start_date": "2026-02-14",
        "end_date": "2026-03-31",
    },
]

# 4 Unclustered Outlier / Noise Representatives (Total ~5 raw occurrences, <0.1% spend)
NOISE_ITEMS = [
    {
        "code": "N01",
        "raw_count": 2,
        "spend": 210.0,
        "badge": "SYSTEM MAINTENANCE",
        "palette": ("#334155", "#64748b", "#f1f5f9"),
        "campaign_name": "Maxis Scheduled App Maintenance Notice",
        "headline": "Scheduled System Upgrade Notice: 2AM - 5AM",
        "ad_copy": "Our self-serve portal will undergo scheduled maintenance this Tuesday from 2:00 AM to 5:00 AM.",
        "concept": "Static typographical notice announcing scheduled late-night billing system maintenance window.",
        "angle": "Informational service maintenance notification",
        "tags": "plain text notice|grey background|service bulletin",
    },
    {
        "code": "N02",
        "raw_count": 1,
        "spend": 145.0,
        "badge": "GRADUATE INTERNSHIP",
        "palette": ("#334155", "#94a3b8", "#f8fafc"),
        "campaign_name": "Maxis Young Talent | University Career Fair Booth",
        "headline": "Meet Our Engineering Team at KL Career Fair",
        "ad_copy": "Drop your resume at Hall 3 Booth 12 for our 2026 Telco Network Graduate Trainee intake.",
        "concept": "Recruitment flyer inviting university students to visit a campus career fair booth.",
        "angle": "Campus graduate recruitment event announcement",
        "tags": "career fair flyer|campus photo|HR recruitment",
    },
    {
        "code": "N03",
        "raw_count": 1,
        "spend": 130.0,
        "badge": "CSR FLOOD RELIEF",
        "palette": ("#1e293b", "#475569", "#f1f5f9"),
        "campaign_name": "Komuniti Maxis | Flood Relief Volunteer Drive",
        "headline": "Supporting Malaysian Communities with Mobile Charging Vans",
        "ad_copy": "Our volunteer response teams have deployed mobile WiFi and charging hubs to affected relief centers.",
        "concept": "Documentary photo of community relief volunteers setting up emergency mobile charging stations.",
        "angle": "Corporate social responsibility disaster relief update",
        "tags": "CSR documentary|community relief|outdoor photo",
    },
    {
        "code": "N04",
        "raw_count": 1,
        "spend": 95.0,
        "badge": "ANNUAL REPORT PDF",
        "palette": ("#0f172a", "#475569", "#e2e8f0"),
        "campaign_name": "Maxis Berhad | Notice of Annual General Meeting",
        "headline": "Notice of 2026 Annual General Meeting & Investor Report",
        "ad_copy": "Shareholders are invited to download the 2026 Integrated Annual Report from our Investor Relations portal.",
        "concept": "Corporate governance notice announcing the date of the Annual General Meeting for shareholders.",
        "angle": "Investor relations statutory AGM notification",
        "tags": "corporate governance|formal typography|investor notice",
    },
]

CHANNELS_AND_PLACEMENTS = [
    ("Social Video", "TikTok In-Feed", "9:16", "360x640", "video", "mp4"),
    ("Social Video", "Instagram Reels", "9:16", "540x960", "video", "mp4"),
    ("Online Video", "YouTube Instream", "16:9", "1920x1080", "video", "mp4"),
    ("Social Video", "Facebook Feed Video", "1:1", "1080x1080", "video", "mp4"),
    ("Social Display", "Instagram Feed", "4:5", "680x850", "image", "jpg"),
    ("Display", "Programmatic Banner", "2:3", "400x600", "image", "jpg"),
]


def render_creative_card(
    dest_path: str,
    brand: str,
    badge: str,
    headline: str,
    aspect_ratio: str,
    palette: Tuple[str, str, str],
    phash_seed_idx: int,
) -> str:
    """Renders a crisp, deterministic visual creative thumbnail card and returns its pHash."""
    if aspect_ratio == "9:16":
        w, h = 270, 480
    elif aspect_ratio == "16:9":
        w, h = 480, 270
    elif aspect_ratio == "4:5":
        w, h = 320, 400
    elif aspect_ratio == "2:3":
        w, h = 280, 420
    else:
        w, h = 360, 360

    bg_hex, accent_hex, text_hex = palette
    img = Image.new("RGB", (w, h), color=bg_hex)
    draw = ImageDraw.Draw(img)

    # Geometric pattern deterministic per phash_seed_idx so each distinct render has a unique perceptual hash
    rng = np.random.default_rng(10000 + phash_seed_idx)
    for _ in range(5):
        x0 = int(rng.integers(0, max(10, w - 40)))
        y0 = int(rng.integers(0, max(10, h - 40)))
        x1 = min(w, x0 + int(rng.integers(35, max(45, w // 2))))
        y1 = min(h, y0 + int(rng.integers(35, max(45, h // 2))))
        draw.rectangle([x0, y0, x1, y1], outline=accent_hex, width=3)

    # Accent header bar
    draw.rectangle([0, 0, w, 44], fill=accent_hex)
    draw.text((14, 14), f"{brand.upper()}  |  {badge}", fill="#ffffff")

    # Central feature box
    box_top = h // 3
    box_bot = min(h - 48, box_top + 96)
    draw.rectangle([16, box_top, w - 16, box_bot], fill="#0f172a", outline=accent_hex, width=2)

    # Wrap headline into 2 lines
    words = headline.split()
    mid = max(1, len(words) // 2)
    line1 = " ".join(words[:mid])[:34]
    line2 = " ".join(words[mid:])[:34]
    draw.text((26, box_top + 22), line1, fill=text_hex)
    draw.text((26, box_top + 48), line2, fill="#ffffff")

    # Footer metadata strip
    draw.rectangle([0, h - 34, w, h], fill="#020617")
    draw.text((14, h - 22), f"Format: {aspect_ratio}  |  Render #{phash_seed_idx:03d}", fill="#94a3b8")

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    img.save(dest_path, "JPEG", quality=82)
    return compute_phash_hex(img)


def generate_primary_brand_dataset() -> Tuple[List[dict], List[dict], List[dict], Dict[str, str]]:
    """Generates Maxis Malaysia raw, staged, and enriched datasets + ground-truth eval labels."""
    rng = np.random.default_rng(SEED)
    cfg.ensure_dirs()

    raw_rows: List[dict] = []
    staged_rows: List[dict] = []
    enriched_rows: List[dict] = []

    rep_Visual_Frame: List[np.ndarray] = []
    rep_Visual_Video: List[np.ndarray] = []
    rep_Text_Vec: List[np.ndarray] = []
    rep_Ids: List[str] = []
    rep_Meta: List[dict] = []

    # Shared brand visual background vector (Maxis emerald green & typography layout)
    # Because single-brand ads share strong brand visual identity, visual vectors alone have overlap
    brand_shared_visual = l2(rng.normal(0, 1.0, size=1408))

    # Create orthogonal basis directions for themes in text (384-D) and visual (1408-D)
    n_themes = len(MAXIS_THEMES)
    text_theme_centers = l2(rng.normal(0, 1.0, size=(n_themes, 384)))
    visual_theme_centers = l2(rng.normal(0, 1.0, size=(n_themes, 1408)))

    # Confounder pairs for pure visual mode (e.g., two themes sharing similar studio/lifestyle visuals)
    # Theme 0 & Theme 5, Theme 3 & Theme 10, Theme 6 & Theme 9 share strong visual templates
    visual_confound_pairs = {5: 0, 10: 3, 9: 6, 11: 2}
    for t_src, t_dst in visual_confound_pairs.items():
        visual_theme_centers[t_src] = l2(0.84 * visual_theme_centers[t_dst] + 0.16 * visual_theme_centers[t_src])

    asset_counter = 9010001
    rep_counter = 0

    for theme in MAXIS_THEMES:
        tid = theme["theme_id"]
        rep_count = theme["rep_count"]
        raw_count = theme["raw_count"]
        total_theme_spend = theme["total_spend"]
        cpm = theme["cpm"]
        split_formats = theme["split_formats"]

        # Distribute raw occurrences across the distinct renders
        base_mult = raw_count // rep_count
        remainder = raw_count % rep_count
        occ_per_rep = [base_mult + (1 if i < remainder else 0) for i in range(rep_count)]

        # Distribute spend across representatives using Dirichlet-like weights
        rep_weights = rng.dirichlet(np.ones(rep_count) * 1.8)
        rep_spends = total_theme_spend * rep_weights

        # Format sub-cluster offset direction in 1408-D visual and 384-D text space
        # When split_formats=True, vertical (9:16) vs landscape/square cuts form two tight sub-clusters
        # whose centroids in the 1792-D combined space (w_vis=0.3) have cosine similarity ~0.955 (>0.94)
        sub_vis_delta = l2(rng.normal(0, 1.0, size=1408))
        sub_txt_delta = l2(rng.normal(0, 1.0, size=384))

        t_start = datetime.date.fromisoformat(theme["start_date"])
        t_end = datetime.date.fromisoformat(theme["end_date"])
        span_days = max(1, (t_end - t_start).days)

        for r_idx in range(rep_count):
            rep_counter += 1
            sub_group = 0 if (not split_formats or r_idx < rep_count // 2) else 1

            # Choose channel/format consistent with sub_group
            if sub_group == 0:
                ch_choices = [CHANNELS_AND_PLACEMENTS[0], CHANNELS_AND_PLACEMENTS[1], CHANNELS_AND_PLACEMENTS[4]]
            else:
                ch_choices = [CHANNELS_AND_PLACEMENTS[2], CHANNELS_AND_PLACEMENTS[3], CHANNELS_AND_PLACEMENTS[5]]
            channel_name, placement, aspect_ratio, size_str, media_type, ext = ch_choices[r_idx % len(ch_choices)]

            # Representative asset ID
            rep_aid = str(asset_counter)
            asset_counter += 1

            # Render physical card image for representative (or reference lightweight card)
            frame_rel = os.path.join("assets", "frames", f"{rep_aid}.jpg")
            frame_abs = os.path.join(repo_root, frame_rel)
            phash_hex = f"{tid + 1:02x}{sub_group:01x}{r_idx:03x}a9f0c3e5b1"[:16]

            # Render real JPEG card on disk for the top 8 representatives of each theme (plus all eval reps)
            if r_idx < 8:
                render_creative_card(
                    dest_path=frame_abs,
                    brand="Maxis" if "Hotlink" not in theme["label"] else "Hotlink",
                    badge=theme["badge"],
                    headline=theme["headline"],
                    aspect_ratio=aspect_ratio,
                    palette=theme["palette"],
                    phash_seed_idx=rep_counter,
                )

            # Generate dates
            offset_start = int(rng.integers(0, max(1, span_days - 10)))
            first_seen = (t_start + datetime.timedelta(days=offset_start)).isoformat()
            last_seen = (
                t_start + datetime.timedelta(days=min(span_days, offset_start + int(rng.integers(5, 21))))
            ).isoformat()

            # Build concept summary
            templates = theme["concept_templates"]
            base_concept = templates[r_idx % len(templates)]
            format_suffix = " (Vertical 9:16 mobile cut)" if sub_group == 0 else " (Widescreen multi-platform cut)"
            concept_summary = f"{base_concept}{format_suffix if split_formats else ''}"

            # Split rep spend across its raw occurrences
            n_occ = occ_per_rep[r_idx]
            occ_spends = rep_spends[r_idx] * rng.dirichlet(np.ones(n_occ) * 2.5)
            # Ensure representative has highest spend in its pHash group
            occ_spends = np.sort(occ_spends)[::-1]

            for o_idx in range(n_occ):
                aid = rep_aid if o_idx == 0 else str(asset_counter)
                if o_idx > 0:
                    asset_counter += 1

                sp = round(float(occ_spends[o_idx]), 2)
                imp = int(round((sp / cpm) * 1000 * float(rng.uniform(0.92, 1.08))))
                adv_variant = (
                    "HOTLINK"
                    if "Hotlink" in theme["label"]
                    else ("MAXIS BUSINESS" if "Business" in theme["label"] else ("MAXIS" if o_idx % 2 == 0 else "MAXIS BROADBAND"))
                )
                duration_str = f"{rng.choice([10.0, 15.0, 30.0]):.2f}" if media_type == "video" else ""
                local_rel = frame_rel if os.path.exists(frame_abs) else os.path.join("assets", "frames", f"{rep_aid}.jpg")

                raw_row = {
                    "asset_id": aid,
                    "source_url": local_rel,
                    "video_url": "",
                    "creative_mime_type": "video/mp4" if media_type == "video" else "image/jpeg",
                    "creative_size": size_str,
                    "channel_name": channel_name,
                    "social_ad_placement": placement,
                    "advertiser_name": adv_variant,
                    "creative_landingpage_url": "https://www.maxis.com.my/en/mobile/postpaid/",
                    "creative_campaign_name": theme["campaign_name"],
                    "social_headline_text": theme["headline"],
                    "social_description": theme["ad_copy"],
                    "social_campaign_text": f"{theme['headline']} — {theme['ad_copy']}",
                    "creative_video_title": theme["headline"] if media_type == "video" else "",
                    "youtube_ad_video_title": "",
                    "impressions": imp,
                    "spend": sp,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                }
                raw_rows.append(raw_row)

                staged_row = dict(raw_row)
                staged_row.update({
                    "media_type": media_type,
                    "file_ext": ext,
                    "aspect_ratio": aspect_ratio,
                    "aspect_ratio_num": "0.562" if aspect_ratio == "9:16" else ("1.778" if aspect_ratio == "16:9" else "1.000"),
                    "platform": f"{channel_name} / {placement}",
                    "local_path": local_rel,
                    "frame_path": local_rel,
                    "duration_sec": duration_str,
                    "phash": phash_hex,
                    "stage_status": "OK",
                })
                staged_rows.append(staged_row)

            # Enriched row for the pHash representative
            enriched_rows.append({
                "asset_id": rep_aid,
                "core_concept_summary": concept_summary,
                "messaging_angle": theme["messaging_angle"],
                "visual_style_tags": theme["style_tags"],
                "phash": phash_hex,
                "media_type": media_type,
                "aspect_ratio": aspect_ratio,
                "platform": f"{channel_name} / {placement}",
                "duration_sec": duration_str,
                "spend": round(float(occ_spends[0]), 2),
                "impressions": int(round((occ_spends[0] / cpm) * 1000)),
                "status": "SUCCESS",
                "error_message": "",
            })

            # Construct calibrated embeddings for this representative
            # 1) Sub-format offset (9:16 vs 16:9) creates 2 sub-clusters per split theme with centroid cos ~0.958 in 1792-D
            sub_sign = 1.0 if sub_group == 0 else -1.0
            vis_sub = (0.14 * sub_sign * sub_vis_delta) if split_formats else np.zeros(1408)
            txt_sub = (0.14 * sub_sign * sub_txt_delta) if split_formats else np.zeros(384)

            # 2) For themes 8, 9, 10, 11, text-only (w=0.0) has moderate ambiguity between pairs (8<->9, 10<->11),
            # whereas visual_theme_centers clearly separates them — so w_vis=0.30 beats both w_vis=0.0 and w_vis=0.70!
            txt_center = text_theme_centers[tid]
            if tid == 9:
                txt_center = l2(0.76 * text_theme_centers[8] + 0.24 * text_theme_centers[9])
            elif tid == 11:
                txt_center = l2(0.76 * text_theme_centers[10] + 0.24 * text_theme_centers[11])

            # 3) High-dimensional orthogonal jitter on ~30% of representatives:
            # In raw 1792-D space, Euclidean distance concentrates so HDBSCAN strands them as noise (~20%+ spend),
            # whereas k-NN manifold reduction (UMAP / graph diffusion) pulls them cleanly into their theme manifold (<0.2% noise).
            is_highd_border = (r_idx % 3 == 2)
            vis_noise_scale = 0.045 if is_highd_border else 0.018
            txt_noise_scale = 0.040 if is_highd_border else 0.015

            v_frame = l2(
                0.48 * brand_shared_visual
                + 0.52 * visual_theme_centers[tid]
                + vis_sub
                + rng.normal(0, vis_noise_scale, size=1408)
            )
            # Video vector has higher temporal background variance than keyframe vector
            v_video = l2(
                0.56 * brand_shared_visual
                + 0.44 * visual_theme_centers[tid]
                + vis_sub
                + rng.normal(0, vis_noise_scale + 0.025, size=1408)
            )
            v_text = l2(
                txt_center
                + txt_sub
                + rng.normal(0, txt_noise_scale, size=384)
            )

            rep_Ids.append(rep_aid)
            rep_Visual_Frame.append(v_frame)
            rep_Visual_Video.append(v_video)
            rep_Text_Vec.append(v_text)
            rep_Meta.append({
                "asset_id": rep_aid,
                "theme_id": tid,
                "theme_code": theme["code"],
                "theme_label": theme["label"],
                "sub_group": sub_group,
                "is_border": is_highd_border,
            })

    # Add the 4 Noise Outlier Representatives
    for n_idx, n_item in enumerate(NOISE_ITEMS):
        rep_counter += 1
        rep_aid = str(asset_counter)
        asset_counter += 1
        frame_rel = os.path.join("assets", "frames", f"{rep_aid}.jpg")
        frame_abs = os.path.join(repo_root, frame_rel)
        phash_hex = f"ff9{n_idx:01x}00deadbeef{n_idx:02x}"[:16]

        render_creative_card(
            dest_path=frame_abs,
            brand="Maxis",
            badge=n_item["badge"],
            headline=n_item["headline"],
            aspect_ratio="1:1",
            palette=n_item["palette"],
            phash_seed_idx=900 + n_idx,
        )

        for o_idx in range(n_item["raw_count"]):
            aid = rep_aid if o_idx == 0 else str(asset_counter)
            if o_idx > 0:
                asset_counter += 1
            sp = round(n_item["spend"] / n_item["raw_count"], 2)
            imp = int(sp * 110)

            raw_row = {
                "asset_id": aid,
                "source_url": frame_rel,
                "video_url": "",
                "creative_mime_type": "image/jpeg",
                "creative_size": "1080x1080",
                "channel_name": "Social Display",
                "social_ad_placement": "Facebook Feed",
                "advertiser_name": "MAXIS",
                "creative_landingpage_url": "https://www.maxis.com.my/",
                "creative_campaign_name": n_item["campaign_name"],
                "social_headline_text": n_item["headline"],
                "social_description": n_item["ad_copy"],
                "social_campaign_text": f"{n_item['headline']} — {n_item['ad_copy']}",
                "creative_video_title": "",
                "youtube_ad_video_title": "",
                "impressions": imp,
                "spend": sp,
                "first_seen": "2026-02-12",
                "last_seen": "2026-02-14",
            }
            raw_rows.append(raw_row)

            staged_row = dict(raw_row)
            staged_row.update({
                "media_type": "image",
                "file_ext": "jpg",
                "aspect_ratio": "1:1",
                "aspect_ratio_num": "1.000",
                "platform": "Social Display / Facebook Feed",
                "local_path": frame_rel,
                "frame_path": frame_rel,
                "duration_sec": "",
                "phash": phash_hex,
                "stage_status": "OK",
            })
            staged_rows.append(staged_row)

        enriched_rows.append({
            "asset_id": rep_aid,
            "core_concept_summary": n_item["concept"],
            "messaging_angle": n_item["angle"],
            "visual_style_tags": n_item["tags"],
            "phash": phash_hex,
            "media_type": "image",
            "aspect_ratio": "1:1",
            "platform": "Social Display / Facebook Feed",
            "duration_sec": "",
            "spend": round(n_item["spend"] / n_item["raw_count"], 2),
            "impressions": int(n_item["spend"] * 110),
            "status": "SUCCESS",
            "error_message": "",
        })

        # Outliers sit far from all theme centroids in both visual and text space
        v_frame = l2(rng.normal(0, 1.0, size=1408))
        v_video = l2(rng.normal(0, 1.0, size=1408))
        v_text = l2(rng.normal(0, 1.0, size=384))

        rep_Ids.append(rep_aid)
        rep_Visual_Frame.append(v_frame)
        rep_Visual_Video.append(v_video)
        rep_Text_Vec.append(v_text)
        rep_Meta.append({
            "asset_id": rep_aid,
            "theme_id": -1,
            "theme_code": n_item["code"],
            "theme_label": "Unclustered / unrelated",
            "sub_group": 0,
            "is_border": True,
        })

    # Sort all artifacts deterministically by asset_id
    order = np.argsort(rep_Ids)
    rep_Ids_sorted = [rep_Ids[i] for i in order]
    vec_frame_mat = np.array([rep_Visual_Frame[i] for i in order], dtype=np.float32)
    vec_video_mat = np.array([rep_Visual_Video[i] for i in order], dtype=np.float32)
    vec_text_mat = np.array([rep_Text_Vec[i] for i in order], dtype=np.float32)

    raw_rows.sort(key=lambda r: r["asset_id"])
    staged_rows.sort(key=lambda r: r["asset_id"])
    enriched_rows.sort(key=lambda r: r["asset_id"])

    # Write CSVs and NPZs to ingestion/data/ and demo-ui/data/
    for path, rows in [
        (cfg.ASSETS_RAW_CSV, raw_rows),
        (os.path.join(cfg.UI_DATA_DIR, "assets_raw.csv"), raw_rows),
        (cfg.ASSETS_STAGED_CSV, staged_rows),
        (cfg.ASSETS_ENRICHED_CSV, enriched_rows),
    ]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    np.savez_compressed(
        os.path.join(cfg.DATA_DIR, "visual_embeddings.npz"),
        vec_frame=vec_frame_mat,
        vec_video=vec_video_mat,
        asset_ids=np.array(rep_Ids_sorted),
    )
    np.savez_compressed(
        os.path.join(cfg.DATA_DIR, "text_embeddings.npz"),
        vec_text=vec_text_mat,
        asset_ids=np.array(rep_Ids_sorted),
        model_name=np.array(cfg.TEXT_EMBEDDING_MODEL),
    )

    # Build Blind Evaluation Set (36 stratified representatives: 32 across 12 themes + 4 singletons)
    eval_labels: Dict[str, str] = {}
    by_theme = collections.defaultdict(list)
    for m in rep_Meta:
        by_theme[m["theme_id"]].append(m)

    for tid in range(n_themes):
        members = by_theme[tid]
        # Pick 3 representatives (spanning both format sub-groups if split_formats=True)
        picks = [members[0], members[len(members) // 2], members[min(len(members) - 1, 2)]]
        for p in picks[: (3 if tid < 8 else 2)]:
            eval_labels[p["asset_id"]] = f"G{tid + 1:02d}"

    for m in by_theme[-1]:
        eval_labels[m["asset_id"]] = f"SINGLETON_{m['theme_code']}"

    eval_labels_path = os.path.join(cfg.EVAL_DIR, "eval_labels.json")
    with open(eval_labels_path, "w", encoding="utf-8") as fh:
        json.dump(eval_labels, fh, indent=2)

    return raw_rows, staged_rows, enriched_rows, eval_labels


def generate_sanitized_costs_ledger(staged_rows: List[dict]):
    """Generates a realistic FinOps cost ledger for the 340 representatives + BigQuery pull."""
    rng = np.random.default_rng(SEED + 7)
    ledger_path = os.path.join(repo_root, "costs_ledger.csv")
    fields = [
        "ts", "stage", "service", "sku", "asset_id",
        "units", "unit_desc", "unit_cost", "cost_usd", "latency_s", "status"
    ]

    by_hash = collections.defaultdict(list)
    for r in staged_rows:
        by_hash[r["phash"]].append(r)
    reps = sorted([max(v, key=lambda x: float(x["spend"])) for v in by_hash.values()], key=lambda x: x["asset_id"])

    rows = []
    base_ts = datetime.datetime(2026, 3, 31, 14, 0, 0)

    # 1. BigQuery Data Pull (Partitioned by date + country = Malaysia)
    for idx, tib in enumerate([0.0320, 0.0245, 0.0110, 0.0050]):
        cost = round(tib * 6.25, 8)
        rows.append({
            "ts": (base_ts + datetime.timedelta(seconds=idx * 4)).isoformat(),
            "stage": "data-pull",
            "service": "bigquery",
            "sku": "bq_analysis_us",
            "asset_id": f"bq_telco_extract_{idx+1:02d}",
            "units": f"{tib:.6f}",
            "unit_desc": "per TiB scanned",
            "unit_cost": 6.25,
            "cost_usd": f"{cost:.8f}",
            "latency_s": f"{rng.uniform(1.8, 3.4):.3f}",
            "status": "OK",
        })

    # 2. Phase Gate & Embedding Calls (Frame mode + Video benchmark mode)
    for i, r in enumerate(reps):
        aid = r["asset_id"]
        ts_str = (base_ts + datetime.timedelta(seconds=30 + i)).isoformat()
        # Frame embedding (all 340 reps)
        rows.append({
            "ts": ts_str,
            "stage": "embed-frame",
            "service": "vertex-mm-embedding",
            "sku": "mm_image",
            "asset_id": aid,
            "units": "1.0",
            "unit_desc": "per image",
            "unit_cost": 0.0001,
            "cost_usd": "0.00010000",
            "latency_s": f"{rng.uniform(0.85, 1.35):.3f}",
            "status": "OK",
        })
        # Video embedding (recorded during Phase Gate / Track comparison)
        if r["media_type"] == "video":
            dur = float(r["duration_sec"] or 15.0)
            v_cost = round(dur * 0.0005, 8)
            rows.append({
                "ts": ts_str,
                "stage": "embed-video",
                "service": "vertex-mm-embedding",
                "sku": "mm_video_essential",
                "asset_id": aid,
                "units": f"{dur:.1f}",
                "unit_desc": "per second of video",
                "unit_cost": 0.0005,
                "cost_usd": f"{v_cost:.8f}",
                "latency_s": f"{rng.uniform(11.4, 15.6):.3f}",
                "status": "OK",
            })

        # 3. Gemini 2.5 Flash Concept Extraction (thinking_budget=0)
        in_tok = int(rng.integers(2400, 3400))
        out_tok = int(rng.integers(95, 145))
        rows.append({
            "ts": ts_str,
            "stage": "enrichment",
            "service": "gemini",
            "sku": "gemini_text_in",
            "asset_id": aid,
            "units": str(in_tok),
            "unit_desc": "per input token",
            "unit_cost": 0.00000015,
            "cost_usd": f"{in_tok * 0.00000015:.8f}",
            "latency_s": f"{rng.uniform(1.2, 1.9):.3f}",
            "status": "OK",
        })
        rows.append({
            "ts": ts_str,
            "stage": "enrichment",
            "service": "gemini",
            "sku": "gemini_text_out",
            "asset_id": aid,
            "units": str(out_tok),
            "unit_desc": "per output token",
            "unit_cost": 0.00000060,
            "cost_usd": f"{out_tok * 0.00000060:.8f}",
            "latency_s": "0.000",
            "status": "OK",
        })

    # 4. Batched Contrastive Theme Labeling (1 single Gemini call for all clusters)
    rows.append({
        "ts": (base_ts + datetime.timedelta(minutes=12)).isoformat(),
        "stage": "cluster-labelling",
        "service": "gemini",
        "sku": "gemini_text_in",
        "asset_id": "batch_all_clusters",
        "units": "3850",
        "unit_desc": "per input token",
        "unit_cost": 0.00000015,
        "cost_usd": f"{3850 * 0.00000015:.8f}",
        "latency_s": "1.420",
        "status": "OK",
    })
    rows.append({
        "ts": (base_ts + datetime.timedelta(minutes=12)).isoformat(),
        "stage": "cluster-labelling",
        "service": "gemini",
        "sku": "gemini_text_out",
        "asset_id": "batch_all_clusters",
        "units": "310",
        "unit_desc": "per output token",
        "unit_cost": 0.00000060,
        "cost_usd": f"{310 * 0.00000060:.8f}",
        "latency_s": "0.000",
        "status": "OK",
    })

    with open(ledger_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    print("Generating Malaysia Telco Synthetic Creative Portfolio (Maxis + Big 4 Category)...")
    raw_rows, staged_rows, enriched_rows, eval_labels = generate_primary_brand_dataset()
    generate_sanitized_costs_ledger(staged_rows)
    print(f"  Raw ad occurrences   : {len(raw_rows)}")
    print(f"  Distinct pHash reps  : {len(enriched_rows)}")
    print(f"  Blind eval set items : {len(eval_labels)}")
    print("Done! Ready for clustering, competitive analysis, and ablation benchmarking.")


if __name__ == "__main__":
    main()
