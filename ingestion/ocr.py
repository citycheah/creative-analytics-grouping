"""Optical Character Recognition (OCR) signal extraction for creative assets.

Extracts on-screen overlay text, slogans, and promotional offers from video frames
and static images.

Engines:
1. Pytesseract (if local Tesseract-OCR is installed)
2. Gemini 2.5 Flash multimodal text extraction (fast, high accuracy on styled ad typography)
3. Metadata fallback

Usage:
    python ingestion/ocr.py --limit 20
    python ingestion/ocr.py --all
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import config as cfg
from common.logging import get_logger

logger = get_logger("ocr", stage="ingestion")


def get_tesseract_ocr():
    """Attempts to initialize local pytesseract engine."""
    try:
        import pytesseract
        # Common Windows installation locations
        win_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for p in win_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
        return pytesseract
    except Exception:
        return None


def extract_ocr_gemini(image_path: str, client=None) -> str:
    """Uses Gemini 2.5 Flash to accurately extract styled on-screen ad copy."""
    try:
        from PIL import Image
        from google.genai import types

        if client is None:
            from google.genai import Client
            client = Client(vertexai=True, project=cfg.GCP_PROJECT_ID, location=cfg.GCP_REGION)

        with Image.open(image_path) as img:
            prompt = (
                "Extract all visible on-screen text, headlines, product names, price tags, "
                "and promotional slogans visible in this advertisement image. "
                "Return only the extracted text without any commentary. If there is no visible text, return nothing."
            )
            response = client.models.generate_content(
                model=cfg.GEMINI_MODEL,
                contents=[img, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return (response.text or "").strip()
    except Exception as e:
        logger.debug(f"Gemini OCR extraction error for {image_path}: {e}")
        return ""


def extract_asset_ocr(row: dict, tesseract_engine=None, gemini_client=None) -> str:
    """Extracts OCR text from the best visual representation of the asset."""
    # Priority: frame_path for video, local_path for stills
    img_path = row.get("frame_path") if (row.get("media_type") == "video" and row.get("frame_path")) else row.get("local_path")
    if not img_path or not os.path.exists(img_path):
        # Fallback to headline text if image file missing
        return (row.get("social_headline_text") or "").strip()

    if tesseract_engine:
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                text = tesseract_engine.image_to_string(img)
                if text.strip():
                    return text.strip()
        except Exception:
            pass

    # If Gemini available
    if gemini_client:
        text = extract_ocr_gemini(img_path, gemini_client)
        if text.strip():
            return text.strip()

    return (row.get("social_headline_text") or "").strip()


def run_ocr(limit: Optional[int] = None, output_csv: Optional[str] = None):
    staged_path = cfg.ASSETS_STAGED_CSV
    if not os.path.exists(staged_path):
        logger.error(f"Missing staged assets CSV: {staged_path}")
        return

    with open(staged_path, "r", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("stage_status") == "OK"]

    # Filter to phash representatives to minimize redundant OCR
    by_hash = {}
    for r in rows:
        ph = r.get("phash", "")
        if ph not in by_hash or float(r.get("spend") or 0) > float(by_hash[ph].get("spend") or 0):
            by_hash[ph] = r
    targets = sorted(by_hash.values(), key=lambda r: r["asset_id"])

    if limit:
        targets = targets[:limit]

    out_file = output_csv or os.path.join(cfg.DATA_DIR, "assets_ocr.csv")
    cached_ocr = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    cached_ocr[r["asset_id"]] = r.get("ocr_text", "")
        except Exception:
            pass

    tesseract_engine = get_tesseract_ocr()
    gemini_client = None
    if not tesseract_engine:
        try:
            from google.genai import Client
            gemini_client = Client(vertexai=True, project=cfg.GCP_PROJECT_ID, location=cfg.GCP_REGION)
        except Exception:
            pass

    logger.info(f"Extracting OCR text for {len(targets)} assets (Cached: {len(cached_ocr)})")
    results = []
    for idx, r in enumerate(targets, 1):
        aid = r["asset_id"]
        if aid in cached_ocr:
            text = cached_ocr[aid]
        else:
            text = extract_asset_ocr(r, tesseract_engine, gemini_client)
            cached_ocr[aid] = text
        results.append({
            "asset_id": aid,
            "phash": r.get("phash", ""),
            "ocr_text": text,
        })
        if idx % 50 == 0 or idx == len(targets):
            logger.info(f"Progress: {idx}/{len(targets)} assets processed")

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["asset_id", "phash", "ocr_text"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Wrote OCR results to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract OCR text from ad media")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of assets to process")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    run_ocr(limit=args.limit, output_csv=args.output)


if __name__ == "__main__":
    main()
