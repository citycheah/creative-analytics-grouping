"""Cost attribution for the creative-clustering pipeline.

Two independent mechanisms, because they fail in different ways:

1. **Billing labels** - attached to every billable API call (Vertex predict,
   Gemini generate_content, BigQuery jobs). These flow to the Cloud Billing
   export and let FinOps filter spend to this pipeline stage.

2. **Local ledger** - every billable call is also appended to a CSV with its
   exact billable units. Billing export lags ~24h and label propagation to
   publisher-model SKUs is not guaranteed, so the ledger is the source of truth
   we can read *immediately* and reconcile against the invoice later.

Rates were read from the Cloud Billing Catalog API
(service C7E2-9256-1C43 "Vertex AI", currency USD).
Re-run `python common/costs.py --refresh-rates` to re-check live SKU pricing.
"""

import csv
import datetime
import os
import threading

# --- Standard label set -----------------------------------------------------
# GCP label rules: lowercase letters, digits, hyphens, underscores only.
LABELS = {
    "project": "creative-theme-clustering",
    "domain": "malaysia-telco-intelligence",
}


def labels(stage: str) -> dict:
    """Label set for one pipeline stage, e.g. labels('embedding')."""
    return {**LABELS, "stage": stage}


# --- Verified rates (USD) ---------------------------------------------------
# Source: Cloud Billing Catalog API. Units as billed by Google Cloud.
RATES = {
    "mm_image":            (0.0001, "per image"),
    "mm_video_essential":  (0.0005, "per second of video"),
    "mm_video_standard":   (0.001,  "per second of video"),
    "mm_video_plus":       (0.002,  "per second of video"),
    "mm_text":             (0.0000002, "per input token"),
    # BigQuery on-demand analysis.
    "bq_analysis_euw3":    (8.125, "per TiB scanned"),
    "bq_analysis_us":      (6.25,  "per TiB scanned"),
    # Gemini 2.5 Flash, per token.
    "gemini_text_in":      (0.00000015, "per input token"),
    "gemini_video_in":     (0.00000015, "per input token"),
    "gemini_image_in":     (0.00000030, "per input token"),
    "gemini_text_out":     (0.00000060, "per output token"),
    # Thinking-mode output is 5.8x standard output. gemini-2.5-flash enables
    # thinking BY DEFAULT, so structured-extraction jobs must set
    # thinking_budget=0 or silently pay this rate.
    "gemini_text_out_thinking": (0.0000035, "per output token"),
}
RATES_VERIFIED_ON = "2026-08-13"

BQ_JOB_REGION = "US"

_LEDGER_LOCK = threading.Lock()
_HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_PATH = os.path.join(os.path.dirname(_HERE), "costs_ledger.csv")

_FIELDS = ["ts", "stage", "service", "sku", "asset_id", "units", "unit_desc",
           "unit_cost", "cost_usd", "latency_s", "status"]


def record(stage, service, sku, units, asset_id="", latency_s=None,
           status="OK", ledger_path=None):
    """Append one billable call to the ledger. Thread-safe.

    `sku` should be a key of RATES where the rate is known; if it is not, the
    row is still written with cost_usd blank so the units are never lost.
    """
    rate, unit_desc = RATES.get(sku, (None, ""))
    cost = round(units * rate, 8) if rate is not None else None
    row = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "stage": stage, "service": service, "sku": sku,
        "asset_id": asset_id, "units": units, "unit_desc": unit_desc,
        "unit_cost": rate if rate is not None else "",
        "cost_usd": f"{cost:.8f}" if cost is not None else "",
        "latency_s": f"{latency_s:.3f}" if latency_s is not None else "",
        "status": status,
    }
    path = ledger_path or LEDGER_PATH
    with _LEDGER_LOCK:
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    return cost


def summary(ledger_path=None):
    """Print spend to date, grouped by stage and SKU."""
    import collections
    path = ledger_path or LEDGER_PATH
    if not os.path.exists(path):
        print("No ledger yet - nothing has been billed.")
        return
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by = collections.defaultdict(lambda: [0, 0.0, 0, 0.0])  # calls, units, unknown, cost
    for r in rows:
        k = (r["stage"], r["sku"])
        by[k][0] += 1
        by[k][1] += float(r["units"] or 0)
        if r["cost_usd"] == "":
            by[k][2] += 1
        else:
            by[k][3] += float(r["cost_usd"])

    print(f"Ledger: {path}")
    print(f"Rates verified: {RATES_VERIFIED_ON}\n")
    print(f"{'stage':<14}{'sku':<22}{'calls':>7}{'units':>12}{'cost USD':>12}")
    print("-" * 67)
    total, unpriced = 0.0, 0
    for (stage, sku), (calls, units, unk, cost) in sorted(by.items()):
        total += cost
        unpriced += unk
        print(f"{stage:<14}{sku:<22}{calls:>7}{units:>12,.1f}{cost:>12.4f}")
    print("-" * 67)
    print(f"{'TOTAL':<43}{'':>12}{total:>12.4f}")
    errs = sum(1 for r in rows if r["status"] != "OK")
    if errs:
        print(f"\n{errs} failed call(s) recorded (billed only if the API charged them).")
    if unpriced:
        print(f"{unpriced} row(s) have no verified rate - units recorded, cost blank.")


def refresh_rates():
    """Re-read embedding SKU prices from the Cloud Billing Catalog API."""
    import json
    import subprocess
    import urllib.request
    token = subprocess.run(["gcloud", "auth", "print-access-token"],
                           capture_output=True, text=True).stdout.strip()
    page, found = "", []
    for _ in range(10):
        url = ("https://cloudbilling.googleapis.com/v1/services/C7E2-9256-1C43"
               f"/skus?pageSize=5000&currencyCode=USD&pageToken={page}")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            d = json.load(resp)
        for s in d.get("skus", []):
            desc = s.get("description", "")
            if "Embeddings for multimodal" in desc:
                for pi in s.get("pricingInfo", []):
                    for t in pi.get("pricingExpression", {}).get("tieredRates", []):
                        u = t["unitPrice"]
                        p = int(u.get("units", 0)) + int(u.get("nanos", 0)) / 1e9
                        if p > 0:
                            found.append((desc, p))
        page = d.get("nextPageToken", "")
        if not page:
            break
    print(f"Live catalog rates ({datetime.date.today()}):")
    for desc, p in sorted(found):
        print(f"  {p:.9f}  {desc}")
    print(f"\nHardcoded in RATES (verified {RATES_VERIFIED_ON}):")
    for k, (r, u) in RATES.items():
        print(f"  {r:.9f}  {k}  ({u})")


if __name__ == "__main__":
    import sys
    if "--refresh-rates" in sys.argv:
        refresh_rates()
    else:
        summary()
