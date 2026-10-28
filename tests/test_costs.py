"""Unit tests for common.costs module."""

import csv
import os
import tempfile
import pytest

from common import costs


def test_costs_record_and_summary(tmp_path):
    test_ledger = str(tmp_path / "test_costs_ledger.csv")

    # Record billable call with known rate (e.g. mm_image: 0.0001 per image)
    cost1 = costs.record(
        stage="embedding",
        service="vertex",
        sku="mm_image",
        units=10,
        asset_id="test_asset_1",
        latency_s=0.25,
        status="OK",
        ledger_path=test_ledger,
    )
    assert cost1 == pytest.approx(0.001)

    # Record unpriced or unknown SKU
    cost2 = costs.record(
        stage="enrichment",
        service="custom",
        sku="unknown_sku",
        units=5,
        asset_id="test_asset_2",
        latency_s=1.2,
        status="OK",
        ledger_path=test_ledger,
    )
    assert cost2 is None

    # Check that CSV file exists and has rows
    assert os.path.exists(test_ledger)
    with open(test_ledger, "r", encoding="utf-8") as fh:
        reader = list(csv.DictReader(fh))
        assert len(reader) == 2
        assert reader[0]["stage"] == "embedding"
        assert reader[0]["sku"] == "mm_image"
        assert float(reader[0]["units"]) == 10.0
        assert float(reader[0]["cost_usd"]) == pytest.approx(0.001)
        assert reader[0]["status"] == "OK"

        assert reader[1]["stage"] == "enrichment"
        assert reader[1]["sku"] == "unknown_sku"
        assert reader[1]["cost_usd"] == ""  # blank cost for unpriced
