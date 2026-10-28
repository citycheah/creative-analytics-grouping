"""Unit tests for Malaysia Telco Competitive Creative Intelligence engine."""

import pytest

from embedding_clustering_helpers import compute_hhi, compute_jaccard_overlap, compute_theme_sov


def test_hhi_concentration_bounds():
    # Equal split across 4 themes: HHI = 4 * (0.25^2) = 0.25
    assert compute_hhi([25.0, 25.0, 25.0, 25.0]) == pytest.approx(0.25)

    # Monopoly / single hero theme: HHI = 1.0
    assert compute_hhi([100000.0]) == pytest.approx(1.0)

    # Equal split across 10 themes: HHI = 10 * (0.10^2) = 0.10
    assert compute_hhi([10.0] * 10) == pytest.approx(0.10)

    # Empty or zero spend
    assert compute_hhi([]) == 0.0
    assert compute_hhi([0.0, 0.0]) == 0.0


def test_jaccard_theme_overlap():
    maxis_themes = {"CAT_01", "CAT_02", "CAT_03", "CAT_08"}
    celcomdigi_themes = {"CAT_01", "CAT_02", "CAT_03", "CAT_10"}

    # 3 shared out of 5 total unique -> 3/5 = 0.60
    assert compute_jaccard_overlap(maxis_themes, celcomdigi_themes) == pytest.approx(0.60)
    assert compute_jaccard_overlap(set(), set()) == 0.0


def test_theme_sov_attribution():
    spends = {
        "Maxis": 200000.0,
        "CelcomDigi": 300000.0,
        "U Mobile": 150000.0,
        "Unifi": 150000.0,
    }
    sov = compute_theme_sov(spends)
    assert sov["Maxis"] == pytest.approx(25.0)
    assert sov["CelcomDigi"] == pytest.approx(37.5)
    assert sov["U Mobile"] == pytest.approx(18.75)
    assert sov["Unifi"] == pytest.approx(18.75)
    assert sum(sov.values()) == pytest.approx(100.0)
