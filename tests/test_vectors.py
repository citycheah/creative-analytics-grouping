"""Unit tests for common.vectors module."""

import numpy as np
import pytest

from common.vectors import (
    combine,
    compute_centroids,
    l2,
    merge_by_centroid,
    score,
)


def test_l2_zero_vector():
    v0 = np.array([0.0, 0.0, 0.0])
    res = l2(v0)
    assert np.allclose(res, 0.0)

    # 2D with zero vector row
    m = np.array([[0.0, 0.0], [3.0, 4.0]])
    res2 = l2(m)
    assert np.allclose(res2[0], 0.0)
    assert np.allclose(res2[1], [0.6, 0.8])


def test_l2_1d_vector():
    v = np.array([3.0, 4.0])
    normalized = l2(v)
    assert np.isclose(np.linalg.norm(normalized), 1.0)
    assert np.allclose(normalized, [0.6, 0.8])


def test_l2_batch_normalization():
    rng = np.random.default_rng(42)
    mat = rng.normal(size=(20, 10))
    normalized = l2(mat)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0)


def test_combine_weights():
    vis = np.array([[1.0, 0.0], [0.0, 1.0]])
    txt = np.array([[0.0, 1.0], [1.0, 0.0]])

    # w_vis >= 1.0: pure visual
    pure_vis = combine(vis, txt, 1.0)
    assert np.allclose(pure_vis, l2(vis))

    # w_vis <= 0.0: pure text
    pure_txt = combine(vis, txt, 0.0)
    assert np.allclose(pure_txt, l2(txt))

    # w_vis = 0.5: weighted mix
    mixed = combine(vis, txt, 0.5)
    assert mixed.shape == (2, 4)
    norms = np.linalg.norm(mixed, axis=1)
    assert np.allclose(norms, 1.0)


def test_compute_centroids():
    X = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])
    pred = np.array([0, 0, 1, -1])  # cluster 0 has 2 points, 1 has 1 point, -1 is noise
    cens = compute_centroids(X, pred)

    assert -1 not in cens
    assert 0 in cens
    assert 1 in cens
    assert np.allclose(cens[0], [1.0, 0.0])
    assert np.allclose(cens[1], [0.0, 1.0])


def test_merge_by_centroid_cases():
    # 2 clusters very close (cosine ~ 1.0), 1 cluster orthogonal
    X = np.array([
        [1.0, 0.01],
        [1.0, -0.01],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    # Clusters: 0 and 1 are practically identical, 2 is orthogonal
    pred = np.array([0, 0, 1, 2])
    merged = merge_by_centroid(X, pred, thresh=0.90)

    # 0 and 1 should merge into a single cluster
    assert merged[0] == merged[2]
    # cluster 2 should remain separate
    assert merged[0] != merged[3]


def test_merge_by_centroid_no_merge():
    # Orthogonal clusters
    X = np.eye(3)
    pred = np.array([0, 1, 2])
    merged = merge_by_centroid(X, pred, thresh=0.90)
    assert len(set(merged)) == 3


def test_merge_by_centroid_chain():
    # Cluster 0 close to Cluster 1, Cluster 1 close to Cluster 2
    X = np.array([
        [1.0, 0.0],
        [0.99, 0.1],
        [0.98, 0.2],
    ])
    pred = np.array([0, 1, 2])
    merged = merge_by_centroid(X, pred, thresh=0.95)
    # They should chain-merge together
    assert len(set(merged)) == 1


def test_score_ari_ami():
    truth = {"a1": "T1", "a2": "T1", "b1": "T2", "b2": "T2"}
    # Perfect match
    pred_perfect = {"a1": 0, "a2": 0, "b1": 1, "b2": 1}
    ari, ami = score(pred_perfect, truth)
    assert np.isclose(ari, 1.0)
    assert np.isclose(ami, 1.0)

    # List signature
    ids = ["a1", "a2", "b1", "b2"]
    pred_list = [0, 0, 1, 1]
    ari2, ami2 = score(ids, pred_list, truth)
    assert np.isclose(ari2, 1.0)
    assert np.isclose(ami2, 1.0)
