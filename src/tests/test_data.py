"""Test plan for Cora loading and invariants."""

from __future__ import annotations

import torch
from torch_geometric.utils import contains_self_loops

from data import AssertGraphInvariants, LoadCora


def test_shapes_and_class_count() -> None:
    data = LoadCora()
    assert data.num_nodes == 2708
    assert tuple(data.x.shape) == (2708, 1433)
    assert int(data.y.max()) + 1 == 7


def test_undirected() -> None:
    data = LoadCora()
    assert data.is_undirected()


def test_raw_graph_has_no_self_loops() -> None:
    data = LoadCora()
    assert not contains_self_loops(data.edge_index)


def test_mask_sizes_and_disjointness() -> None:
    data = LoadCora()
    assert int(data.train_mask.sum()) == 140
    assert int(data.val_mask.sum()) == 500
    assert int(data.test_mask.sum()) == 1000
    assert not bool((data.train_mask & data.val_mask).any())
    assert not bool((data.train_mask & data.test_mask).any())
    assert not bool((data.val_mask & data.test_mask).any())


def test_row_normalization_sums_to_one_or_zero() -> None:
    # zero-sum rows would divide by zero during row normalization; Cora has
    # none, but this test reports the count so a future dataset swap would
    # surface the case rather than assume it away
    data = LoadCora(normalizeFeatures=True)
    rowSums = data.x.sum(dim=1)
    isOne = torch.isclose(rowSums, torch.ones_like(rowSums), atol=1e-5)
    isZero = torch.isclose(rowSums, torch.zeros_like(rowSums), atol=1e-5)
    zeroRowCount = int((~isOne & isZero).sum())
    print(f"\nzero-sum rows after NormalizeFeatures: {zeroRowCount}")
    assert bool((isOne | isZero).all()), "every row sum must be 1 or 0"
    assert not torch.isnan(data.x).any()
    assert not torch.isinf(data.x).any()


def test_unnormalized_contrast_has_integer_row_sums() -> None:
    data = LoadCora(normalizeFeatures=False)
    rowSums = data.x.sum(dim=1)
    assert torch.allclose(rowSums, rowSums.round())
    assert set(torch.unique(data.x).tolist()) <= {0.0, 1.0}


def test_determinism_across_loads() -> None:
    first = LoadCora()
    second = LoadCora()
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.train_mask, second.train_mask)
    assert torch.equal(first.val_mask, second.val_mask)
    assert torch.equal(first.test_mask, second.test_mask)


def test_assert_graph_invariants_raises_on_wrong_node_count() -> None:
    data = LoadCora()
    corrupted = data.clone()
    corrupted.x = corrupted.x[:-1]
    corrupted.y = corrupted.y[:-1]
    try:
        AssertGraphInvariants(corrupted)
    except ValueError:
        return
    raise AssertionError("expected AssertGraphInvariants to raise on a node-count mismatch")
