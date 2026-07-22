"""Cora loading, preprocessing, and graph-invariant validation.

Sole source of the graph, features, labels, and split masks used anywhere in the
repo. Row-normalizes Cora's node features as a fixed preprocessing step, not an
ablation axis.
"""

from __future__ import annotations

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import contains_self_loops

NUM_NODES = 2708
NUM_FEATURES = 1433
NUM_CLASSES = 7
TRAIN_SIZE = 140
VAL_SIZE = 500
TEST_SIZE = 1000


def LoadCora(root: str = "data", normalizeFeatures: bool = True) -> Data:
    """Loads the public-split Cora citation network via PyG's Planetoid.

    `normalizeFeatures` exists only for a contrast check in the test suite;
    every experiment calls this with the default enabled.
    """
    transform = NormalizeFeatures() if normalizeFeatures else None
    dataset = Planetoid(root=root, name="Cora", split="public", transform=transform)
    data = dataset[0]
    AssertGraphInvariants(data)
    return data


def AssertGraphInvariants(data: Data) -> None:
    """Fails loudly if `data` does not match the graph shape this study assumes.

    Raising explicitly rather than using `assert`, so the check cannot be
    silently compiled out under `python -O`.
    """
    if data.num_nodes != NUM_NODES:
        raise ValueError(f"expected {NUM_NODES} nodes, got {data.num_nodes}")

    if tuple(data.x.shape) != (NUM_NODES, NUM_FEATURES):
        raise ValueError(f"expected x shape ({NUM_NODES}, {NUM_FEATURES}), got {tuple(data.x.shape)}")

    numClasses = int(data.y.max().item()) + 1
    if numClasses != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} classes, got {numClasses}")

    if not data.is_undirected():
        raise ValueError("expected an undirected graph")

    # confirming the loader ships the raw graph: A~ = A + I is GCNConv's job, not
    # this component's, so a self-loop here would mean metrics/ later
    # double-augments when it adds its own
    if contains_self_loops(data.edge_index):
        raise ValueError("expected no self-loops in the raw edge_index")

    trainCount = int(data.train_mask.sum())
    valCount = int(data.val_mask.sum())
    testCount = int(data.test_mask.sum())
    if (trainCount, valCount, testCount) != (TRAIN_SIZE, VAL_SIZE, TEST_SIZE):
        raise ValueError(
            f"expected mask sizes ({TRAIN_SIZE}, {VAL_SIZE}, {TEST_SIZE}), "
            f"got ({trainCount}, {valCount}, {testCount})"
        )

    # checking pairwise disjointness with vectorized boolean ANDs, not a node loop
    trainValOverlap = bool((data.train_mask & data.val_mask).any())
    trainTestOverlap = bool((data.train_mask & data.test_mask).any())
    valTestOverlap = bool((data.val_mask & data.test_mask).any())
    if trainValOverlap or trainTestOverlap or valTestOverlap:
        raise ValueError("expected train/val/test masks to be pairwise disjoint")
