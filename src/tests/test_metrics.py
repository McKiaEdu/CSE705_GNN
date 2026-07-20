"""Test plan for metrics_spec.md."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor
from torch_geometric.utils import add_self_loops

from data import LoadCora
from metrics import (
    BuildAugmentedOperator,
    FitContractionSlope,
    MeanAverageDistance,
    OversmoothingMetrics,
    SelectComparableBand,
)
from models import GcnModel, LastLayerReadout


def _RandomUndirectedGraph(numNodes: int, numEdges: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(0, numNodes, (numEdges,), generator=generator)
    dst = torch.randint(0, numNodes, (numEdges,), generator=generator)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    edgeIndex = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return torch.unique(edgeIndex, dim=1)


def _CycleGraph(numNodes: int) -> Tensor:
    """A regular graph (every node has degree 2): d~_i is uniform after
    augmentation, so Delta~'s null space reduces to the constant vector here,
    unlike on Cora's irregular degree sequence (D-036)."""
    src = torch.arange(numNodes)
    dst = (src + 1) % numNodes
    edgeIndex = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return torch.unique(edgeIndex, dim=1)


@pytest.fixture(scope="module")
def cora():
    return LoadCora()


@pytest.fixture(scope="module")
def coraMetrics(cora):
    return OversmoothingMetrics(cora.edge_index, cora.num_nodes)


def test_collapse_floor_on_regular_graph() -> None:
    # Delta~ = I - D~^-1/2 A~ D~^-1/2's null space is sqrt(d~)-proportional in
    # general (D-036); it reduces to the literal "identical rows" case from
    # metrics_spec.md's test plan only on a REGULAR graph, where d~_i is uniform
    numNodes = 10
    edgeIndex = _CycleGraph(numNodes)
    metric = OversmoothingMetrics(edgeIndex, numNodes)
    identicalRows = torch.ones(numNodes, 8) * 3.7
    assert metric.DirichletEnergy(identicalRows) == pytest.approx(0.0, abs=1e-6)
    assert MeanAverageDistance(identicalRows) == pytest.approx(0.0, abs=1e-6)


def test_collapse_floor_on_cora_is_sqrt_degree_weighted(cora, coraMetrics) -> None:
    # on Cora's irregular graph (degree 2..169), the true zero-energy state is
    # rows proportional to sqrt(d~_i), not identical rows (D-036). MAD is still
    # exactly 0 here too, since cosine distance ignores positive per-node scaling
    _, invSqrtDegree = BuildAugmentedOperator(cora.edge_index, cora.num_nodes)
    sqrtDegree = 1.0 / invSqrtDegree
    proportionalRows = sqrtDegree.unsqueeze(1).repeat(1, 8) * 2.5
    assert coraMetrics.DirichletEnergy(proportionalRows) == pytest.approx(0.0, abs=1e-6)
    assert MeanAverageDistance(proportionalRows) == pytest.approx(0.0, abs=1e-6)


def test_non_degenerate(coraMetrics) -> None:
    torch.manual_seed(0)
    randomEmbedding = torch.randn(coraMetrics.numNodes, 8)
    assert coraMetrics.DirichletEnergy(randomEmbedding) > 1e-3
    assert MeanAverageDistance(randomEmbedding) > 1e-3


def test_energy_matches_dense_reference() -> None:
    numNodes = 20
    edgeIndex = _RandomUndirectedGraph(numNodes, numEdges=40, seed=0)
    metric = OversmoothingMetrics(edgeIndex, numNodes)

    torch.manual_seed(1)
    h = torch.randn(numNodes, 5)

    # independent dense reference, built from the raw edgeIndex, not reusing any
    # of OversmoothingMetrics' internal tensors
    adjacency = torch.zeros(numNodes, numNodes)
    adjacency[edgeIndex[0], edgeIndex[1]] = 1.0
    augmentedAdjacency = adjacency + torch.eye(numNodes)
    degreeTilde = augmentedAdjacency.sum(dim=1)
    invSqrtDegreeTilde = degreeTilde.pow(-0.5)
    normalizedAdjacency = invSqrtDegreeTilde.unsqueeze(1) * augmentedAdjacency * invSqrtDegreeTilde.unsqueeze(0)
    deltaTilde = torch.eye(numNodes) - normalizedAdjacency

    denseEnergy = float(torch.trace(h.T @ deltaTilde @ h) / h.shape[1])
    assert metric.DirichletEnergy(h) == pytest.approx(denseEnergy, abs=1e-5)


def test_scaling_behavior(coraMetrics) -> None:
    torch.manual_seed(2)
    h = torch.randn(coraMetrics.numNodes, 8)
    c = 3.0
    baseEnergy = coraMetrics.DirichletEnergy(h)
    scaledEnergy = coraMetrics.DirichletEnergy(c * h)
    assert scaledEnergy == pytest.approx(c * c * baseEnergy, rel=1e-4)

    baseMad = MeanAverageDistance(h)
    scaledMad = MeanAverageDistance(c * h)
    assert scaledMad == pytest.approx(baseMad, rel=1e-4)


def test_self_loop_guard(cora) -> None:
    alreadyAugmented, _ = add_self_loops(cora.edge_index, num_nodes=cora.num_nodes)
    with pytest.raises(ValueError):
        OversmoothingMetrics(alreadyAugmented, cora.num_nodes)


class _StubJkLikeReadout:
    """See test_models.py's identical stub: a minimal Readout test double with
    FinalLayerIsLogits = False, standing in for mitigations' not-yet-built
    JkReadout."""

    FinalLayerIsLogits = False

    def __init__(self, hiddenDim: int, outDim: int) -> None:
        self.projection = torch.nn.Linear(hiddenDim, outDim)

    def Apply(self, layerEmbeddings: list[Tensor]) -> Tensor:
        return self.projection(layerEmbeddings[-1])


def test_band_derivation_under_jk_and_last_layer_readout(cora) -> None:
    torch.manual_seed(0)
    depth = 4
    hiddenDim = 64

    jkModel = GcnModel(
        numLayers=depth,
        inDim=1433,
        hiddenDim=hiddenDim,
        outDim=7,
        dropout=0.0,
        layerHooks=[],
        readout=_StubJkLikeReadout(hiddenDim, 7),
    )
    jkModel.eval()
    _, jkEmbeddings = jkModel.Forward(cora.x, cora.edge_index)
    assert SelectComparableBand(jkEmbeddings) == [1, 2, 3, 4]

    lastLayerModel = GcnModel(
        numLayers=depth,
        inDim=1433,
        hiddenDim=hiddenDim,
        outDim=7,
        dropout=0.0,
        layerHooks=[],
        readout=LastLayerReadout(),
    )
    lastLayerModel.eval()
    _, lastLayerEmbeddings = lastLayerModel.Forward(cora.x, cora.edge_index)
    assert SelectComparableBand(lastLayerEmbeddings) == [1, 2, 3]


def test_slope_on_synthetic_decay() -> None:
    r = 0.7
    e1 = 10.0
    bandIndices = [1, 2, 3, 4, 5]
    energies = [0.0] * (max(bandIndices) + 1)
    for l in bandIndices:
        energies[l] = e1 * (r ** (l - 1))
    slope = FitContractionSlope(energies, bandIndices)
    assert slope == pytest.approx(math.log(r), abs=1e-6)


def test_slope_refuses_to_guess_at_single_point_band() -> None:
    energies = [0.0, 5.0]
    slope = FitContractionSlope(energies, [1])
    assert math.isnan(slope)


class _StubResidualHook:
    """Minimal LayerHook test double implementing D-024's rule: h + hPrev when
    shapes match, h unchanged otherwise. Stands in for mitigations' real
    ResidualHook, which is not built yet (mitigations comes after train/ in the
    build order)."""

    def Apply(self, h: Tensor, hPrev: Tensor, edgeIndex: Tensor) -> Tensor:
        if h.shape == hPrev.shape:
            return h + hPrev
        return h


def test_depth_qualitative_gate_at_epoch_zero(cora, coraMetrics) -> None:
    """Gated on epoch 0 (pre-training), not the checkpoint — see D-038.

    At the settled hyperparameters (D-029), the unmitigated 32-layer
    checkpoint does NOT show monotonic energy decay: vanishing gradients leave
    early layers near their weight-decayed-toward-zero state while the last
    few layers overfit and blow up, producing an INCREASING energy profile
    instead. That is a verified finding (D-038), not a test bug. The
    checkpoint-based version of this gate is deferred until train/ exists with
    real D-017 checkpoint selection.
    """
    torch.manual_seed(0)
    baselineModel = GcnModel(
        numLayers=32,
        inDim=1433,
        hiddenDim=64,
        outDim=7,
        dropout=0.5,
        layerHooks=[],
        readout=LastLayerReadout(),
    )
    baselineModel.eval()
    with torch.no_grad():
        _, baselineEmbeddings = baselineModel.Forward(cora.x, cora.edge_index)

    torch.manual_seed(0)
    residualModel = GcnModel(
        numLayers=32,
        inDim=1433,
        hiddenDim=64,
        outDim=7,
        dropout=0.5,
        layerHooks=[_StubResidualHook()],
        readout=LastLayerReadout(),
    )
    residualModel.eval()
    with torch.no_grad():
        _, residualEmbeddings = residualModel.Forward(cora.x, cora.edge_index)

    baselineMetrics = coraMetrics.ComputeAll([h.detach() for h in baselineEmbeddings])
    residualMetrics = coraMetrics.ComputeAll([h.detach() for h in residualEmbeddings])

    bandEnergies = [baselineMetrics.dirichletEnergy[l] for l in baselineMetrics.bandIndices]
    isMonotonic = all(a > b for a, b in zip(bandEnergies, bandEnergies[1:]))

    assert isMonotonic
    # shallower means less negative / closer to zero, i.e. a greater slope value
    assert residualMetrics.contractionSlope > baselineMetrics.contractionSlope
