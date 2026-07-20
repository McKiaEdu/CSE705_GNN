"""Dirichlet energy, MAD, and the fitted contraction slope.

The measuring instrument (D-004): holds the fixed augmented graph as state and
turns a models/ layerEmbeddings list into the scalars the study reports. Realizes
D-003 (per-dimension energy, slope over l >= 1), D-004 (augmented metric graph
fixed across architectures), and D-035 (MAD reduction convention).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch_geometric.utils import add_self_loops, contains_self_loops, degree


@dataclass(frozen=True)
class LayerMetrics:
    dirichletEnergy: list[float]
    mad: list[float]
    frobeniusSquared: list[float]
    bandIndices: list[int]
    contractionSlope: float


def BuildAugmentedOperator(edgeIndex: Tensor, numNodes: int) -> tuple[Tensor, Tensor]:
    """Augments edgeIndex to G~ = A + I and returns (augmentedEdgeIndex, d~^-1/2).

    Raises if edgeIndex already contains self-loops, so a caller cannot silently
    double-augment (D-004).
    """
    if contains_self_loops(edgeIndex):
        raise ValueError("edgeIndex must be self-loop-free; OversmoothingMetrics augments internally (D-004)")
    augmentedEdgeIndex, _ = add_self_loops(edgeIndex, num_nodes=numNodes)
    # edge_index is undirected with each edge appearing as two directed entries
    # (data_spec.md confirmed assumption), so out-degree on row 0 equals d~_i
    augmentedDegree = degree(augmentedEdgeIndex[0], num_nodes=numNodes, dtype=torch.float32)
    invSqrtDegree = augmentedDegree.pow(-0.5)
    return augmentedEdgeIndex, invSqrtDegree


def MeanAverageDistance(h: Tensor) -> float:
    """MAD per arXiv:1909.03211 Eq. 1-4 (global variant), verified against the
    source (D-035): cosine-distance matrix, reduced over non-zero entries only,
    at both the row-mean and final-mean stage.
    """
    rowNorm = h.norm(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = h / rowNorm
    distance = 1.0 - normalized @ normalized.T

    # excluding self-pairs (D-035): inert for nonzero rows (D_ii = 0 already, so
    # Eq. 3's non-zero filter drops it regardless), but load-bearing for an
    # all-zero row, where the norm clamp above makes the self-cosine-similarity
    # compute as 0 rather than 1, which would otherwise leave a spurious D_ii = 1
    # that survives the non-zero filter
    numNodes = h.shape[0]
    diagonalMask = torch.eye(numNodes, dtype=torch.bool, device=h.device)
    distance = distance.masked_fill(diagonalMask, 0.0)

    nonZeroMask = distance > 0
    rowSum = distance.sum(dim=1)
    rowCount = nonZeroMask.sum(dim=1).to(distance.dtype)
    # Eq. 3, clamped to 0 rather than NaN when a row has no non-zero distances —
    # the fully collapsed regime this metric exists to measure (D-035)
    rowAverage = torch.where(rowCount > 0, rowSum / rowCount.clamp_min(1.0), torch.zeros_like(rowSum))

    nonZeroRowMask = rowAverage > 0
    validRowCount = int(nonZeroRowMask.sum())
    if validRowCount == 0:
        return 0.0
    # Eq. 4, same 0-not-NaN clamp
    return float(rowAverage.sum() / validRowCount)


def SelectComparableBand(layerEmbeddings: Sequence[Tensor]) -> list[int]:
    """Every index l >= 1 whose width equals the modal width across 1..L.

    Index 0 is excluded unconditionally (representation kind, D-003). Raises if
    the resulting indices are not contiguous, so an unanticipated architecture
    cannot silently produce a gapped band FitContractionSlope would fit across.
    """
    numLayers = len(layerEmbeddings) - 1
    widths = [layerEmbeddings[l].shape[1] for l in range(1, numLayers + 1)]
    # ties (only possible at L=2 under LastLayerReadout, where widths are
    # [hiddenDim, outDim] each occurring once) resolve to the first-seen width —
    # layer 1's hiddenDim — via Counter's insertion-order tie-break, which is the
    # correct band per D-001 C1
    modalWidth = Counter(widths).most_common(1)[0][0]
    bandIndices = [l for l, w in zip(range(1, numLayers + 1), widths) if w == modalWidth]
    expectedRun = list(range(bandIndices[0], bandIndices[-1] + 1))
    if bandIndices != expectedRun:
        raise ValueError(f"comparable band indices are not contiguous: {bandIndices}")
    return bandIndices


_MIN_ENERGY_FOR_LOG = 1e-12


def FitContractionSlope(energies: Sequence[float], bandIndices: Sequence[int]) -> float:
    """Least-squares slope of log(energy) against layer index, over bandIndices.

    Returns nan when the band has fewer than two points (the L=2, LastLayerReadout
    case) rather than silently returning 0.
    """
    if len(bandIndices) < 2:
        return float("nan")
    xs = torch.tensor(bandIndices, dtype=torch.float64)
    # a genuinely collapsed deep GCN produces exact-zero energy under float32
    # (D-037); flooring at a small epsilon keeps the fit well-defined instead of
    # raising on log(0), at the cost of capping how negative the fitted slope can
    # read for a fully collapsed run
    ys = torch.tensor(
        [math.log(max(energies[l], _MIN_ENERGY_FOR_LOG)) for l in bandIndices],
        dtype=torch.float64,
    )
    xMean = xs.mean()
    yMean = ys.mean()
    slope = ((xs - xMean) * (ys - yMean)).sum() / ((xs - xMean) ** 2).sum()
    return float(slope)


class OversmoothingMetrics:
    """Constructed once per run; holds the fixed augmented graph as state."""

    def __init__(self, edgeIndex: Tensor, numNodes: int) -> None:
        self.numNodes = numNodes
        self.augmentedEdgeIndex, self.invSqrtDegree = BuildAugmentedOperator(edgeIndex, numNodes)

    def DirichletEnergy(self, h: Tensor) -> float:
        """Per-dimension energy of one layer against the stored augmented operator.

        E(H) = 0.5 * sum_{(i,j) in E~} || h_i/sqrt(d~_i) - h_j/sqrt(d~_j) ||^2,
        divided by h.shape[1] (D-003). Self-loop terms contribute exactly 0 to the
        sum but change every d~_i, which is the point of D-004.
        """
        scaled = h * self.invSqrtDegree.unsqueeze(1)
        source = scaled[self.augmentedEdgeIndex[0]]
        target = scaled[self.augmentedEdgeIndex[1]]
        squaredDiffNorms = ((source - target) ** 2).sum(dim=1)
        energy = 0.5 * squaredDiffNorms.sum()
        return float(energy / h.shape[1])

    def ComputeAll(self, layerEmbeddings: Sequence[Tensor]) -> LayerMetrics:
        dirichletEnergy = [self.DirichletEnergy(h) for h in layerEmbeddings]
        mad = [MeanAverageDistance(h) for h in layerEmbeddings]
        frobeniusSquared = [float(torch.sum(h * h)) for h in layerEmbeddings]
        bandIndices = SelectComparableBand(layerEmbeddings)
        contractionSlope = FitContractionSlope(dirichletEnergy, bandIndices)
        return LayerMetrics(
            dirichletEnergy=dirichletEnergy,
            mad=mad,
            frobeniusSquared=frobeniusSquared,
            bandIndices=bandIndices,
            contractionSlope=contractionSlope,
        )
