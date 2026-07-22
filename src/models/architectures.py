"""Per-architecture subclasses. Each supplies conv construction; GcnModel, SageModel,
and GatModel supply only BuildLayerConv and reuse the base Forward loop unchanged.
GcniiModel overrides both __init__ and Forward: GCN2Conv's equal-width constraint
does not fit the generic per-layer construction, so it is resolved with uncounted
input/output projections instead.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .gnn_model import BuildConv, GnnModel
from .protocols import LayerHook, Readout


class GcnModel(GnnModel):
    CONV_TYPE = "gcn"

    def BuildLayerConv(self, layerInDim: int, layerOutDim: int, isFinalLogits: bool) -> nn.Module:
        return BuildConv("gcn", layerInDim, layerOutDim)


class SageModel(GnnModel):
    CONV_TYPE = "sage"

    def BuildLayerConv(self, layerInDim: int, layerOutDim: int, isFinalLogits: bool) -> nn.Module:
        return BuildConv("sage", layerInDim, layerOutDim)


class GatModel(GnnModel):
    """8 attention heads throughout. The logit-emitting final layer under
    LastLayerReadout uses a single head so its output is exactly outDim wide with
    no concatenation, matching Velickovic et al."""

    CONV_TYPE = "gat"
    NUM_HEADS = 8

    def BuildLayerConv(self, layerInDim: int, layerOutDim: int, isFinalLogits: bool) -> nn.Module:
        heads = 1 if isFinalLogits else self.NUM_HEADS
        return BuildConv("gat", layerInDim, layerOutDim, heads=heads, concat=True)


class GcniiModel(GnnModel):
    """GCNII (Chen et al. 2020, arXiv:2007.02133) via PyG's GCN2Conv.

    An uncounted Linear(inDim, hiddenDim) input projection and, under
    LastLayerReadout, an uncounted Linear(hiddenDim, outDim) output projection
    wrap a stack of exactly `numLayers` GCN2Conv hops, so depth stays a hop
    count comparable with the other three architectures. `alpha`/`theta` are
    GCN2Conv's own hyperparameters with no baked-in default; the caller
    supplies them.
    """

    CONV_TYPE = "gcnii"

    def __init__(
        self,
        numLayers: int,
        inDim: int,
        hiddenDim: int,
        outDim: int,
        dropout: float,
        layerHooks: Sequence[LayerHook],
        readout: Readout,
        alpha: float,
        theta: float,
    ) -> None:
        # bypassing GnnModel.__init__: its generic BuildLayerConv loop assumes a
        # conv can map any (layerInDim, layerOutDim) pair, which GCN2Conv cannot
        nn.Module.__init__(self)
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.layerHooks = list(layerHooks)
        self.readout = readout
        self.dropoutLayer = nn.Dropout(dropout)
        self.inputProjection = nn.Linear(inDim, hiddenDim)
        self.outputProjection = nn.Linear(hiddenDim, outDim) if readout.FinalLayerIsLogits else None
        self.convs = nn.ModuleList(
            [
                BuildConv("gcnii", hiddenDim, hiddenDim, alpha=alpha, theta=theta, layer=l)
                for l in range(1, numLayers + 1)
            ]
        )

    def Forward(self, x: Tensor, edgeIndex: Tensor) -> tuple[Tensor, list[Tensor]]:
        layerEmbeddings: list[Tensor] = [x]
        # input projection is uncounted: performs the 1433 -> hiddenDim width
        # change GCN2Conv structurally cannot do; its output seeds every
        # layer's fixed initial-residual term x0 but is never tapped into
        # layerEmbeddings, which keeps index 0 raw X
        h = torch.relu(self.inputProjection(x))
        h = self.dropoutLayer(h)
        x0 = h
        for l, conv in enumerate(self.convs, start=1):
            hPrev = h
            h = conv(h, edgeIndex, x0)
            for hook in self.layerHooks:
                h = hook.Apply(h, hPrev, edgeIndex)
            isFinalAndLogits = (l == self.numLayers) and self.readout.FinalLayerIsLogits
            if isFinalAndLogits:
                # output projection stands in for the final conv's width
                # change; replaces the activation rather than following it
                h = self.outputProjection(h)
            else:
                h = torch.relu(h)
            layerEmbeddings.append(h)
            if not isFinalAndLogits:
                h = self.dropoutLayer(h)
        logits = self.readout.Apply(layerEmbeddings)
        return logits, layerEmbeddings
