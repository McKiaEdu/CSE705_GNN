"""GnnModel — the depth-parameterized layer stack shared by every architecture.

Owns the layer loop (D-006); subclasses supply only BuildLayerConv. Mitigations
attach via the layerHooks/readout composition seam, not inheritance (D-009).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCN2Conv, GCNConv, SAGEConv

from .protocols import LayerHook, Readout


class _GcnConvAdapter(nn.Module):
    """Wraps GCNConv to expose the uniform (x, edgeIndex, x0) call signature."""

    def __init__(self, inDim: int, outDim: int) -> None:
        super().__init__()
        self.conv = GCNConv(inDim, outDim)

    def forward(self, x: Tensor, edgeIndex: Tensor, x0: Tensor) -> Tensor:
        return self.conv(x, edgeIndex)


class _SageConvAdapter(nn.Module):
    """Wraps SAGEConv to expose the uniform (x, edgeIndex, x0) call signature."""

    def __init__(self, inDim: int, outDim: int) -> None:
        super().__init__()
        self.conv = SAGEConv(inDim, outDim)

    def forward(self, x: Tensor, edgeIndex: Tensor, x0: Tensor) -> Tensor:
        return self.conv(x, edgeIndex)


class _GatConvAdapter(nn.Module):
    """Wraps GATConv to expose the uniform (x, edgeIndex, x0) call signature.

    `outDim` is the TOTAL output width across heads (D-023); dividing by `heads`
    recovers the per-head width GATConv expects when `concat=True`.
    """

    def __init__(self, inDim: int, outDim: int, heads: int, concat: bool) -> None:
        super().__init__()
        outPerHead = outDim // heads if concat else outDim
        self.conv = GATConv(inDim, outPerHead, heads=heads, concat=concat)

    def forward(self, x: Tensor, edgeIndex: Tensor, x0: Tensor) -> Tensor:
        return self.conv(x, edgeIndex)


class _GcniiConvAdapter(nn.Module):
    """Wraps GCN2Conv to expose the uniform (x, edgeIndex, x0) call signature.

    GCN2Conv requires x and x0 at equal width (D-034); BuildConv enforces that
    here with a clear error rather than silently truncating or padding.
    """

    def __init__(self, inDim: int, outDim: int, alpha: float, theta: float, layer: int) -> None:
        super().__init__()
        if inDim != outDim:
            raise ValueError(f"GCN2Conv requires inDim == outDim, got {inDim} != {outDim}")
        self.conv = GCN2Conv(channels=inDim, alpha=alpha, theta=theta, layer=layer)

    def forward(self, x: Tensor, edgeIndex: Tensor, x0: Tensor) -> Tensor:
        return self.conv(x, x0, edgeIndex)


def BuildConv(convType: str, inDim: int, outDim: int, **kwargs: object) -> nn.Module:
    """Factory dispatching on convType; every returned module accepts (x, edgeIndex, x0)."""
    if convType == "gcn":
        return _GcnConvAdapter(inDim, outDim)
    if convType == "sage":
        return _SageConvAdapter(inDim, outDim)
    if convType == "gat":
        heads = int(kwargs.get("heads", 8))
        concat = bool(kwargs.get("concat", True))
        return _GatConvAdapter(inDim, outDim, heads=heads, concat=concat)
    if convType == "gcnii":
        return _GcniiConvAdapter(
            inDim,
            outDim,
            alpha=float(kwargs["alpha"]),
            theta=float(kwargs["theta"]),
            layer=int(kwargs["layer"]),
        )
    raise ValueError(f"unknown convType: {convType!r}")


class GnnModel(nn.Module):
    """Base class owning the layer loop; subclasses supply BuildLayerConv only."""

    CONV_TYPE: str  # set by each architecture subclass

    def __init__(
        self,
        numLayers: int,
        inDim: int,
        hiddenDim: int,
        outDim: int,
        dropout: float,
        layerHooks: Sequence[LayerHook],
        readout: Readout,
    ) -> None:
        super().__init__()
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.layerHooks = list(layerHooks)
        self.readout = readout
        self.dropoutLayer = nn.Dropout(dropout)

        convs: list[nn.Module] = []
        for l in range(1, numLayers + 1):
            layerInDim = inDim if l == 1 else hiddenDim
            isFinalLogits = (l == numLayers) and readout.FinalLayerIsLogits
            layerOutDim = outDim if isFinalLogits else hiddenDim
            convs.append(self.BuildLayerConv(layerInDim, layerOutDim, isFinalLogits))
        self.convs = nn.ModuleList(convs)

    def BuildLayerConv(self, layerInDim: int, layerOutDim: int, isFinalLogits: bool) -> nn.Module:
        raise NotImplementedError("subclasses supply the per-layer conv construction")

    def ConfigRecord(self) -> dict[str, object]:
        """The model configuration record models_spec.md's Outputs & Artifacts
        section requires: everything train/ needs to reconstruct this run's
        model from its results record alone (D-022).

        `mitigations` reads each hook's NAME if present, falling back to its
        class name, in applied order (D-007's ordering, unsorted — C3
        canonicalizes at aggregation time, not at storage). `readout` reads the
        readout's NAME the same way. Duck-typed rather than importing
        mitigations' hook/readout classes, since models must not depend on
        mitigations (D-006's dependency-inversion direction) — this is the same
        convention mitigations.MitigationNames uses, so the two cannot drift
        apart even though the direction constraint keeps them as two call
        sites rather than one shared function.

        "jk" is appended to `mitigations` when the readout's own NAME is "jk",
        matching the accepted config.mitigations/config.readout redundancy for
        Jumping Knowledge (D-028) — again without a mitigations import.
        """
        readoutName = getattr(self.readout, "NAME", type(self.readout).__name__)
        mitigations = [getattr(hook, "NAME", type(hook).__name__) for hook in self.layerHooks]
        if readoutName == "jk":
            mitigations.append("jk")
        return {
            "convType": self.CONV_TYPE,
            "numLayers": self.numLayers,
            "hiddenDim": self.hiddenDim,
            "dropout": float(self.dropoutLayer.p),
            "mitigations": mitigations,
            "readout": readoutName,
        }

    def Forward(self, x: Tensor, edgeIndex: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Returns (logits [N, C], layerEmbeddings); the contract in D-001 C1/C2."""
        h = x
        x0 = x
        layerEmbeddings: list[Tensor] = [x]
        for l, conv in enumerate(self.convs, start=1):
            hPrev = h
            h = conv(h, edgeIndex, x0)
            for hook in self.layerHooks:
                h = hook.Apply(h, hPrev, edgeIndex)
            isFinalAndLogits = (l == self.numLayers) and self.readout.FinalLayerIsLogits
            if not isFinalAndLogits:
                h = torch.relu(h)
            # tapping after activation, before dropout (D-010/C2): this is the
            # representation the next layer actually receives, independent of the
            # stochastic dropout mask
            layerEmbeddings.append(h)
            if not isFinalAndLogits:
                h = self.dropoutLayer(h)
        logits = self.readout.Apply(layerEmbeddings)
        return logits, layerEmbeddings

    def forward(self, x: Tensor, edgeIndex: Tensor) -> tuple[Tensor, list[Tensor]]:
        # nn.Module's __call__ dispatches to lowercase forward; Forward is the
        # PascalCase contract method every other component calls directly
        return self.Forward(x, edgeIndex)
