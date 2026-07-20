"""LayerHook and Readout protocols — the composition seam between models and mitigations.

Declared here, not in mitigations, so mitigations can depend on models without models
depending on mitigations (D-006's dependency-inversion direction).
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class LayerHook(Protocol):
    def Apply(self, h: Tensor, hPrev: Tensor, edgeIndex: Tensor) -> Tensor:
        """Shape-preserving; called once per layer immediately after the conv."""
        ...


class Readout(Protocol):
    FinalLayerIsLogits: bool
    NAME: str

    def Apply(self, layerEmbeddings: list[Tensor]) -> Tensor:
        """Called once after the layer loop; consumes the whole embedding stack."""
        ...


class LastLayerReadout:
    """Null-object default readout (D-009): the last conv's output IS the logits,
    so the unmitigated baseline and every mitigated variant run one code path."""

    FinalLayerIsLogits = True
    NAME = "lastLayer"

    def Apply(self, layerEmbeddings: list[Tensor]) -> Tensor:
        return layerEmbeddings[-1]
