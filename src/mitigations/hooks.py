"""ResidualHook and PairNormHook — LayerHook implementations (D-006).

Neither carries learnable parameters, so both are plain objects rather than
nn.Module subclasses; GnnModel stores layerHooks as a plain list, not an
nn.ModuleList, so a future parameterized hook would need that storage changed
too before its parameters would reach the optimizer.
"""

from __future__ import annotations

from torch import Tensor


class ResidualHook:
    """h + hPrev when shapes match; h unchanged otherwise (D-024).

    The skip is explicit rather than incidental: a hook that silently returned
    h on any shape mismatch would let a misconfigured run report a residual arm
    whose residual never actually fired, with nothing erroring.
    """

    NAME = "residual"

    def Apply(self, h: Tensor, hPrev: Tensor, edgeIndex: Tensor) -> Tensor:
        if h.shape == hPrev.shape:
            return h + hPrev
        return h


class PairNormHook:
    """PN-SI (Zhao & Akoglu, arXiv:1909.12223, Eq. 10-11 with the scale-
    individual substitution given directly below Eq. 12), verified against the
    paper rather than inferred from the variant name (D-025's open question).

    Center by the column mean over nodes, then scale each row by its own L2
    norm times `scale`: x_i -> scale * (x_i - mean) / ||x_i - mean||_2. The
    row-norm denominator is clamped away from zero -- ReLU produces all-zero
    rows at depth, and the paper's own formula does not need this guard because
    it was not stress-tested at that regime; the clamp is this study's addition.
    """

    NAME = "pairnorm"

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def Apply(self, h: Tensor, hPrev: Tensor, edgeIndex: Tensor) -> Tensor:
        centered = h - h.mean(dim=0, keepdim=True)
        rowNorm = centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return self.scale * centered / rowNorm
