"""JkReadout: Jumping Knowledge as a Readout.

An nn.Module (unlike the parameterless hooks in hooks.py): its Linear layer has
learnable parameters, and GnnModel stores `readout` as a direct attribute, which
PyTorch auto-registers as a submodule so those parameters reach the optimizer.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.nn import JumpingKnowledge


class JkReadout(nn.Module):
    """Max-pools layerEmbeddings[1:] and projects to outDim.

    Max pooling, not concatenation: concatenation would make this Linear
    layer's input width, and therefore the JK arm's parameter count, a
    function of depth, confounding the mitigation's apparent effect with a
    capacity increase at exactly the depths this study is about.

    Index 0 is excluded: it is the raw 1433-dim input and cannot be pooled
    against hiddenDim-wide hidden states.
    """

    FinalLayerIsLogits = False
    NAME = "jk"

    def __init__(self, hiddenDim: int, outDim: int, mode: str = "max") -> None:
        super().__init__()
        self.jump = JumpingKnowledge(mode=mode)
        self.linear = nn.Linear(hiddenDim, outDim)

    def Apply(self, layerEmbeddings: list[Tensor]) -> Tensor:
        band = layerEmbeddings[1:]
        widths = {h.shape[1] for h in band}
        if len(widths) != 1:
            raise ValueError(f"JkReadout requires uniform width across layerEmbeddings[1:], got widths {widths}")
        aggregated = self.jump(band)
        return self.linear(aggregated)
