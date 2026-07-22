"""MitigationNames: the single source of the canonical mitigation name list
`experiments/` writes to config.mitigations and uses to build the results
filename, so the naming convention exists in one place.
"""

from __future__ import annotations

from typing import Sequence

from models import LayerHook, Readout

from .readouts import JkReadout


def MitigationNames(layerHooks: Sequence[LayerHook], readout: Readout) -> list[str]:
    """Sorted canonical names: each hook's NAME, plus "jk" when the readout is
    JkReadout. The redundancy with config.readout is accepted; consistency
    between the two is asserted by experiments/, not here."""
    names = [hook.NAME for hook in layerHooks]
    if isinstance(readout, JkReadout):
        names.append("jk")
    return sorted(names)
