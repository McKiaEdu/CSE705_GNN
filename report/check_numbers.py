"""Re-derives every headline figure from results/*.json and checks that the
report documents still state it.

The counterpart to decisions/applied/a67.py: that script verifies that anchor
*text* still exists, this one verifies that quantitative *claims* still match
the records. Closes the open question recorded in DECISIONS.md D-049.

Two failure modes are caught:

  STALE   a claim the records no longer support, because the sweep changed
  MISSING a claim the records support but the document does not state

Usage:
    python3 report/check_numbers.py                  # check every known document
    python3 report/check_numbers.py <doc.ipynb> ...  # check specific ones
    python3 report/check_numbers.py --list           # print the claims table only

Exit status is non-zero when any document is missing a claim it should carry.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics as st
import sys
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# documents checked when no path is given; a claim is only required in a
# document listed under its own `docs` field, since the IEEE paper carries a
# deliberate subset of the full report's numbers
FULL_REPORT = "report/CSE705_GNN_ARAK_WU1261.ipynb"
IEEE_PAPER = "report/CSE705_GNN_IEEE_ARAK_WU1261.ipynb"
DEFAULT_DOCUMENTS = [FULL_REPORT, IEEE_PAPER]

ARCHITECTURES = ("gcn", "sage", "gat")
DEPTHS = (2, 4, 8, 16, 32)


@dataclass
class Claim:
    """One quantitative statement, its recomputed value, and how it is written."""

    key: str
    value: float
    pattern: str
    source: str
    docs: tuple[str, ...] = (FULL_REPORT, IEEE_PAPER)


def _Load(runId: str) -> dict:
    with open(os.path.join(RESULTS_DIR, f"{runId}.json")) as f:
        return json.load(f)


def _Arm(convType: str, mitigation: str, depth: int) -> list[dict]:
    """Every seed present for one configuration."""
    records = []
    for seed in range(10):
        path = os.path.join(RESULTS_DIR, f"{convType}_{mitigation}_d{depth}_s{seed}.json")
        if os.path.exists(path):
            records.append(json.load(open(path)))
    return records


def _MeanStd(values: list[float]) -> tuple[float, float]:
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def _GeometricMean(values: list[float]) -> float:
    return math.exp(st.mean([math.log(v) for v in values]))


def _BandRatio(record: dict, capture: str) -> float:
    """E_last / E_1 across the comparable band, the relative-energy quantity."""
    band = record["bandIndices"]
    energies = record[capture]["dirichletEnergy"]
    return energies[band[-1]] / energies[band[0]]


def _Escape(value: float, decimals: int) -> str:
    """Regex matching any plausible rendering of `value`.

    A document writes the same quantity as 1.64, 1.640, or 1.30e-2 depending on
    the table it sits in, so an exact-decimal match reports formatting
    differences as if they were stale numbers. Matching a set of renderings
    keeps the check sensitive to the thing that matters, a value that moved,
    without being sensitive to how it was typeset. `decimals` is the preferred
    precision and is always included.
    """
    renderings: set[str] = set()
    for places in {decimals, 2, 3, 4}:
        renderings.add(f"{value:.{places}f}")
        # trailing zeros are usually dropped in prose (0.600 written as 0.6)
        renderings.add(f"{value:.{places}f}".rstrip("0").rstrip("."))
    magnitude = abs(value)
    if magnitude and (magnitude < 0.01 or magnitude >= 1e4):
        for places in (1, 2):
            scientific = f"{value:.{places}e}"
            mantissa, exponent = scientific.split("e")
            renderings.add(f"{mantissa}e{int(exponent)}")
            renderings.add(f"{mantissa}e{'-' if int(exponent) < 0 else '+'}{abs(int(exponent)):02d}")
    renderings.discard("")
    return "(?:" + "|".join(re.escape(r) for r in sorted(renderings, key=len, reverse=True)) + ")"


def BuildClaims() -> list[Claim]:
    """Recomputes every headline figure. One entry per claim the documents make."""
    claims: list[Claim] = []
    both = (FULL_REPORT, IEEE_PAPER)
    full = (FULL_REPORT,)

    # --- depth-2 baselines, stated in both documents ---
    for convType in ARCHITECTURES:
        mean, _ = _MeanStd([r["results"]["testAccuracy"] for r in _Arm(convType, "none", 2)])
        claims.append(
            Claim(f"depth2_acc_{convType}", 100 * mean, _Escape(100 * mean, 2),
                  f"arm A {convType} depth 2, mean test accuracy", both)
        )

    # --- depth-32 unmitigated accuracy and macro-F1 ---
    for convType in ARCHITECTURES:
        records = _Arm(convType, "none", 32)
        mean, _ = _MeanStd([r["results"]["testAccuracy"] for r in records])
        f1Mean, _ = _MeanStd([r["results"]["testMacroF1"] for r in records])
        claims.append(
            Claim(f"depth32_acc_{convType}", 100 * mean, _Escape(100 * mean, 2),
                  f"arm A {convType} depth 32, mean test accuracy", both)
        )
        claims.append(
            Claim(f"depth32_f1_{convType}", f1Mean, _Escape(f1Mean, 4),
                  f"arm A {convType} depth 32, mean macro-F1", both)
        )

    # --- GCN mitigation ablation at depth 32 ---
    for mitigation in ("residual", "pairnorm", "jk", "pairnorm+residual"):
        records = _Arm("gcn", mitigation, 32)
        if not records:
            continue
        mean, _ = _MeanStd([r["results"]["testAccuracy"] for r in records])
        claims.append(
            Claim(f"depth32_gcn_{mitigation}", 100 * mean, _Escape(100 * mean, 2),
                  f"arm B gcn+{mitigation} depth 32, mean test accuracy", both)
        )

    # --- arm D: the winning mitigation on the other two architectures ---
    for convType in ("sage", "gat"):
        records = _Arm(convType, "jk", 32)
        mean, _ = _MeanStd([r["results"]["testAccuracy"] for r in records])
        f1Mean, _ = _MeanStd([r["results"]["testMacroF1"] for r in records])
        claims.append(
            Claim(f"depth32_{convType}_jk", 100 * mean, _Escape(100 * mean, 2),
                  f"arm D {convType}+jk depth 32, mean test accuracy", both)
        )
        claims.append(
            Claim(f"depth32_{convType}_jk_f1", f1Mean, _Escape(f1Mean, 4),
                  f"arm D {convType}+jk depth 32, mean macro-F1", both)
        )
    gcnJkF1, _ = _MeanStd([r["results"]["testMacroF1"] for r in _Arm("gcn", "jk", 32)])
    claims.append(
        Claim("depth32_gcn_jk_f1", gcnJkF1, _Escape(gcnJkF1, 4),
              "arm B gcn+jk depth 32, mean macro-F1", both)
    )

    # --- GCNII across depth ---
    for depth in DEPTHS:
        mean, _ = _MeanStd([r["results"]["testAccuracy"] for r in _Arm("gcnii", "none", depth)])
        claims.append(
            Claim(f"gcnii_acc_d{depth}", mean, _Escape(mean, 4),
                  f"arm C gcnii depth {depth}, mean test accuracy", full)
        )
    gcnii32, gcnii32Std = _MeanStd(
        [r["results"]["testAccuracy"] for r in _Arm("gcnii", "none", 32)]
    )
    claims.append(
        Claim("gcnii_acc_d32_pct", 100 * gcnii32, _Escape(100 * gcnii32, 2),
              "arm C gcnii depth 32, mean test accuracy as a percentage", (IEEE_PAPER,))
    )

    # --- GCNII and GCN checkpoint contraction slopes ---
    # GCN's depth-16 slope is recomputed but not claimed: Section 6.6 quotes
    # depths 4, 8 and 32 as the contrast with GCNII and does not state depth 16
    gcnSlopeDepths = (4, 8, 32)
    for convType, docs in (("gcnii", both), ("gcn", full)):
        for depth in (4, 8, 16, 32):
            if convType == "gcn" and depth not in gcnSlopeDepths:
                continue
            slopes = [
                r["checkpointMetrics"]["contractionSlope"] for r in _Arm(convType, "none", depth)
            ]
            slopes = [s for s in slopes if s is not None and not math.isnan(s)]
            if not slopes:
                continue
            mean = st.mean(slopes)
            claims.append(
                Claim(f"slope_{convType}_d{depth}", mean, _Escape(abs(mean), 3),
                      f"{convType} depth {depth}, mean checkpoint contraction slope", docs)
            )

    # --- epoch-0 collapse: MAD at the last band index ---
    for convType in ARCHITECTURES:
        for depth in DEPTHS:
            records = _Arm(convType, "none", depth)
            mads = [r["epoch0Metrics"]["mad"][r["bandIndices"][-1]] for r in records]
            mean = st.mean(mads)
            if abs(mean) < 1e-3:
                continue  # written as a scientific-notation or "approximately zero" cell
            decimals = 3 if abs(mean) >= 0.01 else 4
            # GraphSAGE's deep MAD sits in the noise around zero; the full
            # report prints it in scientific notation, the IEEE table
            # compresses it to "approximately 0", so it is claimed only where
            # a number is actually written
            docs = full if (convType == "sage" and depth >= 16) else both
            claims.append(
                Claim(f"epoch0_mad_{convType}_d{depth}", mean, _Escape(mean, decimals),
                      f"arm A {convType} depth {depth}, epoch-0 MAD at last band index", docs)
            )

    # --- checkpoint MAD, GCN at depth 32 (the "not collapsed" figure) ---
    ckptMad = st.mean(
        [r["checkpointMetrics"]["mad"][r["bandIndices"][-1]] for r in _Arm("gcn", "none", 32)]
    )
    claims.append(
        Claim("checkpoint_mad_gcn_d32", ckptMad, _Escape(ckptMad, 3),
              "arm A gcn depth 32, mean checkpoint MAD at last band index", both)
    )

    # --- the depth-32 checkpoint energy-ratio spread, in orders of magnitude ---
    ratios = [_BandRatio(r, "checkpointMetrics") for r in _Arm("gcn", "none", 32)]
    spanOrders = math.log10(max(ratios)) - math.log10(min(ratios))
    claims.append(
        Claim("checkpoint_ratio_span_orders", spanOrders, str(round(spanOrders)),
              "arm A gcn depth 32, log10 span of checkpoint E_last/E_1 across seeds", both)
    )

    # --- training-loss floor and minima ---
    claims.append(
        Claim("ln7", math.log(7), _Escape(math.log(7), 4),
              "cross-entropy of a uniform prediction over seven classes", both)
    )
    for convType in ARCHITECTURES:
        minima = [
            min(p["trainLoss"] for p in r["trainingCurve"]) for r in _Arm(convType, "none", 32)
        ]
        mean = st.mean(minima)
        decimals = 2 if convType == "gcn" else 4
        claims.append(
            Claim(f"trainloss_min_{convType}_d32", mean, _Escape(mean, decimals),
                  f"arm A {convType} depth 32, mean minimum training loss", both)
        )

    # --- epochs run at depth 32, as ranges ---
    for convType in ARCHITECTURES:
        epochs = [r["results"]["epochsRun"] for r in _Arm(convType, "none", 32)]
        claims.append(
            Claim(f"epochs_lo_{convType}", min(epochs), str(min(epochs)),
                  f"arm A {convType} depth 32, shortest run", both)
        )
        claims.append(
            Claim(f"epochs_hi_{convType}", max(epochs), str(max(epochs)),
                  f"arm A {convType} depth 32, longest run", both)
        )

    # --- sweep size ---
    total = (
        len(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
        + len(glob.glob(os.path.join(RESULTS_DIR, "fidelity", "*.json")))
        + len(glob.glob(os.path.join(RESULTS_DIR, "hpsearch", "*.json")))
    )
    claims.append(Claim("total_runs", total, str(total), "total records on disk", both))

    return claims


def ReadDocument(path: str) -> str:
    """Notebook or markdown source as one whitespace-normalized string."""
    absolute = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    with open(absolute) as f:
        if absolute.endswith(".ipynb"):
            text = "".join("".join(c["source"]) for c in json.load(f)["cells"])
        else:
            text = f.read()
    return re.sub(r"\s+", " ", text)


def ResolveScope(path: str, override: str | None) -> str:
    """Maps a path to the known document whose claim set applies to it.

    Scoping by literal path would make any unrecognized file match no claims
    and pass vacuously, which is a silent false pass and worse than no check
    at all, so an unresolved path is an error rather than a success.
    """
    if override:
        return {"full": FULL_REPORT, "ieee": IEEE_PAPER}[override]
    for known in (FULL_REPORT, IEEE_PAPER):
        if path == known or os.path.basename(path) == os.path.basename(known):
            return known
    raise SystemExit(
        f"{path}: not a known document, so no claim set applies.\n"
        f"  Pass --as=full or --as=ieee to say which claim set to check it against."
    )


def CheckDocument(path: str, claims: list[Claim], scope: str) -> list[Claim]:
    """Claims that should appear in this document but do not."""
    text = ReadDocument(path)
    return [c for c in claims if scope in c.docs and not re.search(c.pattern, text)]


def main() -> int:
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    claims = BuildClaims()

    if "--list" in sys.argv:
        for claim in claims:
            print(f"  {claim.key:34} {claim.value!s:>14}   {claim.source}")
        print(f"\n{len(claims)} claims recomputed from {RESULTS_DIR}")
        return 0

    override = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--as=")), None)
    documents = arguments or DEFAULT_DOCUMENTS
    problems = 0
    for document in documents:
        scope = ResolveScope(document, override)
        missing = CheckDocument(document, claims, scope)
        expected = sum(1 for c in claims if scope in c.docs)
        print(f"\n{document}")
        print(f"  {expected - len(missing)}/{expected} claims found")
        for claim in missing:
            print(f"  MISSING  {claim.key:30} expected {claim.value}")
            print(f"           {claim.source}")
        problems += len(missing)

    print(f"\n{len(claims)} claims recomputed; {problems} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
