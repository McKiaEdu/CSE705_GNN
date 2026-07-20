"""ExportTable — writes an aggregate table as markdown or LaTeX for direct
inclusion in the report.
"""

from __future__ import annotations

import os

import pandas as pd


def _ToMarkdown(table: pd.DataFrame) -> str:
    # not pandas' own to_markdown(), which pulls in the tabulate dependency;
    # viz_spec.md names only matplotlib/pandas/scikit-learn/torch as runtime deps
    columns = [str(c) for c in table.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in table.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def ExportTable(table: pd.DataFrame, path: str, fmt: str) -> None:
    """Writes `table` to `path` in "md" or "tex" format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt == "md":
        content = _ToMarkdown(table)
    elif fmt == "tex":
        content = table.to_latex(index=False, float_format="%.4f")
    else:
        raise ValueError(f"unknown table format: {fmt!r}")
    with open(path, "w") as f:
        f.write(content)
