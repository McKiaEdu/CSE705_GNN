"""ExportTable: writes an aggregate table as markdown or LaTeX for direct
inclusion in the report.
"""

from __future__ import annotations

import os

import pandas as pd


def _ToMarkdown(table: pd.DataFrame, floatFormat: str) -> str:
    # not pandas' own to_markdown(), which pulls in the tabulate dependency,
    # for one extra table format this project can render without it.
    # Formats by each column's own dtype, not by the runtime type of a value
    # pulled from iterrows(): iterrows() upcasts a whole row to one common
    # dtype when the row mixes int/bool columns with float ones, which would
    # otherwise turn an int column like `seed` into "0.0000" instead of "0".
    columns = [str(c) for c in table.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    isBoolColumn = [pd.api.types.is_bool_dtype(table[c]) for c in table.columns]
    isIntColumn = [pd.api.types.is_integer_dtype(table[c]) for c in table.columns]
    isFloatColumn = [pd.api.types.is_float_dtype(table[c]) for c in table.columns]
    for _, row in table.iterrows():
        cells = []
        for v, isBool, isInt, isFloat in zip(row, isBoolColumn, isIntColumn, isFloatColumn):
            if isBool:
                cells.append(str(bool(v)))
            elif isInt:
                cells.append(str(int(v)))
            elif isFloat:
                cells.append(format(v, floatFormat))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def ExportTable(table: pd.DataFrame, path: str, fmt: str, floatFormat: str = ".4f") -> None:
    """Writes `table` to `path` in "md" or "tex" format.

    `floatFormat` is a Python format-spec (e.g. ".4f", ".2e") applied to every
    float column; the default matches every existing report table. Pass a
    scientific-notation spec for tables whose float values span many orders
    of magnitude, where a fixed decimal count would round small values to
    "0.0000" and erase the thing the table exists to show.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt == "md":
        content = _ToMarkdown(table, floatFormat)
    elif fmt == "tex":
        content = table.to_latex(index=False, float_format=lambda v: format(v, floatFormat))
    else:
        raise ValueError(f"unknown table format: {fmt!r}")
    with open(path, "w") as f:
        f.write(content)
