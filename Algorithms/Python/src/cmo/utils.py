"""
Utilities
=======
src/cmo/utils.py
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import TextType

from .types import floatMat, floatVec


def format_float(
    obj: float | int | list[float | int] | floatVec | floatMat | None,
    dprec: int = 2,
    fprec: int = 6,
    ffmt: str = "f",
    sep: str = ",\n",
    lim: int = 5,
    pfx: str = "",
    spac: str = " ",
) -> str:
    if obj is None:
        return "None"

    def _fmtr(x: float | int) -> str:
        if isinstance(x, int):
            return str(x)
        elif isinstance(x, float):
            _fmt = f"{{:{dprec}.{fprec}{ffmt}}}"
            s = _fmt.format(x)
            # Strip trailing zeros
            if "." in s:
                s = s.rstrip("0").rstrip(".")
                if "." not in s:
                    s += ".0"
            return s
        else:
            return "..."

    if isinstance(obj, (float, int)):
        return _fmtr(obj)

    def _fmt_vec(vec: floatVec | list[float] | list[int]) -> str:
        if isinstance(vec, np.ndarray):
            size = vec.size
        else:
            size = len(vec)
        if size <= lim:
            s = [_fmtr(x) for x in vec]
        else:
            # Show first two and last two items
            s = [_fmtr(x) for x in vec[:2]] + ["..."] + [_fmtr(x) for x in vec[-2:]]
        return "[" + sep.join(s) + "]"

    if isinstance(obj, list):
        return _fmt_vec(obj)
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return _fmt_vec(obj)
        elif obj.ndim == 2:
            rows = [_fmt_vec(row) for row in obj]
            return "[" + f",\n{pfx}{spac}".join(rows) + "]"
    return "..."


def format_time(t: float | None) -> str:
    """Format time in seconds to an appropriate unit"""
    if t is None:
        return "None"
    abs_t = abs(t)
    if abs_t >= 60:
        minutes = int(t // 60)
        seconds = t % 60
        if seconds < 1e-3:
            return f"{minutes} min"
        return f"{minutes} min {round(seconds)} s"
    units: list[tuple[str, float]] = [("s", 1), ("ms", 1e-3), ("\u03bcs", 1e-6)]
    for unit, thresh in units:
        if abs_t >= thresh:
            val = t / thresh
            if unit == "\u03bcs":
                return f"{int(round(val))} {unit}"
            if unit == "min":
                return f"{val:.2f} {unit}"
            return f"{val:.3f} {unit}"
    return f"{int(round(t / 1e-6))} \u03bcs"  # Fallback for very small values


def show_solution(
    x: floatVec,
    fx: float,
    table: Table | None = None,
    title: TextType | None = None,
    console: Console | None = None,
) -> None:
    """Helper to format and display the solution."""
    if table is None:
        table = Table(title=title, title_justify="left", show_header=False)
        table.add_column(style="bold", justify="right")
        table.add_column()
    if console is None:
        console = Console()
    table.add_row("x*", format_float(x, sep=", "))
    table.add_row("f(x*)", format_float(fx))
    console.print(table)
