"""
Utilities
=======
src/cmo/utils.py
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import TextType

from .types import floatVec


def format_float(
    obj: float | floatVec | None,
    dprec: int = 2,
    fprec: int = 16,
    ffmt: str = "f",
    sep: str = ",\n",
    lim: int = 5,
) -> str:
    if obj is None:
        return "None"
    _fmt = f"{{:{dprec}.{fprec}{ffmt}}}"
    if isinstance(obj, float):
        return f"{_fmt.format(obj)}"
    elif isinstance(obj, np.ndarray):
        if obj.size <= lim:
            formatted = [f"{_fmt.format(x)}" for x in obj]
        else:
            # Show first two and last two items
            formatted = (
                [f"{_fmt.format(x)}" for x in obj[:2]]
                + ["..."]
                + [f"{_fmt.format(x)}" for x in obj[-2:]]
            )
        return "[" + sep.join(formatted) + "]"
    return "None"


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
