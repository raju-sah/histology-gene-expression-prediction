"""Execute the Python cells of a generated Colab notebook locally with Colab/HF/GPU pieces faked.

Cells that start with `!` / `%`-magics, mount Drive or log in are skipped; `%%writefile` cells are skipped because
the modules they write are importable from src/hestfm. Overrides are injected after specific cells run.
"""
from __future__ import annotations

import nbformat

SKIP_MARKERS = ("google.colab", "nvidia-smi", "!pip", "%%writefile", "!du")


def run_notebook(path, overrides_after: dict[str, dict], init: dict | None = None, skip_containing=(),
                 replace: dict[str, str] | None = None):
    """overrides_after maps a substring of a cell to a dict of globals set right after that cell runs."""
    nb = nbformat.read(path, 4)
    g = {"__name__": "__nb__"}
    g.update(init or {})
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        src = c.source
        if any(m in src for m in SKIP_MARKERS + tuple(skip_containing)):
            continue
        for a, b in (replace or {}).items():
            src = src.replace(a, b)
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        exec(compile(src, f"<cell>", "exec"), g)
        for marker, ov in overrides_after.items():
            if marker in c.source:
                g.update(ov(g) if callable(ov) else ov)
    return g
