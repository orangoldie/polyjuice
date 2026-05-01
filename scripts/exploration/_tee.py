"""Tee sys.stdout to a file under data/exploration/ so probe output is captured."""
from __future__ import annotations

import sys
from pathlib import Path


def tee_stdout(filename: str) -> None:
    out = Path('data/exploration') / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open('w', encoding='utf-8')
    real = sys.stdout

    class _Tee:
        def write(self, s):
            real.write(s)
            f.write(s)

        def flush(self):
            real.flush()
            f.flush()

    sys.stdout = _Tee()
