from __future__ import annotations

from pathlib import Path

from cradio_mlx.modeling import CRadioMLX


def load(path: str | Path) -> CRadioMLX:
    return CRadioMLX.load(path)
