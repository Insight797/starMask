"""生成 side-by-side 对比预览 PNG：原图 | 无星 | 星点。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _stretch(a: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(a, (lo_p, hi_p))
    x = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
    return (x * 255).astype(np.uint8)


def _stretch_stars(a: np.ndarray) -> np.ndarray:
    hi = float(np.percentile(a, 99.8))
    x = np.clip(a / (hi + 1e-8), 0, 1)
    return (x * 255).astype(np.uint8)


def save_compare_png(
    original: np.ndarray,
    starless: np.ndarray,
    stars: np.ndarray,
    path: Path | str,
) -> Path:
    from PIL import Image

    o = _stretch(original)
    s = _stretch(starless)
    t = _stretch_stars(stars)
    comp = np.concatenate([o, s, t], axis=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(comp).save(path, optimize=True)
    return path
