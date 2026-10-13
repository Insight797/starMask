"""StarMask — 银河摄影缩星 / 星点与银河分离工具。"""

from __future__ import annotations

__version__ = "0.4.1"

from .pipeline import separate_image, SeparateResult

__all__ = ["separate_image", "SeparateResult", "__version__"]
