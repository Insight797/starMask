"""生成合成银河测试图，并写一个最小 RAW/TIFF 夹具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def make_milkyway_synthetic(
    height: int = 800,
    width: int = 1200,
    n_stars: int = 350,
    seed: int = 42,
) -> np.ndarray:
    """合成一张含银河、星点、地景的 RGB 测试图 (float32, [0,1])。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.float32)

    yy, xx = np.mgrid[0:height, 0:width]
    nx = xx / width
    ny = yy / height

    # --- 银河带：斜向高斯亮带 + 尘埃暗纹 ---
    band = np.exp(-0.5 * ((ny - 0.55 * nx - 0.18) / 0.10) ** 2)
    # 大尺度云气
    for scale, amp, col in (
        (40, 0.22, (0.95, 0.85, 0.70)),
        (18, 0.12, (0.75, 0.70, 0.90)),
        (8, 0.06, (0.90, 0.65, 0.55)),
    ):
        noise = rng.normal(0, 1, (height, width)).astype(np.float32)
        cloud = ndi.gaussian_filter(noise, scale)
        cloud = cloud / (np.abs(cloud).max() + 1e-8)
        layer = (band * (0.4 + 0.6 * (cloud * 0.5 + 0.5))).astype(np.float32)
        for c in range(3):
            img[..., c] += amp * layer * col[c]

    # 尘埃带（暗纹）
    dust = np.exp(-0.5 * ((ny - 0.55 * nx - 0.28) / 0.03) ** 2)
    img *= (1.0 - 0.45 * dust)[..., None]

    # --- 地景剪影（下 1/4 山形）---
    horizon = 0.72 + 0.04 * np.sin(nx * 9.0) + 0.03 * np.sin(nx * 23.0 + 1.2)
    ground = ny > horizon
    for c, col in enumerate((0.08, 0.09, 0.11)):
        img[..., c] = np.where(ground, col, img[..., c])

    # --- 星点 ---
    # 测量 FWHM 约 2.4 px
    fwhm = 2.4
    sig = fwhm / 2.355
    half = 5
    for _ in range(n_stars):
        x = rng.uniform(0, width - 1)
        y = rng.uniform(0, height * 0.75)
        # 亮度幂律，多数暗星
        amp = float(rng.power(2.2)) * 0.85 + 0.08
        if rng.random() < 0.04:
            amp = float(rng.uniform(0.7, 1.0))  # 少数亮星
        color = rng.normal([1.0, 0.95, 0.9], 0.05)
        color = np.clip(color, 0.3, 1.2)

        x0, x1 = int(x) - half, int(x) + half + 1
        y0, y1 = int(y) - half, int(y) + half + 1
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-0.5 * (((xs - x) / sig) ** 2 + ((ys - y) / sig) ** 2))
        for c in range(3):
            img[y0:y1, x0:x1, c] += (amp * color[c] * g).astype(np.float32)

    # 少量噪声
    img += rng.normal(0, 0.008, img.shape).astype(np.float32)

    # 模拟地景上不出现亮星
    img[ground] *= 0.3

    return np.clip(img, 0.0, 1.0)


def save_synthetic(path: Path) -> Path:
    from .rawio import save_tiff

    img = make_milkyway_synthetic()
    return save_tiff(path, img)
