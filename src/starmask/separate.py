"""星点 / 银河地景分离算法。

目标（硬指标）：
1. 星点图：只有点源星光，不带上银河/地景结构。
2. 银河图：检不出星点。
3. stars + starless ≈ 原图。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import ndimage as ndi

from .detect import Star, StarField, detect_stars


class Method(str, Enum):
    PSF = "psf"
    MASK = "mask"
    MORPH = "morph"
    STARNET = "starnet"  # StarNet2 神经网络（市面工具同级）


@dataclass
class SeparateResult:
    original: np.ndarray
    starless: np.ndarray
    stars: np.ndarray
    star_field: StarField
    method: str


def _local_background(gray: np.ndarray, fwhm: float) -> np.ndarray:
    return ndi.gaussian_filter(gray.astype(np.float32), sigma=max(fwhm * 4.0, 8.0))


def _fit_star_gauss(
    gray: np.ndarray,
    bg: np.ndarray,
    cx: float,
    cy: float,
    fwhm: float,
) -> tuple[float, float, float, float, float] | None:
    half = int(np.ceil(max(fwhm * 2.2, 3.5)))
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + 2 * half + 1
    y1 = y0 + 2 * half + 1
    H, W = gray.shape
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None

    stamp = gray[y0:y1, x0:x1].astype(np.float64)
    s = stamp - bg[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pos = np.clip(s, 0, None)
    wsum = float(pos.sum())
    if wsum <= 1e-8:
        return None

    x_v = float((pos * xx).sum() / wsum)
    y_v = float((pos * yy).sum() / wsum)
    varx = float((pos * (xx - x_v) ** 2).sum() / wsum)
    vary = float((pos * (yy - y_v) ** 2).sum() / wsum)
    fx = float(np.clip(2.355 * np.sqrt(max(varx - 1 / 12, 1e-4)), fwhm * 0.45, fwhm * 2.0))
    fy = float(np.clip(2.355 * np.sqrt(max(vary - 1 / 12, 1e-4)), fwhm * 0.45, fwhm * 2.0))

    amp = 0.0
    for _ in range(3):
        sx = max(fx / 2.355, 0.35)
        sy = max(fy / 2.355, 0.35)
        g = np.exp(-0.5 * (((xx - x_v) / sx) ** 2 + ((yy - y_v) / sy) ** 2))
        denom = float((g * g).sum())
        if denom <= 1e-12:
            return None
        amp = float((np.clip(s, 0, None) * g).sum() / denom)
        if amp <= 0:
            return None
        w2 = np.clip(s, 0, None)
        wsum2 = float(w2.sum())
        if wsum2 <= 1e-8:
            break
        x_v = float((w2 * xx).sum() / wsum2)
        y_v = float((w2 * yy).sum() / wsum2)
        varx = float((w2 * (xx - x_v) ** 2).sum() / wsum2)
        vary = float((w2 * (yy - y_v) ** 2).sum() / wsum2)
        fx = float(np.clip(2.355 * np.sqrt(max(varx - 1 / 12, 1e-4)), fwhm * 0.4, fwhm * 2.2))
        fy = float(np.clip(2.355 * np.sqrt(max(vary - 1 / 12, 1e-4)), fwhm * 0.4, fwhm * 2.2))
    return amp, fx, fy, x_v, y_v


def _star_color(region: np.ndarray, g: np.ndarray) -> np.ndarray:
    thr = float(g.max()) * 0.35 if g.max() > 0 else 0.0
    core = g >= thr if thr > 0 else g > 0
    if not core.any():
        return np.ones(3, dtype=np.float32)
    color = np.array([float(region[..., c][core].mean()) for c in range(3)], dtype=np.float32)
    cmax = float(color.max())
    return color / cmax if cmax > 1e-6 else np.ones(3, dtype=np.float32)


def _subtract_one(
    residual: np.ndarray,
    stars_layer: np.ndarray,
    gray: np.ndarray,
    bg: np.ndarray,
    st: Star,
    fwhm_ref: float,
    gain: float,
) -> None:
    fit = _fit_star_gauss(gray, bg, st.x, st.y, fwhm_ref)
    if fit is None:
        amp, fwhm_x, fwhm_y, x_v, y_v = st.peak, st.fwhm, st.fwhm, st.x, st.y
    else:
        amp, fwhm_x, fwhm_y, x_v, y_v = fit

    size = max(int(np.ceil(max(fwhm_x, fwhm_y) * 2.6)) | 1, 5)
    half = size // 2
    x0 = int(round(x_v)) - half
    y0 = int(round(y_v)) - half
    H, W = gray.shape
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + size, W), min(y0 + size, H)
    if sx0 >= sx1 or sy0 >= sy1:
        return

    yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
    sx = max(fwhm_x / 2.355, 0.35)
    sy = max(fwhm_y / 2.355, 0.35)
    g = np.exp(-0.5 * (((xx - x_v) / sx) ** 2 + ((yy - y_v) / sy) ** 2)) * (amp * gain)

    excess = np.clip(gray[sy0:sy1, sx0:sx1] - bg[sy0:sy1, sx0:sx1], 0, None)
    g = g * np.clip(np.minimum(1.0, (excess + 0.05 * np.maximum(excess, 0.02)) / np.maximum(g, 1e-8)), 0, 1)

    if residual.ndim == 3:
        region = residual[sy0:sy1, sx0:sx1]
        color = _star_color(region, g)
        bg_patch = bg[sy0:sy1, sx0:sx1]
        for c in range(3):
            layer = g * color[c]
            ch_excess = np.clip(residual[sy0:sy1, sx0:sx1, c] - bg_patch * color[c], 0, None)
            layer = np.minimum(layer, ch_excess + 0.02 * np.maximum(ch_excess, 0.01))
            residual[sy0:sy1, sx0:sx1, c] -= layer
            stars_layer[sy0:sy1, sx0:sx1, c] += layer
    else:
        layer = np.minimum(g, excess + 0.02 * np.maximum(excess, 0.01))
        residual[sy0:sy1, sx0:sx1] -= layer
        stars_layer[sy0:sy1, sx0:sx1] += layer


def _fill_peak(
    residual: np.ndarray,
    stars_layer: np.ndarray,
    st: Star,
    fwhm_ref: float,
) -> bool:
    """对残留点源：核内正超额并入星层（外环中值作背景）。

    注意：stamp 必须比星核大，否则环带为空、背景被星核污染。
    """
    span = max(fwhm_ref, st.fwhm, 1.5)
    half = max(int(np.ceil(span * 3.0)), 6)
    x0 = int(round(st.x)) - half
    y0 = int(round(st.y)) - half
    H, W = residual.shape[:2]
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + 2 * half + 1, W), min(y0 + 2 * half + 1, H)
    if sx0 >= sx1 or sy0 >= sy1:
        return False
    yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
    rr = np.hypot(xx - st.x, yy - st.y)
    core = rr <= max(st.fwhm * 1.25, 1.8)
    ring = (rr > max(st.fwhm * 1.6, 2.5)) & (rr <= half * 0.92)
    if not core.any() or not ring.any():
        return False
    region = residual[sy0:sy1, sx0:sx1]
    touched = False
    if residual.ndim == 3:
        for c in range(3):
            ch = region[..., c]
            bg_val = float(np.median(ch[ring]))
            layer = np.where(core, np.clip(ch - bg_val, 0, None), 0.0).astype(np.float64)
            if layer.max() > 1e-6:
                touched = True
            residual[sy0:sy1, sx0:sx1, c] = ch - layer
            stars_layer[sy0:sy1, sx0:sx1, c] += layer
    else:
        bg_val = float(np.median(region[ring]))
        layer = np.where(core, np.clip(region - bg_val, 0, None), 0.0).astype(np.float64)
        if layer.max() > 1e-6:
            touched = True
        residual[sy0:sy1, sx0:sx1] = region - layer
        stars_layer[sy0:sy1, sx0:sx1] += layer
    return touched


def _star_footprint_mask(
    stars: list[Star], shape: tuple[int, int], fwhm_ref: float, scale: float = 1.55
) -> np.ndarray:
    """星点足迹软掩膜：只在星周围保留，去掉星层里混进的银河微光。"""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)
    if not stars:
        return mask
    half = int(np.ceil(max(fwhm_ref * 2.8 * scale, 4)))
    for st in stars:
        x0 = int(round(st.x)) - half
        y0 = int(round(st.y)) - half
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + 2 * half + 1, W), min(y0 + 2 * half + 1, H)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
        sig = max(st.fwhm / 2.355, 0.4) * scale
        g = np.exp(-0.5 * (((xx - st.x) / sig) ** 2 + ((yy - st.y) / sig) ** 2))
        mask[sy0:sy1, sx0:sx1] = np.maximum(mask[sy0:sy1, sx0:sx1], g.astype(np.float32))
    return mask


def _soft_star_mask(star_field: StarField, shape: tuple[int, int], scale: float = 1.3) -> np.ndarray:
    return _star_footprint_mask(star_field.stars, shape, max(star_field.fwhm_median, 1.2), scale=scale)


def separate_psf(img: np.ndarray, threshold_sigma: float, fwhm: float | None, gain: float, max_stars: int) -> SeparateResult:
    gray = (img.mean(axis=-1) if img.ndim == 3 else img).astype(np.float32)
    if fwhm is None:
        from .detect import estimate_fwhm

        fwhm_ref = estimate_fwhm(gray)
    else:
        fwhm_ref = float(fwhm)
    bg = _local_background(gray, fwhm_ref)

    residual = img.copy().astype(np.float32)
    stars_layer = np.zeros_like(img, dtype=np.float32)
    all_stars: list[Star] = []
    n_cap = max_stars
    # 跨轮去重：网格只做索引，真正判定用距离（与 detect.py 一致）。
    # 此前只要相邻 3×3 格有占用就判重，会把同格内的真实双星/密近星误杀。
    min_sep = max(fwhm_ref * 0.5, 1.0)
    cell = max(min_sep, 1.5)
    occupied: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def _is_dup(x: float, y: float) -> bool:
        kx, ky = int(x / cell), int(y / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py in occupied.get((kx + dx, ky + dy), ()):
                    if (x - px) ** 2 + (y - py) ** 2 < min_sep**2:
                        return True
        return False

    def _mark(x: float, y: float) -> None:
        occupied.setdefault((int(x / cell), int(y / cell)), []).append((x, y))

    # 多轮检测 + PSF 扣除（最多 2 轮补检）
    for thr in (threshold_sigma, max(threshold_sigma * 0.7, 2.0)):
        field = detect_stars(
            residual,
            threshold_sigma=thr,
            fwhm=fwhm_ref,
            max_stars=max(n_cap // 2, 2000),
            min_separation=max(fwhm_ref * 0.5, 1.0),
            reject_texture=True,
        )
        new = 0
        for st in field.stars:
            if _is_dup(st.x, st.y):
                continue
            _subtract_one(residual, stars_layer, gray, bg, st, fwhm_ref, gain)
            all_stars.append(st)
            _mark(st.x, st.y)
            new += 1
        if new == 0 or len(all_stars) >= n_cap:
            break

    # 残留尖峰清干净：迭代到检不出点源为止
    for thr in (3.5, 2.8, 2.2, 1.8, 1.5):
        for _ in range(8):
            field = detect_stars(
                residual,
                threshold_sigma=thr,
                fwhm=fwhm_ref,
                max_stars=5000,
                min_separation=1.0,
                reject_texture=True,
            )
            # 只补扣点状源，避免挖掉地景/银河结
            pts = [
                s
                for s in field.stars
                if s.sharpness >= 0.12 and s.roundness <= 0.75 and s.fwhm <= fwhm_ref * 3.2
            ]
            if not pts:
                break
            progress = False
            for st in pts:
                if _fill_peak(residual, stars_layer, st, fwhm_ref):
                    progress = True
                    all_stars.append(st)
                    _mark(st.x, st.y)
            if not progress:
                break

    # 无星图 = 实际残差（不要用掩膜回灌，否则会把星加回去）
    starless = residual
    stars_clean = np.clip(img.astype(np.float32) - starless, 0, None)

    # 仅剔除地景高饱和色斑（花丛等），不影响天空真星
    if img.ndim == 3:
        h = gray.shape[0]
        mx = img.max(axis=-1)
        mn = img.min(axis=-1)
        sat = (mx - mn) / (mx + 1e-6)
        ground = np.zeros(gray.shape, dtype=bool)
        ground[int(h * 0.62) :] = True
        bad = ground & (sat > 0.5) & (stars_clean.mean(axis=-1) > 0.02)
        if bad.any():
            stars_clean[bad] = 0.0
            starless = img.astype(np.float32) - stars_clean

    field = StarField(
        stars=all_stars[:n_cap],
        image_shape=gray.shape,
        fwhm_median=fwhm_ref,
        threshold=0.0,
    )
    return SeparateResult(
        original=img, starless=starless, stars=stars_clean, star_field=field, method="psf"
    )


def _separate_mask(image: np.ndarray, star_field: StarField, fwhm: float) -> tuple[np.ndarray, np.ndarray]:
    gray = image.mean(axis=-1) if image.ndim == 3 else image
    mask = _soft_star_mask(star_field, gray.shape, scale=1.2)
    radius = max(int(np.ceil(fwhm)), 2)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    selem = (xx**2 + yy**2) <= radius**2
    opened = ndi.grey_opening(gray, footprint=selem)
    m3 = mask[..., None] if image.ndim == 3 else mask
    if image.ndim == 3:
        starless = image.astype(np.float32) * (1.0 - m3) + opened[..., None] * m3
    else:
        starless = image.astype(np.float32) * (1 - mask) + opened * mask
    return starless, image.astype(np.float32) - starless


def _separate_morph(image: np.ndarray, fwhm: float) -> tuple[np.ndarray, np.ndarray]:
    gray = image.mean(axis=-1) if image.ndim == 3 else image
    radius = max(int(np.ceil(fwhm)), 2)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    selem = (xx**2 + yy**2) <= radius**2
    opened = ndi.grey_opening(gray, footprint=selem)
    tophat = gray - opened
    if image.ndim == 3:
        ratio = np.clip(image.astype(np.float32) / (gray[..., None] + 1e-6), 0, 2.0)
        for c in range(3):
            ratio[..., c] = ndi.uniform_filter(ratio[..., c], size=3)
        stars = tophat[..., None] * ratio
    else:
        stars = tophat
    return image.astype(np.float32) - stars, stars


def separate(
    image: np.ndarray,
    method: Method | str = Method.PSF,
    fwhm: float | None = None,
    threshold_sigma: float = 4.5,
    gain: float = 0.95,
    max_stars: int = 12000,
    log=None,
    glow: float = 0.85,
) -> SeparateResult:
    method = Method(method)
    img = np.asarray(image, dtype=np.float32)

    if method == Method.STARNET:
        from .starnet2 import separate_starnet2

        return separate_starnet2(img, log=log, glow=glow)

    if method == Method.PSF:
        result = separate_psf(img, threshold_sigma, fwhm, gain, max_stars)
    elif method == Method.MASK:
        field = detect_stars(img, threshold_sigma=threshold_sigma, fwhm=fwhm, max_stars=max_stars)
        starless, stars = _separate_mask(img, field, field.fwhm_median)
        result = SeparateResult(img, starless, stars, field, method.value)
    else:
        from .detect import estimate_fwhm

        gray = img.mean(axis=-1) if img.ndim == 3 else img
        fw = fwhm or estimate_fwhm(gray)
        field = detect_stars(img, threshold_sigma=threshold_sigma, fwhm=fw, max_stars=max_stars)
        starless, stars = _separate_morph(img, fw)
        result = SeparateResult(img, starless, stars, field, method.value)

    stars = np.clip(result.stars, 0, None)
    starless = result.original - stars
    if glow and method != Method.STARNET:
        # StarNet2 路径内部已恢复光晕；经典方法在这里补
        from .starnet2 import _restore_star_glow

        _restore_star_glow(result.original, starless, stars, fwhm=result.star_field.fwhm_median or 2.5, strength=glow, log=log)
        stars = np.clip(stars, 0, None)
        starless = np.clip(result.original - stars, 0, None)
    return SeparateResult(result.original, starless, stars, result.star_field, result.method)
