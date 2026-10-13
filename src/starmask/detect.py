"""星点检测：在银心/地景混杂的画面里找出点源（恒星）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Star:
    x: float
    y: float
    flux: float
    fwhm: float
    peak: float
    sharpness: float
    roundness: float


@dataclass
class StarField:
    stars: list[Star]
    image_shape: tuple[int, int]
    fwhm_median: float
    threshold: float


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    # 感知亮度，略偏重绿
    w = np.array([0.25, 0.55, 0.20], dtype=np.float32)
    return (image.astype(np.float32) * w).sum(axis=-1)


def estimate_fwhm(gray: np.ndarray, max_lag: int = 8) -> float:
    """用自相关粗估 PSF 的 FWHM（像素）。"""
    g = gray - np.median(gray)
    if not np.isfinite(g).all():
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    h, w = g.shape
    cy, cx = h // 2, w // 2
    half = min(h, w, 512) // 2
    patch = g[cy - half : cy + half, cx - half : cx + half]
    if patch.size < 64:
        patch = g

    patch = patch - patch.mean()
    denom = float(np.dot(patch.ravel(), patch.ravel()))
    if denom <= 1e-12:
        return 2.5

    best = 2.5
    for axis in (0, 1):
        line = patch.mean(axis=1 - axis)
        acc = np.correlate(line, line, mode="full")
        acc = acc[len(acc) // 2 :]
        if acc[0] <= 0:
            continue
        acc = acc / acc[0]
        for lag in range(1, min(max_lag, len(acc) - 1)):
            if acc[lag] < 0.5 <= acc[lag - 1]:
                t = (acc[lag - 1] - 0.5) / (acc[lag - 1] - acc[lag] + 1e-12)
                half_corr = lag - 1 + t
                sigma = half_corr / np.sqrt(2.0)
                fwhm = 2.355 * sigma
                best = min(best, fwhm) if best != 2.5 else fwhm
                break
    return float(np.clip(best, 1.2, 12.0))


def _texture_energy(gray: np.ndarray, fwhm: float) -> np.ndarray:
    """局部高频能量：花丛/岩石等纹理区高，平滑夜空低。"""
    sigma = max(fwhm / 2.355, 0.6)
    hp = gray - ndi.gaussian_filter(gray, sigma=sigma * 2.0)
    # 用局部 RMS 作为纹理（clip 避免浮点负数开方）
    e2 = ndi.uniform_filter(hp.astype(np.float32) ** 2, size=5)
    return np.sqrt(np.clip(e2, 0.0, None))


def detect_stars(
    image: np.ndarray,
    threshold_sigma: float = 4.5,
    fwhm: float | None = None,
    max_stars: int = 12000,
    min_separation: float | None = None,
    reject_texture: bool = True,
) -> StarField:
    """检测点源星点。

    Parameters
    ----------
    image : (H, W, 3) 或 (H, W) 浮点图
    threshold_sigma : 相对于背景噪声的检测阈值
    fwhm : 星点半高全宽（像素）；None 则自动估计
    max_stars : 最多保留的星点数（按亮度）
    min_separation : 星点最小间距（像素）；None 取 0.65 * fwhm
    reject_texture : 拒绝高纹理区域的误检（花丛、地景细节）
    """
    gray = _to_gray(image)
    if fwhm is None:
        fwhm = estimate_fwhm(gray)
    fwhm = float(np.clip(fwhm, 1.2, 12.0))
    if min_separation is None:
        min_separation = max(fwhm * 0.65, 1.2)

    sigma_match = max(fwhm / 2.355, 0.6) * 0.7
    sm = ndi.gaussian_filter(gray, sigma=sigma_match)

    bg = ndi.gaussian_filter(gray, sigma=max(fwhm * 4.0, 8.0))
    residual = sm - bg

    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    noise = 1.4826 * mad if mad > 1e-8 else float(np.std(residual) * 0.5 + 1e-8)

    thr_rel = threshold_sigma * noise
    abs_floor = med + 2.0 * noise
    peaks = (residual > thr_rel) & (sm > abs_floor)

    size = max(3, int(round(fwhm)) | 1)
    footprint = np.ones((size, size), dtype=bool)
    mx = ndi.maximum_filter(sm, footprint=footprint, mode="nearest")
    peaks = peaks & (sm == mx)

    border = max(2, int(round(fwhm)))
    peaks[:border, :] = False
    peaks[-border:, :] = False
    peaks[:, :border] = False
    peaks[:, -border:] = False

    tex = _texture_energy(gray, fwhm) if reject_texture else None
    tex_global = float(np.median(tex)) + 2.5 * float(np.median(np.abs(tex - np.median(tex)))) if tex is not None else 0.0

    ys, xs = np.nonzero(peaks)
    if ys.size == 0:
        return StarField(stars=[], image_shape=gray.shape, fwhm_median=fwhm, threshold=abs_floor)

    candidates: list[Star] = []
    half = max(3, int(np.ceil(fwhm * 1.8)))

    for y, x in zip(ys, xs):
        # 四边对称裁剪：贴边星允许半幅 stamp（此前左/上越界直接丢弃，
        # 右/下却只截断，导致边缘星点系统性缺失）
        y0, y1 = y - half, y + half + 1
        x0, x1 = x - half, x + half + 1
        y0c, x0c = max(y0, 0), max(x0, 0)
        y1c, x1c = min(y1, gray.shape[0]), min(x1, gray.shape[1])
        if y1c - y0c < 3 or x1c - x0c < 3:
            continue
        stamp = gray[y0c:y1c, x0c:x1c].astype(np.float64)
        if stamp.size < 9:
            continue

        local_bg = float(np.median(stamp))
        stamp = stamp - local_bg
        peak = float(stamp.max())
        if peak <= 0:
            continue

        # 纹理拒绝：花丛/碎石/叶片的高频远高于夜空（不误杀星核）
        if tex is not None:
            tval = float(tex[y, x])
            patch_tex = float(tex[y0c:y1c, x0c:x1c].mean())
            # 高纹理且扩展 → 地景，不是点源
            if tval > tex_global * 5.0 and patch_tex > peak * 0.35:
                continue
            if tval > max(tex_global * 3.5, peak * 0.55) and peak < 0.55 and patch_tex > peak * 0.30:
                continue

        yy, xx = np.mgrid[y0c:y1c, x0c:x1c]
        wsum = float(stamp.sum())
        if wsum <= 1e-8:
            continue
        cx = float((stamp * xx).sum() / wsum)
        cy = float((stamp * yy).sum() / wsum)

        varx = float((stamp * (xx - cx) ** 2).sum() / wsum)
        vary = float((stamp * (yy - cy) ** 2).sum() / wsum)
        sigx = np.sqrt(max(varx - 1.0 / 12.0, 1e-4))
        sigy = np.sqrt(max(vary - 1.0 / 12.0, 1e-4))
        fwhm_i = 2.355 * float(np.sqrt(max(sigx * sigy, 1e-4)))

        roundness = abs(sigx - sigy) / max(sigx, sigy, 1e-6)
        r = fwhm_i / 2.0
        rr = np.hypot(xx - cx, yy - cy)
        core = stamp[rr <= r]
        core_frac = float(core.sum() / max(stamp.sum(), 1e-8))
        sharpness = core_frac

        if fwhm_i < fwhm * 0.4 or fwhm_i > fwhm * 3.0:
            continue
        if roundness > 0.7:
            continue
        if sharpness < 0.12 or sharpness > 0.95:
            continue
        if abs(cx - x) > fwhm or abs(cy - y) > fwhm:
            continue

        candidates.append(
            Star(
                x=cx,
                y=cy,
                flux=max(wsum, 0.0),
                fwhm=fwhm_i,
                peak=peak + local_bg,
                sharpness=sharpness,
                roundness=roundness,
            )
        )

    candidates.sort(key=lambda s: s.flux, reverse=True)
    candidates = candidates[:max_stars]

    # 网格索引去重：cell = min_separation，只需检查相邻 9 格
    kept: list[Star] = []
    min_separation = max(min_separation, 1.0)
    cell_size = min_separation
    grid: dict[tuple[int, int], list[Star]] = {}

    for s in candidates:
        gx, gy = int(s.x / cell_size), int(s.y / cell_size)
        conflict = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in grid.get((gx + dx, gy + dy), ()):
                    if (s.x - k.x) ** 2 + (s.y - k.y) ** 2 < min_separation**2:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                break
        if not conflict:
            kept.append(s)
            grid.setdefault((gx, gy), []).append(s)

    fwhms = [s.fwhm for s in kept] or [fwhm]
    return StarField(
        stars=kept,
        image_shape=gray.shape,
        fwhm_median=float(np.median(fwhms)),
        threshold=abs_floor,
    )
