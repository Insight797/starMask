"""分离质量指标：自动验收「星点图无杂质、银河图无星点」。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


@dataclass
class QualityReport:
    n_stars: int
    recon_err: float
    # 星点图纯净度
    stars_bg_mean: float
    stars_bg_p99: float
    stars_frac_bright: float
    stars_structure_leak: float
    # 银河图无星程度（用星点检测数，而不是朴素局部极大）
    starless_n_detected: int
    starless_max_flux: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


def _is_point(st, fwhm: float | None) -> bool:
    return st.sharpness >= 0.18 and st.roundness <= 0.55 and st.fwhm <= (fwhm or 3.0) * 2.4


def evaluate_quality(
    original: np.ndarray,
    starless: np.ndarray,
    stars: np.ndarray,
    fwhm: float | None = None,
    bg_p99_max: float = 0.02,
    leak_max: float = 0.004,
    residual_stars_max: int = 12,
    residual_flux_ratio: float = 0.05,
    recon_err_max: float = 1e-3,
    border: int = 12,
) -> QualityReport:
    from .detect import detect_stars

    orig = np.asarray(original, dtype=np.float32)
    sl = np.asarray(starless, dtype=np.float32)
    st = np.asarray(stars, dtype=np.float32)

    recon_err = float(np.abs(st + sl - orig).mean())
    g_st = st.mean(axis=-1) if st.ndim == 3 else st

    bg_mask = g_st < 0.02
    if bg_mask.any():
        bg_mean = float(g_st[bg_mask].mean())
        bg_p99 = float(np.percentile(g_st[bg_mask], 99))
    else:
        bg_mean, bg_p99 = float(g_st.mean()), float(np.percentile(g_st, 99))
    frac_bright = float((g_st > 0.03).mean())

    # 结构/地景污染：剔除点源足迹后，星层地景低频应接近 0
    # （地景上方的真星点不算污染）
    h = g_st.shape[0]
    ground = g_st[int(h * 0.72) :]
    if ground.size:
        peaks = (ground == ndi.maximum_filter(ground, 5)) & (ground > 0.02)
        point_mask = ndi.maximum_filter(peaks.astype(np.float32), size=11)
        residual_struct = ground * (1.0 - point_mask)
        structure_leak = float(ndi.gaussian_filter(residual_struct, 3.0).mean())
    else:
        structure_leak = 0.0
    leak = structure_leak

    # 银河图残留：排除拼贴/卷积边界伪峰
    g_sl = sl.mean(axis=-1) if sl.ndim == 3 else sl

    def _drop_border(field):
        b = border
        if b <= 0:
            return list(field.stars)
        out = []
        for s in field.stars:
            if b <= s.x < (g_sl.shape[1] - b) and b <= s.y < (g_sl.shape[0] - b):
                out.append(s)
        return out

    field_sl = detect_stars(sl, threshold_sigma=3.5, fwhm=fwhm, max_stars=5000)
    field_or = detect_stars(orig, threshold_sigma=3.5, fwhm=fwhm, max_stars=5000)

    # 地景高饱和色斑不算星
    if sl.ndim == 3:
        mx_s = sl.max(axis=-1)
        mn_s = sl.min(axis=-1)
        sat_s = (mx_s - mn_s) / (mx_s + 1e-6)
    else:
        sat_s = np.zeros(g_sl.shape, dtype=np.float32)

    def _is_sky_point(s) -> bool:
        if not _is_point(s, fwhm):
            return False
        yi, xi = int(s.y), int(s.x)
        if 0 <= yi < sat_s.shape[0] and 0 <= xi < sat_s.shape[1]:
            if sat_s[yi, xi] > 0.55:
                return False
        return True

    res_pts = [s for s in _drop_border(field_sl) if _is_sky_point(s)]
    or_pts = [s for s in _drop_border(field_or) if _is_sky_point(s)]
    n_det = len(res_pts)
    flux_sl = sum(s.flux for s in res_pts)
    flux_or = sum(s.flux for s in or_pts) + 1e-8
    flux_ratio = flux_sl / flux_or
    max_flux = max((s.flux for s in res_pts), default=0.0)

    reasons: list[str] = []
    if bg_p99 > bg_p99_max:
        reasons.append(f"星点图背景不纯 p99={bg_p99:.4f}>{bg_p99_max}")
    if structure_leak > leak_max:
        reasons.append(f"星点图含地景结构 leak={structure_leak:.4f}>{leak_max}")
    if n_det > residual_stars_max:
        reasons.append(f"银河图残留星点 n={n_det}>{residual_stars_max}")
    if flux_ratio > residual_flux_ratio:
        reasons.append(f"银河图残留星流量比 {flux_ratio:.4f}>{residual_flux_ratio}")
    if recon_err > recon_err_max:
        reasons.append(f"互补性误差 {recon_err:.2e}")

    return QualityReport(
        n_stars=0,
        recon_err=recon_err,
        stars_bg_mean=bg_mean,
        stars_bg_p99=bg_p99,
        stars_frac_bright=frac_bright,
        stars_structure_leak=leak,
        starless_n_detected=n_det,
        starless_max_flux=max_flux,
        passed=len(reasons) == 0,
        reasons=reasons,
    )


def format_report(r: QualityReport) -> str:
    status = "PASS" if r.passed else "FAIL"
    lines = [
        f"[{status}] recon_err={r.recon_err:.2e} n_stars={r.n_stars}",
        f"  stars  bg_mean={r.stars_bg_mean:.5f} bg_p99={r.stars_bg_p99:.5f} bright={r.stars_frac_bright:.4f} leak={r.stars_structure_leak:.3f}",
        f"  starless residual_detected={r.starless_n_detected} max_flux={r.starless_max_flux:.4g}",
    ]
    for reason in r.reasons:
        lines.append(f"  ! {reason}")
    return "\n".join(lines)
