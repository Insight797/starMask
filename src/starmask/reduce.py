"""缩星：自动保留高亮度星点，去掉大部分暗弱/低质量星点。

摄影工作流里常见需求——不把星全抹掉，而是只压暗星、留下亮星当视觉锚点。

输出层关系：
    removed + kept = stars
    starless + stars = original
    reduced = starless + kept = original - removed   （最终缩星图）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


@dataclass
class ReduceReport:
    n_stars: int
    n_kept: int
    n_removed: int
    flux_threshold: float
    peak_threshold: float
    keep_ratio_actual: float
    mode: str  # auto | ratio
    reasons: list[str] = field(default_factory=list)


@dataclass
class ReduceResult:
    reduced: np.ndarray
    kept_stars: np.ndarray
    removed_stars: np.ndarray
    keep_mask: np.ndarray
    report: ReduceReport


def _star_scores(stars_img: np.ndarray, fwhm: float = 2.5):
    """在星点层上编星表：返回 [(y, x, flux, peak, radius), …] 按 flux 降序。"""
    from .detect import detect_stars

    field = detect_stars(
        stars_img,
        threshold_sigma=2.8,
        fwhm=fwhm,
        max_stars=12000,
        min_separation=max(fwhm * 0.6, 1.0),
        reject_texture=True,
    )
    rows = []
    for st in field.stars:
        # 低质量过滤：过肥（星云结）、过瘦（噪声）
        if st.fwhm < fwhm * 0.35 or st.fwhm > fwhm * 2.8:
            continue
        if st.roundness > 0.85 or st.sharpness < 0.08:
            continue
        rows.append((float(st.y), float(st.x), float(st.flux), float(st.peak), float(st.fwhm)))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def _otsu_threshold(x: np.ndarray) -> float:
    """一维 Otsu：把 log 通量分成暗/亮两簇，返回分界（原量纲）。"""
    if x.size < 16:
        return float(np.median(x)) if x.size else 0.0
    logx = np.log10(np.clip(x, 1e-8, None))
    lo, hi = float(logx.min()), float(logx.max())
    if hi - lo < 1e-6:
        return float(x[0])
    bins = np.linspace(lo, hi, 129)
    hist, edges = np.histogram(logx, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
        m1 = (np.cumsum(hist * centers)[::-1] / np.maximum(w1[::-1], 1))[::-1]
    # 用总和算类间方差
    total = w0[-1]
    mu = float((hist * centers).sum() / max(total, 1))
    # 重新算 m0/m1 更稳
    s0 = np.cumsum(hist * centers)
    s1 = s0[-1] - s0
    m0 = s0 / np.maximum(w0, 1)
    m1 = s1 / np.maximum(w1, 1)
    var_between = w0 * w1 * (m0 - m1) ** 2
    i = int(np.argmax(var_between))
    return float(10 ** centers[i])


def auto_flux_threshold(fluxes: np.ndarray, min_keep: int = 20, max_keep_frac: float = 0.22):
    """自动找「亮星 / 暗星」分界。

    优先 Otsu 双簇（暗弱场星 vs 高亮星）；若保留比例离谱再找 log 间隙；最后回退 12%。
    返回 (threshold, note)。
    """
    if fluxes.size == 0:
        return 0.0, "empty"
    fx = np.sort(np.asarray(fluxes, dtype=np.float64))
    otsu_note = ""

    def _ok(thr: float) -> bool:
        n_keep = int((fx >= thr).sum())
        frac = n_keep / fx.size
        return n_keep >= min_keep and frac <= max_keep_frac and frac >= min(min_keep / fx.size, 0.02)

    # 1) Otsu on log flux（仅当呈「暗星海 + 亮星」双簇时）
    thr = _otsu_threshold(fx)
    n_keep = int((fx >= thr).sum())
    if _ok(thr):
        return float(thr), f"otsu keep {n_keep}/{fx.size}"
    otsu_note = f"otsu {n_keep}/{fx.size} rejected"

    # 2) 显著间隙（亮端）
    lo = max(min_keep, int(fx.size * 0.02))
    hi = max(lo + 2, int(fx.size * max_keep_frac))
    hi = min(hi, fx.size)
    if hi - lo >= 4:
        logf = np.log10(np.clip(fx[lo:hi], 1e-8, None))
        gaps = np.diff(logf)
        med = float(np.median(gaps)) if gaps.size else 0.0
        best_i, best_w = -1, 0.0
        for i, w in enumerate(gaps):
            if w > max(med * 2.0, 0.15) and w > best_w:
                best_i, best_w = i, float(w)
        if best_i >= 0:
            gthr = float(0.5 * (fx[lo + best_i] + fx[lo + best_i + 1]))
            if _ok(gthr):
                return gthr, f"gap {best_w:.2f} dex keep {int((fx>=gthr).sum())}/{fx.size}"

    # 3) 连续光度函数：按亮度序取前 12% 作为「高亮度星点」
    k = int(np.clip(round(fx.size * 0.10), min_keep, max(1, int(fx.size * max_keep_frac))))
    note = f"{otsu_note} ratio {k}/{fx.size}" if otsu_note else f"ratio {k}/{fx.size}"
    return float(fx[-k]), note


def build_keep_mask(
    shape: tuple[int, int],
    kept_rows,
    scale: float = 2.0,
) -> np.ndarray:
    """亮星软足迹（0~1），保证与星点层相乘后只留亮星。"""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)
    if not kept_rows:
        return mask
    for y, x, _flux, _peak, fwhm in kept_rows:
        half = int(np.ceil(max(fwhm * 3.2 * scale, 8)))
        x0 = int(round(x)) - half
        y0 = int(round(y)) - half
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + 2 * half + 1, W), min(y0 + 2 * half + 1, H)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
        sig = max(fwhm / 2.355, 0.4) * scale
        g = np.exp(-0.5 * (((xx - x) / sig) ** 2 + ((yy - y) / sig) ** 2))
        mask[sy0:sy1, sx0:sx1] = np.maximum(mask[sy0:sy1, sx0:sx1], g.astype(np.float32))
    return mask


def reduce_stars(
    original: np.ndarray,
    starless: np.ndarray,
    stars: np.ndarray,
    keep_ratio: float | None = None,
    auto: bool = True,
    fwhm: float = 2.5,
    min_keep: int = 15,
    log=None,
    beauty_glow: float = 0.0,
) -> ReduceResult:
    """保留亮星、去掉暗星。

    - ``auto=True``：按亮度分布自动分界（推荐）
    - ``keep_ratio``：手动保留最亮的比例（0.05~0.40）；给定后优先于 auto
    """
    orig = np.asarray(original, dtype=np.float32)
    sl = np.asarray(starless, dtype=np.float32)
    st = np.asarray(stars, dtype=np.float32)
    g = st.mean(axis=-1) if st.ndim == 3 else st
    shape = g.shape

    if log:
        log("  编制星表 / 亮度分层 …")
    rows = _star_scores(st, fwhm=fwhm)
    if not rows:
        # 星层为空：缩星=无星图
        zeros = np.zeros_like(st)
        rep = ReduceReport(0, 0, 0, 0.0, 0.0, 0.0, "empty", ["星表为空"])
        return ReduceResult(sl.copy(), zeros, zeros.copy(), np.zeros(shape, np.float32), rep)

    fluxes = np.array([r[2] for r in rows], dtype=np.float64)
    peaks = np.array([r[3] for r in rows], dtype=np.float64)

    mode = "auto"
    if keep_ratio is not None:
        k = int(np.clip(round(len(rows) * float(keep_ratio)), min_keep, len(rows)))
        thr = float(np.sort(fluxes)[-k])
        note = f"ratio {keep_ratio:.0%} keep {k}/{len(rows)}"
        mode = "ratio"
    elif auto:
        thr, note = auto_flux_threshold(fluxes, min_keep=min_keep)
    else:
        thr, note = float(np.median(fluxes)), "median"
        mode = "median"

    kept_rows = [r for r in rows if r[2] >= thr]
    # 保险：至少留 min_keep 颗
    if len(kept_rows) < min_keep and rows:
        kept_rows = rows[: min(min_keep, len(rows))]
        thr = kept_rows[-1][2]
        note += " +min_keep"

    keep_mask = build_keep_mask(shape, kept_rows)
    kept = st * (keep_mask[..., None] if st.ndim == 3 else keep_mask)
    removed = np.clip(st - kept, 0, None)
    # 保证互补：缩星图 = 原图 - 去掉的暗星
    reduced = np.clip(orig - removed, 0, None)
    if beauty_glow and kept_rows:
        if log:
            log(f"  亮星美晕 strength={beauty_glow:.2f}（艺术增亮，互补性略放宽）")
        reduced = enhance_bright_star_glow(
            reduced, kept_rows, strength=float(beauty_glow), fwhm=fwhm
        )
        kept = np.clip(reduced - sl, 0, None)

    if log:
        log(
            f"  保留亮星 {len(kept_rows)}/{len(rows)}（{len(kept_rows)/max(len(rows),1):.0%}），"
            f"去掉暗星 {len(rows) - len(kept_rows)}  [{note}]"
        )

    rep = ReduceReport(
        n_stars=len(rows),
        n_kept=len(kept_rows),
        n_removed=len(rows) - len(kept_rows),
        flux_threshold=float(thr),
        peak_threshold=float(peaks[fluxes >= thr].min()) if (fluxes >= thr).any() else 0.0,
        keep_ratio_actual=len(kept_rows) / max(len(rows), 1),
        mode=mode,
        reasons=[note],
    )
    return ReduceResult(reduced=reduced, kept_stars=kept, removed_stars=removed, keep_mask=keep_mask, report=rep)


def format_reduce(r: ReduceReport) -> str:
    return (
        f"[缩星] 共 {r.n_stars} 星 保留 {r.n_kept}  去掉 {r.n_removed}  "
        f"保留率 {r.keep_ratio_actual:.0%}  阈值 flux={r.flux_threshold:.3g}  ({r.mode}: {', '.join(r.reasons)})"
    )


def enhance_bright_star_glow(
    image: np.ndarray,
    kept_rows,
    strength: float = 0.45,
    fwhm: float = 2.5,
    warmth: float = 0.12,
    max_stars: int = 40,
) -> np.ndarray:
    """给高亮星加柔和美晕（bloom）：宽斯光斑 + 轻微暖调，增强氛围感。

    仅作用于已判定保留的亮星；``strength=0`` 时原样返回。
    注意：会略增亮，``removed + reduced`` 不再严格等于原图（艺术向）。
    """
    from scipy import ndimage as ndi

    if strength <= 0 or not kept_rows:
        return np.asarray(image, dtype=np.float32)

    img = np.asarray(image, dtype=np.float32)
    rows = sorted(kept_rows, key=lambda r: r[2], reverse=True)[:max_stars]
    glow = np.zeros(img.shape[:2], dtype=np.float32)
    h, w = glow.shape

    for y, x, flux, peak, star_fwhm in rows:
        fw = max(star_fwhm, fwhm, 1.2)
        # 美晕半径：明显大于 PSF 翼，才叫「氛围」
        sigma = fw * 2.1
        half = int(np.ceil(sigma * 3.5))
        x0, y0 = int(round(x)) - half, int(round(y)) - half
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + 2 * half + 1, w), min(y0 + 2 * half + 1, h)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
        # 峰值越高晕越明显
        amp = float(peak) * 0.55 * float(strength)
        g = np.exp(-0.5 * (((xx - x) / sigma) ** 2 + ((yy - y) / sigma) ** 2)) * amp
        glow[sy0:sy1, sx0:sx1] = np.maximum(glow[sy0:sy1, sx0:sx1], g.astype(np.float32))

    if img.ndim == 3:
        # 轻微暖调：核偏黄、外缘略青，接近折射镜星芒色
        warm = np.array([1.0 + warmth, 1.0 + warmth * 0.35, 1.0 - warmth * 0.25], dtype=np.float32)
        cool = np.array([1.0 - warmth * 0.2, 1.0, 1.0 + warmth * 0.4], dtype=np.float32)
        mix = ndi.gaussian_filter(glow, max(fwhm * 0.6, 0.8))
        wsum = float(mix.max()) + 1e-8
        core_w = np.clip(glow / wsum, 0, 1)[..., None]
        tint = cool * (1.0 - core_w) + warm * core_w
        out = img + glow[..., None] * tint
    else:
        out = img + glow
    return np.clip(out, 0.0, None)
