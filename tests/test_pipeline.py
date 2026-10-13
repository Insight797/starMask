"""StarMask 流水线测试。"""

from __future__ import annotations

import numpy as np
import pytest

from starmask.detect import detect_stars, estimate_fwhm, _to_gray
from starmask.rawio import save_tiff
from starmask.separate import Method, separate
from starmask.synthetic import make_milkyway_synthetic


@pytest.fixture(scope="module")
def synth() -> np.ndarray:
    return make_milkyway_synthetic(height=600, width=900, n_stars=220, seed=7)


def test_estimate_fwhm_reasonable(synth: np.ndarray) -> None:
    fwhm = estimate_fwhm(_to_gray(synth))
    assert 1.5 <= fwhm <= 5.0


def test_detect_stars_finds_many(synth: np.ndarray) -> None:
    field = detect_stars(synth, threshold_sigma=4.0)
    assert len(field.stars) >= 50
    # 星点应主要在上部天空
    ys = [s.y for s in field.stars]
    assert np.median(ys) < synth.shape[0] * 0.75


def test_psf_separation_reconstructs(synth: np.ndarray) -> None:
    result = separate(synth, method=Method.PSF, threshold_sigma=4.0)
    recon = result.starless + result.stars
    err = np.abs(recon - result.original).mean()
    assert err < 1e-3

    # 无星图应不比原图更「有星」——允许微小数值差
    g0 = result.original.mean(axis=-1)
    g1 = result.starless.mean(axis=-1)
    assert g1.std() <= g0.std() + 0.01


def test_star_layer_mostly_dark_background(synth: np.ndarray) -> None:
    result = separate(synth, method=Method.PSF, threshold_sigma=4.0)
    stars = result.stars.mean(axis=-1)
    # 大部分像素应接近 0（无星背景）
    frac_bright = float((stars > 0.05).mean())
    assert frac_bright < 0.12
    assert stars.max() > 0.1


def test_mask_method_runs(synth: np.ndarray) -> None:
    result = separate(synth, method=Method.MASK, threshold_sigma=4.0)
    assert result.starless.shape == synth.shape
    assert result.stars.shape == synth.shape


def test_morph_method_runs(synth: np.ndarray) -> None:
    result = separate(synth, method=Method.MORPH)
    assert result.starless.shape == synth.shape


def test_save_roundtrip(tmp_path, synth: np.ndarray) -> None:
    p = save_tiff(tmp_path / "t.tif", synth)
    assert p.exists()
    import tifffile

    arr = tifffile.imread(p)
    assert arr.dtype == np.uint16
    assert arr.shape[:2] == synth.shape[:2]


def test_reduce_keeps_bright_removes_faint(synth: np.ndarray) -> None:
    from starmask.reduce import reduce_stars
    from starmask.separate import separate

    result = separate(synth, method=Method.PSF, threshold_sigma=4.0)
    rr = reduce_stars(result.original, result.starless, result.stars, auto=True)
    assert rr.report.n_kept >= 5
    assert rr.report.n_removed >= 1
    # 互补：removed + reduced ≈ original
    err = np.abs(rr.removed_stars + rr.reduced - result.original).mean()
    assert err < 1e-3
    # 缩星图不应比无星图更「有星」到反超原图
    assert rr.reduced.mean() <= result.original.mean() + 0.05
    # 保留的星层比全星层更少（量级上）
    assert rr.kept_stars.mean() < result.stars.mean() + 0.05


def test_reduce_auto_threshold_range() -> None:
    import numpy as np
    from starmask.reduce import auto_flux_threshold

    # 双峰：多数暗星 + 少数亮星
    rng = np.random.default_rng(0)
    faint = rng.lognormal(-4.0, 0.3, 400)
    bright = rng.lognormal(-1.0, 0.2, 30)
    fluxes = np.concatenate([faint, bright])
    thr, note = auto_flux_threshold(fluxes, min_keep=5)
    n_keep = int((fluxes >= thr).sum())
    assert 5 <= n_keep <= 80
    assert thr > np.median(fluxes)
