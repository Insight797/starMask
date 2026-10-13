"""完整处理流水线：读取 → 分离 → 导出。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preview import save_compare_png
from .rawio import load_image, save_dng, save_tiff
from .separate import Method, SeparateResult, separate

__all__ = ["separate_image", "process_file", "assign_stems", "SeparateResult", "Method"]


def separate_image(
    image: np.ndarray,
    method: Method | str = Method.PSF,
    fwhm: float | None = None,
    threshold_sigma: float = 4.5,
    gain: float = 0.95,
    max_stars: int = 12000,
    log=None,
    glow: float = 0.85,
) -> SeparateResult:
    """对内存中的图像做星点/银河分离。"""
    return separate(
        image,
        method=method,
        fwhm=fwhm,
        threshold_sigma=threshold_sigma,
        gain=gain,
        max_stars=max_stars,
        log=log,
        glow=glow,
    )


@dataclass
class ProcessPaths:
    source: Path
    stars: Path
    starless: Path
    mask: Path | None
    n_stars: int
    fwhm: float
    reduced: Path | None = None
    removed: Path | None = None
    kept: Path | None = None
    n_kept: int = 0
    n_removed_stars: int = 0


def assign_stems(paths: list[Path | str]) -> dict[Path, str]:
    """为批量输入分配互不冲突的输出 stem。

    `a/IMG_001.NEF` 与 `b/IMG_001.ARW` 若都用 ``path.stem`` 作输出名，
    后处理的会静默覆盖先处理的。同名时给后面的加上父目录名，仍冲突再加序号。
    返回 {输入路径: 输出 stem}，供 process_file(stem=...) 与预览定位共用。
    """
    used: dict[str, Path] = {}
    out: dict[Path, str] = {}
    for p in paths:
        path = Path(p)
        stem = path.stem
        if stem in used:
            parent = path.parent.name
            stem = f"{parent}_{path.stem}" if parent else path.stem
        base, n = stem, 2
        while stem in used:
            stem = f"{base}_{n}"
            n += 1
        used[stem] = path
        out[path] = stem
    return out


def _save_layer(out_dir: Path, stem: str, kind: str, arr, image_format: str, dng_scene_linear: bool = False) -> Path:
    """按 image_format 保存一层：tif / dng / both（both 返回 tif 路径，同时写 dng）。"""
    fmt = (image_format or "tif").lower()
    if fmt == "dng":
        return save_dng(out_dir / f"{stem}_{kind}.dng", arr, scene_linear=dng_scene_linear)
    tif_path = save_tiff(out_dir / f"{stem}_{kind}.tif", arr)
    if fmt in {"both", "tif+dng", "dng+tif"}:
        save_dng(out_dir / f"{stem}_{kind}.dng", arr, scene_linear=dng_scene_linear)
    return tif_path


def process_file(
    input_path: Path | str,
    output_dir: Path | str | None = None,
    stem: str | None = None,
    method: Method | str = Method.PSF,
    fwhm: float | None = None,
    threshold_sigma: float = 4.5,
    gain: float = 0.95,
    max_stars: int = 12000,
    write_mask: bool = True,
    write_compare: bool = False,
    preview: bool = False,
    log=None,
    reduce_bright: bool = False,
    keep_ratio: float | None = None,
    image_format: str = "tif",
    dng_scene_linear: bool = False,
    glow: float = 0.85,
    beauty_glow: float = 0.0,
) -> ProcessPaths:
    """处理单个文件，导出星点图与无星图。"""
    input_path = Path(input_path)
    out_dir = Path(output_dir) if output_dir else input_path.parent / "starmask_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or input_path.stem

    if log:
        log(f"读取 {input_path.name} …")
    image = load_image(input_path, mode="preview" if preview else "auto")
    if log:
        log(f"分离引擎: {method} …")
    result = separate_image(
        image,
        method=method,
        fwhm=fwhm,
        threshold_sigma=threshold_sigma,
        gain=gain,
        max_stars=max_stars,
        log=log,
        glow=glow,
    )

    if log:
        log("导出 TIFF …")
    stars_path = _save_layer(out_dir, stem, "stars", result.stars, image_format, dng_scene_linear)
    starless_path = _save_layer(out_dir, stem, "starless", result.starless, image_format, dng_scene_linear)

    mask_path = None
    if write_mask and method not in (Method.STARNET, "starnet", "starnet2"):
        from .separate import _soft_star_mask

        gray = result.original.mean(axis=-1) if result.original.ndim == 3 else result.original
        mask = _soft_star_mask(result.star_field, gray.shape)
        mask_path = _save_layer(out_dir, stem, "starmask", mask, image_format if image_format != "dng" else "tif")

    reduced_path = removed_path = kept_path = None
    n_kept = n_removed_stars = 0
    if reduce_bright:
        from .reduce import format_reduce, reduce_stars

        if log:
            log("缩星：自动保留高亮度星点 …")
        rr = reduce_stars(
            result.original,
            result.starless,
            result.stars,
            keep_ratio=keep_ratio,
            auto=keep_ratio is None,
            fwhm=result.star_field.fwhm_median or 2.5,
            log=log,
            beauty_glow=beauty_glow,
        )
        if log:
            log(format_reduce(rr.report))
        reduced_path = _save_layer(out_dir, stem, "reduced", rr.reduced, image_format, dng_scene_linear)
        kept_path = _save_layer(out_dir, stem, "stars_kept", rr.kept_stars, image_format, dng_scene_linear)
        removed_path = _save_layer(out_dir, stem, "stars_removed", rr.removed_stars, image_format, dng_scene_linear)
        n_kept, n_removed_stars = rr.report.n_kept, rr.report.n_removed
        if write_compare:
            save_compare_png(
                result.original,
                rr.reduced,
                rr.removed_stars,
                out_dir / f"{stem}_reduce_compare.png",
            )

    if write_compare:
        save_compare_png(
            result.original,
            result.starless,
            result.stars,
            out_dir / f"{stem}_compare.png",
        )

    return ProcessPaths(
        source=input_path,
        stars=stars_path,
        starless=starless_path,
        mask=mask_path,
        n_stars=len(result.star_field.stars),
        fwhm=result.star_field.fwhm_median,
        reduced=reduced_path,
        removed=removed_path,
        kept=kept_path,
        n_kept=n_kept,
        n_removed_stars=n_removed_stars,
    )
