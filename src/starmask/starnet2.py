"""StarNet2 神经网络去星后端。

调用官方 StarNet2 CLI（starnetastro.com，与 StarXTerminator/市面工具同级的 CNN 端到端模型）。

CLI 约定（2.6.x）:
  starnet2 -i in.tif -o starless.tif -m mask.tif -n stars.tif -s <stride>
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .rawio import _to_unit_float, imread_any, save_tiff
from .separate import SeparateResult
from .detect import StarField, detect_stars, estimate_fwhm

# 仓库内默认安装位置
ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_BINS = [
    ROOT / "third_party" / "starnet2" / "bin" / "starnet2",
    ROOT / "third_party" / "starnet2" / "starnet2",
    ROOT / "third_party" / "starnet2" / "starnet++",
    Path("/usr/local/bin/starnet2"),
    Path("/opt/homebrew/bin/starnet2"),
    Path.home() / "bin" / "starnet2",
]

DOWNLOAD_PAGE = "https://starnetastro.com/cli-tools/starnet/"
MACOS_ARM64_ZIP = (
    "https://download.starnetastro.com/starnet2_macos-arm64_2.6.2-0242_COREML_arm64_cli.zip"
)


def _frozen_search_dirs() -> list[Path]:
    """PyInstaller / .app 内可能的数据目录。"""
    dirs: list[Path] = []
    meipass = getattr(__import__("sys"), "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass))
    exe = Path(__import__("sys").executable)
    for base in (
        exe.parent,
        exe.parent.parent / "Resources",
        exe.parent.parent / "Frameworks",
        exe.parent / "_internal",
        exe.parent.parent / "_internal",
    ):
        dirs.append(base)
    # 开发仓库
    dirs.append(ROOT / "third_party" / "starnet2")
    dirs.append(ROOT / "third_party" / "starnet2" / "bin")
    out: list[Path] = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def find_starnet2() -> Path | None:
    """定位 starnet2 可执行文件。"""
    env = os.environ.get("STARNET2_BIN")
    if env:
        p = Path(env)
        if p.exists() and p.is_file():
            return p.resolve()
    which = shutil.which("starnet2") or shutil.which("starnet++")
    if which:
        return Path(which).resolve()
    for base in _frozen_search_dirs():
        for cand in (
            base / "starnet2_cli" / "starnet2",
            base / "starnet2",
            base / "starnet++",
            base / "starnet2_cli" / "starnet++",
        ):
            try:
                if cand.is_file() or (cand.exists() and cand.resolve().is_file()):
                    rp = cand.resolve()
                    if os.access(rp, os.X_OK) or rp.suffix == "":
                        return rp
            except OSError:
                continue
        # 递归浅找（starnet2_cli/ 一层）
        try:
            if base.exists():
                for p in base.rglob("starnet2"):
                    if p.is_file() or p.is_symlink():
                        try:
                            rp = p.resolve()
                        except OSError:
                            continue
                        if rp.is_file() and os.access(rp, os.X_OK):
                            return rp
        except OSError:
            continue
    for p in CANDIDATE_BINS:
        if p.exists() and os.access(p, os.X_OK):
            return p.resolve()
    return None


def starnet2_available() -> bool:
    return find_starnet2() is not None


def _mop_residual_points(
    starless: np.ndarray,
    stars: np.ndarray,
    fwhm: float = 2.5,
    log=None,
) -> None:
    """CNN 之后仍残留的亮星核：收进星层，保持互补。"""
    from .detect import detect_stars
    from .separate import _fill_peak

    for thr in (3.2, 2.6, 2.1):
        if log:
            log(f"  残留收尾 thr={thr} …")
        field = detect_stars(
            starless,
            threshold_sigma=thr,
            fwhm=fwhm,
            max_stars=4000,
            min_separation=1.0,
            reject_texture=True,
        )
        pts = [
            s
            for s in field.stars
            if s.sharpness >= 0.15 and s.roundness <= 0.60 and s.fwhm <= fwhm * 3.0
        ]
        if not pts:
            break
        progress = False
        for st in pts:
            if _fill_peak(starless, stars, st, fwhm):
                progress = True
        if not progress:
            break


def _restore_star_glow(
    img: np.ndarray,
    starless: np.ndarray,
    stars: np.ndarray,
    fwhm: float = 2.5,
    strength: float = 0.85,
    log=None,
) -> None:
    """给过紧的星点补回光晕：宽斯高斯 max 混合（用自身亮度，非重绘）。"""
    from scipy import ndimage as ndi

    if strength <= 0:
        return
    if log:
        log("  恢复星光晕 …")
    g = stars.mean(axis=-1) if stars.ndim == 3 else stars
    glow = ndi.gaussian_filter(g, sigma=max(fwhm * 1.15, 1.2))
    near = ndi.maximum_filter(g, size=max(int(fwhm * 3.5) | 1, 9)) > 0.01
    add = np.clip(glow - g, 0, None) * float(strength) * near.astype(np.float32)
    if stars.ndim == 3:
        ratio = np.clip(stars / (g[..., None] + 1e-6), 0.0, 2.0)
        for c in range(3):
            ratio[..., c] = ndi.uniform_filter(ratio[..., c], size=3)
        stars += add[..., None] * ratio
    else:
        stars += add
    # 光晕只能来自原图里已有的光：星层不得超过原图，否则
    # starless = clip(img - stars) 会截断，互补性（stars + starless ≈ 原图）被破坏
    np.minimum(stars, img, out=stars)
    np.copyto(starless, np.clip(img - stars, 0, None))


def _reclaim_extended_ground(
    img: np.ndarray,
    starless: np.ndarray,
    stars: np.ndarray,
    log=None,
    fwhm_hint: float = 2.5,
) -> None:
    """星层里不该有地景/建筑等延展结构：低频非点源退回无星图。"""
    if log:
        log("  地景结构回收 …")
    from scipy import ndimage as ndi

    h = stars.shape[0]
    y0 = int(h * 0.62)
    if y0 >= h - 2:
        return

    g = stars[y0:].mean(axis=-1) if stars.ndim == 3 else stars[y0:]
    lp = ndi.gaussian_filter(g, 3.0)

    # 点源足迹：局部峰附近保留（向量化膨胀，避免按标签循环）
    peaks = (g == ndi.maximum_filter(g, 5)) & (g > 0.02)
    # 光晕/PSF 翼可到数倍 FWHM；保护窗过小会把光晕当地景退回，只剩亮核
    half = max(int(np.ceil(fwhm_hint * 4.0)), 16)
    point_mask = ndi.maximum_filter(peaks.astype(np.float32), size=2 * half + 1)

    # 延展结构 = 低频明显、且远离任何点源（阈值略高，避免吃掉淡晕）
    reclaim = np.clip(lp - 0.01, 0, None) * (1.0 - point_mask)
    # 只退真实存在的星层信号，避免互补被破坏
    if stars.ndim == 3:
        sl = stars[y0:]
        scale = np.clip(reclaim / (sl.mean(axis=-1) + 1e-6), 0.0, 1.0)[..., None]
        moved = sl * scale * (1.0 - point_mask[..., None])
        stars[y0:] = sl - moved
        starless[y0:] = starless[y0:] + moved
    else:
        sl = stars[y0:]
        scale = np.clip(reclaim / (sl + 1e-6), 0.0, 1.0)
        moved = sl * scale * (1.0 - point_mask)
        stars[y0:] = sl - moved
        starless[y0:] = starless[y0:] + moved


def separate_starnet2(
    image: np.ndarray,
    stride: int = 256,
    upsample: bool = False,
    bin_path: Path | str | None = None,
    work_dir: Path | str | None = None,
    log=None,
    glow: float = 0.85,
) -> SeparateResult:
    """用 StarNet2 CNN 一次得到无星图，星层 = 原图 - 无星图。"""
    exe = Path(bin_path) if bin_path else find_starnet2()
    if exe is None:
        raise RuntimeError(
            "未找到 StarNet2 CLI。请运行 scripts/install_starnet2.sh 安装，"
            f"或从 {DOWNLOAD_PAGE} 下载后设置 STARNET2_BIN。"
        )

    img = _to_unit_float(image)
    real = exe.resolve()
    # 必须用真实二进制路径：符号链接会让权重自动查找落在错误目录
    if work_dir:
        wd = Path(work_dir).resolve()
        wd.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        # 每次运行独立目录：固定名 starmask_work 会让并发/多开互相覆盖
        wd = Path(tempfile.mkdtemp(prefix="starmask_work_")).resolve()
    in_path = (wd / "starnet_in.tif").resolve()
    out_starless = (wd / "starnet_starless.tif").resolve()
    out_mask = (wd / "starnet_mask.tif").resolve()
    out_stars = (wd / "starnet_stars.tif").resolve()

    if log:
        log("  调用 StarNet2 推理（大图约 30–90 秒，请稍候）…")
    save_tiff(in_path, img)

    # 权重交给 CLI 在真实二进制旁自动查找
    cmd = [
        str(real),
        "-i", str(in_path),
        "-o", str(out_starless),
        "-m", str(out_mask),
        "-n", str(out_stars),
        "-s", str(int(stride)),
    ]
    if upsample:
        cmd.append("-u")

    libdir = real.parent / "lib"
    env = os.environ.copy()
    if libdir.is_dir():
        env["DYLD_LIBRARY_PATH"] = str(libdir) + ":" + env.get("DYLD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(libdir) + ":" + env.get("LD_LIBRARY_PATH", "")

    proc = subprocess.run(
        cmd,
        cwd=str(real.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"StarNet2 失败 (code={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )

    starless = _to_unit_float(imread_any(out_starless))
    # 互补定义：星层 = 原图 − 无星图（保证 stars + starless ≈ 原图）
    stars = np.clip(img - starless, 0, None)

    # 用真实测得的 FWHM，而不是硬编码 2.5（收尾窗口/光晕尺度都依赖它）
    gray = img.mean(axis=-1) if img.ndim == 3 else img
    fwhm_est = float(estimate_fwhm(gray.astype(np.float32)))

    # 残留星核收尾 + 地景延展结构退回（都是 in-place，每次之后重算互补）
    _mop_residual_points(starless, stars, fwhm=fwhm_est, log=log)
    stars = np.clip(img - starless, 0, None)
    _reclaim_extended_ground(img, starless, stars, log=log, fwhm_hint=fwhm_est)
    stars = np.clip(img - starless, 0, None)
    if glow:
        _restore_star_glow(img, starless, stars, fwhm=fwhm_est, strength=glow, log=log)
    starless = np.clip(img - stars, 0, None)

    # 真实星表：在星层上检测点源（此前是空表 + 硬编码 FWHM=2.5，
    # 导致 n_stars 恒为 0、reduce_stars 的保亮星逻辑拿不到坐标）
    field = detect_stars(
        stars,
        threshold_sigma=2.8,
        fwhm=fwhm_est,
        max_stars=12000,
        min_separation=max(fwhm_est * 0.5, 1.0),
        reject_texture=False,
    )
    if not field.stars:
        field = StarField(
            stars=[], image_shape=img.shape[:2], fwhm_median=fwhm_est, threshold=0.0
        )
    return SeparateResult(
        original=img, starless=starless, stars=stars, star_field=field, method="starnet2"
    )


def _find_weights(exe: Path) -> Path | None:
    """查找 StarNet2 权重（.mlpackage / .pb / .onnx）。"""
    real = exe.resolve()
    for base in {real.parent, real.parent.parent, ROOT / "third_party" / "starnet2"}:
        for name in (
            "StarNet2_weights.mlpackage",
            "starnet2_weights.pb",
            "starnet2_weights.mlpackage",
            "weights.mlpackage",
        ):
            p = base / name
            if p.exists():
                return p
        if base.exists():
            for p in base.glob("*.mlpackage"):
                return p
            for p in base.glob("*weights*"):
                return p
    return None


def install_hint() -> str:
    return (
        "安装 StarNet2（与市面去星工具同源的 CNN 引擎）:\n"
        f"  bash scripts/install_starnet2.sh\n"
        f"  或手动下载: {MACOS_ARM64_ZIP}\n"
        f"  解压后: export STARNET2_BIN=/path/to/starnet2\n"
    )
