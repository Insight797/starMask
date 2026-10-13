"""RAW / 常规图像读写。

支持尼康 NEF、索尼 ARW、佳能 CR2/CR3、奥林巴斯 ORF、富士 RAF、宾得 PEF、
适马 X3F、通用 DNG，以及 TIFF/JPEG/PNG。

部分尼康「高效压缩」/ 特殊封装 NEF 无法走 LibRaw 全量 demosaic，此时自动回退到
内嵌高分辨率预览，保证仍可分离星点。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

RAW_SUFFIXES = {
    ".nef",
    ".arw",
    ".cr2",
    ".cr3",
    ".orf",
    ".raf",
    ".pef",
    ".rw2",
    ".srw",
    ".dng",
    ".raw",
    ".nrw",
    ".erf",
    ".mrw",
    ".x3f",
}

IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".exr"}

ReadMode = Literal["preview", "full", "auto"]


def is_raw(path: Path | str) -> bool:
    return Path(path).suffix.lower() in RAW_SUFFIXES


def _stretch_for_display(linear: np.ndarray) -> np.ndarray:
    """把线性数据拉伸到适合观看的范围，保留高位深。"""
    x = np.clip(linear, 0.0, None)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    black = float(np.percentile(finite, 0.1))
    white = float(np.percentile(finite, 99.5))
    if white <= black:
        white = black + 1e-6
    y = (x - black) / (white - black)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def load_image(path: Path | str, mode: ReadMode = "auto") -> np.ndarray:
    """读取 RAW 或常规图像，返回 float32 RGB，形状 (H, W, 3)，范围约 [0, 1]。

    mode:
    - ``full``: 尽量 RAW 全量 demosaic；失败时自动回退内嵌预览
    - ``preview``: 优先内嵌预览
    - ``auto``: 先 full，失败回退 preview（默认）
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    suffix = path.suffix.lower()

    if suffix in RAW_SUFFIXES:
        return _load_raw(path, mode=mode)

    if suffix in {".tif", ".tiff", ".exr"}:
        return _to_rgb_float(imread_any(path))

    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    return _to_rgb_float(arr)


def _load_raw(path: Path, mode: ReadMode = "auto") -> np.ndarray:
    import rawpy

    want_full = mode in ("full", "auto")
    want_preview = mode in ("preview", "auto")
    last_err: Exception | None = None

    with rawpy.imread(str(path)) as raw:
        if want_full:
            try:
                rgb = raw.postprocess(
                    output_color=rawpy.ColorSpace.sRGB,
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    half_size=False,
                )
                arr = rgb.astype(np.float32) / 65535.0
                return _stretch_for_display(arr)
            except Exception as exc:  # noqa: BLE001 — LibRaw 各类不解压格式
                last_err = exc
                if mode == "full":
                    raise RuntimeError(
                        f"RAW 全量解码失败（可能是尼康高效压缩/特殊封装 NEF）: {exc}。"
                        f"可改用 --preview，或在相机里改成「无损压缩 RAW」。"
                    ) from exc

        if want_preview:
            try:
                return _load_raw_thumb(raw)
            except Exception as exc:  # noqa: BLE001
                last_err = exc

    # 再试一次独立打开提取预览（避开上一次 unpack 失败后的状态污染）
    if want_preview:
        try:
            with rawpy.imread(str(path)) as raw:
                return _load_raw_thumb(raw)
        except Exception as exc:  # noqa: BLE001
            last_err = exc

    raise RuntimeError(f"无法读取 RAW: {path} ({last_err})")


def _load_raw_thumb(raw) -> np.ndarray:
    import io
    import rawpy
    from PIL import Image

    thumb = raw.extract_thumb()
    if thumb.format == rawpy.ThumbFormat.JPEG:
        with Image.open(io.BytesIO(thumb.data)) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
        return _to_rgb_float(arr)
    if thumb.format == rawpy.ThumbFormat.BITMAP:
        return _to_rgb_float(thumb.data)
    raise RuntimeError(f"不支持的缩略图格式: {thumb.format}")


def _load_tiff_like(path: Path) -> np.ndarray:
    import tifffile

    arr = tifffile.imread(path)
    return _to_rgb_float(arr)


def _to_unit_float(arr: np.ndarray) -> np.ndarray:
    """任意数组 → float32 [0,1]。

    量程按 dtype 判定，不按峰值猜——峰值启发会把「暗 16-bit」（uint16 但 max≤255）
    误当 8-bit 除以 255，结果约亮 257 倍。float 输入约定 [0,1]；仅当整体明显
    处于 0~255 / 0~65535 量程时才兜底缩放（兼容历史上的已缩放浮点数据）。
    """
    a = np.asarray(arr)
    dt = a.dtype
    x = a.astype(np.float32, copy=True)
    if np.issubdtype(dt, np.integer):
        # uint8 精确；PIL "I" 模式是 int32 装 16-bit 数据，统一按 65535
        x /= 255.0 if dt == np.uint8 else 65535.0
    else:
        peak = float(x.max()) if x.size else 1.0
        if peak > 1.5:
            x /= 65535.0 if peak > 255 else 255.0
    return np.clip(x, 0.0, 1.0)


def _to_rgb_float(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.transpose(a[:3], (1, 2, 0))
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"无法识别的图像形状: {a.shape}")
    return _to_unit_float(a)


def imread_any(path: Path | str) -> np.ndarray:
    """读 TIFF/PNG/JPEG，返回 float32 [0,1]，尽量保留 16-bit 精度。

    1. 优先 tifffile（LZW 等压缩需 imagecodecs）——StarNet2 输出多为 16-bit LZW；
    2. 失败再用 PIL。注意 PIL 对 16-bit RGB 可能静默降为 8-bit，仅作后备。

    归一化统一走 `_to_rgb_float`（按 dtype 定量程），避免各处复制的峰值启发
    把暗 16-bit 误当 8-bit。
    """
    path = Path(path)

    try:
        import tifffile

        return _to_rgb_float(tifffile.imread(path))
    except Exception:
        pass

    from PIL import Image

    with Image.open(path) as im:
        # 尽量保留高位深：不预先 cast，保留 dtype 给量程判定
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            return _to_rgb_float(np.asarray(im))
        if im.mode in ("RGB", "RGBA", "L", "P", "CMYK"):
            im = im.convert("RGB")
            return _to_rgb_float(np.asarray(im))
        im = im.convert("RGB")
        return _to_rgb_float(np.asarray(im))


def save_tiff(path: Path | str, image: np.ndarray) -> Path:
    """保存 16-bit TIFF。

    image 为 float [0,1]（超出 1.0 的值截断）或整型数组（按 dtype 量程换算）。
    **固定比例、不按峰值重归一化**——否则亮度略超 1.0 的星层会被整层压暗，
    破坏 stars + starless ≈ 原图 的跨层比例。
    """
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    a = np.asarray(image)
    dt = a.dtype
    x = a.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if x.min() < 0:
        x = np.clip(x, 0.0, None)

    if np.issubdtype(dt, np.integer):
        maxv = 255.0 if dt == np.uint8 else 65535.0
        scaled = x / maxv * 65535.0
    else:
        # float 约定 [0,1]：截断而非重归一化，保跨层比例
        scaled = np.clip(x, 0.0, 1.0) * 65535.0

    out = np.clip(scaled, 0, 65535).astype(np.uint16)
    if out.shape[-1] == 1:
        out = out[..., 0]
    tifffile.imwrite(path, out, photometric="rgb" if out.ndim == 3 else "minisblack")
    return path


# DNG / TIFF 常用标签号
_TIFF_DNG_TAGS = {
    "NewSubfileType": 254,
    "ImageWidth": 256,
    "ImageLength": 257,
    "BitsPerSample": 258,
    "Compression": 259,
    "PhotometricInterpretation": 262,
    "StripOffsets": 273,
    "SamplesPerPixel": 277,
    "RowsPerStrip": 278,
    "StripByteCounts": 279,
    "PlanarConfiguration": 284,
    "SubIFD": 330,
    "DNGVersion": 50706,
    "DNGBackwardVersion": 50707,
    "UniqueCameraModel": 50708,
    "ColorMatrix1": 50721,
    "AsShotNeutral": 50727,
    "CalibrationIlluminant1": 50778,
    "BlackLevel": 50714,
    "WhiteLevel": 50717,
    "DefaultScale": 50718,
    "DefaultCropOrigin": 50829,  # 注意：真实 DNG 里 ActiveArea=50829，DefaultCropOrigin=50830
    "ActiveArea": 50829,
    "DefaultCropOrigin_real": 50830,
    "DefaultCropSize_real": 50831,
    "PreviewColorSpace": 50966,
}


def _float_to_rational_list(values, denom: int = 1_000_000) -> list[int]:
    """floats → [num0, den0, num1, den1, ...]（TIFF RATIONAL/SRATIONAL 载荷）。

    SRATIONAL 允许负值：分子为有符号 int32，分母为正。
    """
    out: list[int] = []
    for v in values:
        num = int(round(float(v) * denom))
        out.append(num)
        out.append(denom)
    return out


def _patch_photometric_to_linearraw(path: Path) -> None:
    """把 IFD0 的 PhotometricInterpretation(262) 从 RGB(2) 改为 LinearRaw(34892)。"""
    import struct

    data = bytearray(Path(path).read_bytes())
    endian = bytes(data[:2])
    if endian == b"II":
        u16, u32 = "<H", "<I"
    elif endian == b"MM":
        u16, u32 = ">H", ">I"
    else:
        return
    ifd = struct.unpack(u32, data[4:8])[0]
    n = struct.unpack(u16, data[ifd : ifd + 2])[0]
    for i in range(n):
        e = ifd + 2 + i * 12
        tag = struct.unpack(u16, data[e : e + 2])[0]
        if tag == 262:
            struct.pack_into(u16, data, e + 8, 34892)
            break
    Path(path).write_bytes(data)


def save_dng(
    path: Path | str,
    image: np.ndarray,
    camera_model: str = "StarMask Linear",
    scene_linear: bool = False,
) -> Path:
    """保存 16-bit DNG。

    ``scene_linear=False``（默认）: 输入为显示拉伸域 → 转场景线性后写 LinearRaw。
    ``scene_linear=True``: 输入已是场景线性 → 直接写 LinearRaw。
    两者都会是合法 DNG（Photoshop 可开）；默认路径可避免 ACR 二次提亮。

    布局：先写 RGB 三通道（避免被拆成单通道多页），DNG 标签用 RATIONAL/SRATIONAL
    列表载荷，最后把 Photometric 改成 34892。Photoshop / Camera Raw 可直接打开。
    """
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x0 = np.asarray(image)
    dt = x0.dtype
    x = x0.astype(np.float32)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.shape[-1] != 3:
        raise ValueError(f"DNG 导出需要 1/3 通道，得到 {x.shape}")
    x = np.clip(x, 0.0, None)

    # 与 save_tiff 同一口径：固定比例，不按峰值重归一化（保跨层比例）
    if np.issubdtype(dt, np.integer):
        maxv = 255.0 if dt == np.uint8 else 65535.0
        scaled = x / maxv * 65535.0
    else:
        scaled = np.clip(x, 0.0, 1.0) * 65535.0
    data = np.ascontiguousarray(np.clip(scaled, 0, 65535).astype(np.uint16))
    h, w = int(data.shape[0]), int(data.shape[1])

    # ColorMatrix1: XYZ→Camera(sRGB)，SRATIONAL
    cm = [
        3.2404542, -1.5371385, -0.4985314,
        -0.9692660, 1.8760108, 0.0415560,
        0.0556434, -0.2040259, 1.0572252,
    ]
    cam = (camera_model + "\x00").encode("ascii", "ignore")

    extratags = [
        (254, 4, 1, 0, True),  # NewSubfileType
        (274, 3, 1, 1, True),  # Orientation
        (50706, 1, 4, (1, 4, 0, 0), True),  # DNGVersion
        (50707, 1, 4, (1, 1, 0, 0), True),  # DNGBackwardVersion
        (50708, 2, len(cam), cam, True),  # UniqueCameraModel
        (50721, 10, 9, _float_to_rational_list(cm), True),  # ColorMatrix1
        (50728, 5, 3, [1, 1, 1, 1, 1, 1], True),  # AsShotNeutral（勿用 50727=AnalogBalance）
        (50778, 3, 1, 21, True),  # CalibrationIlluminant1 = D65
        (50713, 3, 2, (1, 1), True),  # BlackLevelRepeatDim
        (50714, 5, 1, [0, 1], True),  # BlackLevel 0/1
        (50717, 4, 1, 65535, True),  # WhiteLevel
        (50718, 5, 2, [1, 1, 1, 1], True),  # DefaultScale
        (50719, 5, 2, [0, 1, 0, 1], True),  # DefaultCropOrigin
        (50720, 5, 2, [w, 1, h, 1], True),  # DefaultCropSize
        (50829, 4, 4, (0, 0, h, w), True),  # ActiveArea
    ]

    # Photoshop / ACR 只接受 LinearRaw 或 CFA 的 DNG；RGB 混 DNG 标签会拒绝打开。
    # 处理管线在「显示拉伸」域工作，若直接当场景线性写入，ACR 会再做一次
    # 提亮（线性 0.4 显示后极亮）。因此：默认假定输入为显示域，先转线性再写。
    if not scene_linear:
        # sRGB 近似反函数：显示域 → 场景线性
        data_f = data.astype(np.float32) / 65535.0
        linear = np.where(
            data_f <= 0.04045,
            data_f / 12.92,
            np.power(np.clip(data_f + 0.055, 0, None) / 1.055, 2.4),
        )
        data = np.clip(linear * 65535.0, 0, 65535).astype(np.uint16)

    tifffile.imwrite(
        path,
        data,
        photometric="rgb",
        metadata=None,
        bitspersample=16,
        extratags=extratags,
    )
    _patch_photometric_to_linearraw(path)
    return path
