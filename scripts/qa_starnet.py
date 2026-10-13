"""StarNet2 质量验收 + 与成熟工具输出对照。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starmask.quality import evaluate_quality, format_report  # noqa: E402
from starmask.rawio import load_image, save_tiff  # noqa: E402
from starmask.separate import separate  # noqa: E402
from starmask.preview import save_compare_png  # noqa: E402

CASES = [
    Path("/Users/west/Desktop/DSC_3663.NEF"),
    Path("/Users/west/Desktop/9.20海坨/9.22/P/DSC09547.ARW"),
]
REF_STARLESS = Path("/Users/west/Desktop/output/starless/DSC_3663.tif")
OUT = ROOT / "examples" / "qa_starnet"


def load_ref(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    import tifffile
    a = tifffile.imread(path)
    a = a.astype(np.float32)
    if a.max() > 1.5:
        a = a / 65535.0
    return np.clip(a, 0, 1)


def residual_star_count(gray: np.ndarray) -> int:
    """粗看无星图里还剩多少点源（本地极大 + 点状形状）。"""
    from starmask.detect import detect_stars
    img = gray
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    field = detect_stars(img, threshold_sigma=3.5, max_stars=5000)
    n = 0
    for st in field.stars:
        if st.sharpness >= 0.18 and st.roundness <= 0.55 and st.fwhm <= 7.5:
            n += 1
    return n


def compare_pair(ours_sl: np.ndarray, ref_sl: np.ndarray) -> dict:
    h = min(ours_sl.shape[0], ref_sl.shape[0])
    w = min(ours_sl.shape[1], ref_sl.shape[1])
    a = ours_sl[:h, :w].mean(axis=-1) if ours_sl.ndim == 3 else ours_sl[:h, :w]
    b = ref_sl[:h, :w].mean(axis=-1) if ref_sl.ndim == 3 else ref_sl[:h, :w]
    return {
        "mae": float(np.abs(a - b).mean()),
        "rmse": float(np.sqrt(((a - b) ** 2).mean())),
        "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for path in CASES:
        if not path.exists():
            print(f"SKIP missing {path}")
            continue
        print("=" * 70)
        print(path.name)
        img = load_image(path, mode="auto")
        print(f"  input {img.shape[1]}x{img.shape[0]}  dtype={img.dtype}")

        result = separate(img, method="starnet")
        report = evaluate_quality(
            result.original, result.starless, result.stars, fwhm=2.5
        )
        report.n_stars = -1  # CNN 不显式数星
        print(format_report(report))
        print(f"  residual_pointlike = {residual_star_count(result.starless)}")

        tag = path.stem
        save_tiff(OUT / f"{tag}_stars.tif", result.stars)
        save_tiff(OUT / f"{tag}_starless.tif", result.starless)
        save_compare_png(
            result.original, result.starless, result.stars, OUT / f"{tag}_compare.png"
        )

        # 与成熟工具对照（仅 DSC_3663 有参考）
        if path.stem == "DSC_3663":
            ref = load_ref(REF_STARLESS)
            if ref is not None:
                stats = compare_pair(result.starless, ref)
                print(f"  vs mature starless: {stats}")
                # 四联：原图 | 我们无星 | 成熟无星 | 星点
                from PIL import Image

                def stretch(a):
                    a = a.mean(axis=-1) if a.ndim == 3 else a
                    lo, hi = np.percentile(a, (1, 99.5))
                    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)

                o = stretch(result.original)
                s1 = stretch(result.starless)
                s2 = stretch(ref)
                t = stretch(result.stars)
                def to_u8(x):
                    return (x * 255).astype(np.uint8)
                hh = min(o.shape[0], s1.shape[0], s2.shape[0], t.shape[0])
                ww = min(o.shape[1], s1.shape[1], s2.shape[1], t.shape[1])
                panel = np.concatenate(
                    [to_u8(o[:hh, :ww]), to_u8(s1[:hh, :ww]), to_u8(s2[:hh, :ww]), to_u8(t[:hh, :ww])],
                    axis=1,
                )
                Image.fromarray(panel).save(OUT / f"{tag}_vs_mature.png", optimize=True)
                # 残差放大：我们无星 - 成熟无星
                h2 = min(s1.shape[0], s2.shape[0])
                w2 = min(s1.shape[1], s2.shape[1])
                d1 = s1 if s1.ndim == 2 else s1.mean(-1)
                d2 = s2 if s2.ndim == 2 else s2.mean(-1)
                diff = np.clip(np.abs(d1[:h2, :w2] - d2[:h2, :w2]) * 4, 0, 1)
                Image.fromarray((diff * 255).astype(np.uint8)).save(
                    OUT / f"{tag}_diff_vs_mature.png", optimize=True
                )
                print(f"  mature residual_pointlike = {residual_star_count(ref)}")

        if not report.passed:
            all_ok = False
            print("  FAIL reasons:", report.reasons)
        else:
            print("  PASS")

    print("=" * 70)
    print("ALL PASS" if all_ok else "HAS FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
