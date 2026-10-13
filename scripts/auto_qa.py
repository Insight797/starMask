"""自动验收脚本：对测试 RAW 跑分离并检查质量，直到 PASS 或达到最大轮数。

用法:
  .venv/bin/python scripts/auto_qa.py
"""

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

OUT = ROOT / "examples" / "qa"


def run_once(img: np.ndarray, **kwargs):
    result = separate(img, method="starnet", **kwargs)
    report = evaluate_quality(
        result.original, result.starless, result.stars, fwhm=result.star_field.fwhm_median
    )
    report.n_stars = len(result.star_field.stars)
    return result, report


def score(report) -> tuple:
    """越小越好。"""
    return (
        0 if report.passed else 1,
        report.starless_n_detected,
        report.stars_bg_p99,
        abs(report.stars_structure_leak),
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for path in CASES:
        if not path.exists():
            print(f"SKIP missing {path}")
            continue
        print("=" * 60, path.name)
        img = load_image(path, mode="auto")
        print(f"  image {img.shape[1]}x{img.shape[0]}")

        # StarNet2：单次 CNN 推理；失败时回退 PSF 参数扫描
        attempts = [dict()]

        best = best_rep = best_kwargs = None
        for i, kwargs in enumerate(attempts):
            result, report = run_once(img, **kwargs)
            print(f"\n  try#{i} {kwargs}")
            print("   " + format_report(report).replace("\n", "\n    "))
            if best is None or score(report) < score(best_rep):
                best, best_rep, best_kwargs = result, report, kwargs
            if report.passed:
                break

        tag = path.stem
        save_tiff(OUT / f"{tag}_stars.tif", best.stars)
        save_tiff(OUT / f"{tag}_starless.tif", best.starless)
        save_compare_png(best.original, best.starless, best.stars, OUT / f"{tag}_compare.png")
        print(f"\n  BEST kwargs={best_kwargs}")
        print(format_report(best_rep))
        if not best_rep.passed:
            all_ok = False
            print(f"  => {path.name} NOT PASS")

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
