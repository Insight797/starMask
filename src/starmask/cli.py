"""StarMask 命令行入口。"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .pipeline import assign_stems, process_file
from .rawio import IMAGE_SUFFIXES, RAW_SUFFIXES
from .separate import Method

console = Console()

INPUT_SUFFIXES = RAW_SUFFIXES | IMAGE_SUFFIXES


@click.group()
@click.version_option(__version__, prog_name="starmask")
def main() -> None:
    """StarMask — 银河摄影缩星工具。

    从 RAW/图像中分离「纯星点」与「银河+地景」两张图，用于缩星与银河强化。
    """


@main.command("process")
@click.argument("inputs", nargs=-1, type=click.Path(path_type=Path), required=True)
@click.option("-o", "--output", "output_dir", type=click.Path(path_type=Path), default=None, help="输出目录（默认: 输入旁 starmask_out/）")
@click.option(
    "-m",
    "--method",
    type=click.Choice([m.value for m in Method], case_sensitive=False),
    default=Method.STARNET.value,
    show_default=True,
    help="分离算法: starnet=StarNet2神经网络(默认,市面同级) / psf=PSF拟合 / mask=软掩膜 / morph=形态学",
)
@click.option("--fwhm", type=float, default=None, help="星点半高全宽（像素）。默认自动估计")
@click.option("--threshold", type=float, default=4.5, show_default=True, help="星点检测阈值（倍数噪声）")
@click.option("--gain", type=float, default=0.95, show_default=True, help="星点扣除增益，越小残留越多")
@click.option("--max-stars", type=int, default=12000, show_default=True, help="最多处理的星点数")
@click.option("--preview", is_flag=True, help="强制只读内嵌预览（更快）")
@click.option("--no-mask", is_flag=True, help="不导出星点掩膜")
@click.option("--compare", is_flag=True, help="额外输出 原图|无星|星点 三联对比 PNG")
@click.option("--reduce", "reduce_bright", is_flag=True, help="缩星：自动保留高亮星，去掉暗星（输出 *_reduced.tif）")
@click.option("--keep-ratio", type=float, default=None, help="手动保留最亮星的比例 0.05~0.40（默认自动判断）")
@click.option("--beauty-glow", type=float, default=0.0, help="亮星美晕强度 0~1.5（艺术 bloom，默认 0）")
@click.option("--glow", type=float, default=0.85, show_default=True, help="星光晕恢复强度 0~1.2（0=关闭）")
@click.option("--dng-linear", "dng_scene_linear", is_flag=True, help="DNG 按场景线性写出（LinearRaw，给 LR 管线）。默认显示值，避免 PS 二次提亮")
@click.option(
    "--format",
    "image_format",
    type=click.Choice(["tif", "dng", "both"], case_sensitive=False),
    default="tif",
    show_default=True,
    help="输出格式：tif / dng（线性 DNG，供 Lightroom AI 降噪）/ both",
)
def process_cmd(
    inputs: tuple[Path, ...],
    output_dir: Path | None,
    method: str,
    fwhm: float | None,
    threshold: float,
    gain: float,
    max_stars: int,
    preview: bool,
    no_mask: bool,
    compare: bool,
    reduce_bright: bool,
    keep_ratio: float | None,
    image_format: str,
    dng_scene_linear: bool,
    glow: float,
    beauty_glow: float,
) -> None:
    """处理一个或多个 RAW/图像文件。

    \b
    示例:
      starmask process milkyway.NEF
      starmask process a.ARW b.NEF -o out/
      starmask process shot.CR3 --method psf --threshold 4.5
    """
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            for ext in sorted(INPUT_SUFFIXES):
                files.extend(sorted(p.glob(f"*{ext}")))
                files.extend(sorted(p.glob(f"*{ext.upper()}")))
        else:
            files.append(p)

    # 去重
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        rp = f.resolve()
        if rp not in seen and rp.suffix.lower() in INPUT_SUFFIXES:
            seen.add(rp)
            unique.append(f)

    if not unique:
        console.print("[red]没有找到可处理的 RAW/图像文件。[/red]")
        sys.exit(1)

    table = Table(title=f"StarMask  v{__version__}", show_lines=False)
    table.add_column("文件", style="cyan", overflow="fold")
    table.add_column("星数", justify="right")
    table.add_column("FWHM", justify="right")
    table.add_column("输出", style="green", overflow="fold")

    ok = 0
    # 同名文件（不同目录）分配互斥输出 stem，避免后写的覆盖先写的
    stems = assign_stems(unique)
    for path in unique:
        try:
            with console.status(f"处理 {path.name} ..."):
                info = process_file(
                    path,
                    output_dir=output_dir,
                    stem=stems[path],
                    method=method,
                    fwhm=fwhm,
                    threshold_sigma=threshold,
                    gain=gain,
                    max_stars=max_stars,
                    write_mask=not no_mask,
                    write_compare=compare,
                    preview=preview,
                    reduce_bright=reduce_bright,
                    keep_ratio=keep_ratio,
                    image_format=image_format,
                    dng_scene_linear=dng_scene_linear,
                    glow=glow,
                    beauty_glow=beauty_glow,
                )
            table.add_row(
                path.name,
                str(info.n_stars),
                f"{info.fwhm:.2f}px",
                f"{info.stars.name} + {info.starless.name}",
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001 — CLI 边界统一报错
            table.add_row(path.name, "-", "-", f"[red]失败: {exc}[/red]")

    console.print(table)
    if ok == 0:
        sys.exit(2)
    console.print(f"[bold green]完成 {ok}/{len(unique)}[/bold green]  —  stars + starless ≈ 原图，可直接进 PS 缩星。")


@main.command("gui")
def gui_cmd() -> None:
    """打开图形界面。"""
    from .gui import main as gui_main
    gui_main()


@main.command("info")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def info_cmd(path: Path) -> None:
    """查看文件与自动估计的星点尺度。"""
    from .detect import detect_stars, estimate_fwhm, _to_gray
    from .rawio import load_image
    from .starnet2 import find_starnet2, install_hint

    img = load_image(path, mode="preview")
    gray = _to_gray(img)
    fwhm = estimate_fwhm(gray)
    field = detect_stars(img, fwhm=fwhm)

    console.print(f"[bold]{path.name}[/bold]")
    console.print(f"  尺寸: {img.shape[1]} × {img.shape[0]}")
    console.print(f"  估计 FWHM: {fwhm:.2f} px")
    console.print(f"  检测星点: {len(field.stars)}")
    if field.stars:
        fluxes = sorted((s.flux for s in field.stars), reverse=True)[:5]
        console.print(f"  最亮 5 星流量: {', '.join(f'{f:.3g}' for f in fluxes)}")
    sn = find_starnet2()
    if sn:
        console.print(f"  [green]StarNet2 已就绪[/green]: {sn}")
    else:
        console.print("  [yellow]StarNet2 未安装[/yellow] — " + install_hint().splitlines()[0])


if __name__ == "__main__":
    main()
