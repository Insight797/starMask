"""StarMask 桌面 GUI（tkinter）。

深色摄影风：选图 → 预览 → 一键缩星 → 导出两张互补 TIFF。
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

import numpy as np

from .pipeline import assign_stems, process_file
from .rawio import IMAGE_SUFFIXES, RAW_SUFFIXES
from .separate import Method

INPUT_SUFFIXES = sorted(RAW_SUFFIXES | IMAGE_SUFFIXES)

BG = "#121417"
PANEL = "#1b1e23"
PANEL2 = "#24282e"
INK = "#e8eaed"
MUTED = "#9aa0a6"
ACCENT = "#5ec8ff"
OK = "#7dCEA0"
WARN = "#f0b429"
ERR = "#ff6b6b"


def _to_u8_preview(arr: np.ndarray, mode: str = "stretch") -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        g = a.mean(axis=-1)
    else:
        g = a
    if mode == "stars":
        hi = float(np.percentile(g, 99.7))
        x = np.clip(g / (hi + 1e-8), 0, 1)
    else:
        lo, hi = np.percentile(g, (1.0, 99.5))
        x = np.clip((g - lo) / (hi - lo + 1e-8), 0, 1)
    return (x * 255).astype(np.uint8)


class StarMaskGUI:
    def __init__(self, root) -> None:
        self.root = root
        self.root.title("StarMask — 银河缩星")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 720)
        self.root.geometry("1280x800")

        self.files: list[Path] = []
        # 批量同名文件的输出 stem 映射（assign_stems 产生），预览按它定位输出
        self.out_stems: dict[Path, str] = {}
        # Finder 启动时 cwd 可能是 /，必须用家目录
        self.out_dir = Path.home() / "Desktop" / "StarMask_out"
        self.log_q: queue.Queue[str] = queue.Queue()
        self.busy = False
        self.preview_photo = None
        self._thumbs: list = []

        self._build_style()
        self._build_layout()
        self._poll_log()

    # ---------- UI ----------
    def _build_style(self) -> None:
        import tkinter.ttk as ttk

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=BG, foreground=INK, fieldbackground=PANEL2, borderwidth=0)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Panel.TLabel", background=PANEL, foreground=INK)
        style.configure("TButton", background=PANEL2, foreground=INK, padding=(12, 8))
        style.map("TButton", background=[("active", "#323840")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#0b1220", padding=(14, 10))
        style.map("Accent.TButton", background=[("active", "#8ad7ff")])
        style.configure("TCombobox", fieldbackground=PANEL2, background=PANEL2, foreground=INK)
        style.configure("TCheckbutton", background=BG, foreground=INK)
        style.configure("Horizontal.TProgressbar", troughcolor=PANEL2, background=ACCENT)
        style.configure("TEntry", fieldbackground=PANEL2, foreground=INK)

    def _build_layout(self) -> None:
        import tkinter as tk
        from tkinter import filedialog, ttk

        # 顶栏
        top = ttk.Frame(self.root, padding=(16, 12))
        top.pack(fill="x")
        ttk.Label(top, text="StarMask", style="TLabel", font=("", 16, "bold")).pack(side="left")
        ttk.Label(top, text="  纯星点 + 银河地景  ·  数学互补", style="Muted.TLabel").pack(side="left", padx=(8, 0))

        body = ttk.Frame(self.root, padding=(16, 0, 16, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # 左：文件
        left = ttk.Frame(body, style="Panel.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        ttk.Label(left, text="输入文件", style="Panel.TLabel", font=("", 11, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(
            left, width=28, height=22, bg=PANEL2, fg=INK, selectbackground="#3a4654",
            highlightthickness=0, bd=0, activestyle="none", font=("", 11),
        )
        self.listbox.pack(fill="both", expand=True, pady=(8, 8))
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.preview_selected())

        btns = ttk.Frame(left, style="Panel.TFrame")
        btns.pack(fill="x")
        ttk.Button(btns, text="添加文件…", command=self.add_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="添加文件夹…", command=self.add_folder).pack(fill="x", pady=2)
        ttk.Button(btns, text="清空", command=self.clear_files).pack(fill="x", pady=2)

        # 中：预览
        mid = ttk.Frame(body, style="Panel.TFrame", padding=12)
        mid.grid(row=0, column=1, sticky="nsew", padx=(0, 12))
        mid.rowconfigure(1, weight=1)
        mid.columnconfigure(0, weight=1)
        ttk.Label(mid, text="预览  ·  原图 | 无星 | 星点", style="Panel.TLabel", font=("", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.preview_label = ttk.Label(mid, style="Panel.TLabel", anchor="center")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        self.preview_label.configure(text="先添加 RAW / TIFF，选中即可预览")

        # 右：参数
        right = ttk.Frame(body, style="Panel.TFrame", padding=12, width=280)
        right.grid(row=0, column=2, sticky="nse")
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="处理参数", style="Panel.TLabel", font=("", 11, "bold")).pack(anchor="w")

        ttk.Label(right, text="分离引擎", style="Panel.TLabel").pack(anchor="w", pady=(12, 2))
        self.method = tk.StringVar(value=Method.STARNET.value)
        cb = ttk.Combobox(
            right, textvariable=self.method, state="readonly",
            values=[Method.STARNET.value, Method.PSF.value, Method.MASK.value, Method.MORPH.value],
        )
        cb.pack(fill="x")
        ttk.Label(right, text="starnet = StarNet2 神经网络（推荐）", style="Panel.TLabel").pack(anchor="w")

        ttk.Label(right, text="输出格式", style="Panel.TLabel").pack(anchor="w", pady=(14, 2))
        self.fmt = tk.StringVar(value="dng")
        self.dng_linear = tk.BooleanVar(value=False)
        self.glow = tk.DoubleVar(value=0.85)
        # 默认 0：美晕是艺术增强，会破坏 removed + reduced = 原图 的互补等式；
        # CLI 同为 0.0，需要时再由用户拖上去
        self.beauty = tk.DoubleVar(value=0.0)
        ttk.Combobox(
            right, textvariable=self.fmt, state="readonly",
            values=["dng", "tif", "both"],
        ).pack(fill="x")
        ttk.Label(right, text="dng=便于 LR AI 降噪", style="Panel.TLabel").pack(anchor="w")
        glow_row = ttk.Frame(right, style="Panel.TFrame")
        glow_row.pack(fill="x", pady=(4, 0))
        ttk.Label(glow_row, text="星光晕", style="Panel.TLabel").pack(side="left")
        self.glow_scale = ttk.Scale(glow_row, from_=0.0, to=1.2, variable=self.glow)
        self.glow_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.glow_lbl = ttk.Label(glow_row, text="85%", style="Panel.TLabel", width=5)
        self.glow_lbl.pack(side="left")
        self.glow_scale.configure(command=lambda _v: self.glow_lbl.configure(text=f"{self.glow.get()*100:.0f}%"))
        b_row = ttk.Frame(right, style="Panel.TFrame")
        b_row.pack(fill="x", pady=(4, 0))
        ttk.Label(b_row, text="亮星美晕", style="Panel.TLabel").pack(side="left")
        self.beauty_scale = ttk.Scale(b_row, from_=0.0, to=1.5, variable=self.beauty)
        self.beauty_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.beauty_lbl = ttk.Label(b_row, text="0%", style="Panel.TLabel", width=5)
        self.beauty_lbl.pack(side="left")
        self.beauty_scale.configure(command=lambda _v: self.beauty_lbl.configure(text=f"{self.beauty.get()*100:.0f}%"))
        ttk.Checkbutton(right, text="DNG 场景线性（LR 用）", variable=self.dng_linear).pack(anchor="w")

        ttk.Label(right, text="输出目录", style="Panel.TLabel").pack(anchor="w", pady=(14, 2))
        row = ttk.Frame(right, style="Panel.TFrame")
        row.pack(fill="x")
        self.out_var = tk.StringVar(value=str(self.out_dir))
        ttk.Entry(row, textvariable=self.out_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="…", width=3, command=self.pick_out).pack(side="left", padx=(6, 0))

        self.write_compare = tk.BooleanVar(value=True)
        self.preview_raw = tk.BooleanVar(value=False)
        self.reduce_bright = tk.BooleanVar(value=True)
        self.keep_auto = tk.BooleanVar(value=True)
        self.keep_ratio = tk.DoubleVar(value=0.12)

        ttk.Checkbutton(right, text="缩星：保留亮星、去掉暗星", variable=self.reduce_bright, command=self._toggle_keep).pack(anchor="w", pady=(14, 0))
        ttk.Checkbutton(right, text="亮度自动分层", variable=self.keep_auto, command=self._toggle_keep).pack(anchor="w")
        keep_row = ttk.Frame(right, style="Panel.TFrame")
        keep_row.pack(fill="x", pady=(2, 0))
        ttk.Label(keep_row, text="保留亮星", style="Panel.TLabel").pack(side="left")
        self.keep_scale = ttk.Scale(keep_row, from_=0.05, to=0.40, variable=self.keep_ratio)
        self.keep_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.keep_lbl = ttk.Label(keep_row, text="12%", style="Panel.TLabel", width=5)
        self.keep_lbl.pack(side="left")
        self.keep_scale.configure(command=lambda _v: self.keep_lbl.configure(text=f"{self.keep_ratio.get()*100:.0f}%"))

        ttk.Checkbutton(right, text="导出三联对比图", variable=self.write_compare).pack(anchor="w", pady=(12, 0))
        ttk.Checkbutton(right, text="优先读内嵌预览（更快）", variable=self.preview_raw).pack(anchor="w")

        self.run_btn = ttk.Button(right, text="开始缩星", style="Accent.TButton", command=self.run)
        self.run_btn.pack(fill="x", pady=(18, 8))
        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.pack(fill="x")
        self.status = ttk.Label(right, text="就绪", style="Panel.TLabel")
        self.status.pack(anchor="w", pady=(8, 0))

        # 底部日志
        bottom = ttk.Frame(self.root, style="Panel.TFrame", padding=(16, 8))
        bottom.pack(fill="x")
        self.log = tk.Text(
            bottom, height=5, bg=PANEL2, fg=MUTED, insertbackground=INK,
            highlightthickness=0, bd=0, font=("Menlo", 11),
        )
        self.log.pack(fill="x")

        self._log("准备就绪。默认「缩星」= 保留高亮星 + 去掉暗星，输出 *_reduced.tif；完整去星见 *_starless.tif + *_stars.tif。")

    # ---------- actions ----------
    def _log(self, msg: str) -> None:
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def _poll_log(self) -> None:
        try:
            while True:
                msg = self.log_q.get_nowait()
                if msg.startswith("__STATUS__"):
                    self.status.configure(text=msg[len("__STATUS__"):])
                elif msg.startswith("__PROG__"):
                    self.progress["value"] = float(msg[len("__PROG__"):])
                elif msg.startswith("__DONE__"):
                    self.busy = False
                    self.run_btn.configure(state="normal")
                    self._log(msg[len("__DONE__"):])
                    self.status.configure(text="完成")
                    self.preview_selected(force=True)
                elif msg.startswith("__ERR__"):
                    self.busy = False
                    self.run_btn.configure(state="normal")
                    self._log(msg[len("__ERR__"):])
                    self.status.configure(text="失败")
                else:
                    self._log(msg)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_log)

    def add_files(self) -> None:
        from tkinter import filedialog

        paths = filedialog.askopenfilenames(
            title="选择 RAW / 图像",
            filetypes=[
                ("RAW / 图像", " ".join(f"*{s}" for s in INPUT_SUFFIXES)),
                ("全部文件", "*.*"),
            ],
        )
        self._add_paths([Path(p) for p in paths])

    def add_folder(self) -> None:
        from tkinter import filedialog

        d = filedialog.askdirectory(title="选择文件夹")
        if not d:
            return
        base = Path(d)
        found: list[Path] = []
        for ext in INPUT_SUFFIXES:
            found.extend(base.glob(f"*{ext}"))
            found.extend(base.glob(f"*{ext.upper()}"))
        self._add_paths(sorted(set(found)))

    def clear_files(self) -> None:
        self.files.clear()
        self.listbox.delete(0, "end")
        self.preview_label.configure(image="", text="先添加 RAW / TIFF，选中即可预览")
        self.preview_photo = None

    def _toggle_keep(self) -> None:
        state = "normal" if (self.reduce_bright.get() and not self.keep_auto.get()) else "disabled"
        self.keep_scale.configure(state=state)

    def pick_out(self) -> None:
        from tkinter import filedialog

        d = filedialog.askdirectory(title="选择输出目录", initialdir=self.out_var.get())
        if d:
            self.out_var.set(d)

    def _add_paths(self, paths: list[Path]) -> None:
        seen = {p.resolve() for p in self.files}
        for p in paths:
            rp = p.resolve()
            if rp in seen or p.suffix.lower() not in {s.lower() for s in INPUT_SUFFIXES}:
                continue
            self.files.append(p)
            seen.add(rp)
            self.listbox.insert("end", p.name)
        if self.files:
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(0)
            self.preview_selected()
        self.status.configure(text=f"{len(self.files)} 个文件")

    def _selected_path(self) -> Path | None:
        sel = self.listbox.curselection()
        if not sel:
            return self.files[0] if self.files else None
        return self.files[sel[0]]

    def preview_selected(self, force: bool = False) -> None:
        path = self._selected_path()
        if path is None:
            return
        try:
            self._show_preview_for(path, force=force)
        except Exception as exc:  # noqa: BLE001
            self._log(f"预览失败: {exc}")
            self.status.configure(text=f"预览失败: {exc}")

    def _show_preview_for(self, path: Path, force: bool = False) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        from .rawio import load_image

        out_dir = Path(self.out_var.get())
        # 同名文件经 assign_stems 改过输出名，预览必须用同一映射找结果
        stem = self.out_stems.get(path) or path.stem
        starless_p = out_dir / f"{stem}_starless.tif"
        stars_p = out_dir / f"{stem}_stars.tif"
        if not starless_p.exists():
            starless_p = out_dir / f"{stem}_starless.dng"
        if not stars_p.exists():
            stars_p = out_dir / f"{stem}_stars.dng"
        reduced_p = out_dir / f"{stem}_reduced.tif"
        removed_p = out_dir / f"{stem}_stars_removed.tif"
        if not reduced_p.exists():
            reduced_p = out_dir / f"{stem}_reduced.dng"
        if not removed_p.exists():
            removed_p = out_dir / f"{stem}_stars_removed.dng"

        if reduced_p.exists() and removed_p.exists():
            import tifffile

            orig = load_image(path, mode="preview")
            # 输出恒为 16-bit：固定除 65535，不用峰值启发（暗 16-bit 会被误判）
            sl = tifffile.imread(reduced_p).astype(np.float32) / 65535.0
            st = tifffile.imread(removed_p).astype(np.float32) / 65535.0
            title = "原图 | 缩星（保留亮星）| 去掉的暗星"
        elif starless_p.exists() and stars_p.exists():
            import tifffile

            orig = load_image(path, mode="preview")
            sl = tifffile.imread(starless_p).astype(np.float32) / 65535.0
            st = tifffile.imread(stars_p).astype(np.float32) / 65535.0
            title = "原图 | 无星 | 星点（已处理）"
        else:
            orig = load_image(path, mode="preview")
            sl = orig
            st = np.zeros_like(orig)
            title = "原图预览（尚未处理）"

        tiles = [
            _to_u8_preview(orig, "stretch"),
            _to_u8_preview(sl, "stretch"),
            _to_u8_preview(st, "stars"),
        ]
        h = min(t.shape[0] for t in tiles)
        w = min(t.shape[1] for t in tiles)
        panel = np.concatenate([t[:h, :w] for t in tiles], axis=1)
        im = Image.fromarray(panel)
        # 适配预览区
        max_w, max_h = 860, 520
        im.thumbnail((max_w, max_h))
        self.preview_photo = ImageTk.PhotoImage(im)
        self.preview_label.configure(image=self.preview_photo, text=title)
        self.preview_label.image = self.preview_photo

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as exc:  # noqa: BLE001 — tk 会吞掉回调异常，必须显式弹出
            self.busy = False
            try:
                self.run_btn.configure(state="normal")
            except Exception:
                pass
            msg = f"{exc}\n{traceback.format_exc()}"
            self._log("启动失败: " + msg)
            self.status.configure(text="启动失败")
            try:
                from tkinter import messagebox

                messagebox.showerror("StarMask", f"无法开始处理：\n{exc}")
            except Exception:
                pass

    def _run_inner(self) -> None:
        if self.busy:
            self._log("正在处理中，请稍候…（大图 StarNet2 约 30–90 秒）")
            return
        if not self.files:
            self._log("请先添加文件。")
            self.status.configure(text="请先添加文件")
            return
        method = self.method.get()
        if method == Method.STARNET.value:
            from .starnet2 import starnet2_available, install_hint

            if not starnet2_available():
                self._log("未找到 StarNet2，已回退 psf。安装：" + install_hint().splitlines()[0])
                method = Method.PSF.value

        raw_out = self.out_var.get().strip() or str(Path.home() / "Desktop" / "StarMask_out")
        out_dir = Path(raw_out).expanduser()
        # 相对路径相对家目录，避免 cwd=/
        if not out_dir.is_absolute():
            out_dir = Path.home() / out_dir
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            out_dir = Path.home() / "Desktop" / "StarMask_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.out_var.set(str(out_dir))
            self._log(f"原输出目录不可写，已改到 {out_dir}")
        self.busy = True
        self.run_btn.configure(state="disabled")
        self.progress["value"] = 0
        files = list(self.files)
        # 同名文件（不同目录）分配互斥输出 stem；预览用同一映射找回结果
        self.out_stems = assign_stems(files)
        keep_ratio = None if self.keep_auto.get() else float(self.keep_ratio.get())
        opts = dict(
            method=method,
            write_compare=bool(self.write_compare.get()),
            preview=bool(self.preview_raw.get()),
            reduce_bright=bool(self.reduce_bright.get()),
            keep_ratio=keep_ratio,
            image_format=str(self.fmt.get()),
            dng_scene_linear=bool(self.dng_linear.get()),
            glow=float(self.glow.get()),
            beauty_glow=float(self.beauty.get()),
        )

        self._log(f"已开始处理 {len(files)} 个文件 → {out_dir}")
        self._log("引擎: %s ｜ 缩星: %s ｜ 格式: %s" % (method, opts["reduce_bright"], opts["image_format"]))
        self.status.configure(text="处理中…")

        def worker() -> None:
            n = len(files)
            try:
                for i, path in enumerate(files):
                    self.log_q.put(f"处理 {path.name} …")
                    self.log_q.put(f"__STATUS__{path.name}（{i + 1}/{n}）")
                    info = process_file(
                        path,
                        output_dir=out_dir,
                        stem=self.out_stems.get(path),
                        log=lambda m: self.log_q.put(m),
                        **opts,
                    )
                    line = f"  ✓ {info.starless.name}  +  {info.stars.name}"
                    if info.reduced is not None:
                        line += (
                            f"  |  缩星 {info.reduced.name}"
                            f"  （保留亮星 {info.n_kept} / 去掉 {info.n_removed_stars}）"
                        )
                    self.log_q.put(line)
                    self.log_q.put(f"__PROG__{(i + 1) / n * 100:.1f}")
                self.log_q.put(f"__DONE__完成 {n}/{n} → {out_dir}")
            except Exception as exc:  # noqa: BLE001
                self.log_q.put(f"__ERR__失败: {exc}\n{traceback.format_exc()}")

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    import tkinter as tk

    root = tk.Tk()
    StarMaskGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
