"""StarMask.app 入口：设置 StarNet2 路径后打开 GUI。"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _setup_frozen_env() -> None:
    if not getattr(sys, "frozen", False):
        return
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    # onedir：捆绑数据在 _MEIPASS / exe 旁
    for cand in (
        base / "starnet2_cli",
        Path(sys.executable).resolve().parent / "starnet2_cli",
        Path(sys.executable).resolve().parent.parent / "Resources" / "starnet2_cli",
    ):
        exe = cand / "starnet2"
        if exe.exists():
            os.environ["STARNET2_BIN"] = str(exe)
            lib = cand / "lib"
            if lib.is_dir():
                os.environ["DYLD_LIBRARY_PATH"] = str(lib) + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
            break


def main() -> None:
    _setup_frozen_env()
    # 预加载关键模块，避免点按钮时才 ImportError（tk 会吞异常）
    import starmask.pipeline  # noqa: F401
    import starmask.reduce  # noqa: F401
    import starmask.rawio  # noqa: F401
    import starmask.starnet2  # noqa: F401

    from starmask.gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
