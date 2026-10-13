#!/usr/bin/env python3
"""自动递增版本号（修 bug 后请务必运行）。

用法:
  python scripts/bump_version.py           # 补丁号 +1 (0.3.0 -> 0.3.1)
  python scripts/bump_version.py minor     # 次版本 +1 (0.3.0 -> 0.4.0)
  python scripts/bump_version.py 0.3.1     # 指定版本
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "pyproject.toml",
    ROOT / "src" / "starmask" / "__init__.py",
    ROOT / "packaging" / "StarMask.spec",
    ROOT / "scripts" / "build_dmg.sh",
    ROOT / "packaging" / "使用说明.txt",
]


def read_version() -> str:
    t = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version = "([^"]+)"', t, re.M)
    return m.group(1) if m else "0.0.0"


def next_version(cur: str, kind: str | None) -> str:
    if kind in (None, "patch", "fix", "bugfix"):
        a, b, c = (list(map(int, cur.split("."))) + [0, 0, 0])[:3]
        return f"{a}.{b}.{c + 1}"
    if kind == "minor":
        a, b, c = (list(map(int, cur.split("."))) + [0, 0, 0])[:3]
        return f"{a}.{b + 1}.0"
    if kind == "major":
        a, b, c = (list(map(int, cur.split("."))) + [0, 0, 0])[:3]
        return f"{a + 1}.0.0"
    return kind


def apply(old: str, new: str) -> None:
    subs = [
        (f'version = "{old}"', f'version = "{new}"'),
        (f'__version__ = "{old}"', f'__version__ = "{new}"'),
        (f'CFBundleShortVersionString": "{old}",', f'CFBundleShortVersionString": "{new}",'),
        (f'CFBundleVersion": "{old}",', f'CFBundleVersion": "{new}",'),
        (f"StarMask-{old}-macOS-arm64.dmg", f"StarMask-{new}-macOS-arm64.dmg"),
        (f"StarMask 银河缩星 {old}", f"StarMask 银河缩星 {new}"),
        (old, new),
    ]
    for path in FILES:
        if not path.exists():
            continue
        t = path.read_text()
        orig = t
        for a, b in subs:
            t = t.replace(a, b)
        if t != orig:
            path.write_text(t)
            print(f"updated {path.relative_to(ROOT)}")


def main() -> int:
    cur = read_version()
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    new = next_version(cur, arg)
    if new == cur:
        print("version unchanged", cur)
        return 0
    apply(cur, new)
    print(f"version: {cur} -> {new}")
    print("记得: bash scripts/build_dmg.sh  重打安装包")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
