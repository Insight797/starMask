# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

imagecodecs_datas, imagecodecs_binaries, imagecodecs_hidden = collect_all("imagecodecs")

# 相对于 spec 文件定位项目根目录
PROJECT_ROOT = Path(SPECPATH).parent
STARNET2_PKG = PROJECT_ROOT / "third_party" / "starnet2" / "pkg" / "starnet2_macos-arm64_2.6.2-0242_COREML_arm64_cli"
assert STARNET2_PKG.is_dir(), f"StarNet2 包不存在: {STARNET2_PKG}"

a = Analysis(
    [str(PROJECT_ROOT / "packaging" / "entry_gui.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=imagecodecs_binaries,
    datas=[
        *imagecodecs_datas,
        (str(STARNET2_PKG), "starnet2_cli"),
    ],
    hiddenimports=[
        *imagecodecs_hidden,
        "imagecodecs",
        "tifffile",
        "starmask",
        "starmask.gui",
        "starmask.cli",
        "starmask.pipeline",
        "starmask.rawio",
        "starmask.separate",
        "starmask.starnet2",
        "starmask.detect",
        "starmask.quality",
        "starmask.reduce",
        "starmask.preview",
        "PIL.ImageTk",
        "tkinter",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "_pytest", "pip", "PyInstaller"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StarMask",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="StarMask",
)

app = BUNDLE(
    coll,
    name="StarMask.app",
    icon=str(PROJECT_ROOT / "packaging" / "StarMask.icns"),
    bundle_identifier="com.starmask.app",
    info_plist={
        "CFBundleName": "StarMask",
        "CFBundleDisplayName": "StarMask 银河缩星",
        "CFBundleShortVersionString": "0.4.1",
        "CFBundleVersion": "0.4.1",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
    },
)
