#!/bin/bash
# 打 macOS arm64 离线 DMG（方案 A：含 StarNet2）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_PY="${BUILD_PY:-/Users/west/miniforge3/bin/python3}"
VENV=/tmp/sm_build
PKG="$ROOT/third_party/starnet2/pkg/starnet2_macos-arm64_2.6.2-0242_COREML_arm64_cli"
OUT="$ROOT/StarMask-0.4.1-macOS-arm64.dmg"

"$BUILD_PY" -m venv "$VENV" 2>/dev/null || true
"$VENV/bin/pip" install -q -U pip wheel
"$VENV/bin/pip" install -q numpy scipy rawpy tifffile imageio click rich pillow pyinstaller

cd "$ROOT/packaging"
if [ ! -f StarMask.icns ] && [ -f "$ROOT/assets/starmask_logo.png" ]; then
  "$VENV/bin/python" -c "
from PIL import Image
from pathlib import Path
import subprocess, shutil
im = Image.open(Path('$ROOT/assets/starmask_logo.png')).convert('RGBA')
iconset = Path('logo/StarMask.iconset')
shutil.rmtree(iconset, ignore_errors=True)
iconset.mkdir(parents=True)
for s in (16, 32, 128, 256, 512):
    im.resize((s, s), Image.Resampling.LANCZOS).save(iconset / f'icon_{s}x{s}.png')
    im.resize((s * 2, s * 2), Image.Resampling.LANCZOS).save(iconset / f'icon_{s}x{s}@2x.png')
subprocess.check_call(['iconutil', '-c', 'icns', str(iconset), '-o', 'StarMask.icns'])
"
fi
rm -rf dist build dmg_stage
"$VENV/bin/python" -m PyInstaller --noconfirm --clean StarMask.spec

# 从源包完整铺 StarNet2（必须是真实文件，不能是断链）
APP="dist/StarMask.app"
RES="$APP/Contents/Resources/starnet2_cli"
rm -rf "$RES" "$APP/Contents/Frameworks/starnet2_cli" "dist/StarMask/_internal/starnet2_cli"
mkdir -p "$RES"
cp -R "$PKG/LICENSE.txt" "$PKG/README.txt" "$PKG/StarNet2_weights.mlpackage" "$RES/"
cp -R "$PKG/lib" "$RES/lib"
cp "$PKG/starnet2" "$RES/starnet2"
chmod +x "$RES/starnet2"
# 依赖元数据放 Resources（勿放 Frameworks，会破坏 codesign）
for s in /tmp/sm_build/lib/python3.12/site-packages/*.dist-info; do
  base=$(basename "$s")
  case "$base" in
    imageio*|tifffile*|rawpy*|pillow*|numpy*|scipy*|click*|rich*|imageio_*) 
      [ -d "$APP/Contents/Resources/$base" ] || cp -R "$s" "$APP/Contents/Resources/$base"
      ;;
  esac
done
rm -rf "$APP/Contents/Frameworks/"*.dist-info

codesign --force --deep --sign - dist/StarMask.app

STAGE="$ROOT/packaging/dmg_stage"
mkdir -p "$STAGE"
cp -R dist/StarMask.app "$STAGE/"
cp "$ROOT/packaging/dmg_stage/使用说明.txt" "$STAGE/" 2>/dev/null || true
# 若说明不在，从仓库复制
test -f "$STAGE/使用说明.txt" || cp "$ROOT/packaging/使用说明.txt" "$STAGE/" 2>/dev/null || true

rm -f "$OUT"
hdiutil create -volname "StarMask 银河缩星 0.4.1" -srcfolder "$STAGE" \
  -ov -format UDZO -imagekey zlib-level=9 "$OUT"
ls -lh "$OUT"
