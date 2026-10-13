#!/usr/bin/env bash
# 安装官方 StarNet2 CLI（Apple Silicon / Intel Mac / Linux）
# 用法: bash scripts/install_starnet2.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/starnet2"
mkdir -p "$DEST/bin"

ARCH="$(uname -m)"
OS="$(uname -s)"
echo "检测: $OS $ARCH"

URL=""
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  URL="https://download.starnetastro.com/starnet2_macos-arm64_2.6.2-0242_COREML_arm64_cli.zip"
  NAME="starnet2_macos-arm64_cli.zip"
elif [[ "$OS" == "Darwin" ]]; then
  URL="https://download.starnetastro.com/starnet2_macos-x64_2.6.2-0242_ORT_x64_cli.zip"
  NAME="starnet2_macos-x64_cli.zip"
elif [[ "$OS" == "Linux" ]]; then
  URL="https://download.starnetastro.com/starnet2_linux_2.6.2-0242_ORT_x64_cli.zip"
  NAME="starnet2_linux_cli.zip"
else
  echo "请从 https://starnetastro.com/cli-tools/starnet/ 手动下载" >&2
  exit 1
fi

ZIP="$DEST/$NAME"
echo "下载 StarNet2 CLI: $URL"
echo "（约 120MB，网络慢时请耐心等待；支持断点续传）"
curl --http1.1 -L -C - --retry 20 --retry-all-errors --retry-delay 5 \
  --progress-bar -o "$ZIP" "$URL"

echo "解压..."
unzip -o "$ZIP" -d "$DEST/pkg"

# 找到可执行文件
BIN="$(find "$DEST/pkg" -type f \( -name 'starnet2' -o -name 'starnet++' \) | head -1 || true)"
if [[ -z "$BIN" ]]; then
  # 有些包把二进制放在带版本号的目录
  BIN="$(find "$DEST/pkg" -type f -perm -111 ! -name '*.dylib' ! -name '*.so' ! -name '*.txt' | head -5)"
  echo "候选文件:"; echo "$BIN"
  BIN="$(echo "$BIN" | head -1)"
fi
if [[ -z "$BIN" ]]; then
  echo "未在包内找到 starnet2 可执行文件" >&2
  exit 1
fi

chmod +x "$BIN"
ln -sfn "$BIN" "$DEST/bin/starnet2"
echo "已安装: $DEST/bin/starnet2"
echo "可选: export STARNET2_BIN=\"$DEST/bin/starnet2\""
"$DEST/bin/starnet2" --version || "$DEST/bin/starnet2" -h | head -5 || true
echo "完成。之后: starmask process 图.tif --method starnet"
