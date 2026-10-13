#!/bin/bash
cd "$(dirname "$0")"
export STARNET2_BIN="${STARNET2_BIN:-$PWD/third_party/starnet2/bin/starnet2}"
export DYLD_LIBRARY_PATH="$PWD/third_party/starnet2/pkg/starnet2_macos-arm64_2.6.2-0242_COREML_arm64_cli/lib:${DYLD_LIBRARY_PATH}"
exec .venv/bin/python -m starmask.cli gui
