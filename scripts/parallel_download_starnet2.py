"""并行分段下载大文件（绕过单连接慢速问题）。"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import urllib.request

URL = "https://download.starnetastro.com/starnet2_macos-arm64_2.6.2-0242_COREML_arm64_cli.zip"
OUT = Path(__file__).resolve().parents[1] / "third_party" / "starnet2" / "starnet2_macos-arm64_cli.zip"
N = 8  # 并行分段


def fetch_range(i: int, start: int, end: int, parts: list[bytes | None]) -> None:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            parts[i] = r.read()
        print(f"part {i} done ({end-start+1} bytes)", flush=True)
    except Exception as e:
        print(f"part {i} fail: {e}", flush=True)
        parts[i] = b""


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 取总大小
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        total = int(r.headers["Content-Length"])
    print(f"total {total/1e6:.1f} MB, {N} segments", flush=True)

    chunk = (total + N - 1) // N
    parts: list[bytes | None] = [None] * N
    threads = []
    for i in range(N):
        start = i * chunk
        end = min(start + chunk - 1, total - 1)
        t = threading.Thread(target=fetch_range, args=(i, start, end, parts), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=600)

    if any(p is None for p in parts):
        print("some segments failed", flush=True)
        return 1
    data = b"".join(p for p in parts if p is not None)
    if len(data) != total:
        print(f"size mismatch {len(data)} != {total}", flush=True)
        return 1
    OUT.write_bytes(data)
    print(f"saved {OUT} ({OUT.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
