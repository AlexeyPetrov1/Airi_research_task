"""Probe tar member names with small byte ranges; never downloads the archive."""

from __future__ import annotations

import argparse
import requests


def headers_at(url: str, offset: int, length: int = 1 << 20):
    end = offset + length - 1
    response = requests.get(url, headers={"Range": f"bytes={offset}-{end}"}, timeout=90)
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != length:
        raise RuntimeError((response.status_code, response.headers.get("Content-Range"), len(response.content)))
    data = response.content
    found = []
    first = (-offset) % 512
    for i in range(first, len(data) - 512, 512):
        h = data[i : i + 512]
        if h[257:262] != b"ustar":
            continue
        try:
            recorded = int(h[148:156].strip(b" \0") or b"0", 8)
            actual = sum(h[:148]) + sum(b" " * 8) + sum(h[156:])
            if recorded != actual:
                continue
            size = int(h[124:136].strip(b" \0") or b"0", 8)
            name = h[:100].split(b"\0")[0].decode("utf8")
            prefix = h[345:500].split(b"\0")[0].decode("utf8")
            if prefix:
                name = prefix + "/" + name
            found.append((offset + i, name, size, h[156:157].decode("ascii")))
        except (UnicodeError, ValueError):
            continue
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("size", type=int)
    p.add_argument("--samples", type=int, default=9)
    args = p.parse_args()
    for j in range(args.samples):
        offset = int((args.size - (1 << 20)) * j / (args.samples - 1))
        offset -= offset % 512
        members = headers_at(args.url, offset)
        print(j, offset, len(members), members[:1], members[-1:], flush=True)


if __name__ == "__main__":
    main()
