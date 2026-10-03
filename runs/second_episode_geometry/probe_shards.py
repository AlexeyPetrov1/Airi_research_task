"""Read only the first and last 64 KiB from each EgoDex track tar shard."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from probe_remote_tar import headers_at


REV = "c3dec07d796ddeccdc8f5a35bf4920b3ee044feb"
BASE = f"https://huggingface.co/datasets/allenai/molmo-motion-1m/resolve/{REV}/"
TREE = f"https://huggingface.co/api/datasets/allenai/molmo-motion-1m/tree/{REV}/egodex/tracks"


def probe(entry):
    url = BASE + entry["path"]
    size = entry["size"]
    begin = headers_at(url, 0, 65536)
    end = headers_at(url, size - 65536, 65536)
    return entry["path"], size, begin[:1], end[-1:]


def main():
    entries = requests.get(TREE, timeout=30).json()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(probe, e) for e in entries]
        for f in as_completed(futures):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
