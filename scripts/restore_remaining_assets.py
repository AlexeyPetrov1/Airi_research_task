"""Restore large research inputs from the verified GitHub Release manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        while data := source.read(8 * 1024**2):
            result.update(data)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("report/remaining_assets_manifest.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--only", help="Restore only this original relative path")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.root.resolve()
    for entry in manifest["release_files"]:
        if args.only and entry["path"] != args.only:
            continue
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Manifest path escapes the destination directory")
        if target.is_file():
            if target.stat().st_size == entry["bytes"] and digest(target) == entry["sha256"]:
                print("VERIFIED", entry["path"], flush=True)
                continue
            raise FileExistsError(f"Existing file has different bytes: {target}")
        if args.verify_only:
            raise FileNotFoundError(target)
        if sum(part["bytes"] for part in entry.get("parts", [])) != entry["bytes"]:
            raise RuntimeError(f"Release upload is still incomplete for: {entry['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".restore-download")
        with temporary.open("wb") as output:
            for part in entry["parts"]:
                checksum = hashlib.sha256()
                received = 0
                with urllib.request.urlopen(part["url"], timeout=120) as source:
                    while data := source.read(1024**2):
                        output.write(data)
                        checksum.update(data)
                        received += len(data)
                if received != part["bytes"] or checksum.hexdigest() != part["sha256"]:
                    raise ValueError(f"Release asset checksum failed: {part['url']}")
        if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
            raise ValueError(f"Reconstructed file checksum failed: {entry['path']}")
        temporary.rename(target)
        print("RESTORED", entry["path"], flush=True)


if __name__ == "__main__":
    main()
