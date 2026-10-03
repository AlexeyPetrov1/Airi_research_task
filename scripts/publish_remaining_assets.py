"""Publish oversized local research files as resumable GitHub Release assets.

Credentials are obtained from Git's configured credential helper, never stored.
Parts are streamed directly from the source; no second checkpoint copy is made.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.parse
import urllib.request

PART_BYTES = 1024**3
LOCK = threading.Lock()


class SliceReader:
    def __init__(self, path, offset, size):
        self.file = open(path, "rb")
        self.file.seek(offset)
        self.remaining = size
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.file.read(min(size, self.remaining))
        self.remaining -= len(data)
        self.digest.update(data)
        return data

    def close(self):
        self.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--repository", default="AlexeyPetrov1/Airi_research_task")
    parser.add_argument("--tag", default="research-local-assets-2026-10-03")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    credentials = subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
        text=True, capture_output=True, check=True,
    )
    fields = dict(line.split("=", 1) for line in credentials.stdout.splitlines() if "=" in line)
    token = fields["password"]
    api_root = f"https://api.github.com/repos/{args.repository}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    def api(path, payload=None, method=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(api_root + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=90) as response:
            return None if response.status == 204 else json.load(response)

    try:
        release = api(f"/releases/tags/{args.tag}")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        release = api("/releases", {
            "tag_name": args.tag, "target_commitish": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "name": "Complete local research inputs and model checkpoints",
            "body": "Large research inputs and model weights. Files larger than 1 GiB are split into ordered parts. The repository manifest contains original paths, sizes, SHA-256 hashes and restore instructions.",
            "draft": False, "prerelease": False,
        })
    upload_root = release["upload_url"].split("{")[0]
    remote_assets = {}
    page = 1
    while True:
        items = api(f"/releases/{release['id']}/assets?per_page=100&page={page}")
        remote_assets.update({asset["name"]: asset for asset in items})
        if len(items) < 100:
            break
        page += 1
    state_path = args.plan.with_suffix(".uploaded.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"release_url": release["html_url"], "assets": {}}

    def save(asset, digest):
        assert asset["state"] == "uploaded"
        if asset.get("digest"):
            assert asset["digest"] == "sha256:" + digest
        with LOCK:
            state["assets"][asset["name"]] = {"url": asset["browser_download_url"], "bytes": asset["size"], "sha256": digest, "id": asset["id"]}
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print("VERIFIED", asset["name"], asset["size"], flush=True)

    def upload(job):
        path, offset, size, name = job
        existing = remote_assets.get(name)
        if existing and existing["state"] == "uploaded" and existing["size"] == size:
            reader = SliceReader(path, offset, size)
            try:
                while reader.read(4 * 1024**2):
                    pass
                save(existing, reader.digest.hexdigest())
                return
            finally:
                reader.close()
        if existing:
            api(f"/releases/assets/{existing['id']}", method="DELETE")
        for attempt in range(3):
            reader = SliceReader(path, offset, size)
            url = urllib.parse.urlsplit(upload_root + "?name=" + urllib.parse.quote(name))
            conn = http.client.HTTPSConnection(url.hostname, timeout=1200, blocksize=1024**2)
            try:
                print("UPLOADING", name, size, flush=True)
                conn.request("POST", url.path + "?" + url.query, body=reader,
                             headers={**headers, "Content-Type": "application/octet-stream", "Content-Length": str(size)})
                response = conn.getresponse()
                body = response.read()
                if response.status != 201:
                    raise RuntimeError(f"Asset upload returned HTTP {response.status}: {body[:300]!r}")
                assert reader.remaining == 0
                asset = json.loads(body)
                assert asset["size"] == size
                save(asset, reader.digest.hexdigest())
                return
            except (OSError, http.client.HTTPException) as exc:
                print("RETRY", name, type(exc).__name__, flush=True)
                if attempt == 2:
                    raise
                time.sleep(5)
            finally:
                reader.close()
                conn.close()

    jobs = []
    for entry in plan["release_files"]:
        size = entry["bytes"]
        for number, offset in enumerate(range(0, size, PART_BYTES)):
            name = entry["asset_prefix"] + (f".part{number:03d}" if size > PART_BYTES else "")
            jobs.append((entry["path"], offset, min(PART_BYTES, size-offset), name))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(upload, jobs))
    print("COMPLETE", len(jobs), release["html_url"], flush=True)


if __name__ == "__main__":
    main()
