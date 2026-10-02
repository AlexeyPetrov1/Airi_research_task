"""List a remote ZIP's member names using byte ranges, without fetching its payload.

Usage: python scripts/inspect_remote_zip.py URL [substring]
"""

import io
import sys
import zipfile

import requests


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, block_size: int = 4 * 1024 * 1024):
        self.url = url
        self.session = requests.Session()
        response = self.session.head(url, timeout=30)
        response.raise_for_status()
        self.length = int(response.headers["Content-Length"])
        if "bytes" not in response.headers.get("Accept-Ranges", ""):
            probe = self.session.get(url, headers={"Range": "bytes=0-0"},
                                     timeout=30, stream=True)
            valid = (probe.status_code == 206 and
                     probe.headers.get("Content-Range", "").startswith("bytes 0-0/"))
            probe.close()
            if not valid:
                raise RuntimeError("server does not support byte ranges")
        self.block_size = block_size
        self.pos = 0
        self.blocks = {}
        self.transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        else:
            raise ValueError(whence)
        if self.pos < 0:
            raise ValueError("negative seek")
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.pos
        size = min(size, self.length - self.pos)
        chunks = []
        while size:
            start = (self.pos // self.block_size) * self.block_size
            if start not in self.blocks:
                end = min(start + self.block_size, self.length) - 1
                for attempt in range(3):
                    # Check headers before reading the body: a transient 200
                    # response could otherwise download the entire archive.
                    response = self.session.get(
                        self.url,
                        headers={"Range": f"bytes={start}-{end}",
                                 "Cache-Control": "no-cache"},
                        timeout=60,
                        stream=True,
                    )
                    valid = (response.status_code == 206 and
                             response.headers.get("Content-Range", "").startswith(
                                 f"bytes {start}-{end}/"))
                    if valid:
                        data = response.content
                        response.close()
                        if len(data) != end - start + 1:
                            raise RuntimeError("incomplete range response")
                        self.blocks[start] = data
                        self.transferred += len(data)
                        break
                    response.close()
                else:
                    raise RuntimeError(f"invalid range response at {start}-{end}")
            block = self.blocks[start]
            offset = self.pos - start
            count = min(size, len(block) - offset)
            chunks.append(block[offset : offset + count])
            self.pos += count
            size -= count
        return b"".join(chunks)


def main():
    url = sys.argv[1]
    needle = sys.argv[2] if len(sys.argv) > 2 else ""
    remote = HTTPRangeFile(url)
    with zipfile.ZipFile(remote) as archive:
        matches = [
            member for member in archive.infolist() if needle.lower() in member.filename.lower()
        ]
        print(f"members={len(archive.infolist())} matched={len(matches)}")
        print(f"archive_bytes={remote.length} range_bytes={remote.transferred}")
        for member in matches[:30]:
            print(member.filename, member.file_size, member.compress_size)


if __name__ == "__main__":
    main()
