"""Download selected PandaSet scenes from the public archive using HTTP ranges."""
import argparse
import io
import re
import shutil
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath

ARCHIVE = "https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123aea3b3132c67f4b395ec6120f63e190271/pandaset.zip"


class RangeArchive(io.RawIOBase):
    def __init__(self, url):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.url = url
        self.client = requests.Session()
        self.client.mount("https://", HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)
        )))
        with self.client.get(url, headers={"Range": "bytes=0-0"}, timeout=60, stream=True) as r:
            r.raise_for_status()
            header = r.headers.get("Content-Range", "")
            if r.status_code != 206 or not re.fullmatch(r"bytes 0-0/\d+", header):
                raise RuntimeError("The archive server must support byte ranges")
            self.length = int(header.split("/")[-1])
        self.position = 0
        self.chunk = 4 * 1024 * 1024
        self.block = lru_cache(maxsize=6)(self._block)

    def _block(self, start):
        end = min(start + self.chunk, self.length) - 1
        with self.client.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120, stream=True) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.length}":
                raise RuntimeError("Unexpected archive response range")
            return response.content

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = max(0, (0, self.position, self.length)[whence] + offset)
        return self.position

    def read(self, size=-1):
        end = self.length if size < 0 else min(self.length, self.position + size)
        pieces = []
        while self.position < end:
            start = self.position // self.chunk * self.chunk
            index = self.position - start
            take = min(end - self.position, self.chunk - index)
            part = self.block(start)[index:index + take]
            if len(part) != take:
                raise IOError("Incomplete download from PandaSet archive")
            pieces.append(part)
            self.position += take
        return b"".join(pieces)

    def close(self):
        self.client.close()
        self.block.cache_clear()
        super().close()


def scene_files(archive, scenes):
    for record in sorted(archive.infolist(), key=lambda info: info.header_offset):
        path = PurePosixPath(record.filename)
        if record.is_dir() or path.is_absolute() or ".." in path.parts:
            continue
        for i, part in enumerate(path.parts[:-1]):
            if part in scenes and path.parts[i + 1] in {"camera", "lidar", "annotations", "meta"}:
                yield record, Path(*path.parts[i:])
                break
        else:
            if len(path.parts) >= 2 and path.parts[-2] in scenes and path.suffix == ".json":
                yield record, Path(*path.parts[-2:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", default=ARCHIVE)
    args = parser.parse_args()
    if not all(re.fullmatch(r"[0-9]{3}", scene) for scene in args.scenes):
        parser.error("Use three-digit PandaSet scene IDs")
    with RangeArchive(args.archive) as stream, zipfile.ZipFile(stream) as archive:
        records = list(scene_files(archive, set(args.scenes)))
        found = {path.parts[0] for _, path in records}
        if found != set(args.scenes):
            parser.error(f"Missing selected scenes: {sorted(set(args.scenes) - found)}")
        print(f"{len(records)} files / {sum(r.file_size for r, _ in records) / 1024**3:.2f} GiB extracted", flush=True)
        for number, (record, relative) in enumerate(records, 1):
            dest = args.output / relative
            if dest.is_file() and dest.stat().st_size == record.file_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(dest.suffix + ".part")
            with archive.open(record) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(dest)
            if number % 100 == 0 or number == len(records):
                print(f"{number}/{len(records)} {relative}", flush=True)


if __name__ == "__main__":
    main()
