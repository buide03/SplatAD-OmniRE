"""Serve the local replay and metrics dashboard, including byte-range video seeking."""
import argparse
import json
import mimetypes
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ASSETS = Path(__file__).resolve().parents[1] / "dashboard"


class Dashboard(SimpleHTTPRequestHandler):
    def __init__(self, *args, artifacts, **kwargs):
        self.artifacts = artifacts
        super().__init__(*args, directory=str(ASSETS), **kwargs)

    def do_GET(self):
        self.route(head=False)

    def do_HEAD(self):
        self.route(head=True)

    def route(self, head):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/runs":
            runs = []
            for meta in self.artifacts.rglob("run.json"):
                if not meta.resolve().is_relative_to(self.artifacts):
                    continue
                try:
                    record = json.loads(meta.read_text(encoding="utf-8"))
                    if {"scene", "variant", "videos", "step"} <= record.keys():
                        runs.append({**record, "path": meta.parent.relative_to(self.artifacts).as_posix()})
                except (OSError, ValueError, TypeError):
                    continue
            runs.sort(key=lambda row: (row["scene"], row["variant"], row["step"] or 0))
            data = json.dumps(runs).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if not head:
                self.wfile.write(data)
        elif path.startswith("/artifacts/"):
            target = (self.artifacts / path.removeprefix("/artifacts/")).resolve()
            if not target.is_relative_to(self.artifacts) or not target.is_file():
                self.send_error(404)
                return
            self.artifact(target, head)
        elif path in ("/", "/index.html", "/styles.css", "/app.js", "/charts.js", "/pose-monitor.js",
                      "/quality-monitor.js", "/training-monitor.js"):
            if head:
                super().do_HEAD()
            else:
                super().do_GET()
        else:
            self.send_error(404)

    def artifact(self, path, head):
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                first, last = match.groups()
                if first:
                    start, end = int(first), min(int(last), end) if last else end
                elif last:
                    start = max(0, size - int(last))
                else:
                    start = size
            if not match or not 0 <= start <= end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        remaining = max(0, end - start + 1)
        self.send_header("Content-Length", str(remaining))
        self.end_headers()
        if not head:
            with path.open("rb") as file:
                file.seek(start)
                try:
                    while remaining:
                        chunk = file.read(min(remaining, 256 * 1024))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except ConnectionError:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = partial(Dashboard, artifacts=args.artifacts.resolve())
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Dashboard http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
