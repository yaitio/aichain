"""Download what sources.toml pins into data/sources, once, and say when a file changed."""

import hashlib
import time
import urllib.request
from pathlib import Path

try:
    import tomllib
except ImportError:                       # Python 3.10: aichain installs tomli
    import tomli as tomllib


def fetch_sources(toml_path, into, *, user_agent=None, min_interval=0.0) -> dict:
    """{id: path} for every [[source]]; a file already present is not fetched again."""
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    paths, last = {}, 0.0
    for src in tomllib.loads(Path(toml_path).read_text())["source"]:
        path = into / src["file"]
        if not path.exists():
            time.sleep(max(0.0, min_interval - (time.monotonic() - last)))
            request = urllib.request.Request(
                src["url"], headers={"User-Agent": user_agent or "aichain-cookbook/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
            except OSError as exc:
                raise SystemExit(f"cannot download {src['id']}: {src['url']}\n  {exc}") from None
            last = time.monotonic()
            part = path.with_name(path.name + ".part")
            part.write_bytes(data)
            part.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not src.get("sha256"):
            print(f'sha256 of {src["id"]}: "{digest}" — copy it into sources.toml')
        elif digest != src["sha256"]:
            print(f"source changed: {src['id']} (expected {src['sha256'][:12]}…, got {digest[:12]}…)")
        paths[src["id"]] = path
    return paths
