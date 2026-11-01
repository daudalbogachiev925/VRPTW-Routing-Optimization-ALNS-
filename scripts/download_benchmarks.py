"""Download Solomon CVRPTW benchmark instances from SINTEF.

The repo already ships c101, c201, r101, rc101 — this script exists to add more
from the public SINTEF repository. SINTEF requires interactive download
(there's no stable direct URL), so this script uses a curated mirror.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from src.config import get_app_config, get_settings

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vrptw.download")

# Public mirror of Solomon instances (CVRPLIB does not host Solomon directly;
# this GitHub raw mirror mirrors the SINTEF format).
MIRROR_BASE = "https://raw.githubusercontent.com/avral/vrptw/master/data"

DEFAULT_INSTANCES = ["c101", "c201", "r101", "rc101", "c102", "r201", "rc201"]


def download(name: str, dest_dir: Path) -> Path | None:
    url = f"{MIRROR_BASE}/{name}.txt"
    out = dest_dir / f"{name}.txt"
    if out.exists():
        logger.info("Skip existing: %s", out)
        return out
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            out.write_bytes(resp.content)
        logger.info("Downloaded %s (%.1f KB)", out, out.stat().st_size / 1024)
        return out
    except Exception as exc:
        logger.warning("Failed to download %s: %s", name, exc)
        return None


def main() -> int:
    cfg = get_app_config()
    dest = Path(cfg.benchmark.instances_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in DEFAULT_INSTANCES:
        download(name, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
