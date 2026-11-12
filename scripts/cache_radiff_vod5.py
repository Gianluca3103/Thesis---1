"""Cache the paper's uncensored five-sweep VoD radar inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.radiff_radar_data import cache_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(cache_split(args.vod_root, args.cache_root,
                                 args.split, args.limit), indent=2), flush=True)


if __name__ == "__main__":
    main()
