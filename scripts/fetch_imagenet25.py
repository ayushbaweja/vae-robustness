#!/usr/bin/env python3
"""Download 25 deterministic ImageNet sample images.

Source: https://github.com/EliSchwartz/imagenet-sample-images
One image per class; we take a uniform stride across the sorted list
(seed=42, 25 picks) so the choice is reproducible.
"""

import json
import urllib.request
import random
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "resources" / "test-images-imagenet25"
N = 25
SEED = 42

API_URL = "https://api.github.com/repos/EliSchwartz/imagenet-sample-images/contents/"
RAW_PREFIX = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"


def main():
    DEST.mkdir(parents=True, exist_ok=True)

    print("Listing repo contents...")
    req = urllib.request.Request(API_URL, headers={"User-Agent": "vae-robustness"})
    with urllib.request.urlopen(req) as r:
        listing = json.loads(r.read())
    names = sorted(x["name"] for x in listing if x["name"].endswith(".JPEG"))
    print(f"  found {len(names)} sample images")

    rng = random.Random(SEED)
    picks = sorted(rng.sample(names, N))
    print(f"  selected {N} (seed={SEED}):")
    for p in picks:
        print(f"    {p}")

    print("\nDownloading...")
    for name in picks:
        out = DEST / name
        if out.exists() and out.stat().st_size > 0:
            print(f"  skip {name} (exists)")
            continue
        url = RAW_PREFIX + name
        req = urllib.request.Request(url, headers={"User-Agent": "vae-robustness"})
        with urllib.request.urlopen(req) as r, open(out, "wb") as f:
            f.write(r.read())
        print(f"  saved {name} ({out.stat().st_size} B)")

    print(f"\nDone. {len(list(DEST.glob('*.JPEG')))} images in {DEST}")


if __name__ == "__main__":
    main()
