#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["SO400M", "H"])
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "8bit-affine"])
    parser.add_argument("--image-sizes", nargs="+", type=int, default=[256, 384, 512, 768, 1024])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--out", type=Path, default=Path("reports/benchmark-matrix.json"))
    args = parser.parse_args()

    rows = [
        {
            "model": model,
            "dtype": dtype,
            "image_size": image_size,
            "batch_size": batch_size,
        }
        for model, dtype, image_size, batch_size in itertools.product(
            args.models, args.dtypes, args.image_sizes, args.batch_sizes
        )
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
