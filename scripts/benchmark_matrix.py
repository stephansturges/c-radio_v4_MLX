#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cradio_mlx.benchmark import MLXSO400MBenchmarkRequest, write_mlx_so400m_benchmark


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    variant: str
    checkpoint: Path
    dtype: str
    quantization: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--so400m-bf16", type=Path, default=Path("bundles/c-radiov4-so400m-bf16"))
    parser.add_argument("--so400m-8bit", type=Path, default=Path("bundles/c-radiov4-so400m-8bit"))
    parser.add_argument(
        "--so400m-mxfp8",
        type=Path,
        default=Path("bundles/c-radiov4-so400m-mxfp8"),
    )
    parser.add_argument(
        "--so400m-cider-w8a8",
        type=Path,
        default=Path("bundles/c-radiov4-so400m-cider-w8a8"),
    )
    parser.add_argument(
        "--so400m-cider-w8a8-g128",
        type=Path,
        default=Path("bundles/c-radiov4-so400m-cider-w8a8-g128"),
    )
    parser.add_argument(
        "--so400m-cider-w8a8-p9999",
        type=Path,
        default=Path("bundles/c-radiov4-so400m-cider-w8a8-p9999"),
    )
    parser.add_argument("--h-bf16", type=Path, default=Path("bundles/c-radiov4-h-bf16"))
    parser.add_argument("--h-8bit", type=Path, default=Path("bundles/c-radiov4-h-8bit"))
    parser.add_argument("--h-mxfp8", type=Path, default=Path("bundles/c-radiov4-h-mxfp8"))
    parser.add_argument(
        "--h-cider-w8a8",
        type=Path,
        default=Path("bundles/c-radiov4-h-cider-w8a8"),
    )
    parser.add_argument(
        "--h-cider-w8a8-g128",
        type=Path,
        default=Path("bundles/c-radiov4-h-cider-w8a8-g128"),
    )
    parser.add_argument(
        "--h-cider-w8a8-p9999",
        type=Path,
        default=Path("bundles/c-radiov4-h-cider-w8a8-p9999"),
    )
    parser.add_argument(
        "--include-cider",
        action="store_true",
        help="Include optional Cider W8A8 cases. Requires Cider on Apple M5+.",
    )
    parser.add_argument("--image-sizes", nargs="+", type=int, default=[256, 512, 768])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-materialize", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cider-fusion", choices=["off", "auto", "required"], default="off")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/benchmark_matrix"))
    parser.add_argument("--summary", type=Path, default=Path("reports/benchmark-matrix.json"))
    args = parser.parse_args()

    cases = [
        BenchmarkCase("so400m-bf16", "so400m", args.so400m_bf16, "bfloat16", "none"),
        BenchmarkCase("so400m-8bit", "so400m", args.so400m_8bit, "bfloat16", "8bit-affine"),
        BenchmarkCase("so400m-mxfp8", "so400m", args.so400m_mxfp8, "bfloat16", "mxfp8"),
        BenchmarkCase("h-bf16", "h", args.h_bf16, "bfloat16", "none"),
        BenchmarkCase("h-8bit", "h", args.h_8bit, "bfloat16", "8bit-affine"),
        BenchmarkCase("h-mxfp8", "h", args.h_mxfp8, "bfloat16", "mxfp8"),
    ]
    if args.include_cider:
        cases.extend(
            [
                BenchmarkCase(
                    "so400m-cider-w8a8",
                    "so400m",
                    args.so400m_cider_w8a8,
                    "bfloat16",
                    "cider-w8a8",
                ),
                BenchmarkCase(
                    "so400m-cider-w8a8-g128",
                    "so400m",
                    args.so400m_cider_w8a8_g128,
                    "bfloat16",
                    "cider-w8a8-g128",
                ),
                BenchmarkCase(
                    "so400m-cider-w8a8-p9999",
                    "so400m",
                    args.so400m_cider_w8a8_p9999,
                    "bfloat16",
                    "cider-w8a8-p9999",
                ),
                BenchmarkCase(
                    "h-cider-w8a8",
                    "h",
                    args.h_cider_w8a8,
                    "bfloat16",
                    "cider-w8a8",
                ),
                BenchmarkCase(
                    "h-cider-w8a8-g128",
                    "h",
                    args.h_cider_w8a8_g128,
                    "bfloat16",
                    "cider-w8a8-g128",
                ),
                BenchmarkCase(
                    "h-cider-w8a8-p9999",
                    "h",
                    args.h_cider_w8a8_p9999,
                    "bfloat16",
                    "cider-w8a8-p9999",
                ),
            ]
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not (case.checkpoint / "model.safetensors").exists():
            rows.append(
                {
                    "label": case.label,
                    "variant": case.variant,
                    "checkpoint": str(case.checkpoint),
                    "quantization": case.quantization,
                    "benchmark_state": "skipped_missing_checkpoint",
                }
            )
            continue

        report_path = args.out_dir / f"{case.label}.json"
        cider_fusion = args.cider_fusion if case.quantization.startswith("cider") else "off"
        write_mlx_so400m_benchmark(
            MLXSO400MBenchmarkRequest(
                checkpoint=case.checkpoint,
                image=args.images,
                report=report_path,
                variant=case.variant,
                image_size=args.image_sizes,
                dtype=case.dtype,
                batch_size=args.batch_sizes,
                warmups=args.warmups,
                repeats=args.repeats,
                materialize_outputs=not args.no_materialize,
                compile_forward=args.compile,
                cider_fusion=cider_fusion,
            )
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for row in report["rows"]:
            rows.append(
                {
                    "label": case.label,
                    "variant": case.variant,
                    "checkpoint": str(case.checkpoint),
                    "quantization": case.quantization,
                    "dtype": case.dtype,
                    "materialize_outputs": report["materialize_outputs"],
                    "compile_forward": report["compile_forward"],
                    "cider_fusion": report["cider_fusion"],
                    "report": str(report_path),
                    **row,
                }
            )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
