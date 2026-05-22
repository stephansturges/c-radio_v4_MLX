from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cradio_mlx.acceleration import inspect_acceleration
from cradio_mlx.benchmark import (
    BenchmarkRequest,
    MLXSO400MBenchmarkRequest,
    PyTorchBenchmarkRequest,
    write_benchmark_stub,
    write_mlx_so400m_benchmark,
    write_pytorch_benchmark,
)
from cradio_mlx.bundles import inspect_bundle
from cradio_mlx.compat.hf_mapping import write_weight_audit
from cradio_mlx.config import get_model_config
from cradio_mlx.convert import ConversionRequest, convert_checkpoint
from cradio_mlx.encoder import CRadioEncoder
from cradio_mlx.golden import (
    ReferenceCaptureRequest,
    capture_reference_outputs,
    save_embedding_result_npz,
)
from cradio_mlx.metrics import write_embedding_comparison
from cradio_mlx.mlx_so400m import MLXHEncoder, MLXSO400MEncoder
from cradio_mlx.quantize import QuantizationRequest, quantize_bundle


def _image_size(values: list[int]) -> int | tuple[int, int]:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--image-size expects one value or HEIGHT WIDTH")


def _dump(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cradio-mlx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("device-info", help="Report available acceleration backends.")

    spatial = subparsers.add_parser("spatial-shape", help="Print spatial grid metadata.")
    spatial.add_argument("--model-id", default="nvidia/C-RADIOv4-H")
    spatial.add_argument("--image-size", nargs="+", type=int, default=[512])

    inspect = subparsers.add_parser("inspect-bundle", help="Inspect a local bundle directory.")
    inspect.add_argument("--model", required=True, type=Path)

    convert = subparsers.add_parser("convert", help="Create or convert an MLX bundle.")
    convert.add_argument("--hf-path", required=True, type=Path)
    convert.add_argument("--mlx-path", required=True, type=Path)
    convert.add_argument("--model-id", required=True)
    convert.add_argument("--revision", required=True)
    convert.add_argument("--dtype", default="bfloat16")
    convert.add_argument("--manifest-only", action="store_true")

    quantize = subparsers.add_parser("quantize", help="Plan or run bundle quantization.")
    quantize.add_argument("--model", required=True, type=Path)
    quantize.add_argument("--out", required=True, type=Path)
    quantize.add_argument("--bits", type=int, default=8)
    quantize.add_argument("--group-size", type=int, default=64)
    quantize.add_argument("--mode", default="affine")
    quantize.add_argument("--clip-percentile", type=float)
    quantize.add_argument("--smooth-scales", type=Path)
    quantize.add_argument("--dry-run", action="store_true")

    pytorch = subparsers.add_parser("pytorch-ref", help="Run the PyTorch reference backend.")
    pytorch.add_argument("--model-id", default="nvidia/C-RADIOv4-H")
    pytorch.add_argument("--revision")
    pytorch.add_argument("--image", required=True, type=Path)
    pytorch.add_argument("--image-size", nargs="+", type=int, default=[512])
    pytorch.add_argument("--dtype", default="bfloat16")
    pytorch.add_argument("--device", default="auto")
    pytorch.add_argument("--save-npz", type=Path)

    embed = subparsers.add_parser("embed", help="Encode images with an accelerated backend.")
    embed.add_argument("--model-id", default="nvidia/C-RADIOv4-SO400M")
    embed.add_argument("--revision")
    embed.add_argument("--backend", default="pytorch", choices=["pytorch", "mlx-so400m", "mlx-h"])
    embed.add_argument("--checkpoint", type=Path)
    embed.add_argument("--image", required=True, type=Path)
    embed.add_argument("--image-size", nargs="+", type=int, default=[512])
    embed.add_argument("--dtype", default="bfloat16")
    embed.add_argument("--device", default="auto")
    embed.add_argument("--compile", action="store_true")
    embed.add_argument("--cider-fusion", choices=["off", "auto", "required"], default="off")
    embed.add_argument("--save-npz", type=Path)

    capture = subparsers.add_parser(
        "capture-reference",
        help="Persist PyTorch reference summary/spatial outputs for a golden image.",
    )
    capture.add_argument("--model-id", default="nvidia/C-RADIOv4-H")
    capture.add_argument("--revision")
    capture.add_argument("--image", required=True, type=Path)
    capture.add_argument("--out", required=True, type=Path)
    capture.add_argument("--image-size", nargs="+", type=int, default=[512])
    capture.add_argument("--dtype", default="bfloat16")
    capture.add_argument("--device", default="auto")

    audit = subparsers.add_parser("audit-weights", help="List safetensors keys and shapes.")
    audit.add_argument("--hf-path", required=True, type=Path)
    audit.add_argument("--out", required=True, type=Path)

    compare = subparsers.add_parser("compare", help="Compare two embedding .npz files.")
    compare.add_argument("--reference", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--out", required=True, type=Path)

    bench = subparsers.add_parser("benchmark", help="Write a benchmark report scaffold.")
    bench.add_argument("--model", required=True, type=Path)
    bench.add_argument("--image-size", type=int, default=512)
    bench.add_argument("--batch-size", nargs="+", type=int, default=[1])
    bench.add_argument("--report", required=True, type=Path)

    torch_bench = subparsers.add_parser(
        "pytorch-benchmark",
        help="Benchmark the PyTorch accelerated reference backend.",
    )
    torch_bench.add_argument("--model-id", default="nvidia/C-RADIOv4-SO400M")
    torch_bench.add_argument("--revision")
    torch_bench.add_argument("--image", required=True, type=Path)
    torch_bench.add_argument("--image-size", nargs="+", type=int, default=[256])
    torch_bench.add_argument("--dtype", default="float32")
    torch_bench.add_argument("--device", default="auto")
    torch_bench.add_argument("--warmups", type=int, default=1)
    torch_bench.add_argument("--repeats", type=int, default=3)
    torch_bench.add_argument("--report", required=True, type=Path)

    mlx_bench = subparsers.add_parser(
        "mlx-benchmark",
        help="Benchmark a native MLX C-RADIOv4 backend.",
    )
    mlx_bench.add_argument("--checkpoint", required=True, type=Path)
    mlx_bench.add_argument("--variant", choices=["so400m", "h"], default="so400m")
    mlx_bench.add_argument(
        "--revision",
        default=None,
    )
    mlx_bench.add_argument("--image", required=True, type=Path, nargs="+")
    mlx_bench.add_argument("--image-size", nargs="+", type=int, default=[512])
    mlx_bench.add_argument("--image-sizes", nargs="+", type=int)
    mlx_bench.add_argument("--batch-size", nargs="+", type=int, default=[1])
    mlx_bench.add_argument("--dtype", default="bfloat16")
    mlx_bench.add_argument("--warmups", type=int, default=1)
    mlx_bench.add_argument("--repeats", type=int, default=3)
    mlx_bench.add_argument("--no-materialize", action="store_true")
    mlx_bench.add_argument("--compile", action="store_true")
    mlx_bench.add_argument("--cider-fusion", choices=["off", "auto", "required"], default="off")
    mlx_bench.add_argument("--report", required=True, type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "device-info":
        _dump(inspect_acceleration().to_dict())
        return 0

    if args.command == "spatial-shape":
        image_size = _image_size(args.image_size)
        config = get_model_config(args.model_id)
        grid_h, grid_w = config.grid_shape(image_size)
        _dump(
            {
                "model_id": config.model_id,
                "image_size": config.validate_image_size(image_size),
                "patch_size": config.patch_size,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "spatial_tokens": grid_h * grid_w,
            }
        )
        return 0

    if args.command == "inspect-bundle":
        _dump(inspect_bundle(args.model))
        return 0

    if args.command == "convert":
        path = convert_checkpoint(
            ConversionRequest(
                hf_path=args.hf_path,
                mlx_path=args.mlx_path,
                model_id=args.model_id,
                revision=args.revision,
                dtype=args.dtype,
                manifest_only=args.manifest_only,
            )
        )
        _dump({"manifest": str(path)})
        return 0

    if args.command == "quantize":
        path = quantize_bundle(
            QuantizationRequest(
                model=args.model,
                out=args.out,
                bits=args.bits,
                group_size=args.group_size,
                mode=args.mode,
                clip_percentile=args.clip_percentile,
                smooth_scales=args.smooth_scales,
                dry_run=args.dry_run,
            )
        )
        _dump({"manifest": str(path)})
        return 0

    if args.command == "pytorch-ref":
        from cradio_mlx.compat.pytorch_ref import PyTorchReferenceRunner

        result = PyTorchReferenceRunner.from_model_id(
            args.model_id,
            revision=args.revision,
            dtype=args.dtype,
            device=args.device,
        ).encode(args.image, image_size=_image_size(args.image_size))
        saved = save_embedding_result_npz(result, args.save_npz) if args.save_npz else None
        _dump(
            {
                "summary_shape": tuple(result.summary.shape),
                "spatial_shape": tuple(result.spatial.shape),
                "grid_h": result.grid_h,
                "grid_w": result.grid_w,
                "metadata": result.metadata,
                "saved_npz": str(saved) if saved else None,
            }
        )
        return 0

    if args.command == "embed":
        if args.backend in {"mlx-so400m", "mlx-h"}:
            if args.checkpoint is None:
                raise SystemExit(f"--checkpoint is required for --backend {args.backend}")
            encoder_cls = MLXHEncoder if args.backend == "mlx-h" else MLXSO400MEncoder
            result = encoder_cls.load(
                args.checkpoint,
                dtype=args.dtype,
                revision=args.revision,
                compile_forward=args.compile,
                cider_fusion=args.cider_fusion,
            ).encode_image(args.image, image_size=_image_size(args.image_size))
        else:
            result = CRadioEncoder.from_pretrained(
                args.model_id,
                revision=args.revision,
                backend=args.backend,
                dtype=args.dtype,
                device=args.device,
            ).encode_image(args.image, image_size=_image_size(args.image_size))
        saved = save_embedding_result_npz(result, args.save_npz) if args.save_npz else None
        _dump(
            {
                "summary_shape": tuple(result.summary.shape),
                "spatial_shape": tuple(result.spatial.shape),
                "grid_h": result.grid_h,
                "grid_w": result.grid_w,
                "metadata": result.metadata,
                "saved_npz": str(saved) if saved else None,
            }
        )
        return 0

    if args.command == "capture-reference":
        paths = capture_reference_outputs(
            ReferenceCaptureRequest(
                model_id=args.model_id,
                revision=args.revision,
                image=args.image,
                out=args.out,
                image_size=_image_size(args.image_size),
                dtype=args.dtype,
                device=args.device,
            )
        )
        _dump({key: str(path) for key, path in paths.items()})
        return 0

    if args.command == "audit-weights":
        path = write_weight_audit(args.hf_path, args.out)
        _dump({"audit": str(path)})
        return 0

    if args.command == "compare":
        path = write_embedding_comparison(args.reference, args.candidate, args.out)
        _dump({"comparison": str(path)})
        return 0

    if args.command == "benchmark":
        path = write_benchmark_stub(
            BenchmarkRequest(
                model=args.model,
                image_size=args.image_size,
                batch_sizes=args.batch_size,
                report=args.report,
            )
        )
        _dump({"report": str(path)})
        return 0

    if args.command == "pytorch-benchmark":
        path = write_pytorch_benchmark(
            PyTorchBenchmarkRequest(
                model_id=args.model_id,
                revision=args.revision,
                image=args.image,
                image_size=_image_size(args.image_size),
                dtype=args.dtype,
                device=args.device,
                warmups=args.warmups,
                repeats=args.repeats,
                report=args.report,
            )
        )
        _dump({"report": str(path)})
        return 0

    if args.command == "mlx-benchmark":
        image_size = args.image_sizes if args.image_sizes else _image_size(args.image_size)
        image = args.image[0] if len(args.image) == 1 else args.image
        path = write_mlx_so400m_benchmark(
            MLXSO400MBenchmarkRequest(
                checkpoint=args.checkpoint,
                revision=args.revision,
                variant=args.variant,
                image=image,
                image_size=image_size,
                dtype=args.dtype,
                batch_size=args.batch_size,
                warmups=args.warmups,
                repeats=args.repeats,
                materialize_outputs=not args.no_materialize,
                compile_forward=args.compile,
                cider_fusion=args.cider_fusion,
                report=args.report,
            )
        )
        _dump({"report": str(path)})
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
