# Core ML Fast-Kill Results

Core ML was tested as a fixed-shape proof, not as a production backend. The script converts
the local PyTorch checkpoint to an ML Program with `torch.export`, benchmarks selected
compute units, and writes a JSON gate result.

Install:

```sh
python -m pip install -e '.[reference,coreml]'
```

SO400M command:

```sh
python scripts/coreml_fastkill.py \
  --checkpoint checkpoints/c-radiov4-so400m \
  --model-id nvidia/C-RADIOv4-SO400M \
  --variant so400m \
  --image data/golden_images/smoke.jpg \
  --image-size 512 \
  --batch-size 1 \
  --baseline-p50-ms 28.47 \
  --compute-units ALL CPU_AND_GPU \
  --warmups 3 \
  --repeats 8 \
  --skip-convert \
  --package reports/experiments/so400m-512-b1.mlpackage \
  --out reports/experiments/coreml-so400m-fastkill-rerun.json
```

H command:

```sh
python scripts/coreml_fastkill.py \
  --checkpoint checkpoints/c-radiov4-h \
  --model-id nvidia/C-RADIOv4-H \
  --variant h \
  --image data/golden_images/smoke.jpg \
  --image-size 512 \
  --batch-size 1 \
  --baseline-p50-ms 42.68 \
  --compute-units ALL CPU_AND_GPU \
  --warmups 3 \
  --repeats 8 \
  --skip-convert \
  --package reports/experiments/h-512-b1.mlpackage \
  --out reports/experiments/coreml-h-fastkill-rerun.json
```

## Results

| Model | Compute unit | p50 | p95 | Summary cosine | Spatial cosine | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SO400M | `ALL` | 25.12 ms | 25.25 ms | 0.999998 | 0.999995 | kill |
| SO400M | `CPU_AND_GPU` | 25.01 ms | 25.26 ms | 0.999998 | 0.999995 | kill |
| H | `ALL` | 31.42 ms | 31.83 ms | 0.999998 | 0.999991 | continue |
| H | `CPU_AND_GPU` | 31.73 ms | 31.90 ms | 0.999998 | 0.999991 | continue |

TorchScript conversion failed on the RADIO patch generator shape logic:

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

`torch.export(...).run_decompositions({})` converted both models.

Interpretation:

- SO400M Core ML is fast and precise, but it did not clear the planned `20%` speed gate
  versus the best Cider fused path (`28.47 ms` baseline, `22.78 ms` required).
- H Core ML cleared the speed and precision gates versus the best Cider fused path
  (`42.68 ms` baseline, `34.14 ms` required).
- `ALL` and `CPU_AND_GPU` are effectively identical. There is no evidence here of an
  ANE-specific speedup; this looks like a Core ML GPU scheduling win.
