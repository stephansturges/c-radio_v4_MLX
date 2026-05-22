# API

The stable user-facing API is embedding-first:

```python
from cradio_mlx import CRadioEncoder

model = CRadioEncoder.from_pretrained(
    "nvidia/C-RADIOv4-SO400M",
    revision="c0457f5dc26ca145f954cd4fc5bb6114e5705ad8",
    dtype="bfloat16",
)
result = model.encode_image("example.jpg", image_size=512)
grid = result.spatial_as_grid()
```

The standalone MLX `CRadioMLX.encode_image` method is not implemented yet. Use
`CRadioEncoder` for the accelerated PyTorch/MPS path while the MLX architecture is being built.

Native MLX backends are available for SO400M and H:

```python
from cradio_mlx import MLXHEncoder, MLXSO400MEncoder

so400m = MLXSO400MEncoder.load("checkpoints/c-radiov4-so400m", dtype="bfloat16")
so400m_result = so400m.encode_image("example.jpg", image_size=512)

h = MLXHEncoder.load("checkpoints/c-radiov4-h", dtype="bfloat16")
h_result = h.encode_image("example.jpg", image_size=512)
```
