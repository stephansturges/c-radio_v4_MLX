#!/usr/bin/env bash
set -euo pipefail

bundle_path="${1:?usage: scripts/release_bundle.sh <bundle_path>}"

python -m cradio_mlx.cli inspect-bundle --model "$bundle_path"

echo "Release upload is intentionally not automated yet."
echo "Next step: add signed tag, PyPI publish, and Hugging Face upload once weights pass parity."
