#!/bin/bash
# One-time setup for the buddy-nesso voice demo (Apple Silicon Mac).
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Convert the HF checkpoint to MLX format (skipped if already done)
if [ ! -d models/buddy-nesso-mlx ]; then
  echo "Converting buddy-nesso to MLX format..."
  .venv/bin/python -m mlx_lm convert \
    --hf-path giux78/buddy-nesso-sft-v1 \
    --mlx-path models/buddy-nesso-mlx
fi

echo
echo "✅ Setup done. Run the demo with:"
echo "   .venv/bin/python buddy_voice.py"
echo
echo "First run downloads the models from Hugging Face (~2 GB total)."
echo "macOS will ask for microphone permission for your terminal — allow it."
