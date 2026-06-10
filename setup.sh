#!/bin/bash
# One-time setup for the buddy-nesso voice demo (Apple Silicon Mac).
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo
echo "✅ Setup done. Run the demo with:"
echo "   .venv/bin/python buddy_voice.py"
echo
echo "First run downloads the models from Hugging Face (~2 GB total)."
echo "macOS will ask for microphone permission for your terminal — allow it."
