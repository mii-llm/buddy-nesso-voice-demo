# 🧸 Buddy Nesso — Voice Demo

Talk to [giux78/buddy-nesso-sft-v1](https://huggingface.co/giux78/buddy-nesso-sft-v1) with your voice, fully **local** on an Apple Silicon Mac. Italian 🇮🇹 and English 🇬🇧.

```
🎤 mic ──▶ mlx-whisper (speech-to-text) ──▶ buddy-nesso via mlx-lm ──▶ macOS `say` (text-to-speech) ──▶ 🔊
```

No cloud, no API keys — everything runs on the Mac's GPU/Neural-friendly MLX stack plus the built-in macOS voices.

**Requirements:** a Mac with Apple Silicon (M1 or newer), Python 3.10+, and ~3 GB of free disk for the models.

## Setup (one time)

```bash
git clone https://github.com/mii-llm/buddy-nesso-voice-demo.git
cd buddy-nesso-voice-demo
./setup.sh
```

## Run

```bash
.venv/bin/python buddy_voice.py
```

- **Press ENTER** → recording starts; speak, then **press ENTER again** to stop.
- Or **type a message** instead of speaking (handy for testing).
- `q` to quit.

The first run downloads the buddy model in MLX format ([giux78/buddy-nesso-sft-v1-mlx](https://huggingface.co/giux78/buddy-nesso-sft-v1-mlx), ~0.8 GB) and Whisper large-v3-turbo (~1.6 GB) from Hugging Face. macOS will ask for **microphone permission** for your terminal app — allow it.

Language is auto-detected from the child's speech: Italian replies are spoken with the **Alice** voice, English with **Samantha**.

## Useful options

```bash
# Use a different model checkpoint (HF repo id or local MLX path)
.venv/bin/python buddy_voice.py --model /path/to/another-model

# Force Italian (skips language auto-detection — more reliable with small kids)
.venv/bin/python buddy_voice.py --lang it

# Smaller/faster Whisper if the download is too big (slightly worse with kids' speech)
.venv/bin/python buddy_voice.py --whisper mlx-community/whisper-small-mlx

# Tweak generation / speech
.venv/bin/python buddy_voice.py --max-tokens 150 --temp 0.6 --rate 160

# Text-only (no spoken output)
.venv/bin/python buddy_voice.py --no-tts
```

## How it works

1. `sounddevice` records 16 kHz mono audio from the default mic (push-to-talk).
2. `mlx-whisper` transcribes it and detects the language (it/en).
3. The transcript is appended to the chat history (ChatML format, with the
   kid-safety system prompt recommended on the model card) and `mlx-lm`
   streams the reply from the 400M buddy-nesso model.
4. Completed sentences are spoken **while the model is still generating**,
   so the buddy starts answering almost immediately.

## Using the GGUF instead (optional)

If you prefer your GGUF file with llama.cpp, run a local server and the model can
be swapped behind an OpenAI-compatible API:

```bash
brew install llama.cpp
llama-server -m buddy-nesso-sft-v1.gguf --port 8080 --jinja
```

The current script uses MLX directly (simpler, no server), but porting it means
replacing the `mlx_lm` calls with `openai` chat-completion calls to `http://localhost:8080/v1`.

## Notes for the POC

- ⚠️ Per the model card this is a **research checkpoint**, not a production
  child-safety model — demo with adult supervision.
- Better voices: System Settings → Accessibility → Spoken Content →
  Manage Voices… and download the **Enhanced/Premium** versions of Alice (Italian)
  and Samantha (English); `say` picks them up automatically.
- If a young child's speech is misrecognized, force the language with `--lang it`
  and keep the mic close.
