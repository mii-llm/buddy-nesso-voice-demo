#!/usr/bin/env python3
"""Voice buddy demo for giux78/buddy-nesso-sft-v1.

Push-to-talk loop, fully local on Apple Silicon:
  mic -> mlx-whisper (STT, it/en) -> buddy-nesso via mlx-lm -> macOS `say` (TTS)

Usage:
  python buddy_voice.py                       # defaults, downloads models on first run
  python buddy_voice.py --model /path/to/mlx  # use your local MLX conversion
  python buddy_voice.py --lang it             # force Italian instead of auto-detect

At the prompt you can also just type a message instead of speaking.
"""

import argparse
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # whisper expects 16 kHz mono

# Prefer a local MLX conversion if present, otherwise pull the published
# MLX repo from Hugging Face (no conversion needed).
LOCAL_MLX = Path(__file__).parent / "models" / "buddy-nesso-mlx"
DEFAULT_MODEL = str(LOCAL_MLX) if LOCAL_MLX.exists() else "giux78/buddy-nesso-sft-v1-mlx"

SYSTEM_PROMPT = """You are Nesso, a gentle story and play buddy for children under 8.
Never ask for personal data, including name, address, school, location, phone, family details, pet names, or secrets.
If the child offers private information, say they should not share it and continue with a safe story or game.
Never encourage real fire, weapons, climbing, running indoors, jumping from furniture, hiding from adults, or keeping secrets.
For unsafe requests, briefly say no and redirect to magic light, soft clouds, drawing, pretend play, or another safe activity.
Keep replies short: 1-4 simple sentences.
Ask at most one question.
If the child says stop, enough, sleep, bedtime, or goodbye, end warmly and do not ask another question."""

# Preferred macOS voices per language, first available wins.
VOICE_PREFS = {
    "it": ["Alice", "Federica", "Emma", "Luca"],
    "en": ["Samantha", "Karen", "Moira", "Daniel"],
}

ITALIAN_HINTS = re.compile(
    r"\b(ciao|storia|favola|perché|però|grazie|sono|voglio|raccontami|gioco|notte|sonno|sì)\b",
    re.IGNORECASE,
)


def list_say_voices():
    """Return the set of installed `say` voice names."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return set()
    voices = set()
    for line in out.splitlines():
        m = re.match(r"^(.*?)\s{2,}[a-z]{2,3}[_-]", line)
        if m:
            voices.add(m.group(1).strip())
    return voices


class Speaker:
    """Speaks sentences sequentially in a background thread via macOS `say`."""

    def __init__(self, rate=175, enabled=True):
        self.rate = rate
        self.enabled = enabled
        self.available = list_say_voices()
        self.queue = queue.Queue()
        self.current = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def voice_for(self, lang):
        for name in VOICE_PREFS.get(lang, VOICE_PREFS["en"]):
            if not self.available or name in self.available:
                return name
        return None

    def say(self, text, lang):
        text = clean_for_tts(text)
        if text and self.enabled:
            self.queue.put((text, lang))

    def _worker(self):
        while True:
            text, lang = self.queue.get()
            cmd = ["say", "-r", str(self.rate)]
            voice = self.voice_for(lang)
            if voice:
                cmd += ["-v", voice]
            try:
                self.current = subprocess.Popen(cmd + [text])
                self.current.wait()
            except FileNotFoundError:
                pass
            finally:
                self.current = None
                self.queue.task_done()

    def stop(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
        if self.current:
            self.current.terminate()

    def wait(self):
        self.queue.join()


def clean_for_tts(text):
    """Strip markdown/emoji/special tokens so `say` reads plain words."""
    text = re.sub(r"<\|[^|]*\|>", " ", text)
    text = re.sub(r"[*_#`~>]", " ", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def record_audio():
    """Record from the default mic until the user presses ENTER."""
    frames = []

    def callback(indata, _frames, _time, status):
        if status:
            print(f"  (mic: {status})", file=sys.stderr)
        frames.append(indata.copy())

    print("🔴 Recording... press ENTER when you finish speaking.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        callback=callback):
        input()
    if not frames:
        return None
    audio = np.concatenate(frames)[:, 0]
    if len(audio) < SAMPLE_RATE // 4 or np.abs(audio).max() < 0.01:
        return None  # too short or silence
    return audio


def guess_lang(text, fallback="en"):
    return "it" if ITALIAN_HINTS.search(text) else fallback


def build_prompt(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        # Manual ChatML fallback (matches the model's chat_template.jinja).
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        return "".join(parts) + "<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser(description="Voice demo for buddy-nesso")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HF repo or local path (MLX or transformers format)")
    parser.add_argument("--whisper", default="mlx-community/whisper-large-v3-turbo",
                        help="Whisper model for STT (try mlx-community/whisper-small-mlx "
                             "for a smaller download)")
    parser.add_argument("--lang", choices=["auto", "it", "en"], default="auto",
                        help="Conversation language (auto = detect from speech)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rate", type=int, default=175, help="Speech rate (words/min)")
    parser.add_argument("--no-tts", action="store_true", help="Disable spoken output")
    args = parser.parse_args()

    print(f"Loading buddy model ({args.model})...")
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(args.model)

    print(f"Loading whisper ({args.whisper})... first run downloads it.")
    import mlx_whisper
    # Warm up / trigger download with a short silent clip.
    mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                           path_or_hf_repo=args.whisper)

    speaker = Speaker(rate=args.rate, enabled=not args.no_tts)
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    lang = "en" if args.lang == "auto" else args.lang

    print("\n✨ Nesso is ready! Press ENTER to talk, type a message, or 'q' to quit.\n")

    while True:
        try:
            typed = input("👦 ENTER = talk | type = chat | q = quit > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye! 🌙")
            break
        if typed.lower() in {"q", "quit", "exit"}:
            print("Bye! 🌙")
            break

        speaker.stop()  # stop any leftover speech before listening

        if typed:
            user_text = typed
            if args.lang == "auto":
                lang = guess_lang(user_text, fallback=lang)
        else:
            audio = record_audio()
            if audio is None:
                print("  (I didn't hear anything, let's try again!)\n")
                continue
            print("👂 Understanding...")
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=args.whisper,
                language=None if args.lang == "auto" else args.lang,
            )
            user_text = result["text"].strip()
            if not user_text:
                print("  (I didn't catch that, let's try again!)\n")
                continue
            if args.lang == "auto" and result.get("language") in VOICE_PREFS:
                lang = result["language"]
            print(f'👦 You said: "{user_text}"')

        messages.append({"role": "user", "content": user_text})
        # Keep the system prompt plus the last few exchanges (4096-token context).
        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]

        prompt = build_prompt(tokenizer, messages)
        print("🧸 Nesso: ", end="", flush=True)
        reply, spoken_upto = "", 0
        for response in stream_generate(model, tokenizer, prompt=prompt,
                                        max_tokens=args.max_tokens, sampler=sampler):
            print(response.text, end="", flush=True)
            reply += response.text
            # Speak completed sentences while the rest is still generating.
            for m in re.finditer(r"[.!?…]+[\s\"')\]]*", reply[spoken_upto:]):
                sentence = reply[spoken_upto:spoken_upto + m.end()]
                speaker.say(sentence, lang)
                spoken_upto += m.end()
        print("\n")
        if reply[spoken_upto:].strip():
            speaker.say(reply[spoken_upto:], lang)
        messages.append({"role": "assistant", "content": reply.strip()})

        try:
            speaker.wait()  # let Nesso finish talking before listening again
        except KeyboardInterrupt:
            speaker.stop()


if __name__ == "__main__":
    main()
