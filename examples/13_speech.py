"""
13_speech.py — Voice both ways through one interface: text to an mp3, and back to text.

Keys: OPENAI_API_KEY. The mp3 goes to AICHAIN_EXAMPLES_DIR (the temp directory by default).
"""

import os
import tempfile
from pathlib import Path

from yait_aichain.tools import speech, transcribe

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"
OUT.mkdir(parents=True, exist_ok=True)

sentence = "The library speaks every provider's language, and this sentence travelled both ways."
audio = speech("openai", output_dir=str(OUT)).run(sentence, {"voice": "nova"})
heard = transcribe("openai").run(audio)

print(f"said:  {sentence}")
print(f"heard: {heard.strip()}")
print(f"audio: {audio}")
