"""
39_video.py — A video takes minutes: submit returns a job at once, collect waits for it.

Keys: any of XAI_API_KEY, GOOGLE_AI_API_KEY, DASHSCOPE_API_KEY, MINIMAX_API_KEY. Videos
go to AICHAIN_EXAMPLES_DIR (the temp directory by default). The cheapest clip each
provider makes: about $0.02, $0.20, $0.10 or $0.25.
"""

import json
import os
import tempfile
from pathlib import Path

from yait_aichain.tools import video

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"

PROVIDERS = [("XAI_API_KEY", "xai", "grok-imagine-video-1.5-lite", {"duration": 1, "resolution": "480p"}),
             ("GOOGLE_AI_API_KEY", "google", "veo-3.1-lite-generate-preview", {"duration": 4}),
             ("DASHSCOPE_API_KEY", "qwen", "wan2.2-t2v-plus", {"resolution": "480p"}),
             ("MINIMAX_API_KEY", "minimax", "MiniMax-H3-Max", {"resolution": "480p", "duration": 5})]
ready = [p for p in PROVIDERS if os.getenv(p[0])]
if not ready:
    raise SystemExit("set one of: " + ", ".join(p[0] for p in PROVIDERS))
_, service, model, options = ready[0]

clip = video(service, model=model, output_dir=str(OUT))
job = clip.submit("A paper boat drifting across a puddle, soft evening light", options=options)
print("submitted:", json.dumps({k: job[k] for k in ("service", "model", "id")}))

# A job is plain data: store it, queue it, collect it in another process.
job = json.loads(json.dumps(job))
result = clip.collect(job)
made = result["videos"][0]
print(f"saved: {made['path']} ({made['duration']} s, {made['resolution']})")
print(f"cost: ${result['cost']} — {result['cost_basis']}")
print("what an agent reads:", clip.render(result))
