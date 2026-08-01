"""
23_private_model.py — Run a model on hardware you control.

No API key, no vendor, no data leaving the machine. The only prerequisite is
a server of your own; every OpenAI-compatible one works, so pick whichever
suits your hardware:

    vllm serve Qwen/Qwen3-0.6B          # :8000  — vLLM, or vllm-metal on a Mac
    ollama run llama3.3                 # :11434 — set PRIVATE_BASE_URL below
    llama-server -m model.gguf          # :8080

Qwen3 is a reasoning model: it thinks out loud in a <think> block before
answering, and that block arrives as part of the reply. Add the server-side
parser to get a clean answer instead:

    vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3

The flag belongs to the server, not to this library — see the "No quirk
handling" limit in docs/getting-started/private-models.md. Without it the
example still runs; you just see the model's reasoning before its answer.

Required env vars:
    (none)

Optional env vars:
    PRIVATE_BASE_URL     server address, if not http://localhost:8000
    PRIVATE_TEST_MODEL   model name, if you did not start Qwen3-0.6B
"""

import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill

BASE_URL = os.getenv("PRIVATE_BASE_URL", "http://localhost:8000")
NAME     = os.getenv("PRIVATE_TEST_MODEL", "Qwen/Qwen3-0.6B")

# The name after "private/" is passed to the server verbatim — a Hugging Face
# id for vLLM, a tag for Ollama. There is no registry to keep in sync, because
# the server is what decides which models exist.
model = Model(
    f"private/{NAME}",
    # Give reasoning models room: the thinking is billed against max_tokens
    # like any other output, so a budget sized for the answer alone returns a
    # truncated monologue instead. Measured on this very example — 200 tokens
    # produced no answer at all, 1024 leaves plenty of headroom.
    options={"max_tokens": 1024, "temperature": 0.3},
    client_options={"url": BASE_URL},   # or set PRIVATE_BASE_URL and drop this
)

# This is the point of the exercise: the document below never leaves the
# machine. Nothing is sent to a vendor, because there is no vendor in the path.
CONTRACT = """
    Party A shall deliver 400 units by 14 March. Late delivery incurs a
    penalty of 2% of invoice value per week, capped at 10%. Party B may
    terminate on 30 days written notice after two consecutive late deliveries.
"""

skill = Skill(model=model, input={
    "messages": [{
        "role": "user",
        "parts": ["Summarise the obligations in this contract:\n{text}"],
    }],
})

print(skill.run(variables={"text": CONTRACT}))

# Token counts arrive as usual, but cost is None — and that is an answer, not
# a gap. A model you host has no price per token: its economics are GPU-hours
# divided by throughput, which depends on your hardware, not on the model. The
# counts are here so you can work out your own rate.
usage = skill.last_usage
print(f"\ntokens: {usage.input_tokens} in / {usage.output_tokens} out")
print(f"cost:   {usage.cost}   (None — you are paying in electricity)")

# Everything else in the library treats this like any other model. Swapping
# back to a cloud provider is the same one-word change it always is:
#
#     Model("claude-sonnet-4-6")     # data leaves, quality rises, cost appears
#     Model(f"private/{NAME}")       # data stays
