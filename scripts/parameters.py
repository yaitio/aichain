"""
The parameter matrix: what every universal option actually does, per provider.

Built by constructing real requests, not by reading code or docs. For each
(parameter, model) cell the request is built twice — once without the option,
once with it — and the two bodies are diffed. The difference is the fact; the
docs and the conformance test are both written from it, so neither can drift
from the code on its own.

Outcomes, and which of them are acceptable:

    passed     the value went out under its own name               ok
    renamed    same value, provider's spelling (top_k → topK)       ok
    converted  same intent, different shape (reasoning → thinking)  ok
    swapped    the MODEL changed (deepseek-chat → -reasoner)        ok only with a warning
    dropped    the request is byte-identical: the option vanished   ok only with a warning
    refused    the library raised before the wire, naming why       ok — the fourth fate
    error      building failed for another reason                   shown as is

A cell that is dropped or swapped in silence is a defect. The library may
adapt a request to a provider; it may not do so without saying.

    python scripts/parameters.py            # rewrite the snapshot and the docs
    python scripts/parameters.py --check    # exit 1 if behaviour moved
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yait_aichain import Model                                   # noqa: E402

SNAPSHOT = ROOT / "tests" / "fixtures" / "parameters_matrix.json"
DOC      = ROOT / "docs" / "reference" / "parameters.md"

# ── What is probed ───────────────────────────────────────────────────────────

#: Model options: the universal knobs, with the value each probe sends.
#: Every value is chosen to match no provider's default — a probe equal to the
#: default diffs to nothing and reads as "dropped" (Google's default top_k is
#: 40, and the first version of this script probed with 40).
TEXT_OPTIONS = {
    "temperature":   0.31,
    "top_p":         0.87,
    "top_k":         37,
    "max_tokens":    321,
    "reasoning":     "high",
    "cache_control": True,
}

#: Image output-format keys. The vocabulary is not neutral yet — several of
#: these are one provider's own word — which is part of what the matrix is
#: for: it shows which word each provider actually reads.
IMAGE_FORMAT = {
    "size":               "1024x1024",
    "aspect_ratio":       "16:9",
    "quality":            "high",
    "background":         "transparent",
    "output_format":      "png",
    "output_compression": 50,
    "seed":               42,
    "input_fidelity":     "high",
}

#: One representative per provider, and a second where the same provider
#: routes models differently (OpenAI chat vs Responses; DeepSeek chat vs
#: reasoner). Behaviour is recorded per model, not per provider, because it
#: differs within a provider.
TEXT_MODELS = [
    "gpt-4o", "gpt-5.5",
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
    "deepseek-chat", "deepseek-reasoner",
    "kimi-k2-turbo-preview",
    "grok-3",
    "qwen-max",
    "sonar",
]
IMAGE_MODELS = [
    "gpt-image-2.5-flare",
    "gemini-2.5-flash-image",
    "flux-2-pro",
    "reve-image",
    "recraftv3",
    "grok-imagine-image",
    "wan2.2-t2i-flash",
]

# A system prompt is included because that is where a cache mark goes: the
# placement rule keeps it off the variable message, so a conversation with no
# stable prefix has nothing to mark and cache_control correctly does nothing.
TEXT_MSGS  = [{"role": "system", "parts": [{"type": "text", "text": "be brief"}]},
              {"role": "user",   "parts": [{"type": "text", "text": "hi"}]}]
TEXT_OUT   = {"format": {"type": "text"}}
IMAGE_OUT  = {"format": {"type": "image"}}


# ── Probing ──────────────────────────────────────────────────────────────────

def _leaves(obj, path=()) -> dict:
    """Flatten a request body into {path: value}; lists are indexed."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_leaves(v, path + (str(k),)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(_leaves(v, path + (f"[{i}]",)))
        return out
    return {path: obj}


def _reset_once_per_process_warnings():
    """The library warns once per process about a dropped or stripped
    parameter. The matrix asks "does this call say so?", which must not depend
    on whether some earlier call in the same process already did."""
    from yait_aichain.clients._families import _openai_compat, google
    _openai_compat._WARNED_REJECTS.clear()
    google._WARNED_STRIP = False


def _build(model_name, options, out):
    _reset_once_per_process_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = Model(model_name, api_key="k", options=options or None)
        path, body = m.to_request(TEXT_MSGS, out)
    # The model actually named on the wire: in the body for most providers,
    # in the URL for Google.
    sent = body.get("model") if isinstance(body, dict) else None
    if not sent and isinstance(path, str) and "/models/" in path:
        sent = path.split("/models/")[1].split(":")[0]
    return body, sent, [str(w.message) for w in caught]


def probe(model_name, param, value, *, where) -> dict:
    """One cell: what happened to *param* on *model_name*."""
    if where == "options":
        base_args = ({}, TEXT_OUT)
        probe_args = ({param: value}, TEXT_OUT)
    else:
        base_args = ({}, IMAGE_OUT)
        probe_args = ({}, {"format": {**IMAGE_OUT["format"], param: value}})

    try:
        base_body, base_model, _ = _build(model_name, *base_args)
    except Exception as exc:
        return {"outcome": "error", "detail": f"baseline: {type(exc).__name__}: {exc}"[:160],
                "warned": False}
    try:
        body, sent_model, warned = _build(model_name, *probe_args)
    except ValueError as exc:
        # Raised by the library before anything was sent: the fourth fate.
        return {"outcome": "refused", "detail": str(exc)[:160], "warned": False}
    except Exception as exc:
        return {"outcome": "error", "detail": f"{type(exc).__name__}: {exc}"[:160],
                "warned": False}

    before, after = _leaves(base_body), _leaves(body)
    added = {p: v for p, v in after.items() if before.get(p) != v}
    swapped = sent_model != base_model
    warned = any(param in w for w in warned)

    if swapped:
        return {"outcome": "swapped", "detail": f"{base_model} → {sent_model}",
                "warned": warned}
    if not added:
        return {"outcome": "dropped", "detail": "", "warned": warned}

    names = sorted({p[-1] for p in added})
    same_value = [p for p, v in added.items() if v == value]
    if any(p[-1] == param for p in same_value):
        return {"outcome": "passed", "detail": "", "warned": warned}
    if same_value:
        return {"outcome": "renamed", "detail": ", ".join(names), "warned": warned}
    return {"outcome": "converted", "detail": ", ".join(names), "warned": warned}


def verdict(cell: dict) -> str:
    if cell["outcome"] in ("dropped", "swapped") and not cell["warned"]:
        return "defect: silent"
    return "ok"


def build_matrix() -> dict:
    rows = {}
    for model in TEXT_MODELS:
        rows[model] = {p: probe(model, p, v, where="options")
                       for p, v in TEXT_OPTIONS.items()}
    for model in IMAGE_MODELS:
        rows[model] = {p: probe(model, p, v, where="format")
                       for p, v in IMAGE_FORMAT.items()}
    return rows


# ── Outputs ──────────────────────────────────────────────────────────────────

_MARK = {"passed": "✓", "renamed": "→", "converted": "≈",
         "swapped": "⇄", "dropped": "—", "refused": "⊘", "error": "✗"}


def render_doc(matrix: dict) -> str:
    lines = [
        "# Parameters, per provider",
        "",
        "<!-- GENERATED by scripts/parameters.py — do not edit; rerun it. -->",
        "",
        "What each universal option actually does on each model, established by",
        "building the request and diffing it against one built without the",
        "option. The same run feeds the conformance test, so this page cannot",
        "say one thing while the code does another.",
        "",
        "| mark | meaning |",
        "|---|---|",
        "| ✓ passed | sent under its own name |",
        "| → renamed | same value, the provider's spelling |",
        "| ≈ converted | same intent, a different shape |",
        "| ⇄ swapped | **the model changed** |",
        "| — dropped | the option had no effect on the request |",
        "| ⊘ refused | the library declined before the wire, saying why |",
        "| ✗ error | building the request failed for another reason |",
        "",
        "A cell marked **silent** is a defect: the library adapted the request",
        "without saying so. It may adapt; it may not do it quietly.",
        "",
    ]

    def table(title, models, params):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| model | " + " | ".join(f"`{p}`" for p in params) + " |")
        lines.append("|---|" + "---|" * len(params))
        for m in models:
            cells = []
            for p in params:
                c = matrix[m][p]
                mark = _MARK[c["outcome"]]
                text = mark
                if c["detail"] and c["outcome"] in ("renamed", "converted", "swapped"):
                    text += f" `{c['detail']}`"
                if verdict(c) != "ok":
                    text += " **silent**"
                if c["outcome"] in ("error", "refused"):
                    text += f" <sub>{c['detail'][:60]}</sub>"
                cells.append(text)
            lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
        lines.append("")

    table("Model options (text)", TEXT_MODELS, list(TEXT_OPTIONS))
    table("Output format (image)", IMAGE_MODELS, list(IMAGE_FORMAT))

    silent = [(m, p) for m in matrix for p, c in matrix[m].items()
              if verdict(c) != "ok"]
    lines.append("## Defects")
    lines.append("")
    lines.append(f"{len(silent)} silent cells. Each is an option a caller set "
                 "and did not get, with nothing said.")
    lines.append("")
    for m, p in silent:
        c = matrix[m][p]
        lines.append(f"- `{m}` · `{p}` — {c['outcome']}"
                     + (f" ({c['detail']})" if c["detail"] else ""))
    lines.append("")
    return "\n".join(lines)


def main(argv) -> int:
    matrix = build_matrix()
    if "--check" in argv:
        if not SNAPSHOT.exists():
            print("no snapshot; run without --check first"); return 1
        old = json.loads(SNAPSHOT.read_text())
        diffs = [(m, p, old.get(m, {}).get(p), c)
                 for m in matrix for p, c in matrix[m].items()
                 if old.get(m, {}).get(p) != c]
        for m, p, was, now in diffs:
            print(f"  {m} · {p}: {was} → {now}")
        print(f"{len(diffs)} cell(s) moved" if diffs else "matrix unchanged")
        return 1 if diffs else 0

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(matrix, indent=1, ensure_ascii=False) + "\n")
    DOC.write_text(render_doc(matrix))
    silent = sum(1 for m in matrix for c in matrix[m].values() if verdict(c) != "ok")
    total = sum(len(v) for v in matrix.values())
    print(f"{total} cells, {silent} silent; wrote {SNAPSHOT.name} and {DOC.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
