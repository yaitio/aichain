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

#: One representative per provider, and a second wherever the same provider
#: routes models differently (OpenAI chat vs Responses, DeepSeek chat vs
#: reasoner, grok-3 vs grok-3-mini). Behaviour is recorded per model, not per
#: provider, because it differs within one.
#:
#: The set has to be wide enough to exercise what the data claims: three
#: providers first showed as "claims an option no model delivers", and in all
#: three the declaration was right and this list was short — grok-3-mini,
#: QwQ-32B and gpt-image-1.5 were the models that take them.
TEXT_MODELS = [
    "gpt-4o", "gpt-5.5",
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
    "deepseek-chat", "deepseek-reasoner",
    "kimi-k2-turbo-preview",
    "grok-3", "grok-3-mini",
    "qwen-max", "QwQ-32B", "qwen3-32b",
    "sonar",
]
IMAGE_MODELS = [
    "gpt-image-2.5-flare", "gpt-image-1.5",
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

#: Models probed a second time with an input image, because several format
#: keys exist only on the edits path — `input_fidelity` is read there and
#: nowhere else, and probing generation alone reported it as claimed and
#: never delivered.
EDIT_MODELS = ["gpt-image-2.5-flare", "gpt-image-1.5", "grok-imagine-image",
               "recraftv3"]
_PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
        "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
EDIT_MSGS = [{"role": "user", "parts": [
    {"type": "text", "text": "make it darker"},
    {"type": "image", "source": {"kind": "base64", "mime": "image/png",
                                 "data": _PNG}}]}]
#: The format type alone is how a picture is asked for. `modalities` is sent
#: as well because a caller may write either, and both have to keep working —
#: Google once read only the second, which is why it first measured as
#: reading none of the image keys: the probe was not touching the branch.
IMAGE_OUT  = {"format": {"type": "image"}, "modalities": ["image"]}


# ── Probing ──────────────────────────────────────────────────────────────────

def _leaves(obj, path=()) -> dict:
    """Flatten a request body into {path: value}; lists are indexed."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_leaves(v, path + (str(k),)))
        return out
    if isinstance(obj, (list, tuple)):
        out = {}
        for i, v in enumerate(obj):
            # A multipart body is a list of (name, value) pairs. Keyed by
            # position, adding one field shifts every later one and the diff
            # reports unrelated parameters as changed — every edit row read
            # as "converted" when the option had simply been appended.
            if (isinstance(v, (list, tuple)) and len(v) == 2
                    and isinstance(v[0], str)):
                out.update(_leaves(v[1], path + (v[0],)))
            else:
                out.update(_leaves(v, path + (f"[{i}]",)))
        return out
    return {path: obj}


def _reset_once_per_process_warnings():
    """The library warns once per process about a dropped or stripped
    parameter. The matrix asks "does this call say so?", which must not depend
    on whether some earlier call in the same process already did."""
    from yait_aichain.clients._families import _openai_compat, google
    from yait_aichain.models import _adaptation
    _openai_compat._WARNED_REJECTS.clear()
    google._WARNED_STRIP = False
    _adaptation.reset_warnings()


def _build(model_name, options, out, messages=None):
    _reset_once_per_process_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = Model(model_name, api_key="k", options=options or None)
        path, body = m.to_request(messages or TEXT_MSGS, out)
    # The model actually named on the wire: in the body for most providers,
    # in the URL for Google.
    sent = body.get("model") if isinstance(body, dict) else None
    if not sent and isinstance(path, str) and "/models/" in path:
        sent = path.split("/models/")[1].split(":")[0]
    return body, sent, list(m.last_adaptations)


def probe(model_name, param, value, *, where) -> dict:
    """One cell: what happened to *param* on *model_name*."""
    msgs = EDIT_MSGS if where == "edit" else None
    if where == "options":
        base_args = ({}, TEXT_OUT)
        probe_args = ({param: value}, TEXT_OUT)
    else:
        base_args = ({}, IMAGE_OUT)
        probe_args = ({}, {**IMAGE_OUT,
                           "format": {**IMAGE_OUT["format"], param: value}})

    try:
        base_body, base_model, _ = _build(model_name, *base_args, messages=msgs)
    except Exception as exc:
        return {"outcome": "error", "detail": f"baseline: {type(exc).__name__}: {exc}"[:160],
                "warned": False}
    try:
        body, sent_model, warned = _build(model_name, *probe_args,
                                          messages=msgs)
    except ValueError as exc:
        # Raised by the library before anything was sent: the fourth fate.
        return {"outcome": "refused", "detail": str(exc)[:160], "warned": False}
    except Exception as exc:
        return {"outcome": "error", "detail": f"{type(exc).__name__}: {exc}"[:160],
                "warned": False}

    before, after = _leaves(base_body), _leaves(body)
    added = {p: v for p, v in after.items() if before.get(p) != v}
    swapped = sent_model != base_model
    # What the library said about THIS option, not merely that it said
    # something: a notice of the wrong kind is its own defect — three
    # providers that honour `reasoning` were briefly being told they had no
    # such control.
    # Kind plus whether the notice names a replacement. The generic "this
    # provider has no such control" carries no replacement, and saying that
    # about an option the provider did honour is its own defect.
    said = sorted({f"{a.kind}+named" if a.sent is not None else a.kind
                   for a in warned if a.option == param})

    if swapped:
        return {"outcome": "swapped", "detail": f"{base_model} → {sent_model}",
                "said": said}
    if not added:
        return {"outcome": "dropped", "detail": "", "said": said}

    names = sorted({p[-1] for p in added})
    # A multipart field is always a string on the wire, so 50 arrives as
    # "50". Comparing strictly reported a delivered option as converted and
    # then demanded a notice naming a replacement that never happened.
    same_value = [p for p, v in added.items()
                  if v == value or str(v) == str(value)]
    if any(p[-1] == param for p in same_value):
        return {"outcome": "passed", "detail": "", "said": said}
    if same_value:
        return {"outcome": "renamed", "detail": ", ".join(names), "said": said}
    return {"outcome": "converted", "detail": ", ".join(names), "said": said}


#: What each outcome must be reported as. A rename needs no notice — the value
#: the caller set is the value that went out.
_EXPECTED = {
    "passed":    (),
    "renamed":   (),
    # Something went out in the option's place, so the notice has to say what.
    # A bare "no such control" here is a lie: it was honoured, differently.
    "converted": ("adapted+named", "declined+named", "swapped+named"),
    "swapped":   ("swapped+named",),
    # Nothing went out. Either notice is honest; neither needs a replacement.
    "dropped":   ("declined", "declined+named", "adapted", "adapted+named"),
    "refused":   (),
    # An error is not a pass. Every edit cell was raising NameError while the
    # summary reported zero defects, because a cell that could not be built
    # counted as nothing rather than as unknown.
    "error":     ("__never__",),
}


def unfulfilled(matrix: dict) -> list:
    """Cells a provider claims and does not deliver.

    The declaration in the provider data says what a provider has a control
    for; the probe says what actually reached the request. Where the two
    disagree the library is promising something it does not do — the gap the
    matrix alone could never see, because measurement can only report what is,
    never what was meant.
    """
    import yait_aichain.models._base as base
    from yait_aichain.models._options import accepted_by

    # Compared per provider, not per model. A model narrowing an option its
    # provider does have — gpt-4o does not reason, a reasoner refuses
    # temperature — is expected and is reported at the time. What is worth
    # flagging is an option a provider claims and *none* of its models
    # delivers: then either the declaration is wrong or nothing was ever
    # wired to it.
    delivered, claimed = {}, {}
    for model, cells in matrix.items():
        prov = base._resolve_provider(model)
        accepts = accepted_by(prov)
        if accepts is None:
            continue
        claimed[prov] = accepts
        for option, cell in cells.items():
            if cell["outcome"] != "dropped":
                delivered.setdefault(prov, set()).add(option)

    out = []
    for prov, accepts in sorted(claimed.items()):
        probed = {o for cells in matrix.values() for o in cells}
        for option in sorted(accepts & probed):
            if option not in delivered.get(prov, set()):
                out.append((prov, option))
    return out


def verdict(cell: dict) -> str:
    need = _EXPECTED[cell["outcome"]]
    said = cell.get("said") or []
    if not need:
        return "ok"
    if not said:
        return "defect: silent"
    if not set(said) & set(need):
        return f"defect: called it {'/'.join(said)}, it was {cell['outcome']}"
    return "ok"


def build_matrix() -> dict:
    rows = {}
    for model in TEXT_MODELS:
        rows[model] = {p: probe(model, p, v, where="options")
                       for p, v in TEXT_OPTIONS.items()}
    for model in IMAGE_MODELS:
        rows[model] = {p: probe(model, p, v, where="format")
                       for p, v in IMAGE_FORMAT.items()}
    for model in EDIT_MODELS:
        rows[f"{model} (edit)"] = {p: probe(model, p, v, where="edit")
                                   for p, v in IMAGE_FORMAT.items()}
    return rows


# ── Outputs ──────────────────────────────────────────────────────────────────

_MARK = {"passed": "✓", "renamed": "→", "converted": "≈",
         "swapped": "⇄", "dropped": "—", "refused": "⊘", "error": "✗"}


def _vocabulary_section(matrix: dict) -> list:
    """What a caller may ask for, and what values are allowed.

    Generated rather than written: the vocabulary, the per-provider
    declarations and the value scales all come out of the same data the
    conformance test reads, so a page that disagrees with the code cannot be
    produced.
    """
    import yait_aichain.models._base as base
    from yait_aichain.models._options import (UNIVERSAL_OPTIONS,
                                              UNIVERSAL_FORMAT, accepted_by,
                                              allowed_values, value_map)

    lines = ["## 1. What you can ask for", "",
             "The universal vocabulary. **Model options** are set once on the "
             "`Model` and are a closed set — an unknown name raises at "
             "construction. **Format keys** are set per call in "
             "`output={\"format\": …}` and are open: a provider may read "
             "names of its own, so an unknown one is reported and the request "
             "still goes.", ""]

    models_by_provider: dict = {}
    for name in matrix:
        clean = name.replace(" (edit)", "")
        models_by_provider.setdefault(base._resolve_provider(clean), []).append(clean)

    def _draws(provider: str, model: str) -> bool:
        """Whether this model renders images at all."""
        from yait_aichain.models._data import PROVIDERS
        caps = ((PROVIDERS.get(provider) or {}).get("models", {})
                .get(model, {}).get("caps") or ())
        return any("image" in c.split("-to-")[-1] for c in caps)

    for title, vocab in (("Model options", UNIVERSAL_OPTIONS),
                         ("Format keys", UNIVERSAL_FORMAT)):
        lines += [f"### {title}", "",
                  "| option | what it does | values |", "|---|---|---|"]
        for option, meta in vocab.items():
            # Allowed sets differ by model where a provider says so; show the
            # union and name the exception rather than pretending one set.
            seen = {}
            for prov, models in models_by_provider.items():
                # Only models that have the control at all: a provider-wide
                # value list otherwise reads as if gpt-4o took `quality`.
                declared = accepted_by(prov)
                if declared is not None and option not in declared:
                    continue
                for m in models:
                    # A format key belongs to a model that draws; the
                    # declaration is per provider, and openai's text models
                    # were being listed as taking `quality`.
                    if vocab is UNIVERSAL_FORMAT and not _draws(prov, m):
                        continue
                    v = allowed_values(option, prov, m)
                    if v:
                        seen.setdefault(tuple(v), []).append(m)

            def _show(vals):
                # Two numbers are the range convention, not a pair of choices.
                if len(vals) == 2 and all(isinstance(v, (int, float))
                                          and not isinstance(v, bool)
                                          for v in vals):
                    return f"`{vals[0]}`–`{vals[1]}`"
                return ", ".join(f"`{v}`" for v in vals)

            if not seen:
                values = "any"
            elif len(seen) == 1:
                vals, = seen
                values = _show(vals)
            else:
                values = "; ".join(
                    _show(vals) + f" ({', '.join(sorted(set(ms)))})"
                    for vals, ms in seen.items())
            lines.append(f"| `{option}` | {meta['what']} | {values} |")
        lines.append("")

    lines += ["## 2. What each provider declares it takes", "",
              "From the provider data, not from the code — `accepts` and "
              "`format_accepts`. A provider that declares nothing is not a "
              "provider that takes nothing: absence of a claim is not a claim.",
              "", "| provider | takes |", "|---|---|"]
    for prov in sorted(models_by_provider):
        declared = accepted_by(prov)
        lines.append(f"| `{prov}` | " +
                     (", ".join(f"`{o}`" for o in sorted(declared)) if declared
                      else "*declares nothing*") + " |")
    lines.append("")

    scales = []
    for prov in sorted(models_by_provider):
        for option in list(UNIVERSAL_OPTIONS) + list(UNIVERSAL_FORMAT):
            table = value_map(option, prov)
            if table:
                scales.append((prov, option, table))
    if scales:
        lines += ["### The same level on each provider's own scale", "",
                  "A universal name is only half the promise. These are the "
                  "tables the value is put through — declared in the provider "
                  "data, never in a client.", "",
                  "| provider | option | our value → theirs |", "|---|---|---|"]
        for prov, option, table in scales:
            pairs = ", ".join(f"`{k}`→`{v}`" for k, v in table.items())
            lines.append(f"| `{prov}` | `{option}` | {pairs} |")
        lines.append("")
    return lines


def render_doc(matrix: dict) -> str:
    lines = [
        "# Parameters, per provider",
        "",
        "<!-- GENERATED by scripts/parameters.py — do not edit; rerun it. -->",
        "",
        "Three questions, one source. **What you may ask for**, **what each "
        "provider takes**, and **what actually happens when you ask** — the "
        "last established by building the request and diffing it against one "
        "built without the option. The same run feeds the conformance test, "
        "so this page cannot say one thing while the code does another.",
        "",
    ]
    lines += _vocabulary_section(matrix)
    lines += [
        "## 3. What happens when you ask",
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
        lines.append(f"### {title}")
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
                v = verdict(c)
                if v != "ok":
                    text += " **" + ("silent" if v.endswith("silent")
                                     else "mis-said") + "**"
                if c["outcome"] in ("error", "refused"):
                    text += f" <sub>{c['detail'][:60]}</sub>"
                cells.append(text)
            lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
        lines.append("")

    table("Model options (text)", TEXT_MODELS, list(TEXT_OPTIONS))
    table("Output format (image)", IMAGE_MODELS, list(IMAGE_FORMAT))
    table("Output format (image edits)", [f"{m} (edit)" for m in EDIT_MODELS],
          list(IMAGE_FORMAT))

    claimed = unfulfilled(matrix)
    if claimed:
        lines.append("## Claimed but not delivered")
        lines.append("")
        lines.append("The provider data says this provider has the control; "
                     "the request says nothing arrived. Either the "
                     "declaration is wrong or the option was never wired — "
                     "the matrix cannot tell which, only that they disagree.")
        lines.append("")
        for m, p in claimed:
            lines.append(f"- `{m}` · `{p}` — claimed by the provider, "
                         "delivered by none of its probed models")
        lines.append("")

    silent = [(m, p) for m in matrix for p, c in matrix[m].items()
              if verdict(c) != "ok"]
    lines.append("## Defects")
    lines.append("")
    lines.append(f"{len(silent)} cell(s) where the library changed the request "
                 "without saying so, or described the change wrongly.")
    lines.append("")
    for m, p in silent:
        c = matrix[m][p]
        lines.append(f"- `{m}` · `{p}` — {c['outcome']}"
                     + (f" ({c['detail']})" if c["detail"] else "")
                     + f" — {verdict(c)}")
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
    gaps = len(unfulfilled(matrix))
    print(f"{total} cells, {silent} unreported, {gaps} claimed-not-delivered; "
          f"wrote {SNAPSHOT.name} and {DOC.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
