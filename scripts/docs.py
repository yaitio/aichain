"""
Documentation generated from the data it describes.

The 2.6.1 hand sweep went stale on four pages within a week; `parameters.md`
did not, because a run produces it. Every table on these pages that has a
source of truth in the code is produced from that source here, between
markers, and `tests/test_docs_generated.py` fails when a committed page and
its regeneration differ.

    python scripts/docs.py            # rewrite every generated block
    python scripts/docs.py --check    # exit 1 when a page is stale

A marker is ``<!-- g:NAME -->…<!-- /g:NAME -->``. Everything outside markers is
prose and belongs to whoever wrote it; everything inside belongs to this file.
A page that lost a marker, or carries one this file does not know, is an error
rather than a block that quietly stops updating.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yait_aichain import Chain, Pool, _errors_policy                  # noqa: E402
from yait_aichain.models import _options, registry                    # noqa: E402
from yait_aichain.models._data import PROVIDERS as DATA               # noqa: E402
import yait_aichain.tools as tools                                    # noqa: E402
from yait_aichain.tools import Tool                                   # noqa: E402

MARK = re.compile(r"<!-- g:(?P<name>[\w-]+) -->(?P<body>.*?)<!-- /g:(?P=name) -->", re.S)


# ── Data ──────────────────────────────────────────────────────────────────────

def label(provider: str) -> str:
    return DATA[provider]["provider"]["label"]


def cloud() -> list:
    return registry.providers()


TASK_DESCRIPTION = {
    "text-to-text":   "Text prompt → text response. Chat, reasoning, code.",
    "text-to-image":  "Text prompt → image.",
    "image-to-text":  "Image (+ optional text) → text response.",
    "image-to-image": "Image (+ instruction) → edited image.",
}
TASK_TITLE = {"text-to-text": "Text → Text", "text-to-image": "Text → Image",
              "image-to-text": "Image → Text", "image-to-image": "Image → Image"}

# Where a tool's key comes from, for the services that are not model providers.
# The one hand-kept table in this file: no class carries a sign-up URL.
SERVICE_URL = {
    "BRAVE_SEARCH_API_KEY": ("Brave Search", "https://brave.com/search/api"),
    "SERPAPI_API_KEY":      ("SerpAPI", "https://serpapi.com"),
    "COHERE_API_KEY":       ("Cohere", "https://dashboard.cohere.com"),
    "VOYAGE_API_KEY":       ("Voyage", "https://dash.voyageai.com"),
}

_KEYS = {k: "doc-key" for k in (
    "OPENAI_API_KEY", "GOOGLE_AI_API_KEY", "XAI_API_KEY", "PERPLEXITY_API_KEY",
    "BRAVE_SEARCH_API_KEY", "SERPAPI_API_KEY", "DASHSCOPE_API_KEY")}


def tool_classes() -> dict:
    """Exported Tool classes → every public name they are exported under."""
    found: dict = {}
    for name in sorted(dir(tools)):
        cls = getattr(tools, name)
        if inspect.isclass(cls) and issubclass(cls, Tool) and cls is not Tool:
            found.setdefault(cls, []).append(name)
    return found


def env_of(cls) -> list:
    keys = getattr(cls, "_ENV_KEY", None)
    if isinstance(keys, str):
        return [keys]
    if isinstance(keys, (tuple, list)):
        return list(keys)
    try:
        source = inspect.getsource(cls)
        module = inspect.getsource(inspect.getmodule(cls))
    except (OSError, TypeError):
        return []
    found = set(re.findall(r'environ\.get\("([A-Z0-9_]+_(?:KEY|TOKEN))"', source))
    # The search tools keep the name in a module constant and read
    # `os.environ.get(_ENV_KEY)`; the class body names the constant, not the key.
    if re.search(r"environ\.get\(\s*_ENV_KEY\s*\)", source):
        found |= set(re.findall(r'^_ENV_KEY\s*=\s*"([A-Z0-9_]+)"', module, re.M))
    return sorted(found)


def instance(cls):
    params = inspect.signature(cls.__init__).parameters
    with mock.patch.dict(os.environ, _KEYS):
        if "store" in params:
            return cls(mock.MagicMock())
        if all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
               for p in list(params.values())[1:]):
            return cls()
    return None                     # the schema is whatever the caller builds


def headline(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    first = doc.strip().split("\n\n")[0]
    text = " ".join(first.split()).replace("``", "`")
    return re.sub(r":(?:class|func|meth|attr):`~?([^`]+)`", r"`\1`", text)


def signature(cls, shown: str) -> str:
    parts, star = [], False
    for p in list(inspect.signature(cls.__init__).parameters.values())[1:]:
        if p.name.startswith("_"):
            continue
        if p.kind is p.KEYWORD_ONLY and not star:
            parts.append("*"); star = True
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            parts.append(("*" if p.kind is p.VAR_POSITIONAL else "**") + p.name); continue
        ann = p.annotation
        ann = "" if ann is p.empty else (ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann)))
        ann = ann.strip("'\"")
        text = p.name + (f": {ann}" if ann else "")
        if p.default is not p.empty:
            text += f" = {p.default!r}"
        parts.append(text)
    if not parts:
        return f"{shown}()"
    return f"{shown}(\n" + "".join(f"    {x},\n" for x in parts) + ")"


def type_of(schema: dict) -> str:
    kind = schema.get("type", "any")
    if kind == "array":
        return f"list[{type_of(schema.get('items', {}))}]"
    if isinstance(kind, list):
        return " | ".join(kind)
    return kind


def cell(text) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


# ── Blocks ────────────────────────────────────────────────────────────────────

def b_count_models() -> str:
    return f"{len(registry.models())} models from {len(cloud())} cloud providers"


def b_count_providers() -> str:
    return f"{len(cloud())} cloud providers"


def b_provider_list() -> str:
    return (f"**{len(cloud())} cloud providers + your own** — "
            + ", ".join(label(p) for p in cloud()))


def b_modalities() -> str:
    rows = ["| Modality | Providers |", "|---|---|"]
    for task in registry.TASKS:
        names = ", ".join(label(p) for p in registry.providers(task=task))
        if task == "text-to-text":
            names += ", [your own private server](getting-started/private-models.md)"
        rows.append(f"| **{TASK_TITLE[task]}** | {names} |")
    search = sorted(n for c, ns in tool_classes().items()
                    for n in ns if c.__module__.startswith("yait_aichain.tools.search.")
                    and n.startswith("search"))
    rows.append("| **Web search** (a tool) | " + ", ".join(f"`{n}`" for n in search) + " |")
    return "\n".join(rows)


def _price(price) -> str:
    if not isinstance(price, dict):
        return ""
    if {"input", "output"} <= set(price):
        return f"${price['input']:g} in / ${price['output']:g} out per 1M tokens"
    return ", ".join(f"{k} ${v:g}" if isinstance(v, (int, float)) else f"{k} {v}"
                     for k, v in price.items())


def b_registry_models() -> str:
    out = ["## Tasks", "", "| Task | What it is |", "|---|---|"]
    out += [f"| `{t}` | {TASK_DESCRIPTION[t]} |" for t in registry.TASKS]
    out += ["", "## Providers", "",
            ", ".join(f"`{p}`" for p in cloud()) + ".", "",
            "The `private` provider is registry-less by design: its catalogue belongs "
            "to the server you run (vLLM, Ollama, LM Studio, …), so any name it serves "
            "works and none is listed here. See "
            "[Private models](../getting-started/private-models.md).", ""]
    out += ["| Provider | " + " | ".join(f"`{t}`" for t in registry.TASKS) + " |",
            "|---|" + "---|" * len(registry.TASKS)]
    for p in cloud():
        out.append(f"| {label(p)} | " + " | ".join(
            str(len(registry.models(provider=p, task=t))) or "—" if registry.models(provider=p, task=t) else "—"
            for t in registry.TASKS) + " |")
    for p in cloud():
        models = DATA[p].get("models", {})
        out += ["", "---", "", f"## {label(p)}", "",
                f"`provider = \"{p}\"` · key `{DATA[p]['provider']['env_key']}`"]
        for t in registry.TASKS:
            names = registry.models(provider=p, task=t)
            if not names:
                continue
            width = max(len(n) for n in names) + 2
            lines = [f"{n:<{width}}# {_price(models.get(n, {}).get('price'))}".rstrip(" #")
                     for n in names]
            out += ["", f"### `{t}`", "", "```", *lines, "```"]
    return "\n".join(out)


def b_registry_constants() -> str:
    return "\n".join([
        "| Name | Value |", "|---|---|",
        f"| `TASKS` | `{registry.TASKS!r}` |",
        f"| `PROVIDERS` | `{registry.PROVIDERS!r}` |",
        f"| `OPTIONS` | `{registry.OPTIONS!r}` |",
    ])


def _tool_env() -> dict:
    """env var → tool names that read it, embeddings and rerankers included."""
    found: dict = {}
    for cls, names in tool_classes().items():
        for key in env_of(cls):
            found.setdefault(key, set()).add(names[-1])
    import yait_aichain.tools.embedding as emb
    import yait_aichain.tools.reranking as rer
    for module in (emb, rer):
        for name in dir(module):
            cls = getattr(module, name)
            if inspect.isclass(cls) and getattr(cls, "_ENV_KEY", None) and not name.startswith("_"):
                for key in env_of(cls):
                    found.setdefault(key, set()).add(name)
    return {k: sorted(v) for k, v in sorted(found.items())}


def b_env_providers() -> str:
    rows = ["| Variable | Provider | Where to get it |", "|---|---|---|"]
    for p in cloud():
        d = DATA[p]["provider"]
        url = d.get("keys_url")
        rows.append(f"| `{d['env_key']}` | {label(p)} | {f'<{url}>' if url else '—'} |")
    return "\n".join(rows)


def b_env_resolve() -> str:
    lines = []
    for p in cloud():
        first = (registry.models(provider=p, task="text-to-text")
                 or registry.models(provider=p))[0]
        lines.append((f'Model("{first}")', f"# reads {DATA[p]['provider']['env_key']}"))
    width = max(len(a) for a, _ in lines) + 2
    body = [f"{a:<{width}}{b}" for a, b in lines]
    body += ["", 'Model("gpt-5.5", api_key="sk-…")    # explicit — the environment is not read']
    return "```python\n" + "\n".join(body) + "\n```"


def b_env_tools() -> str:
    provider_keys = {DATA[p]["provider"]["env_key"] for p in DATA}
    rows = ["| Variable | Read by | Same key as a model provider |", "|---|---|---|"]
    for key, names in _tool_env().items():
        rows.append(f"| `{key}` | " + ", ".join(f"`{n}`" for n in names)
                    + f" | {'yes' if key in provider_keys else '—'} |")
    return "\n".join(rows)


def b_exports() -> str:
    keys = [DATA[p]["provider"]["env_key"] for p in cloud()]
    keys += [k for k in _tool_env() if k not in keys]
    return "```bash\n" + "\n".join(f'export {k}="…"' for k in keys) + "\n```"


def b_install_keys() -> str:
    rows = ["| Service | Environment variable | Get a key |", "|---|---|---|"]
    for p in cloud():
        d = DATA[p]["provider"]
        url = d.get("keys_url")
        rows.append(f"| {label(p)} | `{d['env_key']}` | {f'<{url}>' if url else '—'} |")
    for key, (service, url) in SERVICE_URL.items():
        rows.append(f"| {service} | `{key}` | <{url}> |")
    return "\n".join(rows)


# Tool pages: file → (title, [classes]). A tool exported but on no page fails
# the build — the reference used to cover seven of thirty-two.
def _cls(name):
    return getattr(tools, name)


TOOL_PAGES = {
    "perplexity-search.md": ("Perplexity search", ["searchPerplexity"]),
    "brave-search.md":      ("Brave search", ["searchBrave"]),
    "serp-api.md":          ("SerpAPI search", ["searchSerp"]),
    "openai-web-search.md": ("OpenAI web search", ["searchOpenAI"]),
    "markitdown.md":        ("Anything → Markdown", ["convertToMD"]),
    "mistletoe.md":         ("Markdown → HTML", ["convertToHTML"]),
    "weasyprint.md":        ("HTML → PDF", ["convertToPDF"]),
    "speech.md":            ("Speech: text ↔ audio",
                             ["ttsOpenAI", "ttsGoogle", "ttsXAI", "ttsQwen",
                              "sttOpenAI", "sttGoogle", "sttXAI", "sttQwen"]),
    "local.md":             ("Local files and shell",
                             ["LocalBrowseTool", "LocalReadTool", "LocalWriteTool", "LocalRunTool"]),
    "vectordb.md":          ("Vector store tools",
                             ["VectorChunkTool", "VectorUpsertTool", "VectorQueryTool",
                              "VectorFetchTool", "VectorDeleteTool"]),
    "rest-api.md":          ("REST endpoint as a tool", ["RestApiTool"]),
    "mcp.md":               ("MCP server tools", ["MCPTool"]),
    "wait-gate.md":         ("Pause for a signal", ["Wait", "Gate"]),
}
#: Exported, and deliberately without a page of their own.
NOT_PAGED = {"Search": "abstract base of the search tools",
             "convertToSpeech": "base of the tts* tools",
             "convertToText": "base of the stt* tools"}
GROUP = {"search": "Search", "convert": "Conversion", "local": "Local",
         "vectordb": "Vector store", "rest_api": "HTTP", "mcp": "MCP", "_wait": "Pause"}


def _group(cls) -> str:
    part = cls.__module__.split("yait_aichain.tools.")[-1].split(".")[0]
    return GROUP.get(part, part)


def _wire(cls, names) -> str:
    inst = instance(cls)
    return (inst.name if inst is not None and inst.name else "") or names[-1]


def _check_paging():
    paged = {_cls(n) for _, classes in TOOL_PAGES.values() for n in classes}
    missing = [ns[-1] for c, ns in tool_classes().items()
               if c not in paged and not set(ns) & set(NOT_PAGED)]
    if missing:
        raise SystemExit(f"tools with no reference page: {missing} — add them to TOOL_PAGES")


def b_tools_index() -> str:
    _check_paging()
    rows = ["| Group | Tool | Class | What it does | Key | Risk |", "|---|---|---|---|---|---|"]
    by_class = tool_classes()
    for page, (_, classes) in TOOL_PAGES.items():
        for n in classes:
            cls = _cls(n)
            names = by_class[cls]
            klass = next((x for x in names if x[0].isupper()), names[0])
            wire = _wire(cls, names)
            anchor = "" if len(classes) == 1 else "#" + re.sub(r"[^a-z0-9-]", "", n.lower())
            key = ", ".join(f"`{k}`" for k in env_of(cls)) or "—"
            rows.append(f"| {_group(cls)} | [`{wire or n}`]({page}{anchor}) | `{klass}` | "
                        f"{cell(headline(cls))} | {key} | `{getattr(cls, 'risk', '')}` |")
    return "\n".join(rows)


def _params_table(inst) -> list:
    schema = inst.parameters or {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    rows = ["| Name | Type | Required | Description |", "|---|---|---|---|"]
    for name, spec in props.items():
        if name == "options" and spec.get("properties"):
            inner_req = set(spec.get("required", []))
            for opt, ospec in spec["properties"].items():
                desc = ospec.get("description", "")
                if "enum" in ospec:
                    desc += " One of " + ", ".join(f"`{v}`" for v in ospec["enum"]) + "."
                rows.append(f"| `options.{opt}` | `{type_of(ospec)}` | "
                            f"{'✓' if opt in inner_req else ''} | {cell(desc)} |")
            continue
        desc = spec.get("description", "")
        if "enum" in spec:
            desc += " One of " + ", ".join(f"`{v}`" for v in spec["enum"]) + "."
        rows.append(f"| `{name}` | `{type_of(spec)}` | {'✓' if name in required else ''} | {cell(desc)} |")
    return rows if len(rows) > 2 else ["No parameters."]


def _tool_section(n, level) -> list:
    cls = _cls(n)
    names = tool_classes()[cls]
    klass = next((x for x in names if x[0].isupper()), names[0])
    inst = instance(cls)
    wire = (inst.name if inst is not None and inst.name else None)
    h = "#" * level
    title = f"`{wire}` — `{klass}`" if wire and wire != klass else f"`{klass}`"
    out = [f"{h} {title}", "", headline(cls), ""]
    also = [x for x in names if x != klass]
    facts = [f"| Import | `from yait_aichain.tools import {klass}` |",
             f"| Risk class | `{getattr(cls, 'risk', '')}` |"]
    if also:
        facts.append("| Also exported as | " + ", ".join(f"`{x}`" for x in also) + " |")
    if env_of(cls):
        facts.append("| Key | " + " or ".join(f"`{k}`" for k in env_of(cls)) + " |")
    out += ["| | |", "|---|---|", *facts, "", "```python", signature(cls, klass), "```", ""]
    if inst is None:
        out += ["The call schema is built from the constructor arguments, so it is "
                "whatever the caller declares.", ""]
    else:
        out += _params_table(inst) + [""]
        opts = inst.parameters.get("properties", {}).get("options", {}).get("properties", {})
        if "input" in inst.parameters.get("properties", {}):
            example = '    input="…",'
            if opts:
                first = next(iter(opts))
                example += f"\n    options={{\"{first}\": …}},"
            out += ["```python", f"result = {klass}{'(store)' if 'store' in inspect.signature(cls.__init__).parameters else '()'}(",
                    example, ")", "```", ""]
    return out


def tool_page_block(page: str) -> str:
    title, classes = TOOL_PAGES[page]
    if len(classes) == 1:
        return "\n".join(_tool_section(classes[0], 1)).rstrip()
    out = [f"# {title}", ""]
    for n in classes:
        out += _tool_section(n, 2)
    return "\n".join(out).rstrip()


POLICY = {
    "chain": {
        "raise":   "Propagate the exception. The run is marked failed and nothing after the step runs.",
        "stop":    "End the run without raising; the error is in `history` and `run()` returns the last successful output (or `None`).",
        "skip":    "Record the error, emit a `RuntimeWarning`, and run the next step. A later step that reads the failed step's output gets a stale or absent value.",
        "collect": "As `skip`, without the warning — for a long generated chain that expects a few steps to fail.",
    },
    "pool": {
        "raise":   "The first failing item propagates out of `run()`.",
        "stop":    "Start no further items; items already running finish. Every failed or unstarted item is `None` in the results.",
        "skip":    "That item's result is `None`, a `RuntimeWarning` names it, the others continue.",
        "collect": "That item's result is `None` and its error is in `history`; the others continue, silently.",
    },
}


def b_policy(primitive: str) -> str:
    table = POLICY[primitive]
    if set(table) != set(_errors_policy.POLICIES):
        raise SystemExit(f"POLICY[{primitive!r}] does not cover {_errors_policy.POLICIES}")
    cls, param = (Chain, "on_step_error") if primitive == "chain" else (Pool, "on_error")
    default = inspect.signature(cls.__init__).parameters[param].default
    rows = ["| Mode | Behaviour |", "|---|---|"]
    for mode in ("raise", "stop", "skip", "collect"):
        mark = " (default)" if mode == default else ""
        rows.append(f"| `\"{mode}\"`{mark} | {table[mode]} |")
    return "\n".join(rows)


def b_pool_signature() -> str:
    return "```python\n" + signature(Pool, "Pool") + "\n```"


def b_model_options() -> str:
    rows = ["| Key | What it asks for | Providers with a control for it |", "|---|---|---|"]
    for key, spec in _options.UNIVERSAL_OPTIONS.items():
        takers = [label(p) for p in cloud() if key in (registry.accepts(p) or ())]
        rows.append(f"| `{key}` | {cell(spec['what'])} | {', '.join(takers) or '—'} |")
    rows += ["", "`output[\"format\"]` keys, asked per call rather than per model. "
             "✓ marks a key whose loss changes *what* you get rather than how good it "
             "is — the set `Model(on_unsupported=\"requirements\")` raises for.", "",
             "| Key | What it asks for | ✓ | Providers that read it |", "|---|---|---|---|"]
    for key, spec in _options.UNIVERSAL_FORMAT.items():
        takers = [label(p) for p in cloud()
                  if key in (DATA[p]["provider"].get("options", {}).get("format_accepts") or ())]
        rows.append(f"| `{key}` | {cell(spec['what'])} | "
                    f"{'✓' if key in _options.REQUIREMENT else ''} | {', '.join(takers) or '—'} |")
    return "\n".join(rows)


# ── Pages ─────────────────────────────────────────────────────────────────────

PAGES = {
    "README.md": {"count-models": b_count_models, "count-providers": b_count_providers,
                  "exports": b_exports},
    "docs/index.md": {"provider-list": b_provider_list, "modalities": b_modalities},
    "docs/reference/model-registry.md": {"registry-models": b_registry_models,
                                         "registry-constants": b_registry_constants},
    "docs/reference/environment-variables.md": {"env-providers": b_env_providers,
                                                "env-resolve": b_env_resolve,
                                                "env-tools": b_env_tools,
                                                "exports": b_exports},
    "docs/getting-started/installation.md": {"install-keys": b_install_keys},
    "docs/tools-reference/index.md": {"tools-index": b_tools_index},
    "docs/primitives/chain.md": {"policy": lambda: b_policy("chain")},
    "docs/primitives/pool.md": {"policy": lambda: b_policy("pool"),
                                "pool-signature": b_pool_signature},
    "docs/primitives/models.md": {"model-options": b_model_options,
                                  "count-models": b_count_models},
}
for _page in TOOL_PAGES:
    PAGES[f"docs/tools-reference/{_page}"] = {"tool": (lambda p=_page: tool_page_block(p))}


def render(rel: str, text: str) -> str:
    blocks = PAGES[rel]
    seen = [m.group("name") for m in MARK.finditer(text)]
    unknown = sorted(set(seen) - set(blocks))
    missing = sorted(set(blocks) - set(seen))
    if unknown or missing:
        raise SystemExit(f"{rel}: unknown markers {unknown}, missing markers {missing}")

    def fill(m):
        name, body = m.group("name"), m.group("body")
        content = blocks[name]()
        inner = f"\n{content}\n" if (body.startswith("\n") or "\n" in content) else content
        return f"<!-- g:{name} -->{inner}<!-- /g:{name} -->"

    return MARK.sub(fill, text)


def targets() -> dict:
    out = {}
    for rel in PAGES:
        path = ROOT / rel
        text = path.read_text() if path.exists() else f"<!-- g:tool -->\n<!-- /g:tool -->\n"
        out[path] = render(rel, text)
    return out


def main(argv) -> int:
    results = targets()
    stale = [p for p, t in results.items() if not p.exists() or p.read_text() != t]
    if "--check" in argv:
        for p in stale:
            print(f"stale: {p.relative_to(ROOT)} — run python scripts/docs.py")
        return 1 if stale else 0
    for p, t in results.items():
        p.write_text(t)
    print(f"{len(results)} page(s), {len(stale)} rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
