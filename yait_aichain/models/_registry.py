"""
models._registry
================

Canonical registry of the model families officially supported by this library,
organised by provider and task.

The registry is **reference data only** — the :class:`~models.Model` factory
accepts any valid model name regardless of whether it appears here.  Use the
registry for discovery, documentation, and light validation in application code.

Tasks
-----
``"text-to-text"``
    Text prompt → text response.  Includes chat, instruction-following,
    reasoning, and code-generation models.

``"text-to-image"``
    Text prompt → image.  Dedicated image-generation endpoints
    (``/v1/images/generations`` for OpenAI / xAI; ``generateContent`` with
    ``responseModalities: ["IMAGE"]`` for Google).

``"image-to-text"``
    Image (+ optional text) → text response.  Models that accept image parts
    in the universal message format.

Quick reference
---------------
::

    from models import registry

    registry.models(provider="anthropic", task="text-to-text")
    # → Claude model list

    registry.providers(task="text-to-image")
    # → ["google", "openai", "xai"]

    registry.models(provider="kimi")
    # → ["kimi-k2-0905-preview", "kimi-k2-thinking", …]

    registry.tasks("gpt-4o")
    # → ["image-to-text", "text-to-text"]

    registry.is_supported("gpt-image-1", "text-to-image")  # True
    registry.is_supported("gpt-image-1", "text-to-text")   # False
"""

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

TASKS: tuple[str, ...] = (
    "text-to-text",
    "text-to-image",
    "image-to-text",
)

PROVIDERS: tuple[str, ...] = (
    "openai",
    "anthropic",
    "google",
    "xai",
    "perplexity",
    "kimi",
    "deepseek",
    "qwen",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Canonical registry mapping provider → task → list of supported model names.
#:
#: A provider key is absent for tasks it does not support (e.g. Anthropic has
#: no ``"text-to-image"`` entry).  Model lists are ordered roughly by
#: capability (most capable first) within each family.
REGISTRY: dict[str, dict[str, list[str]]] = {

    # ── OpenAI ─────────────────────────────────────────────────────────────
    "openai": {
        "text-to-text": [
            # GPT-5 series (Responses API)
            "gpt-5",
            "gpt-5-mini",
            # GPT-4.1 series
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            # GPT-4o series
            "gpt-4o",
            "gpt-4o-mini",
            # o-series reasoning models
            "o3",
            "o4-mini",
            "o1",
        ],
        "text-to-image": [
            # GPT Image series  (dall-e-2 / dall-e-3 deprecated 05/12/2026 — excluded)
            "gpt-image-1",
            "gpt-image-1.5",
            "gpt-image-1-mini",
        ],
        "image-to-text": [
            # GPT-5 series
            "gpt-5",
            "gpt-5-mini",
            # GPT-4.1 series
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            # GPT-4o series
            "gpt-4o",
            "gpt-4o-mini",
            # o-series (all support image input)
            "o3",
            "o4-mini",
            "o1",
        ],
    },

    # ── Anthropic ──────────────────────────────────────────────────────────
    "anthropic": {
        "text-to-text": [
            # Claude 4 series
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
        # All Claude models accept image input; list mirrors text-to-text.
        "image-to-text": [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
        # Anthropic has no image-generation models.
    },

    # ── Google AI ──────────────────────────────────────────────────────────
    "google": {
        "text-to-text": [
            # Gemini 3.1 series
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview",
            # Gemini 3 series
            "gemini-3-flash-preview",
            # Gemini 2.5 series (GA — -preview suffix dropped)
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        "text-to-image": [
            # Gemini image-generation models
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
        ],
        # All Gemini chat models accept image (and video/audio) input.
        "image-to-text": [
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
    },

    # ── xAI ────────────────────────────────────────────────────────────────
    "xai": {
        "text-to-text": [
            # Grok 4 series (always-on reasoning)
            "grok-4-0709",
            "grok-4-fast-reasoning",
            "grok-4-1-fast-reasoning",
            # Grok 3 series
            "grok-3",
            "grok-3-fast",
            "grok-3-mini",
            "grok-3-mini-fast",
        ],
        "text-to-image": [
            # Grok image-generation models
            "grok-imagine-image-pro",
            "grok-imagine-image",
        ],
        # All Grok 3/4 text models accept image input.
        "image-to-text": [
            "grok-4-0709",
            "grok-4-fast-reasoning",
            "grok-4-1-fast-reasoning",
            "grok-3",
            "grok-3-fast",
            "grok-3-mini",
            "grok-3-mini-fast",
        ],
    },

    # ── Perplexity ─────────────────────────────────────────────────────────
    "perplexity": {
        "text-to-text": [
            # Sonar search models
            "sonar-pro",
            "sonar",
            # Sonar reasoning model (built-in chain-of-thought)
            "sonar-reasoning-pro",
            # Deep research
            "sonar-deep-research",
        ],
        # Perplexity has no image-generation or image-input models.
    },

    # ── DeepSeek ───────────────────────────────────────────────────────────
    "deepseek": {
        "text-to-text": [
            # Standard chat model (DeepSeek-V3) — full parameter support
            "deepseek-chat",
            # Always-on CoT model (DeepSeek-R1) — reasoning_content in response
            "deepseek-reasoner",
        ],
        # DeepSeek has no image-generation or image-input models.
    },

    # ── Qwen (Alibaba DashScope) ───────────────────────────────────────────
    "qwen": {
        "text-to-text": [
            # Qwen flagship and workhorses
            "qwen-max",
            "qwen-plus",
            "qwen-turbo",
            # Qwen3 series — latest generation (text + optional thinking)
            "qwen3-235b-a22b",
            "qwen3-72b",
            "qwen3-32b",
            "qwen3-14b",
            "qwen3-8b",
            # QwQ reasoning model (always-on chain-of-thought)
            "QwQ-32B",
        ],
        "text-to-image": [
            # Wanx image-generation models
            "wanx2.1-t2i-turbo",
            "wanx2.1-t2i-plus",
            "wanx-v1",
        ],
        # Vision-language models that accept image input
        "image-to-text": [
            "qwen-vl-max",
            "qwen-vl-plus",
            "qwen2.5-vl-max",
            "qwen2.5-vl-plus",
        ],
    },

    # ── Kimi (Moonshot AI) ─────────────────────────────────────────────────
    "kimi": {
        "text-to-text": [
            # K2.5 series — multimodal, thinking toggle
            "kimi-k2.5",
            # K2 series — text-only
            "kimi-k2-0905-preview",
            "kimi-k2-turbo-preview",
            # K2 thinking series — always-on reasoning
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
        ],
        # kimi-k2.5 accepts image and video input via the standard
        # OpenAI-compatible multimodal message format.
        "image-to-text": [
            "kimi-k2.5",
        ],
        # Kimi has no image-generation models.
    },
}


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def models(
    provider: "str | None" = None,
    task:     "str | None" = None,
) -> list[str]:
    """
    Return registered model names, optionally filtered by *provider* and/or
    *task*.

    Parameters
    ----------
    provider : str | None, optional
        One of :data:`PROVIDERS`.  When omitted, models from all providers
        are included.
    task : str | None, optional
        One of :data:`TASKS`.  When omitted, all tasks are included.

    Returns
    -------
    list[str]
        Sorted, deduplicated list of model names.

    Raises
    ------
    ValueError
        If *provider* or *task* is not a recognised value.

    Examples
    --------
    ::

        registry.models(task="text-to-image")
        registry.models(provider="openai", task="text-to-text")
        registry.models(provider="anthropic")
    """
    if provider is not None and provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Valid providers: {list(PROVIDERS)}"
        )
    if task is not None and task not in TASKS:
        raise ValueError(
            f"Unknown task {task!r}. "
            f"Valid tasks: {list(TASKS)}"
        )

    result: set[str] = set()
    for prov, task_map in REGISTRY.items():
        if provider is not None and prov != provider:
            continue
        for t, names in task_map.items():
            if task is not None and t != task:
                continue
            result.update(names)

    return sorted(result)


def providers(task: "str | None" = None) -> list[str]:
    """
    Return provider names that have at least one registered model, optionally
    limited to providers that support *task*.

    Parameters
    ----------
    task : str | None, optional
        One of :data:`TASKS`.  When omitted, all providers are returned.

    Returns
    -------
    list[str]
        Provider names in their canonical order (same as :data:`PROVIDERS`).

    Raises
    ------
    ValueError
        If *task* is not a recognised value.

    Examples
    --------
    ::

        registry.providers()                      # all five providers
        registry.providers(task="text-to-image")  # ["google", "openai", "xai"]
    """
    if task is not None and task not in TASKS:
        raise ValueError(
            f"Unknown task {task!r}. "
            f"Valid tasks: {list(TASKS)}"
        )

    return [
        p for p in PROVIDERS
        if p in REGISTRY and (task is None or task in REGISTRY[p])
    ]


def tasks(model_name: str) -> list[str]:
    """
    Return the tasks supported by *model_name* across all providers.

    Parameters
    ----------
    model_name : str
        Model identifier as it appears in the registry.

    Returns
    -------
    list[str]
        Sorted list of task names.  Empty list when the model is not in the
        registry (no error is raised).

    Examples
    --------
    ::

        registry.tasks("gpt-4o")
        # ["image-to-text", "text-to-text"]

        registry.tasks("gpt-image-1")
        # ["text-to-image"]

        registry.tasks("unknown-model")
        # []
    """
    found: set[str] = set()
    for task_map in REGISTRY.values():
        for task, names in task_map.items():
            if model_name in names:
                found.add(task)
    return sorted(found)


def is_supported(model_name: str, task: "str | None" = None) -> bool:
    """
    Return ``True`` when *model_name* appears in the registry.

    Parameters
    ----------
    model_name : str
        Model identifier to look up.
    task : str | None, optional
        When provided, also require the model to be registered for this
        specific task.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If *task* is not a recognised value.

    Examples
    --------
    ::

        registry.is_supported("grok-imagine-image")                   # True
        registry.is_supported("grok-imagine-image", "text-to-image")  # True
        registry.is_supported("grok-imagine-image", "text-to-text")   # False
        registry.is_supported("unknown-model")                        # False
    """
    if task is not None and task not in TASKS:
        raise ValueError(
            f"Unknown task {task!r}. "
            f"Valid tasks: {list(TASKS)}"
        )

    for task_map in REGISTRY.values():
        for t, names in task_map.items():
            if model_name in names:
                if task is None or t == task:
                    return True
    return False
