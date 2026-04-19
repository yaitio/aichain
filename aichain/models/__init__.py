"""
models
======

Public API for the aichain 2.0 model layer.

Typical usage
-------------
::

    from models import Model

    # Auto-detected provider — returns the correct subclass
    gpt    = Model("gpt-4o")
    sonnet = Model("claude-sonnet-4-5")
    gemini = Model("gemini-2.0-flash")
    grok   = Model("grok-3")
    sonar  = Model("sonar-pro")

    # Override generation parameters
    fast = Model("gpt-4o-mini", options={"temperature": 0.3, "max_tokens": 2048})

    # Override client settings (proxy, timeout, custom URL)
    corp = Model("gpt-4o", client_options={"proxy": {"url": "http://proxy:3128"}})

    # Explicit API key (falls back to env var when omitted)
    model = Model("claude-sonnet-4-5", api_key="sk-ant-...")

    # Use a provider subclass directly
    from models import OpenAIModel
    m = OpenAIModel("gpt-4.1", options={"max_tokens": 32768})

    # Query the model registry
    from models import registry
    registry.models(task="text-to-image")           # all text-to-image models
    registry.providers(task="text-to-image")        # ["google", "openai", "xai"]
    registry.tasks("gpt-4o")                        # ["image-to-text", "text-to-text"]
    registry.is_supported("gpt-image-1", "text-to-image")  # True
"""

from ._base       import Model
from ._openai     import OpenAIModel
from ._anthropic  import AnthropicModel
from ._google     import GoogleAIModel
from ._xai        import XAIModel
from ._perplexity import PerplexityModel
from ._kimi       import KimiModel
from ._deepseek   import DeepSeekModel
from ._qwen       import QwenModel
from .            import _registry as registry

__all__ = [
    "Model",
    "OpenAIModel",
    "AnthropicModel",
    "GoogleAIModel",
    "XAIModel",
    "PerplexityModel",
    "KimiModel",
    "DeepSeekModel",
    "QwenModel",
    "registry",
]
