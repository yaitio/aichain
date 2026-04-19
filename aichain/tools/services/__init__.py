"""
tools.services
==============

External service connectors — all use ``run(input, options=None)``.

Tools
-----
serviceTranslate    DeepL translation + rephrase (two functions: run / rephrase).
serviceSocialPost   Late social media publishing (two functions: run / accounts).

Backward-compatible aliases
---------------------------
DeepLTranslateTool, DeepLRephraseTool, LatePublishTool, LateAccountsTool
"""

from .translate    import serviceTranslate
from .social_post  import serviceSocialPost

# Backward-compatible aliases
DeepLTranslateTool = serviceTranslate
DeepLRephraseTool  = serviceTranslate   # rephrase() is now a method on serviceTranslate
LatePublishTool    = serviceSocialPost
LateAccountsTool   = serviceSocialPost  # accounts() is now a method on serviceSocialPost

__all__ = [
    "serviceTranslate",
    "serviceSocialPost",
    "DeepLTranslateTool",
    "DeepLRephraseTool",
    "LatePublishTool",
    "LateAccountsTool",
]
