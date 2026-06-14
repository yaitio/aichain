"""
clients._families.qwen
======================

Qwen (DashScope) rides the OpenAI Chat Completions *format* (inherited from
OpenAIClient, incl. its qwen quirk branch) but resolves its base URL by
region. It overrides only construction.
"""

from __future__ import annotations

import os

from .openai import OpenAIClient

REGION_URLS = {
    "ap": "https://dashscope-intl.aliyuncs.com",
    "us": "https://dashscope-us.aliyuncs.com",
    "cn": "https://dashscope.aliyuncs.com",
    "hk": "https://cn-hongkong.dashscope.aliyuncs.com",
}
_DEFAULT_REGION = "ap"


def resolve_qwen_base_url(region: "str | None" = None) -> str:
    r = (region or os.environ.get("DASHSCOPE_REGION") or _DEFAULT_REGION).lower().strip()
    if r not in REGION_URLS:
        raise ValueError(
            f"Unknown DashScope region {r!r}. "
            f"Valid regions: {', '.join(sorted(REGION_URLS))}."
        )
    return REGION_URLS[r]


class QwenClient(OpenAIClient):

    def __init__(self, api_key: str, *, data: dict, **client_opts) -> None:
        # Region-resolved base URL unless an explicit url is given.
        if not client_opts.get("url"):
            client_opts["url"] = resolve_qwen_base_url(client_opts.pop("region", None))
        super().__init__(api_key, data=data, **client_opts)
