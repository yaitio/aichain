"""
tools._security
===============

Shared defensive helpers for tools that act on caller/LLM-supplied URLs and
output paths. In an agentic setting these inputs are untrusted, so the tools
default to safe behaviour — no SSRF to private hosts, no ``file://`` reads,
optional confinement of output writes — with explicit opt-outs for trusted
deployments via environment variables:

* ``AICHAIN_ALLOW_PRIVATE_URLS=1`` — allow fetching private / internal hosts.
* ``AICHAIN_OUTPUT_ROOT=/path``    — confine all tool output writes here.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def has_url_scheme(value: object) -> bool:
    """True when *value* is a string carrying any URL scheme (http/file/...)."""
    if not isinstance(value, str):
        return False
    try:
        return bool(urlparse(value).scheme)
    except ValueError:
        return False


def assert_url_scheme(url: str) -> None:
    """Raise ``ValueError`` unless *url* uses the http or https scheme."""
    scheme = urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(
            f"Refusing non-http(s) URL scheme {scheme or '(none)'!r}: {url!r}"
        )


def assert_safe_url(url: str, *, allow_private: "bool | None" = None) -> None:
    """
    Raise ``ValueError`` unless *url* is a safe outbound target.

    Blocks non-http(s) schemes (e.g. ``file:``, ``gopher:``) and — unless
    *allow_private* (defaulting to the ``AICHAIN_ALLOW_PRIVATE_URLS`` env flag)
    — any host that resolves to a private / loopback / link-local / reserved /
    multicast address. These are the classic SSRF and cloud-metadata vectors.
    """
    assert_url_scheme(url)
    host = urlparse(url).hostname
    if not host:
        raise ValueError(f"URL has no host: {url!r}")

    if allow_private is None:
        allow_private = _env_flag("AICHAIN_ALLOW_PRIVATE_URLS")
    if allow_private:
        return

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve host {host!r}: {exc}") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ValueError(
                f"Refusing to reach non-public address {ip} (host {host!r}) — "
                f"blocked to prevent SSRF. Set AICHAIN_ALLOW_PRIVATE_URLS=1 to "
                f"allow trusted internal targets."
            )


def confine_output_path(path: str) -> str:
    """
    Return the absolute *path*, confined to ``AICHAIN_OUTPUT_ROOT`` when set.

    When the env var is unset (default), the path is returned as-is — current
    behaviour. When it is set, a path resolving outside that root raises
    ``PermissionError``, so an LLM-chosen ``output_path`` cannot overwrite
    arbitrary files.
    """
    resolved = os.path.realpath(path)
    root = os.environ.get("AICHAIN_OUTPUT_ROOT", "").strip()
    if not root:
        return resolved
    root_real = os.path.realpath(root)
    if resolved != root_real and not resolved.startswith(root_real + os.sep):
        raise PermissionError(
            f"Output path {path!r} resolves to {resolved!r}, outside the "
            f"configured AICHAIN_OUTPUT_ROOT {root_real!r}."
        )
    return resolved
