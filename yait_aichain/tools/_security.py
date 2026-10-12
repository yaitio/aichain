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
from urllib.parse import urljoin, urlparse


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def assert_url_scheme(url: str) -> None:
    """Raise ``ValueError`` unless *url* uses the http or https scheme."""
    scheme = urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(
            f"Refusing non-http(s) URL scheme {scheme or '(none)'!r}: {url!r}"
        )


def _public(ip) -> bool:
    """Whether *ip* is a public address, looking through IPv6 forms that carry
    an IPv4 one — ``::ffff:127.0.0.1`` and the NAT64 ``64:ff9b::/96``. Public
    is ``is_global``, which also excludes the shared ``100.64.0.0/10`` range
    that a list of private networks misses (Alibaba's metadata service is
    ``100.100.100.200``)."""
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in ipaddress.ip_network("64:ff9b::/96"):
            ip = ipaddress.ip_address(int(ip) & 0xFFFFFFFF)
    return ip.is_global and not ip.is_multicast


def assert_safe_url(url: str, *, allow_private: bool | None = None) -> None:
    """
    Raise ``ValueError`` unless *url* is a safe outbound target.

    Blocks non-http(s) schemes (e.g. ``file:``, ``gopher:``), a URL carrying
    credentials or a backslash — the two places where this parser and the one
    that opens the connection disagree about the host (``http://127.0.0.1\\@
    example.com/``) — and, unless *allow_private* (defaulting to the
    ``AICHAIN_ALLOW_PRIVATE_URLS`` env flag), any host that resolves to an
    address that is not public. These are the classic SSRF and cloud-metadata
    vectors. A redirect is a new URL: see `safe_request` and `safe_get`, which
    check every hop.
    """
    assert_url_scheme(url)
    if "\\" in url:
        raise ValueError(f"Refusing a URL with a backslash in it: {url!r}")
    parts = urlparse(url)
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"Refusing a URL that carries credentials: {url!r}")
    host = parts.hostname
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
        ip = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not _public(ip):
            raise ValueError(
                f"Refusing to reach non-public address {ip} (host {host!r}) — "
                f"blocked to prevent SSRF. Set AICHAIN_ALLOW_PRIVATE_URLS=1 to "
                f"allow trusted internal targets."
            )


#: Statuses that send a client elsewhere, and how many hops are followed.
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 5


def safe_request(http, method: str, url: str, **kwargs):
    """A urllib3 request whose every hop passes `assert_safe_url`.

    urllib3 follows redirects itself and checks nothing on the way, so a
    public URL answering ``302 Location: http://169.254.169.254/`` reached the
    metadata service. Redirects are followed here instead, at most
    `MAX_REDIRECTS`, each target checked before it is asked; a 303 becomes a
    GET without a body, as browsers do.
    """
    for _ in range(MAX_REDIRECTS + 1):
        assert_safe_url(url)
        response = http.request(method, url, redirect=False, **kwargs)
        location = response.headers.get("location") if response.status in _REDIRECTS else None
        if not location:
            return response
        url = urljoin(url, location)
        if response.status == 303:
            method = "GET"
            kwargs.pop("body", None)
    raise ValueError(f"more than {MAX_REDIRECTS} redirects, last to {url!r}")


def safe_get(session, url: str, **kwargs):
    """`safe_request` for a ``requests`` session: every hop checked."""
    for _ in range(MAX_REDIRECTS + 1):
        assert_safe_url(url)
        response = session.get(url, allow_redirects=False, **kwargs)
        location = response.headers.get("location") if response.status_code in _REDIRECTS else None
        if not location:
            return response
        url = urljoin(url, location)
    raise ValueError(f"more than {MAX_REDIRECTS} redirects, last to {url!r}")


def within(root: str, path: str) -> bool:
    """True when the resolved *path* is *root* or inside it.

    Compared by path components, not by string prefix: a prefix would put
    ``/tmp/sandbox-evil`` inside ``/tmp/sandbox``, and a root of ``/`` would
    contain nothing, since every path then had to start with ``//``.
    """
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:                 # different drives on Windows
        return False


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
    if not within(root_real, resolved):
        raise PermissionError(
            f"Output path {path!r} resolves to {resolved!r}, outside the "
            f"configured AICHAIN_OUTPUT_ROOT {root_real!r}."
        )
    return resolved
