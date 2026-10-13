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
import stat
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
    vectors. A redirect is a new URL: see `safe_request`, which checks every hop.
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


def safe_request(http, method: str, url: str, *, allow_private: bool | None = None,
                 cross_host: bool = False, **kwargs):
    """A urllib3 request whose every hop passes `assert_safe_url`.

    urllib3 follows redirects itself and checks nothing on the way, so a
    public URL answering ``302 Location: http://169.254.169.254/`` reached the
    metadata service. Redirects are followed here instead, at most
    `MAX_REDIRECTS`, each target checked before it is asked; a 303 becomes a
    GET without a body, as browsers do.
    """
    first = urlparse(url).netloc
    for _ in range(MAX_REDIRECTS + 1):
        assert_safe_url(url, allow_private=allow_private)
        response = http.request(method, url, redirect=False, **kwargs)
        location = response.headers.get("location") if response.status in _REDIRECTS else None
        if not location:
            response.final_url = url
            return response
        if not kwargs.get("preload_content", True):
            response.drain_conn()
            response.release_conn()
        nxt = urljoin(url, location)
        # To another host, never: the request carries whatever the caller
        # gave it — an `X-Api-Key`, a token in a header urllib3 does not know
        # to drop, a body a 307 would resend — and the host it was meant for
        # chose to send it elsewhere. Said, with the address, instead.
        # A page fetch may follow (`cross_host=True`), with only its Accept
        # header: example.com sending a reader to www.example.com is the web.
        if urlparse(nxt).netloc != urlparse(url).netloc:
            if not cross_host or method != "GET":
                raise ValueError(f"the server redirected to another host ({nxt}); not followed, "
                                 "so the request's headers and body stay with the host they were for")
            if urlparse(nxt).netloc != first:
                kwargs["headers"] = {k: v for k, v in (kwargs.get("headers") or {}).items()
                                     if k.lower() == "accept"}
        url = nxt
        if response.status == 303:
            method = "GET"
            kwargs.pop("body", None)
    raise ValueError(f"more than {MAX_REDIRECTS} redirects, last to {url!r}")


#: What a file of each kind starts with. A file a model or a variable named is
#: read and sent to a provider; `/proc/self/environ` — the AWS keys on Lambda —
#: is a regular file too, and it starts with none of these.
_MAGIC = {
    "image":    (lambda h: h[:3] == b"\xff\xd8\xff" or h[:8] == b"\x89PNG\r\n\x1a\n"
                 or h[:4] == b"GIF8" or (h[:4] == b"RIFF" and h[8:12] == b"WEBP")
                 or h[4:8] == b"ftyp" or h[:2] == b"BM"),
    "audio":    (lambda h: h[:3] == b"ID3" or (len(h) > 1 and h[0] == 0xFF and h[1] & 0xE0 == 0xE0)
                 or (h[:4] == b"RIFF" and h[8:12] == b"WAVE") or h[:4] in (b"OggS", b"fLaC")
                 or h[4:8] == b"ftyp" or h[:4] == b"\x1aE\xdf\xa3"),
    "video":    (lambda h: h[4:8] == b"ftyp" or h[:4] == b"\x1aE\xdf\xa3"
                 or (h[:4] == b"RIFF" and h[8:12] == b"AVI ")),
    "document": (lambda h: h[:5] == b"%PDF-"),
}

#: The most a local media file may be, read whole into a request.
MAX_INPUT_BYTES = {"image": 20 << 20, "video": 100 << 20, "document": 32 << 20, "audio": 200 << 20}


def check_input_file(path: str, kind: str) -> str:
    """The real path of a local *kind* file it is safe to read and send, or an error.

    A path a model or a variable chose is read and handed to a third party:
    ``images=["/proc/self/environ"]`` sent the environment, and ``/dev/zero``
    exhausted the memory. It must be a regular file, no larger than
    ``MAX_INPUT_BYTES`` for its kind, start the way a file of that kind does,
    and — when ``AICHAIN_INPUT_ROOT`` is set — sit inside that folder.
    """
    real = os.path.realpath(os.path.expanduser(path))
    root = os.environ.get("AICHAIN_INPUT_ROOT", "").strip()
    if root and not within(os.path.realpath(root), real):
        raise PermissionError(f"{path!r} is outside AICHAIN_INPUT_ROOT")
    try:
        info = os.stat(real)
    except FileNotFoundError:
        raise FileNotFoundError(f"no {kind} file at {path!r}") from None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{path!r} is not a regular file, so it is not read as {kind}")
    if info.st_size > MAX_INPUT_BYTES[kind]:
        raise ValueError(f"{path!r} is {info.st_size:,} bytes; a {kind} file may be at most "
                         f"{MAX_INPUT_BYTES[kind]:,}")
    with open(real, "rb") as fh:
        head = fh.read(16)
    if not _MAGIC[kind](head):
        raise ValueError(f"{path!r} does not look like {kind}: its first bytes are not a "
                         f"{kind} format's")
    return real


def clip(text: str, limit: int) -> str:
    """*text* at most *limit* characters: its head and its tail, the middle
    replaced by a marker saying how much was left out. Strictly shorter than
    a text over the limit, so clipping twice changes nothing more."""
    if len(text) <= limit:
        return text
    marker = f"\n… [{len(text) - limit:,} characters left out] …\n"
    keep = max(0, limit - len(marker))
    return text[: keep // 2] + marker + text[len(text) - (keep - keep // 2):]


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
