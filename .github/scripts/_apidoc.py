"""
The API reference, rendered from the code: signatures and docstrings.

Imported by `.github/scripts/docs.py`, which owns the pages and the `--check`. Nothing
here knows what a class does; it reads what the class says. A wrong API page is
therefore a wrong docstring, and is fixed in the code.

Docstrings are numpy-style with some reST: `Parameters` / `Returns` / `Raises`
/ … sections under a dashed underline, ``literal`` and :class:`role` markup,
and `::` before an indented code block. Each becomes its Markdown equivalent.
"""

from __future__ import annotations

import inspect
import re
import textwrap

SECTIONS = {"Parameters", "Returns", "Yields", "Raises", "Attributes", "Examples",
            "Example", "Notes", "Note", "Warnings", "See Also", "Methods", "Usage"}
#: Sections whose entries are `name : type` followed by an indented description.
ENTRY_SECTIONS = {"Parameters", "Returns", "Yields", "Raises", "Attributes"}

_ROLE = re.compile(r":(?:class|func|meth|attr|mod|exc|data|obj):`~?([^`]+)`")


def _inline(text: str) -> str:
    text = _ROLE.sub(lambda m: f"`{m.group(1).split('.')[-1]}`", text)
    return text.replace("``", "`")


def _code_blocks(lines: list) -> list:
    """`::` then an indented block → a fenced Python block."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith("::"):
            head = line.rstrip()[:-2].rstrip()
            if head:
                out.append(head + (":" if not head.endswith((":", ".")) else ""))
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            block = []
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith((" ", "\t"))):
                block.append(lines[j])
                j += 1
            while block and not block[-1].strip():
                block.pop()
            out += ["", "```python", *textwrap.dedent("\n".join(block)).splitlines(), "```", ""]
            i = j
            continue
        out.append(line)
        i += 1
    return out


def _split_sections(doc: str):
    """[(title or None, lines)] in the docstring's order."""
    lines = doc.splitlines()
    parts, title, buf, i = [], None, [], 0
    while i < len(lines):
        if (i + 1 < len(lines) and lines[i].strip() in SECTIONS
                and re.fullmatch(r"\s*-{3,}\s*", lines[i + 1] or "")):
            parts.append((title, buf))
            title, buf, i = lines[i].strip(), [], i + 2
            continue
        buf.append(lines[i])
        i += 1
    parts.append((title, buf))
    return [(t, b) for t, b in parts if t or any(x.strip() for x in b)]


def _entries(lines: list) -> list:
    """`name : type` / indented description → [(name, type, description lines)]."""
    out = []
    for line in lines:
        if line and not line[0].isspace() and line.strip():
            name, _, kind = line.partition(" : ")
            out.append([name.strip(), kind.strip(), []])
        elif out:
            out[-1][2].append(line)
    return out


def _prose(lines: list) -> str:
    out, fenced = [], False
    for line in _code_blocks(textwrap.dedent("\n".join(lines)).splitlines()):
        if line.startswith("```"):
            fenced = not fenced
            out.append(line)
        else:
            out.append(line if fenced else _inline(line))
    return "\n".join(out).strip()


def _default(param) -> str:
    if param.default is inspect.Parameter.empty:
        return ""
    return f"`{param.default!r}`" if len(repr(param.default)) < 40 else "a default"


def render_doc(doc: str, signature: "inspect.Signature | None" = None) -> list:
    params = dict(signature.parameters) if signature else {}
    out = []
    for title, lines in _split_sections(doc or ""):
        if title is None:
            out += [_prose(lines), ""]
            continue
        heading = {"Example": "Examples", "Note": "Notes"}.get(title, title)
        out += [f"**{heading}**", ""]
        if title in ENTRY_SECTIONS:
            for name, kind, desc in _entries(lines):
                label = f"`{name}`" if title != "Raises" else f"`{name}`"
                bits = [b for b in (f"*{_inline(kind)}*" if kind else "",
                                    f"default {_default(params[name])}" if name in params
                                    and _default(params[name]) else "") if b]
                text = _prose(desc)
                item = f"- {label}" + (f" ({', '.join(bits)})" if bits else "")
                if text:
                    first, *rest = text.splitlines()
                    item += f" — {first}"
                    if rest:
                        item += "\n" + textwrap.indent("\n".join(rest), "  ")
                out.append(item)
            out.append("")
        else:
            out += [_prose(lines), ""]
    return out


def _annotation(a) -> str:
    if a is inspect.Parameter.empty or a is inspect.Signature.empty:
        return ""
    return a if isinstance(a, str) else inspect.formatannotation(a)


def _sig(name: str, obj, drop_first: bool = False) -> str:
    """`name(a: T = default, *, b=…) -> R`, annotations as written, defaults as repr."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return f"{name}(…)"
    params = list(sig.parameters.values())
    if drop_first and params and params[0].name in ("self", "cls"):
        params = params[1:]
    parts, star = [], False
    for p in params:
        if p.kind is p.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        if p.kind is p.VAR_POSITIONAL:
            star = True
            parts.append(f"*{p.name}")
            continue
        if p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
            continue
        text = p.name
        ann = _annotation(p.annotation)
        if ann:
            text += f": {ann}"
        if p.default is not p.empty:
            text += f" = {p.default!r}" if ann else f"={p.default!r}"
        parts.append(text)
    ret = _annotation(sig.return_annotation)
    head = f"{name}({', '.join(parts)})" + (f" -> {ret}" if ret and ret != "None" else "")
    if len(head) > 88:
        head = f"{name}(\n" + "".join(f"    {x},\n" for x in parts) + ")" + (f" -> {ret}" if ret and ret != "None" else "")
    return head


def _factory(obj):
    return getattr(obj, "__factory__", None)


def public_methods(cls) -> list:
    """Methods and properties a class defines or overrides, with a docstring."""
    out = []
    for name, member in cls.__dict__.items():
        if name.startswith("_") and name not in ("__call__",):
            continue
        fn = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
        if isinstance(member, property):
            if inspect.getdoc(member):
                out.append((name, member, "property"))
        elif inspect.isfunction(fn) and inspect.getdoc(fn):
            out.append((name, fn, "classmethod" if isinstance(member, classmethod) else "method"))
    return out


def render(name: str, obj) -> list:
    """One public name: its heading, signature and docstring, then its methods."""
    factory = _factory(obj)
    if factory is not None:
        out = [f"## `{name}()`", "", "```python", _sig(name, factory), "```", ""]
        return out + render_doc(inspect.getdoc(factory) or "", inspect.signature(factory))
    if inspect.isclass(obj):
        out = [f"## `{name}`", ""]
        if "__init__" in obj.__dict__ or not issubclass(obj, BaseException):
            out += ["```python", _sig(name, obj.__init__ if "__init__" in obj.__dict__ else obj, drop_first="__init__" in obj.__dict__), "```", ""]
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            sig = None
        out += render_doc(inspect.getdoc(obj) or "", sig)
        for mname, member, kind in public_methods(obj):
            if kind == "property":
                out += [f"### `{name}.{mname}`", "", *render_doc(inspect.getdoc(member))]
                continue
            label = f"{name}.{mname}" if mname != "__call__" else name
            out += [f"### `{label}()`", "", "```python", _sig(label, member, drop_first=True), "```", ""]
            out += render_doc(inspect.getdoc(member) or "", inspect.signature(member))
        return out
    if callable(obj):
        out = [f"## `{name}()`", "", "```python", _sig(name, obj), "```", ""]
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            sig = None
        return out + render_doc(inspect.getdoc(obj) or "", sig)
    value = sorted(obj) if isinstance(obj, (set, frozenset)) else obj   # a set's order is not stable
    return [f"## `{name}`", "", f"`{value!r}`" if len(repr(value)) < 200 else "", ""]


def page(objects: list) -> str:
    """The generated body of one API page: an index, then every name in order."""
    index = " · ".join(f"[`{n}`](#{re.sub(r'[^a-z0-9_-]', '', n.lower())})" for n, _ in objects)
    out = [index, ""]
    for n, obj in objects:
        out += ["---", "", *render(n, obj)]
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
