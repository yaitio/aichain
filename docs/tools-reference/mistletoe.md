<!-- g:tool -->
# `convertToHTML` — `MistletoeTool`

Convert Markdown text to HTML, LaTeX, or normalised Markdown.

| | |
|---|---|
| Import | `from yait_aichain.tools import MistletoeTool` |
| Risk class | `write` |
| Also exported as | `convertToHTML` |

```python
MistletoeTool(
    *args,
    **kwargs,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Markdown text to convert. |
| `options.format` | `string` |  | Output format. Default: 'html'. One of `html`, `latex`, `markdown`. |
| `options.output_path` | `string` |  | Optional file path to write the result. |

```python
result = MistletoeTool()(
    input="…",
    options={"format": …},
)
```
<!-- /g:tool -->

Converts Markdown to **HTML**, **LaTeX**, or **normalised Markdown** with
[mistletoe](https://github.com/miyuchina/mistletoe). Each format has a real
renderer — no regex post-processing.

```python
from yait_aichain.tools import MistletoeTool

tool = MistletoeTool()
html = tool.run(input="# Hello\n\nWorld", options={"format": "html"})
# '<h1>Hello</h1>\n<p>World</p>\n'
```

---

## Supported formats

| `options["format"]` | What you get |
|---|---|
| `html` | HTML fragment (no `<html>`/`<body>` wrapper). |
| `latex` | LaTeX document body, ready for inclusion in a `.tex` file. |
| `markdown` | Normalised / reformatted Markdown — round-trip clean-up. |

---

## Installation

```bash
pip install mistletoe
```

Stateless; no key required.

---

## Usage

### Markdown → LaTeX, saved to a file

```python
tool = MistletoeTool()
tool.run(input="# Introduction\n\nSome **bold** text.",
         options={"format": "latex", "output_path": "out/intro.tex"})
```

### Normalise messy Markdown

```python
tool     = MistletoeTool()
clean_md = tool.run(input=messy_markdown, options={"format": "markdown"})
```

### In a Chain — Markdown → HTML → PDF

```python
from yait_aichain.chain import Chain
from yait_aichain.tools import MistletoeTool, WeasyprintTool

chain = Chain(steps=[
    (write_report_skill, "report_md"),
    (MistletoeTool(),    "report_html", {"input": "report_md"}),
    (WeasyprintTool(),   "pdf_bytes",   {"input": "report_html"}),
])

chain.run(variables={"topic": "quarterly review"})
```

HTML is the default format, so the step needs only its input.

---

## Notes

- Raises `ValueError` when `format` is not one of the three values.
- Raises `ImportError` when `mistletoe` is not installed.

---

## See also

- [`convertToMD`](markitdown.md) — the other direction (files/URLs → Markdown).
- [`convertToPDF`](weasyprint.md) — render HTML to PDF.
