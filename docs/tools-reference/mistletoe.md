# `mistletoe` — `MistletoeTool`

Convert Markdown to **HTML**, **LaTeX**, or **normalised Markdown** using the [mistletoe](https://github.com/miyuchina/mistletoe) library. Each format has a first-class renderer — no regex post-processing.

```python
from tools import MistletoeTool

tool = MistletoeTool()
html = tool.run(text="# Hello\n\nWorld", format="html")
# '<h1>Hello</h1>\n<p>World</p>\n'
```

---

## Supported formats

| `format` | What you get |
|---|---|
| `html` | HTML fragment (no `<html>`/`<body>` wrapper). |
| `latex` | LaTeX document body, ready for inclusion in a `.tex` file. |
| `markdown` | Normalised / reformatted Markdown — round-trip clean-up. |

---

## Installation

```bash
pip install mistletoe
```

Stateless; no env var required.

---

## Parameters

| Name | Type | Required | Notes |
|---|---|---|---|
| `text` | `string` | ✓ | Markdown source text. |
| `format` | `string` | ✓ | `html` / `latex` / `markdown`. |
| `output_path` | `string` | | Save converted output to this path. Parent directories created. |

---

## Usage

### Markdown → HTML

```python
tool = MistletoeTool()
html = tool.run(text="# Hello\n\nWorld", format="html")
```

### Markdown → LaTeX, save to file

```python
tool.run(
    text        = "# Introduction\n\nSome **bold** text.",
    format      = "latex",
    output_path = "out/intro.tex",
)
```

### Normalise messy Markdown

```python
clean_md = tool.run(text=messy_markdown, format="markdown")
```

### In a Chain — Markdown → HTML → PDF

```python
from chain import Chain
from tools import MistletoeTool, WeasyprintTool

chain = Chain(steps=[
    (write_report_skill,   "report_md"),
    (MistletoeTool(),      "report_html", {"text":   "report_md", "format": "format"}),
    (WeasyprintTool(),     "pdf_path",    {"source": "report_html"}),
], variables={"format": "html"})

chain.run(variables={"topic": "quarterly review"})
```

---

## Notes

- Raises `ValueError` if `format` is not one of the three supported values.
- Raises `ImportError` if `mistletoe` isn't installed.
- The tool's own module file is named `mistletoe.py`; it explicitly strips its own directory from `sys.path` before import so it doesn't shadow the installed library.

---

## See also

- [`markitdown`](markitdown.md) — the opposite direction (files/URLs → Markdown).
- [`weasyprint`](weasyprint.md) — render HTML to PDF.
