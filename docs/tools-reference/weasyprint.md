<!-- g:tool -->
# `convertToPDF` — `WeasyprintTool`

Render an HTML document to PDF.

| | |
|---|---|
| Import | `from yait_aichain.tools import WeasyprintTool` |
| Risk class | `write` |
| Also exported as | `convertToPDF` |

```python
WeasyprintTool(
    *args,
    **kwargs,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | HTML content as a string, or path to an existing HTML file. |
| `options.output_path` | `string` |  | Destination file path for the PDF. Returns raw bytes when omitted. |
| `options.base_url` | `string` |  | Base URL for resolving relative asset references in the HTML. |

```python
result = WeasyprintTool()(
    input="…",
    options={"output_path": …},
)
```
<!-- /g:tool -->

Renders HTML to **PDF** with [WeasyPrint](https://weasyprint.org/). Takes a raw
HTML string or a path to an HTML file; returns the saved file's path, or the
PDF bytes when no path is given.

```python
from yait_aichain.tools import WeasyprintTool

tool = WeasyprintTool()
path = tool.run(input="<h1>Hello</h1>", options={"output_path": "hello.pdf"})
```

---

## Installation

```bash
pip install weasyprint
```

Stateless; no key required.

---

## Return value

- **`options["output_path"]` given** — the file is written and its **absolute path** returned.
- **omitted** — the **raw PDF bytes** are returned.

---

## Usage

### From an HTML string

```python
tool = WeasyprintTool()

# Call-style — ToolResult wraps any exception
result = tool(input="<h1>Hello</h1><p>World</p>", options={"output_path": "hello.pdf"})
if result:
    print("Saved to:", result.output)
else:
    print("Error:", result.error)
```

### From an HTML file

```python
tool = WeasyprintTool()
tool.run(input="report.html", options={"output_path": "report.pdf"})
```

### Raw bytes (no file written)

```python
tool      = WeasyprintTool()
pdf_bytes = tool.run(input="<p>Hello</p>")
with open("manual.pdf", "wb") as fh:
    fh.write(pdf_bytes)
```

### With relative assets

```python
tool = WeasyprintTool()
tool.run(input="<link rel='stylesheet' href='styles.css'>…",
         options={"output_path": "out.pdf", "base_url": "/path/to/assets/"})
```

### In a Chain — Markdown → HTML → PDF

```python
from yait_aichain.chain import Chain
from yait_aichain.tools import MistletoeTool, WeasyprintTool

chain = Chain(steps=[
    (report_skill,     "report_md"),
    (MistletoeTool(),  "report_html", {"input": "report_md"}),
    (WeasyprintTool(), "pdf_bytes",   {"input": "report_html"}),
])
```

---

## Notes

- If `input` is a path to an existing file the tool renders that file;
  otherwise it treats `input` as an HTML string.
- Parent directories for `output_path` are created as needed, and confined by
  `AICHAIN_OUTPUT_ROOT` when it is set.

---

## See also

- [`convertToHTML`](mistletoe.md) — Markdown to HTML first.
- [`convertToMD`](markitdown.md) — the round trip: PDF → Markdown.
