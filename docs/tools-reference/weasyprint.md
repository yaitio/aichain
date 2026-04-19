# `weasyprint` — `WeasyprintTool`

Render HTML to **PDF** using [WeasyPrint](https://weasyprint.org/). Accepts either a raw HTML string or a path to an HTML file; returns either the saved PDF's path or raw bytes.

```python
from tools import WeasyprintTool

tool = WeasyprintTool()
path = tool.run(source="<h1>Hello</h1>", output_path="hello.pdf")
```

> Note: the class is **`WeasyprintTool`** (lowercase `p`), matching `tools/__init__.py`.

---

## Installation

```bash
pip install weasyprint
```

Stateless; no env var required.

---

## Parameters

| Name | Type | Required | Notes |
|---|---|---|---|
| `source` | `string` | ✓ | Either an HTML string **or** a path to an `.html` file. Auto-detected. |
| `output_path` | `string` | | Destination `.pdf` path. Parent directories are created. |
| `base_url` | `string` | | Base URL or directory for resolving relative CSS / image / font references. |

---

## Return value

- **`output_path` provided** — the tool writes the file and returns its **absolute path string**.
- **`output_path` omitted** — the tool returns the **raw PDF bytes**.

This dual return type is useful: a Chain step typically wants the path; a direct script might want the bytes in memory.

---

## Usage

### From an HTML string

```python
tool = WeasyprintTool()

# Call-style — ToolResult wraps any exception
result = tool(source="<h1>Hello</h1><p>World</p>", output_path="hello.pdf")
if result:
    print("Saved to:", result.output)
else:
    print("Error:", result.error)

# Direct run
path = tool.run(source="<h1>Hi</h1>", output_path="out.pdf")
```

### From an HTML file

```python
tool.run(source="report.html", output_path="report.pdf")
```

### Raw bytes (no file written)

```python
pdf_bytes = tool.run(source="<p>Hello</p>")
with open("manual.pdf", "wb") as fh:
    fh.write(pdf_bytes)
```

### With relative assets

```python
tool.run(
    source      = "<link rel='stylesheet' href='styles.css'>…",
    output_path = "out.pdf",
    base_url    = "/path/to/assets/",
)
```

### In a Chain — Markdown → HTML → PDF

```python
from chain import Chain
from tools import MistletoeTool, WeasyprintTool

chain = Chain(steps=[
    (report_skill,      "report_md"),
    (MistletoeTool(),   "report_html", {"text": "report_md", "format": "format"}),
    (WeasyprintTool(),  "pdf_path",    {"source": "report_html"}),
], variables={"format": "html"})
```

`chain.accumulated["pdf_path"]` is now the absolute path of the final PDF.

---

## Notes

- The tool module file is named `weasyprint.py`, so it explicitly strips its own directory from `sys.path` before importing the library.
- If `source` is a path that resolves to an existing file, the tool uses `HTML(filename=…)`. Otherwise it uses `HTML(string=…)`.
- `output_path` parent directories are created with `os.makedirs(..., exist_ok=True)`.

---

## See also

- [`mistletoe`](mistletoe.md) — convert Markdown to HTML first.
- [`markitdown`](markitdown.md) — round-trip: PDF → Markdown (and back via Mistletoe → WeasyPrint).
