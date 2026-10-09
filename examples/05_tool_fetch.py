"""
05_tool_fetch.py — A web page's text from its URL.

fetch("local") runs MarkItDown on this machine: free, no key, and it reads
robots.txt first. The result is a Record — fields for code, and the page as
text for a model or a prompt.

Required packages:
    pip install "yait-aichain[fetch]"

Required env vars:
    (none)
"""

from yait_aichain.tools import fetch

found = fetch("local").run(["https://example.com", "https://www.iana.org/help/example-domains"])

for page in found["pages"]:
    print(page["title"] or "(untitled)", "—", page["error"] or f"{len(page['text'])} characters")

print("\nWhat a model reads:\n")
print(str(found)[:600])
