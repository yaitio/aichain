"""
05_tool_convert.py — Convert a URL or file to Markdown.

Required packages:
    pip install markitdown
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tools import convertToMD

tool   = convertToMD()
result = tool.run("https://example.com")
print(result[:500])
