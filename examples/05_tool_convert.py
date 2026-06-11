"""
05_tool_convert.py — Convert a URL or file to Markdown.

Required packages:
    pip install markitdown
"""

from yait_aichain.tools import convertToMD

tool   = convertToMD()
result = tool.run("https://example.com")
print(result[:500])
