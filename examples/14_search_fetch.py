"""
14_search_fetch.py — Search the web, then read the first result as a model would.

Keys: any one of BRAVE_SEARCH_API_KEY, EXA_API_KEY, FIRECRAWL_API_KEY, PARALLEL_API_KEY,
PERPLEXITY_API_KEY, TAVILY_API_KEY; without one, only the fetch runs. fetch("local") needs none.
"""

import os

from yait_aichain.tools import fetch, search

SERVICES = {"BRAVE_SEARCH_API_KEY": "brave", "EXA_API_KEY": "exa", "FIRECRAWL_API_KEY": "firecrawl",
            "PARALLEL_API_KEY": "parallel", "PERPLEXITY_API_KEY": "perplexity",
            "TAVILY_API_KEY": "tavily"}
service = next((name for key, name in SERVICES.items() if os.getenv(key)), None)

if service:
    results = search(service).run("Qdrant documentation")["results"]
    for r in results[:3]:
        print(f"- {r['title']}  {r['url']}")
    url = results[0]["url"]
else:
    print("no search key set: reading https://example.com only")
    url = "https://example.com"

page = fetch("local").run(url)["pages"][0]       # free, and honours robots.txt
if page["error"]:
    raise SystemExit(f"{url}: {page['error']}")    # a page that failed says why
print(f"\n{page['title'] or url}: {len(page['text'])} characters")
print(f"What a model reads:\n{page['text'][:400]}")
