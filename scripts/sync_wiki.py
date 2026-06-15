#!/usr/bin/env python3
"""
Generate the GitHub wiki pages from docs/ into a target directory.

    python scripts/sync_wiki.py <wiki_checkout_dir>

The target dir is a checkout of the repo's `.wiki.git`. Existing `*.md` pages
are replaced so a removed/renamed doc drops its wiki page too. Cross-doc links
are rewritten to wiki page names; links into the repo (examples/, tools-reference)
become absolute GitHub blob URLs. Run by .github/workflows/wiki.yml on docs
changes; safe to run locally too.
"""

import os
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = os.environ.get("GITHUB_REPOSITORY", "yaitio/aichain")
BLOB = f"https://github.com/{REPO}/blob/main"

# Source doc → wiki page name.
SOURCE_PAGES = {
    "docs/index.md":                       "Home",
    "docs/getting-started/quickstart.md":  "Quickstart",
    "docs/getting-started/installation.md":"Installation",
    "docs/getting-started/concepts.md":    "Concepts",
    "docs/primitives/skills.md":           "Skill",
    "docs/primitives/models.md":           "Model",
    "docs/primitives/chain.md":            "Chain",
    "docs/primitives/pool.md":             "Pool",
    "docs/primitives/tools.md":            "Tools",
    "docs/primitives/state.md":            "State",
    "docs/agents/overview.md":             "Agent",
    "docs/reference/model-registry.md":    "Model-Registry",
}

# Link target basename → wiki page name.
PAGE = {
    "skills.md": "Skill", "models.md": "Model", "chain.md": "Chain",
    "pool.md": "Pool", "tools.md": "Tools", "state.md": "State",
    "overview.md": "Agent", "model-registry.md": "Model-Registry",
    "concepts.md": "Concepts", "installation.md": "Installation",
    "quickstart.md": "Quickstart", "index.md": "Home",
}
# Link target basename → repo path (rendered as a GitHub blob URL).
BLOB_PAGES = {
    "agent-as-chain-step.md":   "docs/agents/agent-as-chain-step.md",
    "configuration.md":         "docs/agents/configuration.md",
    "memory.md":                "docs/agents/memory.md",
    "environment-variables.md": "docs/reference/environment-variables.md",
    "yaml-schema.md":           "docs/reference/yaml-schema.md",
}

_LINK = re.compile(r"\]\(([^)]+)\)")


def _fix(target: str) -> str:
    t = target.strip()
    if t.startswith("http") or t.startswith("#"):
        return target
    if "examples/" in t:
        return f"{BLOB}/examples/" + t.split("examples/")[-1]
    if "tools-reference" in t:
        return f"{BLOB}/docs/tools-reference"
    path, _, anchor = t.partition("#")
    base = path.rsplit("/", 1)[-1]
    suffix = f"#{anchor}" if anchor else ""
    if base in PAGE:
        return PAGE[base] + suffix
    if base in BLOB_PAGES:
        return f"{BLOB}/{BLOB_PAGES[base]}{suffix}"
    return target


def _transform(text: str) -> str:
    return _LINK.sub(lambda m: "](" + _fix(m.group(1)) + ")", text)


_SIDEBAR = """### yait-aichain
[[Home]]

**Get started**
- [[Quickstart]]
- [[Installation]]
- [[Concepts]]

**Primitives**
- [[Skill]]
- [[Model]]
- [[Chain]]
- [[Pool]]
- [[Agent]]
- [[Tools]]
- [[State]]

**Reference**
- [[Model-Registry]]

---
[GitHub repo](https://github.com/{repo}) · [PyPI](https://pypi.org/project/yait-aichain/)
"""


def main(target: str) -> int:
    out = pathlib.Path(target)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.md"):
        f.unlink()
    for src, page in SOURCE_PAGES.items():
        (out / f"{page}.md").write_text(_transform((ROOT / src).read_text()))
        print(f"wrote {page}.md  (from {src})")
    (out / "_Sidebar.md").write_text(_SIDEBAR.format(repo=REPO))
    print("wrote _Sidebar.md")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
