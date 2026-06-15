#!/usr/bin/env bash
#
# One-command release. Validates that the tree is clean and the version is
# consistent, runs the (non-network) test suite, then tags the commit and
# pushes the tag — which triggers .github/workflows/publish.yml to build and
# publish the package to PyPI via Trusted Publishing.
#
#   ./scripts/release.sh           # release the version in pyproject.toml
#   ./scripts/release.sh 1.3.1     # assert this exact version, then release
#
# Prerequisite (one-time): a Trusted Publisher configured on PyPI for this
# repo + publish.yml. No tokens are needed locally or in CI.
set -euo pipefail
cd "$(dirname "$0")/.."

PKG_VERSION=$(python3 -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")
VERSION="${1:-$PKG_VERSION}"
TAG="v$VERSION"

# 1. version consistency: argument == pyproject == __init__
if [ "$VERSION" != "$PKG_VERSION" ]; then
    echo "✗ requested $VERSION but pyproject.toml says $PKG_VERSION — bump it first." >&2
    exit 1
fi
INIT_VERSION=$(python3 -c "import re,pathlib; print(re.search(r'__version__ = \"([^\"]+)\"', pathlib.Path('yait_aichain/__init__.py').read_text()).group(1))")
if [ "$INIT_VERSION" != "$PKG_VERSION" ]; then
    echo "✗ yait_aichain/__init__.py ($INIT_VERSION) != pyproject.toml ($PKG_VERSION)." >&2
    exit 1
fi

# 2. clean working tree (release commit must already be made)
if [ -n "$(git status --porcelain)" ]; then
    echo "✗ working tree is not clean — commit the release first." >&2
    exit 1
fi

# 3. tests (skip *Live* — they need real provider keys + network)
echo "→ running tests (excluding *Live*)..."
pytest -k "not Live" -q

# 4. tag (reuse if it already exists and points at HEAD)
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "✓ tag $TAG already exists — reusing it."
else
    git tag -a "$TAG" -m "$VERSION"
    echo "✓ created tag $TAG"
fi

# 5. push branch + tag → GitHub Actions publishes to PyPI
echo "→ pushing $(git rev-parse --abbrev-ref HEAD) and $TAG..."
git push origin HEAD
git push origin "$TAG"

echo "✓ pushed $TAG."
echo "  Watch the publish run: https://github.com/yaitio/aichain/actions"
echo "  Package page:          https://pypi.org/project/yait-aichain/$VERSION/"
