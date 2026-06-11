#!/usr/bin/env bash
set -e

echo "→ Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

echo "→ Building package..."
python -m build

echo "→ Checking distribution..."
twine check dist/*

echo "✓ Build complete: $(ls dist/)"
