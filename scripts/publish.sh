#!/usr/bin/env bash
set -e

TARGET=${1:-pypi}   # ./scripts/publish.sh testpypi  or  ./scripts/publish.sh

bash "$(dirname "$0")/build.sh"

if [ "$TARGET" = "testpypi" ]; then
    echo "→ Publishing to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo "✓ Published: https://test.pypi.org/project/yait-aichain/"
else
    echo "→ Publishing to PyPI..."
    twine upload dist/*
    echo "✓ Published: https://pypi.org/project/yait-aichain/"
fi
