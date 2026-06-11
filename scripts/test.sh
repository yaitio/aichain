#!/usr/bin/env bash
set -e

echo "→ Running tests..."
pytest tests/ -v --tb=short "$@"
