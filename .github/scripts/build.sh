#!/bin/sh
# Build the sdist and the wheel with the pages of this version inside them,
# and fail if the wheel does not carry them. One script for the release and
# for every push, so the check that runs on a push is the build that ships.
set -e
pip install build twine
rm -rf dist yait_aichain/_docs
cp -R docs yait_aichain/_docs
cp llms.txt yait_aichain/_docs/
python -m build
rm -rf yait_aichain/_docs
twine check dist/*
python - <<'PY'
import glob, zipfile
names = set(zipfile.ZipFile(glob.glob("dist/*.whl")[0]).namelist())
wanted = {"yait_aichain/_docs/index.md", "yait_aichain/_docs/llms.txt",
          "yait_aichain/_docs/api/agent.md"}
missing = sorted(wanted - names)
assert not missing, f"the wheel does not carry its docs: {missing}"
print(sum(n.startswith("yait_aichain/_docs/") for n in names), "doc files in the wheel")
PY
