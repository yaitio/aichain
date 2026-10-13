"""
The wheel carries the documentation of its own version.

An agent writing code against an installed library reads either pages fetched
over the network, which describe the latest release, or what it remembers,
which describes an older one. Neither is the version it will run. The release
build copies `docs/` and `llms.txt` into the package, and
`yait_aichain.docs_dir()` finds them; in a checkout it finds the repository's.

Pinned here without building: the package-data patterns cover every page,
`docs_dir()` prefers the packaged copy, and the release and the push run one
build script. The build itself is checked on every push by
`.github/scripts/build.sh`, which fails when the wheel lacks the pages.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath

if sys.version_info >= (3, 11):
    import tomllib
else:                                     # pragma: no cover
    import tomli as tomllib

import yait_aichain

ROOT = Path(__file__).resolve().parents[1]


def _covered(rel: str, patterns) -> bool:
    path = PurePosixPath("_docs") / rel
    for pattern in patterns:
        if pattern.startswith("_docs/**/"):
            # setuptools reads `**` as any depth, at least one directory.
            tail = pattern[len("_docs/**/"):]
            if len(path.parts) > 2 and PurePosixPath(*path.parts[-1:]).match(tail):
                return True
        elif path.match(pattern) and len(path.parts) == len(PurePosixPath(pattern).parts):
            return True
    return False


class TestEveryPageIsPackaged(unittest.TestCase):

    def test_the_patterns_cover_the_docs_and_llms_txt(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        patterns = data["tool"]["setuptools"]["package-data"]["yait_aichain"]
        pages = [p.relative_to(ROOT / "docs").as_posix() for p in (ROOT / "docs").rglob("*.md")]
        self.assertGreater(len(pages), 20)
        missed = [p for p in pages + ["llms.txt"] if not _covered(p, patterns)]
        self.assertEqual(missed, [])

    def test_release_and_push_run_one_build(self):
        wf = ROOT / ".github" / "workflows"
        for name in ("publish.yml", "test.yml"):
            with self.subTest(workflow=name):
                self.assertIn("sh .github/scripts/build.sh", (wf / name).read_text())
        script = (ROOT / ".github" / "scripts" / "build.sh").read_text()
        self.assertIn("cp -R docs yait_aichain/_docs", script)
        self.assertIn("cp llms.txt yait_aichain/_docs/", script)


class TestDocsDir(unittest.TestCase):

    def test_a_checkout_finds_the_repository_docs(self):
        self.assertEqual(yait_aichain.docs_dir(), ROOT / "docs")

    def test_the_packaged_copy_comes_first(self):
        packaged = Path(yait_aichain.__file__).resolve().parent / "_docs"
        self.assertFalse(packaged.exists(), "a stray _docs/ from a build is in the tree")
        try:
            packaged.mkdir()
            (packaged / "index.md").write_text("# packaged\n")
            self.assertEqual(yait_aichain.docs_dir(), packaged)
        finally:
            shutil.rmtree(packaged, ignore_errors=True)

    def test_nothing_there_is_none(self):
        # The function alone, run as if the package lived in an empty folder.
        src = Path(yait_aichain.__file__).read_text()
        body = src[src.index("def docs_dir"):src.index("\n\n\n", src.index("def docs_dir"))]
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "pkg" / "yait_aichain" / "__init__.py"
            fake.parent.mkdir(parents=True)
            ns = {"__file__": str(fake)}
            exec(compile(body, "docs_dir", "exec"), ns)
            self.assertIsNone(ns["docs_dir"]())

if __name__ == "__main__":
    unittest.main()
