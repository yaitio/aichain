"""
Every example runs and prints what it exists to show.

Binding — the right names and arity — is `test_lightness_invariant`'s job.
This one runs the files: each as ``python examples/NN.py`` in a subprocess,
with ``AICHAIN_EXAMPLES_DIR`` pointed at a temporary directory and a
deprecated name an error. Offline examples run on every commit, with no key
and no network; live ones run in the nightly job, each skipped when its key
is not set.

A file in `examples/` with no row here, or a row with no file, fails.
"""

import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

SEARCH_KEYS = {"BRAVE_SEARCH_API_KEY", "EXA_API_KEY", "FIRECRAWL_API_KEY",
               "PARALLEL_API_KEY", "PERPLEXITY_API_KEY", "TAVILY_API_KEY"}
IMAGE_KEYS = {"OPENAI_API_KEY", "GOOGLE_AI_API_KEY", "XAI_API_KEY", "BFL_API_KEY",
              "RECRAFT_API_TOKEN", "DASHSCOPE_API_KEY", "REVE_API_KEY"}


def _said_is_heard(out: str) -> None:
    said = re.search(r"said:\s*(.+)", out).group(1).lower()
    heard = re.search(r"heard:\s*(.+)", out).group(1).lower()
    words = set(re.findall(r"[a-z']+", said))
    shared = words & set(re.findall(r"[a-z']+", heard))
    assert len(shared) >= len(words) / 2, (said, heard)


def _events_in_order(out: str) -> None:
    types = [line.split()[0] for line in out.splitlines() if re.match(r"^[a-z_]+\.[a-z]+", line)]
    assert types[0] == "run.started" and types[-1] == "run.finished", types
    first = {t: types.index(t) for t in set(types)}
    assert (first["tool_call.started"] < first["approval.requested"] < first["approval.decided"]
            < first["tool_call.ended"] < first["text.delta"]), types


def _nothing_in_examples(out: str) -> None:
    allowed = {p.name for p in EXAMPLES.iterdir() if not p.name.startswith(".")}
    assert allowed <= KNOWN_FILES, sorted(allowed - KNOWN_FILES)


#: name → (level, keys, what stdout must show). Keys: a tuple needs all of
#: them, a set any one, and a set inside a tuple any one of that set. An expectation is a substring, a compiled pattern,
#: or a function of the whole output.
RUNS = {
    "01_skill.py": ("live", ("ANTHROPIC_API_KEY",), [re.compile(r"\S")]),
    "02_switch_provider.py": ("live", {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_AI_API_KEY"},
                              [re.compile(r"^\[[\w.-]+\]$", re.M)]),
    "03_structured_output.py": ("live", ("OPENAI_API_KEY",), ['"order_id"', "problems: []"]),
    "04_fallback_models.py": ("live", ("ANTHROPIC_API_KEY",), ["answered by: claude-haiku-5-5"]),
    "05_multi_turn.py": ("live", ("ANTHROPIC_API_KEY",), ["--- turn 3 ---", "total tokens:"]),
    "06_media_input.py": ("live", ("GOOGLE_AI_API_KEY",), ["[image]", "[pdf]", "[video]"]),
    "07_stream.py": ("live", ("ANTHROPIC_API_KEY",), [re.compile(r"pieces: ([2-9]|\d\d+)\s*$")]),
    "08_skill_save_load.py": ("live", ("ANTHROPIC_API_KEY",), ["claude-haiku-5-5", "answer:"]),
    "09_private_model.py": ("live", ("PRIVATE_BASE_URL",), ["cost: None"]),
    "10_what_was_sent.py": ("run", (), ["declined top_k", re.compile(r"cost: \$\d")]),
    "11_custom_tool.py": ("live", (), ['"amount"', "converted:"]),
    "12_image.py": ("live", IMAGE_KEYS, ["✓", _nothing_in_examples]),
    "13_speech.py": ("live", ("OPENAI_API_KEY",), ["said:", "heard:", _said_is_heard]),
    "14_search_fetch.py": ("live", (), ["characters", "What a model reads:"]),
    "15_retrieval.py": ("live", ("OPENAI_API_KEY", "DASHSCOPE_API_KEY"), ["source:"]),
    "16_decide.py": ("live", ("OPENAI_API_KEY",), [re.compile(r"(\$\d.*\n.*){3}")]),
    "17_mcp.py": ("live", (), ["Tools found:", "read_wiki_structure"]),
    "18_rest.py": ("live", (), ["latest  read"]),
    "19_shell.py": ("run", (), ["git status  read  allowed", "git push  privileged  denied"]),
    "20_chain.py": ("live", ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"), ["[writer]", "[reviewer]"]),
    "21_chain_steps.py": ("live", ("ANTHROPIC_API_KEY",), ['"source"', '"required": true', "summary:"]),
    "22_chain_save_load.py": ("live", ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"), ["policy:", "[loaded]"]),
    "23_chain_human.py": ("run", (), ["paused:", "sent:", "draft calls: 1"]),
    "24_chain_webhook.py": ("run", (), ["Paused", "Refunded $42.0", "approved=False"]),
    "25_pool.py": ("live", ("ANTHROPIC_API_KEY",), ["success: True"]),
    "26_pool_chain.py": ("live", ("ANTHROPIC_API_KEY",), ["[https://example.com]"]),
    "27_agent.py": ("live", ("ANTHROPIC_API_KEY", SEARCH_KEYS), ["stopped_by:", re.compile(r"tools: .*fetch")]),
    "28_agent_stop.py": ("live", ("ANTHROPIC_API_KEY",),
                         [re.compile(r"stopped_by: (check:safe_open|answered|step_count)")]),
    "29_agent_stream.py": ("run", (), ["run.started", _events_in_order]),
    "30_agent_team.py": ("live", ("ANTHROPIC_API_KEY", SEARCH_KEYS), ["delegate →", "cost:"]),
    "32_agent_pause.py": ("run", (), ["paused:", "refunded", "stopped_by: answered"]),
    "33_agent_step.py": ("run", (), ["step 1", "step 3", "dangling: ['call-", "answer:"]),
    "34_budget.py": ("run", (), ["refused: scripted for skill step_4", "spent: $0.012 of $0.010", "calls made: 3"]),
    "35_policy.py": ("run", (), ["$42", "executed", "$500", "refused: over $50 needs a ticket",
                                 "card ****4242"]),
    "36_observe.py": ("run", (), [re.compile(r"(run\.started.*\n(.*\n)*){3}"), "llm_call.ended",
                                  "step.ended", "journal:"]),
    "37_tenant_keys.py": ("run", (), [re.compile(r"tenant-a .*sk-…"), re.compile(r"tenant-b .*sk-…"),
                                      "keys in store: 0"]),
    "38_eval.py": ("run", (), ["pass^3", "terse", "verbose", "ledger:"]),
}

#: What `examples/` holds besides the scripts.
KNOWN_FILES = set(RUNS) | {"README.md", "assets", "__pycache__"}

#: A connection to anything but this machine fails an offline example.
_GUARD = textwrap.dedent('''
    import socket
    _real = socket.socket.connect
    def _connect(sock, address, *a, **k):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in ("127.0.0.1", "::1", "localhost") \\
                and sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError(f"an offline example tried to connect to {host}")
        return _real(sock, address, *a, **k)
    socket.socket.connect = _connect
''')


def run_example(name: str, offline: bool, timeout: int = 300) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, AICHAIN_EXAMPLES_DIR=tmp, PYTHONIOENCODING="utf-8")
        if offline:
            guard = Path(tmp) / "guard"
            guard.mkdir()
            (guard / "sitecustomize.py").write_text(_GUARD)
            env["PYTHONPATH"] = os.pathsep.join([str(guard), str(ROOT)])
            for k in list(env):
                if k.endswith(("_API_KEY", "_API_TOKEN")) or k.lower().endswith("_proxy"):
                    env.pop(k)
        else:
            env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
        done = subprocess.run(
            [sys.executable, "-W", "error::DeprecationWarning", "-W", "error::FutureWarning",
             str(EXAMPLES / name)],
            env=env, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        if done.returncode:
            raise AssertionError(f"{name} exited {done.returncode}\n{done.stdout[-2000:]}\n"
                                 f"{done.stderr[-3000:]}")
        return done.stdout


def check_output(case: unittest.TestCase, name: str, out: str) -> None:
    for expected in RUNS[name][2]:
        if callable(expected) and not isinstance(expected, re.Pattern):
            expected(out)
        elif isinstance(expected, re.Pattern):
            case.assertRegex(out, expected, f"{name}:\n{out}")
        else:
            case.assertIn(expected, out, f"{name}:\n{out}")


class TestEveryExampleHasARow(unittest.TestCase):

    def test_files_and_rows_agree(self):
        files = {p.name for p in EXAMPLES.glob("*.py")}
        self.assertEqual(sorted(files - set(RUNS)), [], "an example with no row in RUNS")
        self.assertEqual(sorted(set(RUNS) - files), [], "a row with no example")

    def test_nothing_else_lives_in_examples(self):
        stray = {p.name for p in EXAMPLES.iterdir() if not p.name.startswith(".")} - KNOWN_FILES
        self.assertEqual(sorted(stray), [], "examples write to AICHAIN_EXAMPLES_DIR, not here")

    def test_no_example_uses_what_is_deprecated_or_private(self):
        banned = re.compile(r'os\.getenv\("[A-Z_]*API_KEY"\)|models\._calls|permissions=|approve=|'
                            r'planner_model|cost_budget|\.history\[|from yait_aichain[\w.]*\._')
        for path in sorted(EXAMPLES.glob("*.py")):
            with self.subTest(example=path.name):
                self.assertIsNone(banned.search(path.read_text()))

    def test_each_is_one_idea(self):
        for path in sorted(EXAMPLES.glob("*.py")):
            with self.subTest(example=path.name):
                self.assertLessEqual(len(path.read_text().splitlines()), 60)


class TestOfflineExamplesRun(unittest.TestCase):

    def test_each_runs_and_shows_what_it_is_for(self):
        for name, (level, _, _) in RUNS.items():
            if level != "run":
                continue
            with self.subTest(example=name):
                check_output(self, name, run_example(name, offline=True, timeout=60))


class TestExamplesLive(unittest.TestCase):

    def test_each_runs_against_its_provider(self):
        for name, (level, keys, _) in RUNS.items():
            if level != "live":
                continue
            def have(k):
                return any(os.getenv(x) for x in k) if isinstance(k, (set, frozenset)) else os.getenv(k)
            needed = all(have(k) for k in keys) if isinstance(keys, tuple) else have(keys)
            with self.subTest(example=name):
                if not needed:
                    self.skipTest(f"{name}: {keys} not set")
                check_output(self, name, run_example(name, offline=False))


if __name__ == "__main__":
    unittest.main()
