"""
Can an agent build on yait-aichain from `SKILL.md` and `llms.txt` alone?

Stage 5.3 of `docs/design/cleanup-plan-2026-09-13.md`. A mid-tier model gets
the two files as its only reference and one task at a time; it returns a
program; the program runs in a sandbox with no network, where every model call
is scripted and everything the program built is recorded; the task's check
reads that record. Twenty tasks, three trials, `Pass^3`.

Three prompt-support levels, per the measurement rules — only two count:
  supportive  the task plus a hint naming the construct   (a control)
  neutral     the task alone
  competing   the task plus pressure to cut corners

Controls, run before any money is spent: the reference solutions must pass
their own checks (`oracle` ≈ 1.0) and a program that does nothing must fail
them (`noise` ≈ 0.0). A check the oracle fails is a broken check, not a result.

    python evals/agent_builds/run.py controls
    python evals/agent_builds/run.py smoke            # one task, every level
    python evals/agent_builds/run.py full [--name N]  # 20 × 3 levels × 3 trials
    python evals/agent_builds/run.py report NAME

Sandbox limits, stated rather than implied: programs run as a subprocess of
this interpreter in a temporary directory, with dummy keys, a dead proxy and a
timeout. That stops the library reaching a provider and stops most HTTP
clients; it is not a security boundary against hostile code. The programs are
written in answer to these twenty prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from tasks import TASKS                                          # noqa: E402
from yait_aichain.eval import Case, Eval, Report                 # noqa: E402

MODEL = "gpt-5.4-mini"
RUNS = HERE / "runs"
TIMEOUT = 60
DUMMY_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "XAI_API_KEY",
              "PERPLEXITY_API_KEY", "MOONSHOT_API_KEY", "DEEPSEEK_API_KEY",
              "DASHSCOPE_API_KEY", "BFL_API_KEY", "RECRAFT_API_TOKEN", "REVE_API_KEY")

COMPETING = ("This is a quick throwaway script. Keep it as short as possible and skip "
             "anything optional — limits, approvals, error handling and structure can "
             "all be left out.")

INSTRUCTIONS = ("You write Python programs with the yait-aichain library. Your only "
                "reference is the two documents below.\n\nReturn exactly one complete, "
                "runnable Python program in a single ```python code block. Hard-code any "
                "inputs. API keys are already in the environment. Print what the task "
                "asks for.")


def context() -> str:
    return (INSTRUCTIONS + "\n\n===== SKILL.md =====\n" + (ROOT / "SKILL.md").read_text()
            + "\n\n===== llms.txt =====\n" + (ROOT / "llms.txt").read_text())


def load_env() -> None:
    """Real keys for the one real call, read from .env without ever printing them."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*)$", line)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip().strip("'\"")


def ask(user: str, model_name: str) -> "tuple[str, float, int]":
    """One call through the library's own request path, no Skill templating —
    the reference documents contain `{placeholders}` that must reach the model
    verbatim."""
    from yait_aichain import Model
    from yait_aichain.clients._base import APIError
    from yait_aichain.models._usage import attach_cost, extract_usage
    model = Model(model_name, options={"max_tokens": 12000})
    # Typed parts: `Model.to_request` takes messages already normalised, which
    # is what `Skill` hands it. Bare strings are Skill's shorthand, not the
    # model layer's contract.
    messages = [{"role": "system", "parts": [{"type": "text", "text": context()}]},
                {"role": "user", "parts": [{"type": "text", "text": user}]}]
    output = {"modalities": ["text"], "format": {"type": "text"}}
    path, body = model.to_request(messages, output)
    for attempt in range(4):
        try:
            raw = model.client.send(path, body, model.client._auth_headers())
            break
        except APIError as exc:
            if attempt == 3 or getattr(exc, "status", 0) not in (0, 408, 429, 500, 502, 503, 504):
                raise
            time.sleep(4 * (attempt + 1))
    response = json.loads(raw)
    usage = attach_cost(extract_usage(response), model.name)
    return model.from_response(response, output), (usage.cost or 0.0), (usage.total_tokens or 0)


def extract(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    if blocks:
        return max(blocks, key=len)
    return text if re.match(r"\s*(from|import)\s", text or "") else ""


def sandbox(code: str, prefer: str = "") -> dict:
    if not code.strip():
        return {"code": code, "exit": None, "stdout": "", "stderr": "no code in the reply",
                "trace": {"calls": [], "skills": [], "chains": [], "pools": [], "agents": [],
                          "tool_runs": [], "approvals": [], "methods": {}, "risks": {}}}
    with tempfile.TemporaryDirectory() as tmp:
        prog, trace_path = Path(tmp) / "program.py", Path(tmp) / "trace.json"
        prog.write_text(code)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp,
               "PYTHONPATH": f"{HERE / 'sandbox'}{os.pathsep}{ROOT}",
               "AICHAIN_TRACE": str(trace_path), "AICHAIN_STUB_PREFER": prefer,
               "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
               "ALL_PROXY": "http://127.0.0.1:9", "PYTHONDONTWRITEBYTECODE": "1"}
        env.update({k: "sandbox-key" for k in DUMMY_KEYS})
        try:
            proc = subprocess.run([sys.executable, str(prog)], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=TIMEOUT)
            code_, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            code_, out, err = -9, (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""), f"timeout after {TIMEOUT}s"
        trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
    for key, empty in (("calls", []), ("skills", []), ("chains", []), ("pools", []), ("agents", []),
                       ("tool_runs", []), ("approvals", []), ("methods", {}), ("risks", {})):
        trace.setdefault(key, empty)
    return {"code": code, "exit": code_, "stdout": out[-4000:], "stderr": err[-3000:], "trace": trace}


BY_ID = {t.id: t for t in TASKS}


def cases() -> list:
    return [Case(id=t.id, input=t.text, group=t.group, meta={"hint": t.hint, "prefer": t.prefer})
            for t in TASKS]


def score(case, output):
    return BY_ID[case.id].check(output)


def level_arm(level: str, model_name: str):
    def arm(case):
        user = case.input
        if level == "supportive":
            user += "\n\nHint: " + case.meta["hint"]
        elif level == "competing":
            user += "\n\n" + COMPETING
        reply, cost, tokens = ask(user, model_name)
        result = sandbox(extract(reply), case.meta.get("prefer", ""))
        # The raw reply, because the first full run had three attempts with no
        # extractable code and nothing left to say why — truncation, a refusal
        # or an unfenced program all look the same once only the extraction
        # is kept.
        result["reply"] = (reply or "")[-20000:]
        return {"output": result, "cost": cost, "tokens": tokens}
    return arm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["controls", "smoke", "full", "report"])
    ap.add_argument("name", nargs="?", default="")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.mode == "controls":
        stamp = time.strftime("%Y%m%d-%H%M%S")
        oracle = Eval(cases(), {"oracle": lambda c: sandbox(BY_ID[c.id].reference, BY_ID[c.id].prefer)},
                      score, out=RUNS / f"controls-oracle-{stamp}.jsonl", name="oracle").run(concurrency=args.concurrency)
        noise = Eval(cases(), {"noise": lambda c: sandbox("print('hello')\n")},
                     score, out=RUNS / f"controls-noise-{stamp}.jsonl", name="noise").run(concurrency=args.concurrency)
        print(oracle.table())
        failed = [(r.case, r.meta.get("why")) for r in oracle.records if not r.ok]
        for case_id, why in failed:
            print(f"  oracle failed {case_id}: {why}")
        verdict = Report.controls(oracle, noise)
        print(verdict)
        return 0 if verdict["ok"] else 1

    load_env()
    levels = {lvl: level_arm(lvl, args.model) for lvl in ("supportive", "neutral", "competing")}

    if args.mode == "smoke":
        stamp = time.strftime("%Y%m%d-%H%M%S")
        report = Eval(cases()[:1], levels, score, out=RUNS / f"smoke-{stamp}.jsonl",
                      name="smoke").run()
        print(report.table())
        for r in report.records:
            print(f"  {r.arm}: ok={r.ok} cost=${r.cost:.4f} tokens={r.tokens} why={r.meta.get('why')!r} error={r.error!r}")
        return 0

    if args.mode == "full":
        name = args.name or f"{args.model}-{time.strftime('%Y%m%d')}"
        report = Eval(cases(), levels, score, trials=3, out=RUNS / f"{name}.jsonl",
                      name=name).run(concurrency=args.concurrency)
        print(report.table(k=3))
        print(report.by_group(k=3))
        return 0

    report = Report.of(RUNS / f"{args.name}.jsonl")
    report.reject()
    print(report.table(k=3))
    print(report.by_group(k=3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
