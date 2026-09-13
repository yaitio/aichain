"""
Twenty tasks an agent should be able to build with yait-aichain, and how each is checked.

Every task has a neutral text (the task alone), a supportive hint (names the
construct), a reference solution (the oracle control: if the reference fails
its own check, the check is wrong, not the model), and a check that reads what
the program **did** in the sandbox — the trace — not what it printed about
itself. Printing is checked only where the task asks for a value.

Groups follow the primitive the task is about.
"""

import re
import textwrap
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    group: str
    text: str
    hint: str
    reference: str
    check: object
    prefer: str = ""
    meta: dict = field(default_factory=dict)


def _ok(out):
    return out.get("exit") == 0


def _mine(items):
    return [i for i in items if not i.get("internal")]


def _calls(out):
    return out["trace"].get("calls", [])


def _method(out, name):
    return out["trace"].get("methods", {}).get(name, 0)


def _result(ok, why):
    return {"ok": bool(ok), "score": 1.0 if ok else 0.0, "why": "" if ok else why}


def _need(out, *conditions):
    """Each condition is (bool, reason). The first false one is the verdict."""
    if out.get("exit") is None:
        return _result(False, "no program")
    if not _ok(out):
        tail = (out.get("stderr") or "").strip().splitlines()[-1:] or ["(no stderr)"]
        return _result(False, f"exit {out.get('exit')}: {tail[0][:160]}")
    for ok, why in conditions:
        if not ok:
            return _result(False, why)
    return _result(True, "")


def _src(code):
    return textwrap.dedent(code).strip() + "\n"


TASKS = []


def task(**kw):
    kw["reference"] = _src(kw["reference"])
    TASKS.append(Task(**kw))


# ── Skill ───────────────────────────────────────────────────────────────────

task(id="skill-summarise", group="skill",
     text="Write a program that uses an LLM to summarise a short paragraph (hard-code it) in one sentence, and prints the summary.",
     hint="Use Skill(Model(...), prompt='... {text}') and skill.run(variables={...}).",
     reference='''
        from yait_aichain import Model, Skill
        s = Skill(Model("claude-sonnet-4-6"), prompt="Summarise in one sentence: {text}")
        print(s.run(variables={"text": "Solar panels turn sunlight into electricity using silicon cells."}))
     ''',
     check=lambda o: _need(o, (_mine(o["trace"]["skills"]), "no Skill was built"),
                          (_calls(o), "no model call was made"),
                          ("STUB-" in o["stdout"], "the model's answer was not printed")))

task(id="skill-two-providers", group="skill",
     text="Ask the same question to two different AI providers — one Anthropic model and one OpenAI model — and print both answers.",
     hint="Two Skills (or one Skill run with two Models) that differ only in the model name, e.g. Model('claude-sonnet-4-6') and Model('gpt-5.4-mini').",
     reference='''
        from yait_aichain import Model, Skill
        for name in ("claude-sonnet-4-6", "gpt-5.4-mini"):
            print(name, Skill(Model(name), prompt="What is the capital of Peru?").run())
     ''',
     check=lambda o: _need(o, (len({c["provider"] for c in _calls(o)}) >= 2,
                               f"calls went to {sorted({str(c['provider']) for c in _calls(o)})}, not two providers"),
                          (o["stdout"].count("STUB-") >= 2, "both answers were not printed")))

task(id="skill-json-schema", group="skill",
     text="Extract the person's name, email and company from a hard-coded email signature as structured data with exactly those three fields, then print the company.",
     hint="Use output={'format': {'type': 'json_schema', 'schema': {...}}}; run() then returns a dict.",
     reference='''
        from yait_aichain import Model, Skill
        s = Skill(Model("gpt-5.4-mini"), prompt="Extract name, email and company from: {sig}",
                  output={"format": {"type": "json_schema", "schema": {
                      "type": "object",
                      "properties": {"name": {"type": "string"}, "email": {"type": "string"},
                                     "company": {"type": "string"}},
                      "required": ["name", "email", "company"]}}})
        data = s.run(variables={"sig": "Ana Lima | ana@acme.io | Acme Corp"})
        print(data["company"])
     ''',
     check=lambda o: _need(o, (any(s["format"] == "json_schema" for s in _mine(o["trace"]["skills"])),
                               "no Skill asked for json_schema output"),
                          ("STUB-" in o["stdout"], "the company field was not printed")))

task(id="skill-multiturn", group="skill",
     text="In a single Skill, first ask the model for five product name ideas, then — as a follow-up turn in the same conversation — ask it to pick the best one. Print the final answer.",
     hint="input={'messages': [user turn, {'role': 'assistant'} (an assistant turn with no parts marks 'generate here'), user turn]}.",
     reference='''
        from yait_aichain import Model, Skill
        s = Skill(Model("claude-sonnet-4-6"), input={"messages": [
            {"role": "user", "parts": ["Give five names for a reusable water bottle."]},
            {"role": "assistant"},
            {"role": "user", "parts": ["Pick the best one and say why in one line."]},
        ]})
        print(s.run())
     ''',
     check=lambda o: _need(o, (any(s["messages"] >= 3 and "assistant" in s["roles"] for s in _mine(o["trace"]["skills"])),
                               "no single Skill carried a multi-turn conversation"),
                          (len(_calls(o)) >= 2, "the follow-up turn was never generated")))

# ── Tool ────────────────────────────────────────────────────────────────────

task(id="tool-custom", group="tool",
     text="Define a custom tool that converts Celsius to Fahrenheit, call it directly with 100, and print the result.",
     hint="Subclass Tool with name, description, parameters and run(self, input, options=None); call tool.run(input=100).",
     reference='''
        from yait_aichain.tools import Tool
        class CToF(Tool):
            name = "c_to_f"
            description = "Convert Celsius to Fahrenheit."
            parameters = {"type": "object", "properties": {"input": {"type": "number"}}, "required": ["input"]}
            def run(self, input, options=None):
                return input * 9 / 5 + 32
        print(CToF().run(input=100))
     ''',
     check=lambda o: _need(o, (o["trace"]["tool_runs"], "the tool never ran"),
                          ("212" in o["stdout"], "212 was not printed")))

task(id="tool-options", group="tool",
     text="Define a custom tool that returns the first N words of a text, where N is an optional setting defaulting to 3. Call it with N=2 on 'alpha beta gamma delta' and print the result.",
     hint="Declare the setting under parameters['properties']['options'] and call tool(input=..., options={'n': 2}).",
     reference='''
        from yait_aichain.tools import Tool
        class FirstWords(Tool):
            name = "first_words"
            description = "Return the first N words."
            parameters = {"type": "object", "properties": {
                "input": {"type": "string"},
                "options": {"type": "object", "properties": {"n": {"type": "integer"}}}},
                "required": ["input"]}
            def run(self, input, options=None):
                return " ".join(input.split()[: (options or {}).get("n", 3)])
        print(FirstWords()(input="alpha beta gamma delta", options={"n": 2}).output)
     ''',
     check=lambda o: _need(o, (o["trace"]["tool_runs"], "the tool never ran"),
                          ("alpha beta" in o["stdout"] and "alpha beta gamma" not in o["stdout"],
                           "the first two words were not printed")))

# ── Chain ───────────────────────────────────────────────────────────────────

task(id="chain-two-skills", group="chain",
     text="Build a pipeline: an LLM writes a product description for a given product name, then a second LLM step turns that description into a tweet. Print the tweet.",
     hint="Chain(steps=[(describe_skill, 'description'), (tweet_skill, 'tweet')]) where the second prompt uses {description}.",
     reference='''
        from yait_aichain import Chain, Model, Skill
        describe = Skill(Model("gpt-5.4-mini"), prompt="Describe the product {product}.")
        tweet = Skill(Model("claude-sonnet-4-6"), prompt="Turn this into a tweet: {description}")
        print(Chain(steps=[(describe, "description"), (tweet, "tweet")]).run(variables={"product": "a solar lamp"}))
     ''',
     check=lambda o: _need(o, (any(c["kinds"].count("skill") >= 2 for c in o["trace"]["chains"]),
                               "no Chain of two Skill steps"),
                          (_method(o, "Chain.run"), "the chain was never run"),
                          (len(_calls(o)) >= 2 and "STUB-" in _calls(o)[-1]["text"],
                           "the first step's output did not reach the second step")))

task(id="chain-tool-then-skill", group="chain",
     text="Build a pipeline whose first step is a custom tool that counts the words in an input text, and whose second step asks an LLM to comment on that count. Print the comment.",
     hint="A Tool step first, e.g. (WordCount(), 'count', {'input': 'text'}), then a Skill whose prompt uses {count}.",
     reference='''
        from yait_aichain import Chain, Model, Skill
        from yait_aichain.tools import Tool
        class WordCount(Tool):
            name = "word_count"
            description = "Count words."
            parameters = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
            def run(self, input, options=None):
                return str(len(input.split()))
        comment = Skill(Model("gpt-5.4-mini"), prompt="Comment on a text of {count} words.")
        chain = Chain(steps=[(WordCount(), "count", {"input": "text"}), (comment, "comment")])
        print(chain.run(variables={"text": "one two three four five"}))
     ''',
     check=lambda o: _need(o, (any("tool" in c["kinds"] and "skill" in c["kinds"]
                                   and c["kinds"].index("tool") < c["kinds"].index("skill")
                                   for c in o["trace"]["chains"]), "no Chain with a Tool step before a Skill step"),
                          (o["trace"]["tool_runs"], "the tool step never ran"),
                          (_calls(o) and re.search(r"\d", _calls(o)[-1]["text"]),
                           "the count did not reach the LLM step")))

task(id="chain-skip-failure", group="chain",
     text="Build a three-step pipeline whose middle step is a tool that always raises an error. The pipeline must carry on past the failure with a warning and still run its last step, an LLM call. Print the final output.",
     hint="Chain(..., on_step_error='skip') — carries on and emits a warning.",
     reference='''
        from yait_aichain import Chain, Model, Skill
        from yait_aichain.tools import Tool
        class Broken(Tool):
            name = "broken"
            description = "Always fails."
            parameters = {"type": "object", "properties": {"input": {"type": "string"}}}
            def run(self, input=None, options=None):
                raise RuntimeError("down")
        first = Skill(Model("gpt-5.4-mini"), prompt="Name a colour.")
        last = Skill(Model("gpt-5.4-mini"), prompt="Write a haiku about {colour}.")
        chain = Chain(steps=[(first, "colour"), (Broken(), "x"), (last, "haiku")], on_step_error="skip")
        print(chain.run())
     ''',
     check=lambda o: _need(o, (any(c["on_step_error"] == "skip" for c in o["trace"]["chains"]),
                               "no Chain with on_step_error='skip'"),
                          (len(_calls(o)) >= 1 and "STUB-" in o["stdout"], "the last step's output was not printed")))

task(id="chain-save-load", group="chain",
     text="Build a two-step LLM pipeline, save it to a YAML file, load it back from that file, and run the loaded pipeline. Print the result.",
     hint="chain.save('pipeline.yaml'), then Chain.load('pipeline.yaml').run(...).",
     reference='''
        from yait_aichain import Chain, Model, Skill
        a = Skill(Model("gpt-5.4-mini"), prompt="Name a city.")
        b = Skill(Model("gpt-5.4-mini"), prompt="One fact about {city}.")
        Chain(steps=[(a, "city"), (b, "fact")]).save("pipeline.yaml")
        print(Chain.load("pipeline.yaml").run())
     ''',
     check=lambda o: _need(o, (_method(o, "Chain.save"), "the chain was never saved"),
                          (_method(o, "Chain.load"), "the chain was never loaded"),
                          (_method(o, "Chain.run") and len(_calls(o)) >= 2, "the loaded chain did not run both steps")))

task(id="chain-pause-resume", group="chain",
     text="Build a pipeline that drafts a reply with an LLM, pauses until a human approves the draft, then finishes with a tool that 'sends' the approved text. The pause must be resumable from another process. Simulate the approval in the same script, resume, and print what was sent.",
     hint="A Wait step, store=FileStore('runs'), then chain.resume(result.run_id, signal={...}).",
     reference='''
        from yait_aichain import Chain, Model, Skill
        from yait_aichain.state import FileStore, SuspendedResult
        from yait_aichain.tools import Tool, Wait
        class Send(Tool):
            name = "send"
            description = "Send the reply."
            parameters = {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
            def run(self, reply, options=None):
                return f"sent: {reply}"
        draft = Skill(Model("gpt-5.4-mini"), prompt="Draft a polite reply to: {complaint}")
        chain = Chain(steps=[(draft, "reply"), (Wait(reason="approve?", resume_with={"reply": "str"}), "approval"),
                             (Send(), "sent")], store=FileStore("runs"))
        result = chain.run(variables={"complaint": "late delivery"})
        assert isinstance(result, SuspendedResult)
        print(chain.resume(result.run_id, signal={"reply": result.document["variables"]["reply"]}))
     ''',
     check=lambda o: _need(o, (any(c["store"] not in ("InMemoryStore", None) for c in o["trace"]["chains"]),
                               "no Chain with a persistent store"),
                          (_method(o, "Chain.resume"), "the run was never resumed"),
                          (o["trace"]["tool_runs"], "the send tool never ran")))

# ── Pool ────────────────────────────────────────────────────────────────────

task(id="pool-fanout", group="pool",
     text="Classify the sentiment of six hard-coded reviews in parallel, at most three at a time, and print the labels in the original order.",
     hint="Pool(skill, items=[{'review': r} for r in reviews], max_flows=3).run().",
     reference='''
        from yait_aichain import Model, Pool, Skill
        reviews = ["great", "awful", "fine", "loved it", "broken", "ok"]
        s = Skill(Model("gpt-5.4-mini"), prompt="Positive or negative? {review}")
        print(Pool(s, items=[{"review": r} for r in reviews], max_flows=3).run())
     ''',
     check=lambda o: _need(o, (any(p["items"] == 6 and p["max_flows"] == 3 for p in o["trace"]["pools"]),
                               "no Pool of six items with max_flows=3"),
                          (_method(o, "Pool.run") and len(_calls(o)) >= 6, "the six items were not all run")))

task(id="pool-chain-runner", group="pool",
     text="For three topics, run a two-step pipeline — an outline, then a one-paragraph draft from the outline — for each topic in parallel. Print the three drafts.",
     hint="Pool(chain, items=[{'topic': t} for t in topics]) where chain is a Chain of two Skills.",
     reference='''
        from yait_aichain import Chain, Model, Pool, Skill
        outline = Skill(Model("gpt-5.4-mini"), prompt="Outline {topic}.")
        draft = Skill(Model("gpt-5.4-mini"), prompt="Write a paragraph from: {outline}")
        chain = Chain(steps=[(outline, "outline"), (draft, "draft")])
        for d in Pool(chain, items=[{"topic": t} for t in ("tides", "bees", "rust")]).run():
            print(d)
     ''',
     check=lambda o: _need(o, (any(p["runner"] == "Chain" and p["items"] == 3 for p in o["trace"]["pools"]),
                               "no Pool with a Chain runner over three items"),
                          (len(_calls(o)) >= 6, "not every topic ran both steps")))

# ── Agent ───────────────────────────────────────────────────────────────────

_CALC = '''
    from yait_aichain.tools import Tool
    class Calc(Tool):
        name = "calculator"
        description = "Evaluate a multiplication a*b."
        parameters = {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                      "required": ["a", "b"]}
        def run(self, a, b, options=None):
            return a * b
'''


def _stop(o, kind, name=None, value=None):
    for agent in _mine(o["trace"]["agents"]):
        for c in agent["stop_when"]:
            if c["kind"] == kind and (name is None or c["name"] == name) \
                    and (value is None or (c.get("spec") or {}).get("value") == value):
                return True
    return False


task(id="agent-tool-ceiling", group="agent",
     text="Build an agent that can use a calculator tool you define, allowed at most six steps, and ask it what 17 * 23 is. Print the agent's answer and why the run stopped.",
     hint="Agent(model, tools=[...], stop_when=[step_count(6)]); print result.output and result.stopped_by.",
     reference=_CALC + '''
    from yait_aichain import Agent, Model
    from yait_aichain.agent import step_count
    result = Agent(Model("claude-sonnet-4-6"), tools=[Calc()], stop_when=[step_count(6)]).run("What is 17 * 23?")
    print(result.output, result.stopped_by)
     ''',
     check=lambda o: _need(o, (_stop(o, "ceiling", "step_count", 6), "no step_count(6) in stop_when"),
                          (_method(o, "Agent.run"), "the agent was never run"),
                          (o["trace"]["tool_runs"], "the tool never ran"),
                          ("answered" in o["stdout"], "why the run stopped was not printed")))

task(id="agent-verified-stop", group="agent",
     text="Build an agent with a tool that stores a final answer into a Python dict. The run may count as successful only when that answer has been stored — verified by your code, not by the model saying it is done. Print how the run stopped.",
     hint="stop_when=[check(lambda state: 'answer' in store, name='stored'), step_count(10)]; print result.stopped_by.",
     reference='''
        from yait_aichain import Agent, Model
        from yait_aichain.agent import check, step_count
        from yait_aichain.tools import Tool
        store = {}
        class Save(Tool):
            name = "save_answer"
            description = "Store the final answer."
            parameters = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
            def run(self, input, options=None):
                store["answer"] = input
                return "stored"
        agent = Agent(Model("gpt-5.4-mini"), tools=[Save()],
                      stop_when=[check(lambda state: "answer" in store, name="stored"), step_count(10)])
        print(agent.run("What is the tallest mountain? Store the answer.").stopped_by)
     ''',
     check=lambda o: _need(o, (_stop(o, "check"), "no check(...) in stop_when"),
                          (o["trace"]["tool_runs"], "the storing tool never ran"),
                          ("check:" in o["stdout"], "the run did not stop on the verified check")))

task(id="agent-approval", group="agent", prefer="refund",
     text="Build an agent with a refund tool marked as a financial-risk action. Refunds must be approved by a function that allows amounts up to 50 and refuses larger ones with a reason. Ask the agent to refund 80 for order 42. Print the result.",
     hint="Tool risk='financial'; Agent(..., permissions=PermissionPolicy({'financial': 'approve'}), approve=fn) where fn returns True or ApprovalDecision(False, reason).",
     reference='''
        from yait_aichain import Agent, Model
        from yait_aichain.tools import ApprovalDecision, PermissionPolicy, Tool
        class Refund(Tool):
            name = "issue_refund"
            description = "Refund an order."
            risk = "financial"
            parameters = {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]}
            def run(self, amount, options=None):
                return f"refunded {amount}"
        def approve(request):
            return True if request.arguments.get("amount", 0) <= 50 else ApprovalDecision(False, "over 50")
        agent = Agent(Model("gpt-5.4-mini"), tools=[Refund()],
                      permissions=PermissionPolicy({"financial": "approve"}), approve=approve)
        print(agent.run("Refund 80 for order 42.").output)
     ''',
     check=lambda o: _need(o, (any(a["permissions"] and a["approve"] for a in _mine(o["trace"]["agents"])),
                               "no Agent with both permissions= and approve="),
                          ("financial" in o["trace"]["risks"].values(), "no tool marked financial"),
                          (o["trace"]["approvals"], "the approver was never asked")))

task(id="agent-stall-nudge", group="agent",
     text="Build a research agent with a search-like tool you define. It must be told to change approach when it stops making progress for three steps, and must not run more than ten steps. Run it on any question and print the output.",
     hint="stop_when=[step_count(10), stalled(3)].",
     reference='''
        from yait_aichain import Agent, Model
        from yait_aichain.agent import stalled, step_count
        from yait_aichain.tools import Tool
        class Search(Tool):
            name = "search"
            description = "Search notes."
            parameters = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
            def run(self, input, options=None):
                return f"notes about {input}"
        agent = Agent(Model("gpt-5.4-mini"), tools=[Search()], stop_when=[step_count(10), stalled(3)])
        print(agent.run("Why is the sky blue?").output)
     ''',
     check=lambda o: _need(o, (_stop(o, "nudge"), "no stall/repetition nudge in stop_when"),
                          (_stop(o, "ceiling", "step_count", 10), "no step_count(10) in stop_when")))

task(id="agent-in-chain", group="agent",
     text="Build a pipeline whose first step is an LLM that turns a topic into a research question, and whose second step is an agent — with one tool you define — that answers that question. Print the agent's answer.",
     hint="Chain(steps=[(question_skill, 'task'), (agent, 'answer')]) — an Agent step reads its task from the 'task' variable.",
     reference='''
        from yait_aichain import Agent, Chain, Model, Skill
        from yait_aichain.agent import step_count
        from yait_aichain.tools import Tool
        class Lookup(Tool):
            name = "lookup"
            description = "Look something up."
            parameters = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
            def run(self, input, options=None):
                return f"facts on {input}"
        question = Skill(Model("gpt-5.4-mini"), prompt="Turn {topic} into one research question.")
        agent = Agent(Model("claude-sonnet-4-6"), tools=[Lookup()], stop_when=[step_count(5)])
        print(Chain(steps=[(question, "task"), (agent, "answer")]).run(variables={"topic": "coral reefs"}))
     ''',
     check=lambda o: _need(o, (any("skill" in c["kinds"] and "agent" in c["kinds"] for c in o["trace"]["chains"]),
                               "no Chain with a Skill step and an Agent step"),
                          (o["trace"]["tool_runs"], "the agent inside the chain never used its tool")))

task(id="agent-external-loop", group="agent",
     text="Drive an agent's loop yourself, step by step, instead of calling run(): execute every tool call it asks for and feed each result back until it gives an answer. Use one tool you define. Print the final answer.",
     hint="agent.opening(task), agent.new_state(), agent.step(messages, state), messages.append(reply.as_turn()), agent.execute(call, state), tool_result_turn(call.id, result).",
     reference='''
        from yait_aichain import Agent, Model
        from yait_aichain.models import ToolCallRequest, tool_result_turn
        from yait_aichain.tools import Tool
        class Echo(Tool):
            name = "echo"
            description = "Echo text."
            parameters = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
            def run(self, input, options=None):
                return input
        agent = Agent(Model("gpt-5.4-mini"), tools=[Echo()])
        state, messages = agent.new_state(), agent.opening("Echo 'hi' then answer.")
        while True:
            reply = agent.step(messages, state)
            if not isinstance(reply, ToolCallRequest):
                print(reply)
                break
            messages.append(reply.as_turn())
            for call in reply.calls:
                result, error = agent.execute(call, state)
                messages.append(tool_result_turn(call.id, error or result))
     ''',
     check=lambda o: _need(o, (_method(o, "Agent.step") >= 2, "step() was not called until the answer"),
                          (_method(o, "Agent.execute") >= 1, "no tool call was executed"),
                          (not _method(o, "Agent.run"), "the task was to drive the loop, but run() was called"),
                          ("STUB-" in o["stdout"], "the final answer was not printed")))

task(id="agent-events", group="agent",
     text="Build an agent with one tool, attach something that records every event of the run, and after the run print the list of event types.",
     hint="hooks=[Tracer()] on the Agent; after run(), print [e.type for e in tracer.events].",
     reference='''
        from yait_aichain import Agent, Model, Tracer
        from yait_aichain.tools import Tool
        class Clock(Tool):
            name = "clock"
            description = "Tell the time."
            parameters = {"type": "object", "properties": {"input": {"type": "string"}}}
            def run(self, input=None, options=None):
                return "12:00"
        tracer = Tracer()
        Agent(Model("gpt-5.4-mini"), tools=[Clock()], hooks=[tracer]).run("What time is it?")
        print([e.type for e in tracer.events])
     ''',
     check=lambda o: _need(o, (any(a["hooks"] for a in _mine(o["trace"]["agents"])), "no hook attached to an Agent"),
                          ("tool_call.started" in o["stdout"], "the event types were not printed")))

assert len(TASKS) == 20, len(TASKS)
assert len({t.id for t in TASKS}) == 20
