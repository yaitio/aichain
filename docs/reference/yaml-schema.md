# YAML schema

Skills and Chains can be saved to YAML and reloaded later. API keys are **never** written — they are always resolved from environment variables at load time.

Requires `pyyaml`:
```bash
pip install pyyaml
```

---

## Skill YAML

### Save / load

```python
skill.save("skills/translation.yaml")

from skills import Skill
skill = Skill.load("skills/translation.yaml")
```

### Schema

```yaml
# Top-level fields — all present except variables/options which are omitted when empty.

model_name: "claude-sonnet-4-6"      # required — full model ID
name: "translation_skill"            # optional — human label
description: "Translate text"        # optional

input:                               # required — universal message template
  messages:
    - role: system
      parts:
        - type: text
          text: "You are a professional translator."
    - role: user
      parts:
        - type: text
          text: "Translate this text to {target_language}:\n\n{text}"

output:                              # required — output spec
  modalities:
    - text
  format:
    type: text                       # text | json | json_schema

# Optional — default variable values merged at run time
variables:
  target_language: "French"

# Optional — model/inference options
options:
  temperature: 0.3
  max_tokens: 2000
  reasoning: "low"                   # low | medium | high
```

### `input` object

Mirrors the universal message format accepted by `Skill(input=…)`:

```yaml
input:
  messages:
    - role: system | user | assistant
      parts:
        - type: text
          text: "String with {placeholder} tokens"
        - type: image_url              # multimodal — image from URL
          image_url:
            url: "https://…"
        - type: image_base64           # multimodal — inline base64
          image_base64:
            data: "…"
            media_type: "image/png"
```

### `output` object

```yaml
output:
  modalities:
    - text                    # "text" or "image"
  format:
    type: text                # plain text — run() returns str

  # OR
  format:
    type: json                # run() returns dict parsed from JSON
    # No schema — model decides structure

  # OR
  format:
    type: json_schema         # run() returns validated dict
    json_schema:
      name: "TranslationResult"
      schema:
        type: object
        properties:
          translated_text:
            type: string
          detected_source_lang:
            type: string
        required:
          - translated_text
      strict: true
```

---

## Chain YAML

### Save / load

```python
chain.save("chains/analyse_and_translate.yaml")

from chain import Chain
chain = Chain.load("chains/analyse_and_translate.yaml")
result = chain.run(variables={"topic": "AI safety", "language": "French"})
```

### Schema

```yaml
# Chain-level metadata
name: "analyse_and_translate"           # optional
description: "Analyse then translate"   # optional
on_step_error: "raise"                  # raise (default) | stop | skip
variables:                              # optional — chain-level defaults
  language: "French"

steps:
  - kind: skill                         # "skill" | "tool" | "agent"
    output_key: analysis                # name in accumulated dict
    input_map:                          # optional — rename accumulated vars
      text: raw_content                 # passes accumulated["raw_content"] as "text"
    skill:
      model_name: "gpt-4o"
      name: "analyser"
      input:
        messages:
          - role: user
            parts:
              - type: text
                text: "Analyse:\n\n{text}"
      output:
        modalities: [text]
        format:
          type: text
      variables: {}
      options:
        temperature: 0.5

  - kind: tool
    output_key: translated
    input_map:
      text: analysis                    # pass "analysis" as "text" to the tool
    tool:
      class: "tools.deepl.DeepLTranslateTool"    # fully-qualified class path
      init_args: {}                              # serialisable constructor kwargs

  - kind: agent
    output_key: final_report
    options:
      task_key: analysis                # read task from accumulated["analysis"]
      output_field: output              # extract AgentResult.output
    agent:
      class: "agent._agent.Agent"
      orchestrator: "claude-opus-4-6"
      mode: agile
      max_steps: 8
      max_attempts: 3
      max_tokens: 50000
      verbose: 1
      persona: "You are a senior analyst."
      tools:
        - "tools.search.brave_search.BraveSearchTool"
        - "tools.markitdown.MarkItDownTool"
```

---

## Step reference

### Skill step

| Field | Required | Notes |
|---|---|---|
| `kind` | ✓ | `"skill"` |
| `output_key` | ✓ | Variable name in `accumulated`. |
| `input_map` | | `{param: accumulated_var}` rename map. |
| `options` | | Step-level options (currently unused for Skill steps). |
| `skill.model_name` | ✓ | Full model ID, e.g. `"gpt-4o"`. |
| `skill.input` | ✓ | Universal message template. |
| `skill.output` | ✓ | Output spec with `modalities` + `format`. |
| `skill.name` | | Optional label. |
| `skill.description` | | Optional description. |
| `skill.variables` | | Default variable values. |
| `skill.options` | | Inference options: `temperature`, `max_tokens`, `reasoning`, etc. |

### Tool step

| Field | Required | Notes |
|---|---|---|
| `kind` | ✓ | `"tool"` |
| `output_key` | ✓ | Variable name. If the tool returns a dict, all keys are merged. |
| `input_map` | | Rename map. |
| `tool.class` | ✓ | Fully-qualified Python class path, e.g. `"tools.deepl.DeepLTranslateTool"`. |
| `tool.init_args` | | Serialisable constructor kwargs (no API keys). |

### Agent step

| Field | Required | Notes |
|---|---|---|
| `kind` | ✓ | `"agent"` |
| `output_key` | ✓ | Variable name for the extracted field. |
| `options.task_key` | | Accumulated variable holding the agent's task. Default `"task"`. |
| `options.output_field` | | `AgentResult` attribute to extract. Default `"output"`. |
| `agent.class` | ✓ | Fully-qualified class path. |
| `agent.orchestrator` | | Model name for the agent's orchestrator. |
| `agent.mode` | | `"waterfall"` or `"agile"`. |
| `agent.max_steps` | | Integer. |
| `agent.max_attempts` | | Integer. |
| `agent.max_tokens` | | Integer. |
| `agent.verbose` | | `0`, `1`, or `2`. |
| `agent.persona` | | Prepended to all agent prompts. |
| `agent.tools` | | List of fully-qualified class paths. API keys resolved at load time. |

---

## Notes on serialisation

- **API keys** are never written. On `Chain.load()` / `Skill.load()`, each model or tool constructor reads its key from the matching env var, or from the optional `api_key=` argument.
- **Tool `init_args`** must be JSON-serialisable. Pass non-serialisable state (e.g. LLM clients for `MarkItDownTool`) by constructing the tool in Python before placing it in a Chain.
- **Agent tools** are stored as class paths only; at load time each class is imported and instantiated with no arguments (API keys come from env vars).
- **Parent directories** are created automatically by both `save()` methods.

---

## See also

- [Chain primitives](../primitives/chain.md) — the full step specification in Python.
- [Skill primitives](../primitives/skills.md) — the input/output format.
- [Environment variables](environment-variables.md) — keys resolved at load time.
