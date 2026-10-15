# API: speech

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/media.md).

<!-- g:tool -->
## `speech()`

```python
speech(
    service: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    region: str | None = None,
    name: str | None = None,
    description: str | None = None,
    output_dir: str | None = None,
) -> yait_aichain.tools.speech._tts.Speech
```

A text-to-speech tool for *service* — one of `SERVICES`.

`model` defaults to the provider's (Google's has one and takes none);
`region` is Qwen's alone; `name` and `description` default to the
provider's; `api_key` to its environment variable. `output_dir` is
where the audio files go — the system's temporary directory by default.

```python
speech("openai").run("Hello!", {"voice": "nova"})    # the path of the audio file
```

## `transcribe()`

```python
transcribe(
    service: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> yait_aichain.tools.transcribe._stt.Transcription
```

A speech-to-text tool for *service* — one of `SERVICES`.

`model` defaults to the provider's and is a setting of the tool, not an
option of a call; `api_key` defaults to the provider's environment
variable.

```python
transcribe("openai").run("call.mp3", {"language": "ru"})    # the transcript
```

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `ttsOpenAI`

Text-to-speech via the OpenAI TTS API.

| | |
|---|---|
| Import | `from yait_aichain.tools import ttsOpenAI` |
| Risk class | `draft` |
| Key | `OPENAI_API_KEY` |

```python
ttsOpenAI(
    api_key: str | None = None,
    model: str | None = None,
    *,
    output_dir: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Text to synthesise into speech. |
| `options.voice` | `string` |  | Voice name or ID (provider-specific). |
| `options.speed` | `number` |  | Playback speed multiplier. 0.25–4.0, default 1.0. |
| `options.format` | `string` |  | Audio output format. Default: mp3. One of `mp3`, `wav`, `opus`, `flac`, `pcm`. |

```python
result = ttsOpenAI()(
    input="…",
    options={"voice": …},
)
```

### `ttsGoogle`

Text-to-speech via Google Cloud Text-to-Speech, over its REST API.

| | |
|---|---|
| Import | `from yait_aichain.tools import ttsGoogle` |
| Risk class | `draft` |
| Key | `GOOGLE_AI_API_KEY` or `GOOGLE_API_KEY` or `GOOGLE_CLOUD_API_KEY` |

```python
ttsGoogle(
    api_key: str | None = None,
    model: str | None = None,
    *,
    output_dir: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Text to synthesise into speech. |
| `options.voice` | `string` |  | Voice name or ID (provider-specific). |
| `options.speed` | `number` |  | Playback speed multiplier. 0.25–4.0, default 1.0. |
| `options.format` | `string` |  | Audio output format. Default: mp3. One of `mp3`, `wav`, `opus`, `flac`, `pcm`. |
| `options.language` | `string` |  | BCP-47 language code, e.g. 'en-US'. Google only — it is required there and the other endpoints have no such field, so they do not declare this option at all. |

```python
result = ttsGoogle()(
    input="…",
    options={"voice": …},
)
```

### `ttsXAI`

Text-to-speech via the xAI TTS API.

| | |
|---|---|
| Import | `from yait_aichain.tools import ttsXAI` |
| Risk class | `draft` |
| Key | `XAI_API_KEY` |

```python
ttsXAI(
    api_key: str | None = None,
    model: str | None = None,
    *,
    output_dir: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Text to synthesise into speech. |
| `options.voice` | `string` |  | Voice name or ID (provider-specific). |
| `options.speed` | `number` |  | Playback speed multiplier. 0.25–4.0, default 1.0. |
| `options.format` | `string` |  | Audio output format. Default: mp3. One of `mp3`, `wav`, `opus`, `flac`, `pcm`. |

```python
result = ttsXAI()(
    input="…",
    options={"voice": …},
)
```

### `ttsQwen`

Text-to-speech via the Alibaba DashScope Qwen3-TTS API; the audio URL it answers with is downloaded and saved locally.

| | |
|---|---|
| Import | `from yait_aichain.tools import ttsQwen` |
| Risk class | `draft` |
| Key | `DASHSCOPE_API_KEY` |

```python
ttsQwen(
    api_key: str | None = None,
    region: str | None = None,
    model: str | None = None,
    *,
    output_dir: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Text to synthesise into speech. |
| `options.voice` | `string` |  | Voice name or ID (provider-specific). |
| `options.format` | `string` |  | Audio output format. Default: mp3. One of `mp3`, `wav`, `opus`, `flac`, `pcm`. |

```python
result = ttsQwen()(
    input="…",
    options={"voice": …},
)
```

### `sttOpenAI`

Speech-to-text via the OpenAI Whisper API.

| | |
|---|---|
| Import | `from yait_aichain.tools import sttOpenAI` |
| Risk class | `external` |
| Key | `OPENAI_API_KEY` |

```python
sttOpenAI(
    api_key: str | None = None,
    model: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Path to the audio file to transcribe. |
| `options.language` | `string` |  | Language hint (ISO-639-1 or BCP-47), e.g. 'en', 'en-US'. Improves accuracy. Auto-detected when omitted. |
| `options.timestamps` | `boolean` |  | Include word-level timestamps. When True the output is a JSON string. Default: false. |
| `options.prompt` | `string` |  | Context hint — domain vocabulary or spelling guidance. Supported by OpenAI and xAI. |
| `options.format` | `string` |  | Output format. ``text`` (default) — plain transcript. ``json`` — includes timestamps and segment metadata. One of `text`, `json`. |

```python
result = sttOpenAI()(
    input="…",
    options={"language": …},
)
```

### `sttGoogle`

Speech-to-text via Google Cloud Speech-to-Text, over its REST API.

| | |
|---|---|
| Import | `from yait_aichain.tools import sttGoogle` |
| Risk class | `external` |
| Key | `GOOGLE_AI_API_KEY` or `GOOGLE_API_KEY` or `GOOGLE_CLOUD_API_KEY` |

```python
sttGoogle(
    api_key: str | None = None,
    model: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Path to the audio file to transcribe. |
| `options.language` | `string` |  | Language hint (ISO-639-1 or BCP-47), e.g. 'en', 'en-US'. Improves accuracy. Auto-detected when omitted. |
| `options.timestamps` | `boolean` |  | Include word-level timestamps. When True the output is a JSON string. Default: false. |
| `options.format` | `string` |  | Output format. ``text`` (default) — plain transcript. ``json`` — includes timestamps and segment metadata. One of `text`, `json`. |

```python
result = sttGoogle()(
    input="…",
    options={"language": …},
)
```
<!-- /g:tool -->
