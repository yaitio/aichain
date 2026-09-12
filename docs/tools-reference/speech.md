<!-- g:tool -->
# Speech: text ↔ audio

## `ttsOpenAI`

Text-to-speech via the OpenAI TTS API.

| | |
|---|---|
| Import | `from yait_aichain.tools import ttsOpenAI` |
| Risk class | `write` |
| Key | `OPENAI_API_KEY` |

```python
ttsOpenAI(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Text to synthesise into speech. |
| `options.voice` | `string` |  | Voice name or ID (provider-specific). |
| `options.model` | `string` |  | Model or quality tier (provider-specific). |
| `options.speed` | `number` |  | Playback speed multiplier. 0.25–4.0, default 1.0. |
| `options.format` | `string` |  | Audio output format. Default: mp3. One of `mp3`, `wav`, `opus`, `flac`, `pcm`. |
| `options.output_path` | `string` |  | Destination file path. Auto-generated inside ./audio/ when omitted. |

```python
result = ttsOpenAI()(
    input="…",
    options={"voice": …},
)
```

## `ttsGoogle`

Text-to-speech via Google Cloud Text-to-Speech.

| | |
|---|---|
| Import | `from yait_aichain.tools import ttsGoogle` |
| Risk class | `write` |
| Key | `GOOGLE_AI_API_KEY` or `GOOGLE_API_KEY` |

```python
ttsGoogle(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Text to synthesise into speech. |
| `options.voice` | `string` |  | Voice name or ID (provider-specific). |
| `options.speed` | `number` |  | Playback speed multiplier. 0.25–4.0, default 1.0. |
| `options.format` | `string` |  | Audio output format. Default: mp3. One of `mp3`, `wav`, `opus`, `flac`, `pcm`. |
| `options.language` | `string` |  | BCP-47 language code, e.g. 'en-US'. Google only — it is required there and the other endpoints have no such field, so they do not declare this option at all. |
| `options.output_path` | `string` |  | Destination file path. Auto-generated inside ./audio/ when omitted. |

```python
result = ttsGoogle()(
    input="…",
    options={"voice": …},
)
```

## `ttsXAI`

Text-to-speech via the xAI TTS API.

| | |
|---|---|
| Import | `from yait_aichain.tools import ttsXAI` |
| Risk class | `write` |
| Key | `XAI_API_KEY` |

```python
ttsXAI(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Text to synthesise into speech. |
| `options.voice` | `string` |  | Voice name or ID (provider-specific). |
| `options.model` | `string` |  | Model or quality tier (provider-specific). |
| `options.speed` | `number` |  | Playback speed multiplier. 0.25–4.0, default 1.0. |
| `options.format` | `string` |  | Audio output format. Default: mp3. One of `mp3`, `wav`, `opus`, `flac`, `pcm`. |
| `options.output_path` | `string` |  | Destination file path. Auto-generated inside ./audio/ when omitted. |

```python
result = ttsXAI()(
    input="…",
    options={"voice": …},
)
```

## `ttsQwen`

Text-to-speech via the Alibaba DashScope CosyVoice / Qwen3-TTS API.

| | |
|---|---|
| Import | `from yait_aichain.tools import ttsQwen` |
| Risk class | `write` |
| Key | `DASHSCOPE_API_KEY` |

```python
ttsQwen(
    api_key: str | None = None,
    region: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Text to synthesise into speech. |
| `options.voice` | `string` |  | Voice name or ID (provider-specific). |
| `options.model` | `string` |  | Model or quality tier (provider-specific). |
| `options.format` | `string` |  | Audio output format. Default: mp3. One of `mp3`, `wav`, `opus`, `flac`, `pcm`. |
| `options.output_path` | `string` |  | Destination file path. Auto-generated inside ./audio/ when omitted. |
| `options.region` | `string` |  | DashScope region for this call. Default: ap. One of `ap`, `us`, `cn`, `hk`. |

```python
result = ttsQwen()(
    input="…",
    options={"voice": …},
)
```

## `sttOpenAI`

Speech-to-text via the OpenAI Whisper API.

| | |
|---|---|
| Import | `from yait_aichain.tools import sttOpenAI` |
| Risk class | `write` |
| Key | `OPENAI_API_KEY` |

```python
sttOpenAI(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Path to the audio file to transcribe. |
| `options.language` | `string` |  | Language hint (ISO-639-1 or BCP-47), e.g. 'en', 'en-US'. Improves accuracy. Auto-detected when omitted. |
| `options.model` | `string` |  | Model or tier to use (provider-specific). |
| `options.timestamps` | `boolean` |  | Include word-level timestamps. When True the output is a JSON string. Default: false. |
| `options.prompt` | `string` |  | Context hint — domain vocabulary or spelling guidance. Supported by OpenAI and xAI. |
| `options.format` | `string` |  | Output format. ``text`` (default) — plain transcript. ``json`` — includes timestamps and segment metadata. One of `text`, `json`. |

```python
result = sttOpenAI()(
    input="…",
    options={"language": …},
)
```

## `sttGoogle`

Speech-to-text via Google Cloud Speech-to-Text.

| | |
|---|---|
| Import | `from yait_aichain.tools import sttGoogle` |
| Risk class | `write` |
| Key | `GOOGLE_AI_API_KEY` or `GOOGLE_API_KEY` |

```python
sttGoogle(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Path to the audio file to transcribe. |
| `options.language` | `string` |  | Language hint (ISO-639-1 or BCP-47), e.g. 'en', 'en-US'. Improves accuracy. Auto-detected when omitted. |
| `options.model` | `string` |  | Model or tier to use (provider-specific). |
| `options.timestamps` | `boolean` |  | Include word-level timestamps. When True the output is a JSON string. Default: false. |
| `options.format` | `string` |  | Output format. ``text`` (default) — plain transcript. ``json`` — includes timestamps and segment metadata. One of `text`, `json`. |

```python
result = sttGoogle()(
    input="…",
    options={"language": …},
)
```

## `sttXAI`

Speech-to-text via the xAI STT API.

| | |
|---|---|
| Import | `from yait_aichain.tools import sttXAI` |
| Risk class | `write` |
| Key | `XAI_API_KEY` |

```python
sttXAI(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Path to the audio file to transcribe. |
| `options.language` | `string` |  | Language hint (ISO-639-1 or BCP-47), e.g. 'en', 'en-US'. Improves accuracy. Auto-detected when omitted. |
| `options.model` | `string` |  | Model or tier to use (provider-specific). |
| `options.timestamps` | `boolean` |  | Include word-level timestamps. When True the output is a JSON string. Default: false. |
| `options.prompt` | `string` |  | Context hint — domain vocabulary or spelling guidance. Supported by OpenAI and xAI. |
| `options.format` | `string` |  | Output format. ``text`` (default) — plain transcript. ``json`` — includes timestamps and segment metadata. One of `text`, `json`. |

```python
result = sttXAI()(
    input="…",
    options={"language": …},
)
```

## `sttQwen`

Speech-to-text via the Alibaba DashScope ASR API.

| | |
|---|---|
| Import | `from yait_aichain.tools import sttQwen` |
| Risk class | `write` |
| Key | `DASHSCOPE_API_KEY` |

```python
sttQwen(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Path to the audio file to transcribe. |
| `options.language` | `string` |  | Language hint (ISO-639-1 or BCP-47), e.g. 'en', 'en-US'. Improves accuracy. Auto-detected when omitted. |
| `options.model` | `string` |  | Model or tier to use (provider-specific). |
| `options.timestamps` | `boolean` |  | Include word-level timestamps. When True the output is a JSON string. Default: false. |
| `options.prompt` | `string` |  | Context hint — domain vocabulary or spelling guidance. Supported by OpenAI and xAI. |
| `options.format` | `string` |  | Output format. ``text`` (default) — plain transcript. ``json`` — includes timestamps and segment metadata. One of `text`, `json`. |

```python
result = sttQwen()(
    input="…",
    options={"language": …},
)
```
<!-- /g:tool -->
