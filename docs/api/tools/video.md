<!-- g:tool -->
## `video()`

```python
video(
    service: str,
    *,
    model: str,
    api_key: str | None = None,
    output_dir: str | None = None,
    timeout: float = 600.0,
    name: str | None = None,
    description: str | None = None,
) -> yait_aichain.tools.video._tool.Video
```

A video tool for *service* — one of `SERVICES` — and *model*.

`output_dir` is where videos are saved; without it they go back to the code
in the result. `timeout` is how long a call waits for the video, in seconds.

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `videoGoogle` — `Video`

A video model as a tool — from a prompt, or from an image as the first frame. Made by `tools.video` — `video("google", model="veo-3.1-fast-generate-preview")`.

| | |
|---|---|
| Import | `from yait_aichain.tools import Video` |
| Risk class | `draft` |

```python
Video(
    model: str,
    *,
    api_key: str | None = None,
    output_dir: str | None = None,
    timeout: float = 600.0,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | What the video shows. |
| `images` | `list[string]` |  | One image as the first frame — a file path or a URL. None for text alone. |
| `options.duration` | `integer` |  | Seconds; the nearest length the model makes. |
| `options.resolution` | `string` |  | 480p, 720p, 1080p or 4k; the nearest the model has. |
| `options.aspect_ratio` | `string` |  | e.g. '16:9'; the nearest the model takes. |
| `options.size` | `string` |  | Pixels, e.g. '1280x720'; becomes a ratio and a resolution. |
| `options.seed` | `integer` |  | For a repeatable video where the model honours it. |
| `options.audio` | `boolean` |  | With sound or without, where the model lets you choose. |
| `options.negative_prompt` | `string` |  | What the video should not show, where the model takes it. |

```python
result = Video("<model>")(
    input="…",
    options={"duration": …},
)
```
<!-- /g:tool -->
