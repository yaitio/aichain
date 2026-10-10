# API: image

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/media.md).

<!-- g:tool -->
## `image()`

```python
image(
    service: str,
    *,
    model: str,
    api_key: str | None = None,
    output_dir: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> yait_aichain.tools.image._tool.Image
```

An image tool for *service* — one of `SERVICES` — and *model*.

The model is a setting: an image's look, price and what it can do
(transparency, vectors, edits) come with it. `output_dir` is where images
are saved; without it they go back to the code in the result.

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `imageOpenai` — `Image`

An image model as a tool — generation from a prompt, and editing when the call carries images. Made by `tools.image` — `image("openai", model="gpt-image-1.5")`.

| | |
|---|---|
| Import | `from yait_aichain.tools import Image` |
| Risk class | `draft` |

```python
Image(
    model: str,
    *,
    api_key: str | None = None,
    output_dir: str | None = None,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | What to draw, or what to change in the images given. |
| `images` | `list[string]` |  | Images to edit or take as reference — file paths or URLs. None to generate. |
| `options.size` | `string` |  | Pixels, e.g. '1024x1024'; converted to the nearest shape where a provider takes a ratio. |
| `options.aspect_ratio` | `string` |  | e.g. '16:9'; converted to a size where a provider takes pixels. |
| `options.quality` | `string` |  | The provider's quality tier, e.g. 'low', 'medium', 'high'. |
| `options.background` | `string` |  | 'transparent' where the model can, else 'opaque'. |
| `options.output_format` | `string` |  | png, jpeg or webp. |
| `options.compression` | `integer` |  | 0-100, for jpeg and webp. |
| `options.seed` | `integer` |  | For a repeatable image where the provider honours it. |
| `options.fidelity` | `string` |  | How closely an edit keeps its input: 'low' or 'high'. |

```python
result = Image("<model>")(
    input="…",
    options={"size": …},
)
```
<!-- /g:tool -->
