<!-- g:tool -->
# `imageOpenai` — `Image`

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

## An image model is a tool

```python
from yait_aichain.tools import image

picture = image("openai", model="gpt-image-1.5", output_dir="images/")
out = picture.run("a lighthouse at dawn, watercolor", options={"aspect_ratio": "16:9"})
out["images"][0]["path"], out["cost"]
picture.run("make the sky stormy", images=[out["images"][0]["path"]])   # an edit
Agent(model, tools=[picture])           # the agent draws when it decides to
```

`image(service, model=…)` takes `openai`, `google`, `bfl`, `xai`, `qwen` or
`recraft`, and a model of that provider — `registry.models(task=
"text-to-image")` lists them, `"image-to-image"` the ones that edit. A model is
what can be an agent's mind; an image model cannot, so since 3.9.0 it is a tool.
The request is the one `Skill` and `Model` built before: the same option
vocabulary, converted where a provider takes another spelling (`aspect_ratio` to
pixels, `size` to a ratio), and a notice where it cannot honour one.

- **`images`** — file paths or URLs — makes the call an edit, or gives it
  references.
- **`output_dir`** is where an image is saved, and the code's choice: a model
  never picks where to write. Without it the image goes back in the result
  (`images[0]["base64"]`), and an agent is told one was made and nothing more.
- **What a model reads** is a line — "Made a png image with gpt-image-1.5, saved
  to images/…" — never the bytes, with any notice about an option that was
  adapted or declined.
- **`cost`** is the provider's reported usage at the registry's rate where both
  exist — OpenAI's image models — and `None` with the reason in `cost_basis`
  where they do not.

`Skill(Model("gpt-image-1.5"), output={"modalities": ["image"], …})` — the way
images were made until now — still works and warns; it raises in 4.0.
Which model for what, with prices and speeds: `docs/design/image-edit.md`.
