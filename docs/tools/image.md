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
Which model for what, with prices and speeds: [`design/image-edit.md`](../../design/image-edit.md).
