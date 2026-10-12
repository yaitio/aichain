# Speech, transcription, images

Text in, text out is a model's job. Turning text into audio, audio into text,
or a prompt into a picture is done by services that write no prose and cannot
reason — so here they are tools: things your code, a chain or an agent call,
never an agent's mind.

```python
from yait_aichain.tools import speech, transcribe, image

speech("openai").run("Your order has shipped.", {"voice": "nova"})     # the audio file's path
transcribe("openai").run("call.mp3", {"language": "ru"})               # the transcript

painter = image("openai", model="gpt-image-1.5", output_dir="images")
out = painter.run("a lighthouse at dawn, watercolor", options={"aspect_ratio": "16:9"})
out["images"][0]["path"], out["cost"]
```

→ Every parameter: [API: speech and transcribe](../api/tools/speech.md) · [API: image](../api/tools/image.md)

---

## Speech and transcription

`speech(service)` and `transcribe(service)` take `openai`, `google`, `xai` or
`qwen`. A voice, a speed, a language and a format are asked per call; the
service and the model are fixed when the tool is made. Speech returns the path
of the file it wrote, which is what a chain passes on and what a model can be
told.

## Images: why a tool and not a model

A model, in this library, is what can be an agent's mind: it reads, reasons
and calls tools. An image model does none of that, so since 3.9.0 it is
reached as a tool — `image(service, model=…)` with `openai`, `google`, `bfl`,
`xai`, `qwen` or `recraft`. An agent given one draws when it decides a picture
is needed, the way it searches when it decides it needs to know something.

The request underneath is the one the model layer always built, with one
vocabulary of options for every provider: `size` and `aspect_ratio` are
converted into each other where a provider takes the other, and an option a
provider cannot honour — a `seed`, a transparent background — comes back as a
notice in the result, not as a silent difference in the picture.

Passing `images=[…]` — paths or URLs — makes the call an edit, or gives the
model references. `registry.models(task="image-to-image")` lists the models
that edit.

## Where the bytes go

`output_dir` is where images are saved, and it is a setting of the code: a
model never picks where to write on your disk. Without it, the image comes
back in the result as base64, for the code to handle.

What a model reads is a line — "Made a png image with gpt-image-1.5, saved to
images/…" — and any notice about an option that was adapted. Never the bytes:
a megabyte of base64 in an agent's context costs as much as it is useless.

## What a picture costs

Image models are priced two ways, and the result says which it used:

- **by the token**, where the provider reports usage — OpenAI's image models:
  the reported tokens at the registry's rates, image input apart from text;
- **by the picture**, where it does not — xAI, Google, FLUX, Recraft, Qwen:
  the registry's price per image, at the base size, so a larger picture on a
  tiered model costs more than this says.

`cost_basis` names the basis, and `cost` is `None` — with the reason — for a
model the registry has no price for. → [Cost and budget](../control/cost-and-budget.md)

A `Skill` asked for an image output raises, naming `image(...)`: a Skill makes
text.

---

▶ [`examples/03_skill_multimodal.py`](../../examples/03_skill_multimodal.py) ·
[`examples/19_image_edit.py`](../../examples/19_image_edit.py)
