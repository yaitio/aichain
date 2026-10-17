# Speech, transcription, images, video

Text in, text out is a model's job. Turning text into audio, audio into text,
or a prompt into a picture or a clip is done by services that write no prose and cannot
reason — so here they are tools: things your code, a chain or an agent call,
never an agent's mind.

```python
from yait_aichain.tools import speech, transcribe, image, video

speech("openai").run("Your order has shipped.", {"voice": "nova"})     # the audio file's path
transcribe("openai").run("call.mp3", {"language": "ru"})               # the transcript

painter = image("openai", model="gpt-image-1.5", output_dir="images")
out = painter.run("a lighthouse at dawn, watercolor", options={"aspect_ratio": "16:9"})
out["images"][0]["path"], out["cost"]

clip = video("google", model="veo-3.1-fast-generate-preview", output_dir="videos")
out = clip.run("a paper boat on a puddle", options={"duration": 6})   # waits minutes
out["videos"][0]["path"], out["cost"]
```

→ Every parameter: [API: speech and transcribe](../api/tools/speech.md) · [API: image](../api/tools/image.md) · [API: video](../api/tools/video.md)

---

## Speech and transcription

`speech(service)` takes `openai`, `google`, `xai`, `qwen`, `elevenlabs` and
`minimax`; `transcribe(service)` the first four. A voice, a speed, a language and a format are asked per call; the
service and the model are fixed when the tool is made. Speech returns the path
of the file it wrote, which is what a chain passes on and what a model can be
told. The file goes in the tool's `output_dir`, else `AICHAIN_OUTPUT_ROOT`,
else the system's temporary directory — the one writable place on Lambda.
Qwen's speech endpoint answers WAV whatever is asked, so `wav` is the one
format `speech("qwen")` takes; its host, like every DashScope tool's, follows
`region=` or `DASHSCOPE_REGION` (`ap`, `us`, `cn`, `hk`).

### A voice of its own

`speech("elevenlabs")` and `speech("minimax")` make voices from a description,
not only speak with stock ones. `design(description, text)` returns candidates —
three from ElevenLabs, one from MiniMax per call — each with a sample spoken in
that voice; `keep(voice_id, name, description)` makes the chosen one permanent
(an ElevenLabs design is a preview until then; a MiniMax voice is the account's
at once), and `run(text, {"voice": voice_id})` speaks with it from then on.

```python
voices = speech("elevenlabs", output_dir="voices").design(
    "A young woman speaking Russian: clean, bright, ringing, lively.", "Текст образца…" * 10, seed=7)
kept = speech("elevenlabs").keep(voices[2]["voice_id"], "Linara", "clean, bright, ringing")
speech("elevenlabs", model="eleven_v4").run("[excited] Уже пишем?! [laughing] Ой.", {"voice": kept, "speed": 1.12})
```

`eleven_v4` reads emotion from inline tags — `[excited]`, `[whispers]`,
`[laughing]`, `[pause]` — one emotion a segment works best. `speed` is 0.7–1.2
on ElevenLabs. A seed with the same description gives the same voice back; the
voice itself lives with the provider and is not exported, so keep the
description, the seed and a sample beside the id. Voice design needs a paid
ElevenLabs plan and a key with Text to Speech, Voice Generation and Voices
write; a plan's concurrency limit (two on the smallest) holds design calls too.
MiniMax deletes cloned voices unused for seven days and may treat designed ones
the same; its design costs $30 per million characters of sample.

A transcript is plain text, or a JSON string with words and their times when
`timestamps` or `format="json"` is asked. Word timestamps come from
`whisper-1` alone. A file over a provider's limit is refused with its size and
the limit — OpenAI takes 25 MB, Google's synchronous API a minute and 10 MB.

### Long audio

The library does not cut audio: decoding it takes a dependency the rest of the
library does not need. A recording over the limit is cut by your own tool,
which wraps the library's — here with `pydub` (`pip install pydub`, plus
`audioop-lts` on Python 3.13 and later, and `ffmpeg` on the machine for any
format but WAV):

```python
import os, tempfile
from pydub import AudioSegment
from yait_aichain.tools import tool, transcribe

stt = transcribe("openai")

@tool(risk="external")
def transcribe_long(path: str, minutes: int = 10) -> str:
    """Transcribe a recording of any length, cut into pieces.

    Parameters
    ----------
    path : str
        The audio file.
    minutes : int
        How long each piece is.
    """
    audio, step = AudioSegment.from_file(path), minutes * 60_000
    texts = []
    for start in range(0, len(audio), step):
        piece = os.path.join(tempfile.mkdtemp(), "piece.mp3")
        audio[start:start + step].export(piece, format="mp3")
        texts.append(stt.run(piece).strip())
    return " ".join(texts)
```

Pieces are cut end to end: a cut may split a word, where an overlap would
repeat a few seconds of speech in the joined text. A runnable version is the
`splice` recipe in the cookbooks repository.

## Images: why a tool and not a model

A model, in this library, is what can be an agent's mind: it reads, reasons
and calls tools. An image model does none of that, so since 3.9.0 it is
reached as a tool — `image(service, model=…)` with `openai`, `google`, `bfl`,
`xai`, `qwen`, `recraft`, `krea`, `kling` or `byteplus`. An agent given one draws when it decides a picture
is needed, the way it searches when it decides it needs to know something.

The request underneath is the one the model layer always built, with one
vocabulary of options for every provider: `size` and `aspect_ratio` are
converted into each other where a provider takes the other, and an option a
provider cannot honour — a `seed`, a transparent background — comes back as a
notice in the result, not as a silent difference in the picture.

Passing `images=[…]` — paths or URLs — makes the call an edit, or gives the
model references. `registry.models(task="image-to-image")` lists the models
that edit.

**Krea** runs its own models as jobs — submitted, polled every two seconds,
downloaded — with `KREA_API_KEY`, billed from the workspace's API balance in
dollars, which is separate from the compute units the web app spends (an
empty one answers HTTP 402). `krea-2-medium`, `krea-2-large` and
`krea-2-turbo` generate; they redraw an image given to them rather than follow
an edit, so `images=` is refused there and `krea-3` (early access) is the one
that edits, with up to eight images. Each model takes its own list of ratios
and a resolution tier, `1K` or, on Krea 3, `2K`: a `size` becomes the nearest
ratio and the tier, said in the notices. A job reports no price, so the cost
in the result is Krea's own list applied to the request — the tier, each
reference by its size, `thinking_mode="high"` passed through `extra`.

**Kling** also runs as a job, with `KLING_API_KEY` — one key for its images
and its video, billed from resource packages or a balance in dollars, apart
from the Kling web app. `kling-v3` ($0.028) and `kling-v2-1` ($0.014; an edit
costs twice that) both generate and edit from one image. Each takes eight
ratios, from 21:9 to 9:16: an `aspect_ratio` or a `size` becomes the nearest,
said in the notices, and a size over 1536 pixels asks for the 2K tier.

**BytePlus** serves ByteDance's Seedream with `BYTEPLUS_API_KEY`, a key from
one ModelArk region with each model activated in the console first. Seedream
5.0 pro ($0.045, or $0.09 above 2.6 million pixels), 5.0 flash ($0.018), 5.0
lite ($0.035), 4.5 ($0.04) and 4.0 ($0.03) all generate and edit, with up to
ten images. Each makes a range of pixel counts: a `size` outside it is scaled
into it with its shape kept, an `aspect_ratio` is drawn at the model's base
size, and with neither the model takes the shape from the prompt or the input
image. The price is read from the size that came back. The watermark Seedream
adds by default is turned off.

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

## Video: a job, not a call

A video takes from ten seconds to several minutes to render, so `video()` is
built around a job. `submit()` starts one and returns it at once — a small dict,
`{"service", "model", "id", …}`, that you can store, queue or hand to another
process; `collect(job)` waits for it, downloads the file and prices it; `run()`
is the two together, waiting up to `timeout` (ten minutes by default).

```python
clip = video("xai", model="grok-imagine-video-1.5-lite", output_dir="videos")
job = clip.submit("a paper boat on a puddle", images=["boat.png"], options={"duration": 3})
...                                   # another process, a queue, a cron
result = video("xai", model="grok-imagine-video-1.5-lite", output_dir="videos").collect(job)
```

A wait that runs out raises with the job, so it can be collected again rather
than paid for twice. `images` is one image, the clip's first frame.

| Service | Models | Price per second |
|---|---|---|
| `google` | `veo-3.1-generate-preview`, `-fast-`, `-lite-` | $0.05–0.60 by model and resolution; sound always on |
| `xai` | `grok-imagine-video-1.5`, `-1.5-lite` | $0.08, $0.02; the job reports its own cost |
| `qwen` | `wan3.0-video`, `wan3.0-video-prime`, `wan2.7-t2v`, `wan2.7-i2v`, `wan2.2-t2v-plus` | $0.02–0.28 by model and resolution; Wan 3.0 up to 30 s |
| `minimax` | `MiniMax-H3`, `MiniMax-H3-Max` | $0.05–0.13 by resolution (480p, 768p, 2K) |
| `byteplus` | `dreamina-seedance-2-5-260628`, `-2-0-`, `-2-0-fast-`, `-2-0-mini-`; `seedance-1-0-pro-250528`, `-pro-fast-` | $0.01–0.78 by model and resolution; billed by the tokens the task reports; 2.5 up to 30 s; sound on 2.x when asked, at no extra cost; 2.x refuses a first frame with a real person's face |
| `kling` | `kling-3.0`, `kling-3.0-turbo` | $0.084–0.42 by resolution (720p, 1080p, 4k), 3–15 s; sound on 3.0 when asked ($0.126 at 720p), always on Turbo |

OpenAI closed Sora and its video API on 2026-09-24.

**One vocabulary for the shape**: `duration`, `resolution` (480p, 720p,
1080p, 4k), `aspect_ratio`, `size` (pixels, turned into a ratio and a
resolution), `seed`, `audio`, `negative_prompt`. Each model takes some lengths,
some resolutions and some ratios; the nearest goes, and the notice says what
changed — Veo makes 1080p and 4k at 8 seconds only, Wan 2.2 reads a pixel size
where Wan 2.7 reads a resolution. A control a model lacks is declined and said.

**The file is downloaded, always**: written to `output_dir`, or returned as
bytes in `videos[0]["data"]`. A provider's link is not the result — Google's
needs the key, Wan's lasts a day.

**What it costs**: the provider's own figure where it gives one (xAI), else the
seconds delivered at the registry's rate for the resolution delivered; `cost_basis`
says which. A clip whose estimate is more than a budget in force has left is
warned about and made; a budget with nothing left stops it before it starts.
`image()` warns the same way.

---

▶ [`examples/06_media_input.py`](../../examples/06_media_input.py) ·
[`examples/12_image.py`](../../examples/12_image.py) ·
[`examples/39_video.py`](../../examples/39_video.py)
