# Images in yait-aichain — generation, editing & model selection

Status: **shipped in 1.4.0** · reference guide (was the img2img design doc).

One interface for all image work: `Skill(Model(name), input=messages, output={...})`.
The provider is a one-word change in the model name. Three capabilities:
**generation** (text→image), **editing** (image→image), and **vision** (image→text,
not covered here). 10 providers; `registry.models(task=...)` lists what's available.

Result shape is always: `{"url", "base64", "mime_type", "revised_prompt"}`.

Prices below are USD/image at 1024², official tariffs (June 2026); speed is avg
seconds/image from our blind bench. ⚠️ = approximate.

---

## 1 · Generation (text → image)

```python
from yait_aichain import Model, Skill

Skill(
    model  = Model("gemini-3-pro-image"),     # swap → gpt-image-2 / flux-2-pro / recraftv4_1_vector / grok-imagine-image / wan2.2-t2i-plus
    input  = {"messages": [{"role": "user", "parts": ["a cozy cabin at alpine dawn"]}]},
    output = {"modalities": ["image"], "format": {"type": "image", "size": "1024x1024"}},
).run()
```

**22 models / 6 providers** (`registry.models(task="text-to-image")`):

| Provider | Models | $/img | Speed | Notes |
|---|---|--:|--:|---|
| OpenAI | gpt-image-2 / 1.5 / 1 / 1-mini, chatgpt-image-latest | $0.011–0.21 (by quality) | **60–73s** 🐢 | best text; **slow** (autoregressive); gpt-image-2 has no alpha |
| Google | gemini-3-pro-image, 3.1-flash-image, 2.5-flash-image | $0.039 / $0.067 / $0.134 | ~fast | top prompt-adherence; 3-pro pricey |
| BFL/FLUX | flux-2-pro, flux-pro-1.1(-ultra), flux-dev | $0.04–0.07 | **6s** ⚡ | king of 3D-render / abstract |
| xAI | grok-imagine-image / -pro | $0.02 / $0.05 | 6–7s | cheap + fast + legible text |
| Qwen | wan2.2-t2i-flash / -plus | ~$0.02 / ~$0.03 ⚠️ | async | cheap, multilingual prompts |
| Recraft | recraftv4_1(_vector), recraftv3(_vector) | $0.04 raster / $0.08 vector | 12s | **the only vector/SVG** |

**Capabilities and limits:**
- **Vector/SVG** — Recraft only (`*_vector`, $0.08); `mime_type=image/svg+xml`.
- **Transparent background (alpha)** — `gpt-image-1`/`-mini` only (via `format.background="transparent"`); **`gpt-image-2` cannot** (HTTP 400).
- **Size**: Gemini is priced by resolution tier (1K/2K/4K); OpenAI by fixed sizes per quality; FLUX.2 by megapixels; xAI/Recraft/Qwen are flat per image.

---

## 2 · Editing (image → image)

Same call, but the message carries an **input image** ⇒ it's an edit. Several image
parts = multi-reference (compose / replace by example).

```python
Skill(
    model  = Model("flux-kontext-pro"),       # swap → gemini-3.1-flash-image / qwen-image-edit / gpt-image-1.5 / grok-imagine-image
    input  = {"messages": [{"role": "user", "parts": [
        {"type": "image", "source": {"kind": "file", "path": "product.png"}},     # base
        {"type": "image", "source": {"kind": "file", "path": "tomato.png"}},      # reference (optional)
        "Replace the chest print with the tomato illustration; keep the photo",
    ]}]},
    output = {"modalities": ["image"], "format": {"type": "image"}},
).run()
```

**6 providers** (`registry.models(task="image-to-image")`). Multi-image: OpenAI ≤16,
xAI ≤3, Qwen 1–3, Gemini/FLUX Kontext several. `kind:"file"` auto-loads to base64.

| Provider | Edit models | Type | $/op | Speed |
|---|---|---|--:|--:|
| BFL | `flux-kontext-pro` / `-max` | instruction-edit, **benchmark** | $0.04 / $0.08 | ~10–25s |
| Google | `gemini-3.1-flash-image` … | instruction-edit, **keeps frame 1:1 + reference** | ~$0.07 | ~15s |
| Qwen | `qwen-image-edit(-plus/-max)` | instruction-edit | ~$0.04 ⚠️ | sync |
| OpenAI | `gpt-image-1.5` / `-2` / `-1-mini` | instruction-edit | ~$0.05–0.08 | slow |
| xAI | `grok-imagine-image(-pro)` | instruction-edit (multi-image ≤3) | $0.02 / $0.05 | 6–7s |
| Recraft | `recraftv3(_vector)` | ⚠️ **whole-image variation, does NOT preserve the subject** | $0.04 / $0.08 | 12s |

**Important:**
- **Subject-preserving** ("take THIS object"): OpenAI · Google · xAI · Qwen · FLUX Kontext.
- **Recraft imageToImage = whole-frame variation** — not for "object → into a scene" (use its inpaint for that).
- Empty response (blocked / refused / text instead of an image) → a clear `ValueError`, not a silent `base64=None`.

---

## 3 · Model recommendations — which model for what

From the blind bench (Family 1–4 + bench-7 + edit tests, June 2026).

### Generation by asset family
| Task | Recommended | Why |
|---|---|---|
| **Illustration flat/vector, logo** | **`recraftv4_1_vector`** | clean scalable SVG; correct short text |
| **Illustration character / painterly** | **`gemini-3-pro-image`** | anatomy (hands), quality; budget — `gemini-2.5-flash-image` |
| **3D-render / abstract / hero** | **`flux-pro-1.1`** / `flux-2-pro` | best render, and fastest (6s) |
| **Photo (food/interior/people/street)** | **`gpt-image-2`** or **`gemini-3-pro-image`** (quality); **`grok-imagine-image-pro`** (speed/cost) | grok scored 9 on interior/street at $0.05/7s |
| **Pattern / gradient background** | **`gpt-image-2`** | seamlessness; gradient — any raster |
| **Diagrams / schemes** | **`grok-imagine-image-pro`** (default: correct + large + 7s) | dense numbered → fallback `gpt-image-2` (more correct but small/60s); on a slip — re-roll grok |

### Editing (adaptation)
| Task | Recommended |
|---|---|
| **Product → place into a scene** | **`flux-kontext-pro`** ($0.04, clean compositing) |
| **Change an element, keep the frame 1:1 + use a reference** | **`gemini-3.1-flash-image`** (preserved the original perfectly) |
| Cheap/fast | `grok-imagine-image` ($0.02; keeps frame; multi-ref ≤3) |
| Avoid | `recraftv3` (variation), `gpt-image` (slow) |

### Heuristics — when to use what
- **speed matters** → flux / grok (6–7s); **avoid gpt-image** (60–73s).
- **cheap** → `grok-imagine-image` $0.02, `gpt-image-1-mini` $0.011.
- **text in the image** → short captions: almost all do it; logos — Recraft vector; diagrams — grok-pro.
- **need vector/SVG** → Recraft `*_vector` only.
- **need transparency** → `gpt-image-1`/`-mini` (not `gpt-image-2`).
- **premium quality, time/money no object** → `gemini-3-pro-image` / `gpt-image-2`.

---

## Implementation notes (1.4.0)
- Detection: image-output model/modality **+** ≥1 input image part ⇒ edit; otherwise generation. No new methods.
- OpenAI edits — multipart `/v1/images/edits` (via the `send()` seam + `_post_form`); xAI — JSON `/v1/images/edits` (`image` / `images[]`); Qwen — synchronous `multimodal-generation` (download URL→b64); Google — `generateContent` with an image part + `responseModalities`; BFL — async submit→poll→download.
- `kind:"file"` → base64 in `normalize_input`; SVG mime by signature; empty response → `ValueError`.
