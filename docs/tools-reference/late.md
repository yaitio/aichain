# `late_accounts` and `late_publish`

Two tools wrapping [Late](https://getlate.dev) — a unified API across 14 social platforms:

> Twitter/X, Instagram, LinkedIn, Facebook, TikTok, YouTube, Threads, Reddit, Pinterest, Bluesky, Telegram, Google Business, Snapchat, WhatsApp.

| Tool | Class | Purpose |
|---|---|---|
| `late_accounts` | `LateAccountsTool` | List connected accounts + IDs. |
| `late_publish` | `LatePublishTool` | Create / publish / schedule posts. |

---

## Requirements

| | |
|---|---|
| Env var | `LATE_API_KEY` |
| Get a key | <https://getlate.dev/dashboard/api-keys> |
| Base URL | `https://getlate.dev/api` |

Both constructors raise `ValueError` if no key is found.

---

## Typical agent workflow

1. **`late_accounts`** — discover which accounts are connected and collect their IDs.
2. **`late_publish`** — post using those IDs.

An agent given both tools will usually call `late_accounts` first on its own.

---

## `LateAccountsTool`

### Parameters

| Name | Type | Required | Notes |
|---|---|---|---|
| `profile_id` | `string` | | Filter to a specific profile. |
| `platform` | `string` | | Filter to a specific platform. |

### Output

Plain text:

```
Connected accounts (3):

[1]  twitter       @acme          ID: 64e1f0...  |  Profile: My Brand (64f0...)
[2]  linkedin      Acme Corp      ID: 64e2f0...  |  Profile: My Brand (64f0...)
[3]  instagram     @acme          ID: 64e3f0...  |  Profile: Marketing (65f0...)

Use the ID and platform name from above in late_publish · platforms.
```

### Example

```python
from tools import LateAccountsTool

tool   = LateAccountsTool()
result = tool()
if result:
    print(result.output)
```

---

## `LatePublishTool`

### Parameters

| Name | Type | Required | Notes |
|---|---|---|---|
| `platforms` | `array[object]` | ✓ | Each item: `{platform, account_id, custom_content?}`. |
| `content` | `string` | | Caption / text. Required for text-only posts. |
| `publish_now` | `boolean` | | Default `true`. Set `false` to schedule. |
| `scheduled_at` | `string` | | ISO 8601, e.g. `"2025-06-01T10:00:00Z"`. Required when `publish_now=false`. |
| `timezone` | `string` | | IANA timezone. Default `"UTC"`. |
| `media_urls` | `array[string]` | | Public URLs (image / video / GIF / PDF). Media type inferred from extension. |
| `is_draft` | `boolean` | | Default `false`. Save without publishing or scheduling. |

You must provide `content`, `media_urls`, or both — otherwise the tool raises `ValueError`.

### Per-platform content override

```python
platforms = [
    {"platform": "twitter",  "account_id": "64e1f0...",
     "custom_content": "Short punchy tweet ⚡️"},
    {"platform": "linkedin", "account_id": "64e2f0...",
     "custom_content": "Longer, more formal LinkedIn post with context and a CTA."},
]
```

### Media type inference

Inferred from the URL's MIME type (fallback: file extension):

| Category | Extensions |
|---|---|
| `image` | jpg, jpeg, png, webp |
| `gif` | gif |
| `video` | mp4, mov, avi, webm, m4v, mpeg |
| `document` | pdf |

### Examples

**Publish immediately to Twitter + LinkedIn**

```python
from tools import LatePublishTool

tool = LatePublishTool()
text = tool.run(
    content   = "Big news — we just launched! 🚀",
    platforms = [
        {"platform": "twitter",  "account_id": "64e1f0..."},
        {"platform": "linkedin", "account_id": "64e2f0..."},
    ],
    publish_now = True,
)
print(text)
```

**Schedule a post with an image**

```python
tool.run(
    content      = "Check out our new product drop! 🎉",
    platforms    = [{"platform": "instagram", "account_id": "64e3f0..."}],
    media_urls   = ["https://cdn.example.com/product.jpg"],
    publish_now  = False,
    scheduled_at = "2025-02-01T10:00:00Z",
    timezone     = "America/New_York",
)
```

**Save as a draft**

```python
tool.run(
    content   = "Working on copy — do not publish yet.",
    platforms = [{"platform": "twitter", "account_id": "64e1f0..."}],
    is_draft  = True,
)
```

### Output

```
Post created  ·  status: scheduled  ·  ID: 65a…
Scheduled for: 2025-02-01T10:00:00Z (America/New_York)

Platforms:
  [1]  instagram     @acme               scheduled
        https://instagram.com/p/…
```

Errors per platform are surfaced as `ERROR: <message>` on the corresponding line.

---

## Agent example

```python
from agent import Agent
from models import Model
from tools import LateAccountsTool, LatePublishTool

agent = Agent(
    orchestrator = Model("claude-opus-4-6"),
    tools        = [LateAccountsTool(), LatePublishTool()],
    persona      = (
        "You are a social media manager. Always call late_accounts before "
        "publishing, and tailor copy per platform with custom_content."
    ),
    max_steps    = 6,
)

agent.run(
    "Post an announcement for our Series B. Keep the tweet short; the LinkedIn "
    "post should be 3 paragraphs with a clear CTA to the careers page."
)
```

---

## Notes

- Both tools raise `RuntimeError` on any non-2xx response. Call-style `tool(…)` captures this in `ToolResult.error`.
- Upload local files first via Late's `/v1/media/presign` before passing their public URLs — this tool does not handle uploads.

---

## See also

- [Agent overview](../agents/overview.md) — agents orchestrating multi-tool social workflows.
