## Made by the provider's name

```python
from yait_aichain.tools import speech, transcribe

path = speech("openai").run("Hello!", {"voice": "nova"})            # the audio file's path
text = transcribe("openai").run("call.mp3", {"language": "ru"})     # the transcript
Agent(model, tools=[speech("google"), transcribe("openai")])
```

`speech(service)` and `transcribe(service)` take `openai`, `google`, `xai` or
`qwen`, with `api_key`, `name` and `description`; `speech("qwen")` takes
`region` as well. Until 3.5.0 these were `TTS(provider)` and `STT(provider)` in
`tools.convert`; both names still work, imported from `yait_aichain.tools`.
