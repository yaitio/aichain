"""
``Speech``, the base of the text-to-speech tools, and its providers:
``ttsOpenAI``, ``ttsGoogle``, ``ttsXAI`` and ``ttsQwen``, all plain HTTP.

``run()`` returns the path of the audio file it saved, in the tool's
``output_dir``. See docs/tools/media.md.
"""

from __future__ import annotations


import json
import os
import tempfile
import urllib3
from yait_aichain.clients._base import make_http

from .._base import Tool
from ...clients._http import request, request_json
from .._security import assert_safe_url, confine_output_path


# ── Module-level defaults ─────────────────────────────────────────────────────

_DEFAULT_FORMAT = "mp3"
#: The largest audio file a speech tool saves.
MAX_AUDIO_BYTES = 50 * 1024 * 1024
_DEFAULT_SPEED  = 1.0



def _model_setting(tool, model):
    """The model given, or the provider's default: a setting of the tool, not of a call."""
    default = getattr(type(tool), "_DEFAULT_MODEL", None)
    if model is not None and default is None:
        raise TypeError(f"{type(tool).__name__} has one model and takes no model=")
    return model or default


def _no_model_option(tool, opts: dict) -> None:
    if "model" in opts:
        raise ValueError(f"{tool.name}: model is a setting of the tool, not of a call — "
                         f"make it with model=… ({type(tool).__name__}(model=…), or the factory)")


def _call_options(tool, options) -> dict:
    """A call's options without the unset ones, and with the format checked
    against those this provider writes rather than replaced in silence."""
    _no_model_option(tool, options or {})
    opts = {k: v for k, v in (options or {}).items() if v is not None}
    fmt = opts.get("format", _DEFAULT_FORMAT)
    if fmt not in tool._FORMATS:
        raise ValueError(f"{tool.name} writes {sorted(tool._FORMATS)}; {fmt!r} is not one")
    return opts

def _output_folder(output_dir: str | None) -> str:
    """``output_dir``, else ``AICHAIN_OUTPUT_ROOT``, else the temporary directory,
    which is writable on Lambda where the working directory is not."""
    folder = output_dir or os.environ.get("AICHAIN_OUTPUT_ROOT", "").strip() or tempfile.gettempdir()
    folder = confine_output_path(folder)
    os.makedirs(folder, exist_ok=True)
    return folder


# ── Base class ────────────────────────────────────────────────────────────────


def _without_options(parameters: dict, *keys: str) -> dict:
    """A copy of *parameters* without option *keys*: an option advertised and not
    read is one a model sets and loses. A copy, since the schemas are class
    attributes shared by inheritance."""
    import copy
    out = copy.deepcopy(parameters)
    options = out.get("properties", {}).get("options", {}).get("properties", {})
    for key in keys:
        options.pop(key, None)
    return out


class Speech(Tool):
    """
    Synthesise speech from text and save to an audio file.

    Subclasses implement the provider-specific ``run()`` logic.
    All share the same ``options`` schema.

    Returns
    -------
    str
        Absolute path to the saved audio file.
    """

    name        = "speech"
    description = (
        "Synthesise speech from text using a TTS model.  "
        "Returns the absolute path to the saved audio file."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "Text to synthesise into speech.",
            },
            "options": {
                "type":        "object",
                "description": "TTS options.",
                "properties": {
                    "voice": {
                        "type":        "string",
                        "description": "Voice name or ID (provider-specific).",
                    },
                    "speed": {
                        "type":        "number",
                        "description": "Playback speed multiplier.  0.25–4.0, default 1.0.",
                    },
                    "format": {
                        "type":        "string",
                        "enum":        ["mp3", "wav", "opus", "flac", "pcm"],
                        "description": "Audio output format.  Default: mp3.",
                    },
                    "language": {
                        "type":        "string",
                        "description": (
                            "BCP-47 language code, e.g. 'en-US'.  "
                            "Google only — it is required there and the "
                            "other endpoints have no such field, so they "
                            "do not declare this option at all."
                        ),
                    },
                },
            },
        },
        "required": ["input"],
    }

    def _serialise_init_args(self) -> dict:
        """What `Chain.save` writes: the model and the region, never the key."""
        out = {}
        if getattr(self, "model", None) and self.model != getattr(type(self), "_DEFAULT_MODEL", None):
            out["model"] = self.model
        if getattr(self, "_region", None):
            out["region"] = self._region
        if self.output_dir:
            out["output_dir"] = self.output_dir
        return out

    #: `draft`: it makes a file of its own in a folder the code chose, and
    #: overwrites nothing. A model never names the path.
    risk = "draft"
    priced = True
    output_dir: str | None = None

    def run(self, input: str, options: dict | None = None) -> str:
        raise NotImplementedError(f"{type(self).__name__} must implement run()")

    def _save(self, opts: dict, fmt: str, produce) -> str:
        """Make the file (new and exclusive, so parallel calls never share one), then
        pay for its contents; return the path. An unfilled file is removed. The
        folder is ``output_dir``, never an option a model can set."""
        if "output_path" in opts:
            import warnings
            warnings.warn(f"{self.name}: output_path is going in 5.0 — set the folder with "
                          "output_dir= on the tool; files are named for you",
                          DeprecationWarning, stacklevel=3)
            path = confine_output_path(opts["output_path"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            fd, path = tempfile.mkstemp(dir=_output_folder(self.output_dir),
                                        prefix="speech-", suffix=f".{fmt}")
            os.close(fd)
        try:
            data = produce()
        except BaseException:
            if "output_path" not in opts:
                os.unlink(path)
            raise
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def stream(self, input: str, options: dict | None = None):
        """Real-time streaming synthesis — reserved for a future release."""
        raise NotImplementedError("Streaming TTS is not yet implemented.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ── ttsOpenAI ─────────────────────────────────────────────────────────────────

class _OpenAISpeech(Speech):
    """The OpenAI speech wire (``POST {base}/audio/speech``), which xAI speaks too.
    A provider on it is data: ``_BASE_URL``, ``_ENV_KEY``, ``_LABEL`` and defaults."""

    # The endpoint has no language field — the voice and the model carry it.
    parameters = _without_options(Speech.parameters, "language")

    _BASE_URL: str = ""
    _ENV_KEY:  str = ""
    _LABEL:    str = ""
    _FORMATS = frozenset({"mp3", "wav", "opus", "flac", "pcm", "aac"})

    def __init__(self, api_key: str | None = None, model: str | None = None, *,
                 output_dir: str | None = None) -> None:
        self.model = _model_setting(self, model)
        self.output_dir = output_dir
        self._api_key = api_key or os.environ.get(self._ENV_KEY)
        if not self._api_key:
            raise ValueError(f"No {self._LABEL} API key found.  "
                             f"Pass api_key= or set the {self._ENV_KEY} environment variable.")
        self._http = make_http(timeout=urllib3.Timeout(connect=10.0, read=60.0),
                               retries=urllib3.Retry(total=2, backoff_factor=0.5))

    def run(self, input: str, options: dict | None = None) -> str:
        """
        Synthesise *input* and save it as an audio file; return its path.

        options: ``voice``, ``speed`` (0.25–4.0, default 1.0) and ``format``
        (``"mp3"`` default, ``"wav"``, ``"opus"``, ``"flac"``, ``"pcm"``). The
        file goes in the tool's ``output_dir``, made before the request.
        """
        opts = _call_options(self, options)
        fmt  = opts.get("format", _DEFAULT_FORMAT)
        return self._save(opts, fmt, lambda: request(
            self._http, "POST", f"{self._BASE_URL}/audio/speech",
            json={"model": self.model, "voice": opts.get("voice", self._DEFAULT_VOICE),
                  "input": input, "response_format": fmt,
                  "speed": float(opts.get("speed", _DEFAULT_SPEED))},
            headers={"Authorization": f"Bearer {self._api_key}"},
            what=f"{self._LABEL} speech API").data)


class ttsOpenAI(_OpenAISpeech):
    """
    Text-to-speech via the OpenAI TTS API.

    Models
    ------
    ``tts-1``     Standard quality — low latency.  Default.
    ``tts-1-hd``  High-definition — higher quality, slightly slower.

    Voices
    ------
    alloy · ash · coral · echo · fable · nova · onyx · sage · shimmer

    The API takes up to 4 096 characters a request; split longer texts.

    Parameters
    ----------
    api_key : str | None, optional
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    model : str | None, optional
        ``"tts-1"`` (default) or ``"tts-1-hd"``.  A setting of the tool.

    Examples
    --------
    ::

        tool = speech("openai", model="tts-1-hd", output_dir="audio/")
        path = tool.run(input="Hello!", options={"voice": "echo"})
    """

    name           = "ttsOpenAI"
    _BASE_URL      = "https://api.openai.com/v1"
    _ENV_KEY       = "OPENAI_API_KEY"
    _LABEL         = "OpenAI"
    _DEFAULT_MODEL = "tts-1"
    _DEFAULT_VOICE = "alloy"


# ── ttsGoogle ─────────────────────────────────────────────────────────────────

class ttsGoogle(Speech):
    """
    Text-to-speech via Google Cloud Text-to-Speech, over its REST API.

    One plain HTTPS request to ``text:synthesize``; no Google SDK. The key is
    an API key with the Cloud Text-to-Speech API enabled on its project.

    Voices
    ------
    Full list: https://cloud.google.com/text-to-speech/docs/voices
    Example names: ``"en-US-Neural2-A"``, ``"en-US-Journey-F"``

    Parameters
    ----------
    api_key : str | None, optional
        Google Cloud API key.  Falls back to ``GOOGLE_CLOUD_API_KEY``, then
        ``GOOGLE_AI_API_KEY``, then ``GOOGLE_API_KEY``.

    Examples
    --------
    ::

        from yait_aichain.tools import speech

        tool = speech("google")
        path = tool.run(
            input="Hello, world!",
            options={
                "voice":    "en-US-Neural2-C",
                "language": "en-US",
                "format":   "mp3",
            },
        )
    """

    # Google picks the synthesiser from the voice name; there is no model to
    # name.
    parameters = _without_options(Speech.parameters, "model")

    name              = "ttsGoogle"
    _URL              = "https://texttospeech.googleapis.com/v1/text:synthesize"
    _DEFAULT_LANGUAGE = "en-US"

    # Maps common format names to Google AudioEncoding names and file extensions
    _FORMAT_MAP: dict[str, tuple[str, str]] = {
        "mp3":  ("MP3",      "mp3"),
        "wav":  ("LINEAR16", "wav"),
        "opus": ("OGG_OPUS", "opus"),
        "flac": ("FLAC",     "flac"),
    }
    _FORMATS = frozenset(_FORMAT_MAP)

    def __init__(self, api_key: str | None = None, model: str | None = None, *,
                 output_dir: str | None = None) -> None:
        self.model = _model_setting(self, model)
        self.output_dir = output_dir
        # GOOGLE_CLOUD_API_KEY first: speech is a Cloud API, and an AI Studio
        # key is often restricted to Gemini, which answers
        # API_KEY_SERVICE_BLOCKED here. GOOGLE_AI_API_KEY is the Google
        # provider's key and GOOGLE_API_KEY its older spelling.
        self._api_key = (api_key or os.environ.get("GOOGLE_CLOUD_API_KEY")
                         or os.environ.get("GOOGLE_AI_API_KEY")
                         or os.environ.get("GOOGLE_API_KEY"))
        if not self._api_key:
            raise ValueError("No Google API key found. Pass api_key= or set GOOGLE_CLOUD_API_KEY "
                             "(or GOOGLE_AI_API_KEY); the key's project needs the Text-to-Speech API.")
        self._http = make_http(timeout=urllib3.Timeout(connect=10.0, read=60.0),
                               retries=urllib3.Retry(total=2, backoff_factor=0.5))

    def run(self, input: str, options: dict | None = None) -> str:
        """Synthesise *input* text and save to an audio file.

        Parameters
        ----------
        input : str
            Text to synthesise.
        options : dict | None, optional
            ``voice`` (default: the language's own), ``language`` (``"en-US"``),
            ``speed`` (0.25–4.0, 1.0), ``format`` (``mp3``, ``wav``, ``opus``, ``flac``).

        Returns
        -------
        str
            Absolute path to the saved audio file.
        """
        opts     = _call_options(self, options)
        language = opts.get("language", self._DEFAULT_LANGUAGE)
        fmt      = opts.get("format",   _DEFAULT_FORMAT)
        encoding, ext = self._FORMAT_MAP[fmt]
        voice = {"languageCode": language}
        if opts.get("voice"):
            voice["name"] = opts["voice"]
        body = {"input": {"text": input}, "voice": voice,
                "audioConfig": {"audioEncoding": encoding,
                                "speakingRate": float(opts.get("speed", _DEFAULT_SPEED))}}
        return self._save(opts, ext, lambda: self._synthesise(body))

    def _synthesise(self, body: dict) -> bytes:
        import base64
        data = request_json(self._http, "POST", self._URL, json=body,
                            headers={"x-goog-api-key": self._api_key},
                            what="Google Text-to-Speech API")
        audio = data.get("audioContent")
        if not audio:
            raise RuntimeError(f"Google Text-to-Speech answered with no audio: {json.dumps(data)[:300]}")
        return base64.b64decode(audio)


# ── ttsXAI ────────────────────────────────────────────────────────────────────

class ttsXAI(_OpenAISpeech):
    """
    Text-to-speech via the xAI TTS API.

    xAI exposes an OpenAI-compatible audio endpoint.
    See https://docs.x.ai/developers/model-capabilities/audio/voice for
    available model names and voice IDs.

    Parameters
    ----------
    api_key : str | None, optional
        xAI API key.  Falls back to ``XAI_API_KEY`` env var.
    model : str | None, optional
        Default ``"grok-tts"``.  A setting of the tool.

    Examples
    --------
    ::

        from yait_aichain.tools import speech

        tool = speech("xai")
        path = tool.run(
            input="Hello from xAI!",
            options={"format": "mp3"},
        )
    """

    name           = "ttsXAI"
    _BASE_URL      = "https://api.x.ai/v1"
    _ENV_KEY       = "XAI_API_KEY"
    _LABEL         = "xAI"
    _DEFAULT_MODEL = "grok-tts"    # check docs.x.ai for current model names
    _DEFAULT_VOICE = "default"     # check docs.x.ai for available voice IDs


# ── ttsQwen ───────────────────────────────────────────────────────────────────

class ttsQwen(Speech):
    """
    Text-to-speech via the Alibaba DashScope Qwen3-TTS API; the audio URL it
    answers with is downloaded and saved locally.

    Models
    ------
    ``qwen3-tts-flash``        Qwen3 TTS — fast and natural.  Default.
    ``qwen3-tts-instruct-flash`` Instruction-following TTS.

    Voices include ``Cherry`` (the default), ``Ethan``, ``Nofish``,
    ``Jennifer``, ``Ryan``; see the DashScope docs for the full list. The
    answer is WAV, the one format this tool writes.

    Parameters
    ----------
    api_key : str | None, optional
        DashScope API key.  Falls back to ``DASHSCOPE_API_KEY`` env var.
    region : str | None, optional
        Region selector: ``"ap"`` (default), ``"us"``, ``"cn"``, ``"hk"``.
        Overrides the ``DASHSCOPE_REGION`` env var.
    model : str | None, optional
        Default ``"qwen3-tts-flash"``.  A setting of the tool.

    Examples
    --------
    ::

        from yait_aichain.tools import speech

        tool = speech("qwen", region="us")
        path = tool.run(input="Hello from Qwen TTS!", options={"voice": "Cherry"})
    """

    name           = "ttsQwen"
    # The endpoint answers WAV whatever format is asked, so WAV is the one format.
    _DEFAULT_MODEL = "qwen3-tts-flash"
    _DEFAULT_VOICE = "Cherry"
    _TTS_PATH      = "/api/v1/services/aigc/multimodal-generation/generation"
    _FORMATS       = frozenset({"wav"})

    # `language` and `speed` are dropped from the shared schema: this
    # endpoint reads neither, and an option a model can set and watch vanish
    # is worse than one it never saw. `region` is a setting of the tool, not
    # an option of a call, so the schema does not carry it either.
    parameters = _without_options(Speech.parameters, "language", "speed")

    def __init__(
        self,
        api_key: str | None = None,
        region:  str | None = None,
        model:   str | None = None,
        *,
        output_dir: str | None = None,
    ) -> None:
        self.model = _model_setting(self, model)
        self.output_dir = output_dir
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "No DashScope API key found.  "
                "Pass api_key= or set the DASHSCOPE_API_KEY environment variable."
            )
        self._region = region
        self._base_url()        # an unknown region is refused here, not on the first call

        # Build HTTP client (no extra dependency — pure urllib3)
        self._http = make_http(
            timeout = urllib3.Timeout(connect=10.0, read=60.0),
            retries = urllib3.Retry(total=2, backoff_factor=0.5),
        )

    def _base_url(self) -> str:
        from ...clients._families.qwen import resolve_qwen_base_url
        return resolve_qwen_base_url(self._region)

    def run(self, input: str, options: dict | None = None) -> str:
        """Synthesise *input* text and save to an audio file.

        Parameters
        ----------
        input : str
            Text to synthesise.
        options : dict | None, optional
            ``voice`` — default ``"Cherry"``; ``format`` — ``"wav"``, the only one.

        Returns
        -------
        str
            Absolute path to the saved audio file.
        """
        if "region" in (options or {}):
            raise ValueError(f"{self.name}: region is a setting of the tool, not of a call — "
                             "speech('qwen', region=…)")
        opts        = _call_options(self, {"format": "wav", **(options or {})})
        voice       = opts.get("voice",       self._DEFAULT_VOICE)
        model       = self.model
        fmt         = opts.get("format",      "wav")
        url = self._base_url() + self._TTS_PATH

        body = {
            "model": model,
            "input": {"text": input},
            "parameters": {
                "voice":  voice,
                "format": fmt,
            },
        }

        return self._save(opts, fmt, lambda: self._synthesise(url, body))

    def _synthesise(self, url: str, body: dict) -> bytes:
        data = request_json(self._http, "POST", url, json=body, what="DashScope TTS",
                            headers={"Authorization": f"Bearer {self._api_key}",
                                     "X-DashScope-SSE": "disable"})

        # ── Extract audio URL from response ───────────────────────────
        # Shape: {"output": {"audio": {"url": "https://...", ...}}, ...}
        try:
            audio_url = data["output"]["audio"]["url"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"ttsQwen: unexpected response shape — could not find "
                f"output.audio.url: {json.dumps(data)[:300]}"
            ) from exc

        # ── Download audio bytes ──────────────────────────────────────
        # The URL comes from the provider response — guard it (SSRF) and do not
        # follow redirects to an unvalidated host.
        # DashScope hands back an http:// link to its own storage; it is fetched
        # over https, and no larger than an audio file can reasonably be.
        from urllib.parse import urlsplit
        parts = urlsplit(audio_url)
        if parts.scheme == "http" and (parts.hostname or "").endswith(".aliyuncs.com"):
            audio_url = "https://" + audio_url[len("http://"):]
        assert_safe_url(audio_url)
        data = request(self._http, "GET", audio_url, redirect=False,
                       what="DashScope TTS audio download").data
        if len(data) > MAX_AUDIO_BYTES:
            raise ValueError(f"ttsQwen: the audio is over {MAX_AUDIO_BYTES:,} bytes; not saved")
        return data


# ── Voices made from a description: ElevenLabs and MiniMax ─────────────────────

class _DesignedSpeech(Speech):
    """A provider whose voices can be made from a description: ``design()`` returns
    candidates with a sample each, ``keep()`` makes one permanent where the provider
    asks for that, and ``run()`` speaks with a voice id."""

    # The base reads no option; each provider declares the ones its run() reads.
    parameters = _without_options(Speech.parameters, "language", "speed", "format")
    _ENV_KEY = ""
    _LABEL = ""

    def __init__(self, api_key: str | None = None, model: str | None = None, *,
                 output_dir: str | None = None) -> None:
        self.model = _model_setting(self, model)
        self.output_dir = output_dir
        self._api_key = api_key or os.environ.get(self._ENV_KEY)
        if not self._api_key:
            raise ValueError(f"No {self._LABEL} API key found.  "
                             f"Pass api_key= or set the {self._ENV_KEY} environment variable.")
        self._http = make_http(timeout=urllib3.Timeout(connect=10.0, read=180.0),
                               retries=urllib3.Retry(total=2, backoff_factor=0.5))

    def _voice(self, opts: dict) -> str:
        voice = opts.get("voice") or self._DEFAULT_VOICE
        if not voice:
            raise ValueError(f"{self.name} speaks with a voice id: options={{'voice': …}} — "
                             "a stock one, or one made with design() and kept with keep()")
        return voice

    def _sample(self, data: bytes, fmt: str) -> str:
        fd, path = tempfile.mkstemp(dir=_output_folder(self.output_dir), prefix="voice-", suffix=f".{fmt}")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return path

    def keep(self, voice_id: str, name: str, description: str) -> str:
        """The permanent id of a designed voice; where the provider keeps designs as they are, the same id."""
        return voice_id


class ttsElevenLabs(_DesignedSpeech):
    """
    Text-to-speech and voice design via ElevenLabs. ``design(description)`` asks
    for three voices from a description and saves a sample of each; ``keep()``
    turns the chosen one into a voice of the account (a design is a preview until
    then); ``run()`` speaks with any voice id. MP3 out, the format every plan has.

    Models: ``eleven_multilingual_v2`` (default), ``eleven_v3``; designs with
    ``eleven_multilingual_ttv_v2``. Key: ``ELEVENLABS_API_KEY``, with the rights
    Text to Speech, Voice Generation and Voices write.
    """

    name           = "ttsElevenLabs"
    parameters     = _without_options(Speech.parameters, "language")
    _ENV_KEY       = "ELEVENLABS_API_KEY"
    _LABEL         = "ElevenLabs"
    _BASE_URL      = "https://api.elevenlabs.io"
    _DEFAULT_MODEL = "eleven_multilingual_v2"
    _DEFAULT_VOICE = None
    _FORMATS       = frozenset({"mp3"})
    _DESIGN_MODEL  = "eleven_multilingual_ttv_v2"

    def _headers(self) -> dict:
        return {"xi-api-key": self._api_key}

    def run(self, input: str, options: dict | None = None) -> str:
        """Speak *input* with ``options["voice"]``; ``speed`` 0.7–1.2. Returns the file's path."""
        opts = _call_options(self, options)
        fmt = opts.get("format", "mp3")
        body = {"text": input, "model_id": self.model}
        if opts.get("speed") is not None:
            speed = float(opts.get("speed"))
            if not 0.7 <= speed <= 1.2:
                raise ValueError(f"{self.name}: speed is 0.7–1.2 here; {speed} is not")
            body["voice_settings"] = {"speed": speed}
        return self._save(opts, fmt, lambda: request(
            self._http, "POST", f"{self._BASE_URL}/v1/text-to-speech/{self._voice(opts)}?output_format=mp3_44100_128",
            json=body, headers=self._headers(), what="ElevenLabs speech API").data)

    def design(self, description: str, text: str | None = None, seed: int | None = None) -> list:
        """Voices from *description*: ``[{"voice_id", "path", "seconds"}]``, a sample of each spoken
        with *text* (100–1000 characters), or with text the provider writes when there is none."""
        body = {"voice_description": description, "model_id": self._DESIGN_MODEL}
        if text:
            if not 100 <= len(text) <= 1000:
                raise ValueError(f"{self.name}: a design's text is 100–1000 characters; this one is {len(text)}")
            body["text"] = text
        else:
            body["auto_generate_text"] = True
        if seed is not None:
            body["seed"] = seed
        import base64
        made = request_json(self._http, "POST", f"{self._BASE_URL}/v1/text-to-voice/design?output_format=mp3_44100_128",
                            json=body, headers=self._headers(), what="ElevenLabs voice design")
        return [{"voice_id": p["generated_voice_id"], "seconds": p.get("duration_secs"),
                 "path": self._sample(base64.b64decode(p["audio_base_64"]), "mp3")}
                for p in made.get("previews") or []]

    def keep(self, voice_id: str, name: str, description: str) -> str:
        made = request_json(self._http, "POST", f"{self._BASE_URL}/v1/text-to-voice",
                            json={"voice_name": name[:100], "voice_description": description[:1000],
                                  "generated_voice_id": voice_id},
                            headers=self._headers(), what="ElevenLabs voice create")
        return made["voice_id"]


class ttsMiniMax(_DesignedSpeech):
    """
    Text-to-speech and voice design via MiniMax. ``design(description, text)`` makes
    one voice per call and returns it with its sample; the voice is the account's
    at once, so ``keep()`` changes nothing — MiniMax deletes cloned voices unused
    for seven days, and may treat designed ones the same, so a kept voice is one
    that is spoken with. ``run()`` speaks with any voice id.

    Models: ``speech-2.8-hd`` (default), ``speech-2.8-turbo``, ``speech-2.6-hd``.
    Key: ``MINIMAX_API_KEY``. A design is $30 per million characters of sample.
    """

    name           = "ttsMiniMax"
    parameters     = _without_options(Speech.parameters, "language")
    _ENV_KEY       = "MINIMAX_API_KEY"
    _LABEL         = "MiniMax"
    _BASE_URL      = "https://api.minimax.io"
    _DEFAULT_MODEL = "speech-2.8-hd"
    _DEFAULT_VOICE = None
    _FORMATS       = frozenset({"mp3", "wav", "flac"})

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _answer(self, data: dict, what: str) -> dict:
        status = (data.get("base_resp") or {})
        if status.get("status_code", 0) != 0:
            raise RuntimeError(f"{self.name}: {what} refused — {status.get('status_code')} {status.get('status_msg')}")
        return data

    def run(self, input: str, options: dict | None = None) -> str:
        """Speak *input* with ``options["voice"]``; ``speed`` 0.5–2.0; mp3, wav or flac."""
        opts = _call_options(self, options)
        fmt = opts.get("format", _DEFAULT_FORMAT)
        body = {"model": self.model, "text": input, "stream": False, "output_format": "hex",
                "language_boost": "auto",
                "voice_setting": {"voice_id": self._voice(opts), "speed": float(opts.get("speed", _DEFAULT_SPEED)),
                                  "vol": 1, "pitch": 0},
                "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": fmt, "channel": 1}}

        def produce():
            data = self._answer(request_json(self._http, "POST", f"{self._BASE_URL}/v1/t2a_v2", json=body,
                                             headers=self._headers(), what="MiniMax speech API"), "speech")
            return bytes.fromhex(data["data"]["audio"])
        return self._save(opts, fmt, produce)

    def design(self, description: str, text: str, seed: int | None = None) -> list:
        """One voice from *description*: ``[{"voice_id", "path"}]``, its sample spoken with *text*
        (at most 500 characters). Call again for another candidate."""
        if len(text) > 500:
            raise ValueError(f"{self.name}: a design's text is at most 500 characters; this one is {len(text)}")
        data = self._answer(request_json(self._http, "POST", f"{self._BASE_URL}/v1/voice_design",
                                         json={"prompt": description, "preview_text": text},
                                         headers=self._headers(), what="MiniMax voice design"), "voice design")
        return [{"voice_id": data["voice_id"], "path": self._sample(bytes.fromhex(data["trial_audio"]), "mp3")}]
