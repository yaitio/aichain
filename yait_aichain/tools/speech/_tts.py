"""
tools.speech._tts — Speech
==========================

Base class and provider implementations for text-to-speech synthesis.

Supported providers
-------------------
  ttsOpenAI   OpenAI TTS (tts-1, tts-1-hd)
  ttsGoogle   Google Cloud Text-to-Speech
  ttsXAI      xAI TTS
  ttsQwen     Alibaba DashScope CosyVoice / Qwen3-TTS

All implement ``run(input, options=None) -> str`` where the return value
is the absolute path to the saved audio file.

A ``stream()`` placeholder is present on every class.  Real-time
streaming synthesis is reserved for a future release.

Installation
------------
  OpenAI, xAI, Qwen :  nothing extra — plain HTTP
  Google            :  pip install google-cloud-texttospeech

Environment variables
---------------------
  OpenAI  :  OPENAI_API_KEY
  Google  :  GOOGLE_APPLICATION_CREDENTIALS  (path to service-account JSON)
             or GOOGLE_API_KEY
  xAI     :  XAI_API_KEY
  Qwen    :  DASHSCOPE_API_KEY
             DASHSCOPE_REGION   (optional: ap|us|cn|hk; default ap)
"""

from __future__ import annotations

from ..._require import require

import datetime
import json
import os
import urllib3
from yait_aichain.clients._base import make_http

from .._base import Tool
from ...clients._http import request, request_json
from .._security import assert_safe_url, confine_output_path


# ── Module-level defaults ─────────────────────────────────────────────────────

_DEFAULT_FORMAT = "mp3"
_DEFAULT_SPEED  = 1.0



def _model_setting(tool, model):
    """The tool's model: the one given, or the provider's default. A setting,
    fixed for the tool — a call does not choose it."""
    default = getattr(type(tool), "_DEFAULT_MODEL", None)
    if model is not None and default is None:
        raise TypeError(f"{type(tool).__name__} has one model and takes no model=")
    return model or default


def _no_model_option(tool, opts: dict) -> None:
    if "model" in opts:
        raise ValueError(f"{tool.name}: model is a setting of the tool, not of a call — "
                         f"make it with model=… ({type(tool).__name__}(model=…), or the factory)")

def _auto_path(fmt: str) -> str:
    """Return a timestamped output path inside ``./audio/`` (relative to cwd)."""
    stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), "audio")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"speech_{stamp}.{fmt}")


# ── Base class ────────────────────────────────────────────────────────────────


def _without_options(parameters: dict, *keys: str) -> dict:
    """A copy of *parameters* with option *keys* removed.

    The mirror of `_deep_with_option`, and it exists for the same reason:
    these subclasses share one schema and each honours a different part of
    it. Declaring an option a provider has no control for advertises it to
    the model, which then sets it and watches it fall on the floor — a claim
    not established by effect, the same defect the model layer refuses.
    Measured per tool by reading what its `run()` actually
    consumes, not by reading the description.

    A copy, because the schemas are class attributes shared by inheritance.
    """
    import copy
    out = copy.deepcopy(parameters)
    options = out.get("properties", {}).get("options", {}).get("properties", {})
    for key in keys:
        options.pop(key, None)
    return out


def _deep_with_option(parameters: dict, key: str, spec: dict) -> dict:
    """A copy of *parameters* with one more key under ``options``.

    A copy, because the schemas are class attributes shared by inheritance:
    mutating the parent's dict in place would add the key to every sibling —
    which is exactly the mistake this is here to avoid.
    """
    import copy
    out = copy.deepcopy(parameters)
    out.setdefault("properties", {}).setdefault(
        "options", {}).setdefault("properties", {})[key] = spec
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
                    "output_path": {
                        "type":        "string",
                        "description": (
                            "Destination file path.  "
                            "Auto-generated inside ./audio/ when omitted."
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
        return out

    def run(self, input: str, options: dict | None = None) -> str:
        raise NotImplementedError(f"{type(self).__name__} must implement run()")

    def stream(self, input: str, options: dict | None = None):
        """Real-time streaming synthesis — reserved for a future release."""
        raise NotImplementedError("Streaming TTS is not yet implemented.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ── ttsOpenAI ─────────────────────────────────────────────────────────────────

class _OpenAISpeech(Speech):
    """The OpenAI speech wire, which xAI speaks too: ``POST {base}/audio/speech``
    with model, voice, input, format and speed; the answer is the audio.

    A provider on it is data — ``_BASE_URL``, ``_ENV_KEY``, ``_LABEL`` and its
    defaults. Plain HTTP, so neither needs the ``openai`` package.
    """

    # The endpoint has no language field — the voice and the model carry it.
    parameters = _without_options(Speech.parameters, "language")

    _BASE_URL: str = ""
    _ENV_KEY:  str = ""
    _LABEL:    str = ""

    def __init__(self, api_key: str | None = None, model: "str | None" = None) -> None:
        self.model = _model_setting(self, model)
        self._api_key = api_key or os.environ.get(self._ENV_KEY)
        if not self._api_key:
            raise ValueError(f"No {self._LABEL} API key found.  "
                             f"Pass api_key= or set the {self._ENV_KEY} environment variable.")
        self._http = make_http(timeout=urllib3.Timeout(connect=10.0, read=60.0),
                               retries=urllib3.Retry(total=2, backoff_factor=0.5))

    def run(self, input: str, options: dict | None = None) -> str:
        """
        Synthesise *input* and save it as an audio file; return its path.

        options: ``voice``, ``speed`` (0.25–4.0, default 1.0), ``format``
        (``"mp3"`` default, ``"wav"``, ``"opus"``, ``"flac"``, ``"pcm"``) and
        ``output_path`` (generated when omitted). The path is checked before
        the request, so a refused one costs nothing.
        """
        _no_model_option(self, options or {})
        opts = options or {}
        fmt  = opts.get("format", _DEFAULT_FORMAT)
        output_path = confine_output_path(opts.get("output_path") or _auto_path(fmt))
        audio = request(self._http, "POST", f"{self._BASE_URL}/audio/speech",
                        json={"model": self.model, "voice": opts.get("voice", self._DEFAULT_VOICE),
                              "input": input, "response_format": fmt,
                              "speed": float(opts.get("speed", _DEFAULT_SPEED))},
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        what=f"{self._LABEL} speech API").data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(audio)
        return output_path


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

    Note: the OpenAI TTS API accepts up to 4 096 characters per request.
    For longer texts, split into paragraphs and call ``run()`` per chunk.

    Parameters
    ----------
    api_key : str | None, optional
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    model : str | None, optional
        ``"tts-1"`` (default) or ``"tts-1-hd"``.  A setting of the tool.

    Examples
    --------
    ::

        from yait_aichain.tools import speech

        tool = speech("openai", model="tts-1-hd")

        # Call-style — returns ToolResult, never raises
        result = tool(
            input="Hello, world!",
            options={"voice": "nova", "output_path": "hello.mp3"},
        )
        if result:
            print("Saved to:", result.output)

        # Direct run — returns path str, raises on error
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
    Text-to-speech via Google Cloud Text-to-Speech.

    Authentication
    --------------
    Set ``GOOGLE_APPLICATION_CREDENTIALS`` to the path of a service-account
    JSON file (Application Default Credentials), or set ``GOOGLE_API_KEY``
    for API-key authentication.

    Voices
    ------
    Full list: https://cloud.google.com/text-to-speech/docs/voices
    Example names: ``"en-US-Neural2-A"``, ``"en-US-Journey-F"``

    Parameters
    ----------
    api_key : str | None, optional
        Google API key.  Falls back to ``GOOGLE_API_KEY`` env var.
        When absent, Application Default Credentials are used automatically.

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
                "output_path": "hello.mp3",
            },
        )
    """

    # Google picks the synthesiser from the voice name; there is no model to
    # name.
    parameters = _without_options(Speech.parameters, "model")

    name              = "ttsGoogle"
    _DEFAULT_LANGUAGE = "en-US"

    # Maps common format names to Google AudioEncoding names and file extensions
    _FORMAT_MAP: dict[str, tuple[str, str]] = {
        "mp3":  ("MP3",      "mp3"),
        "wav":  ("LINEAR16", "wav"),
        "opus": ("OGG_OPUS", "opus"),
        "flac": ("FLAC",     "flac"),
    }

    def __init__(self, api_key: str | None = None, model: "str | None" = None) -> None:
        self.model = _model_setting(self, model)
        # GOOGLE_AI_API_KEY is the Google provider's key; GOOGLE_API_KEY is
        # the older spelling. Embeddings accept both, and so does this.
        self._api_key = (api_key or os.environ.get("GOOGLE_AI_API_KEY")
                         or os.environ.get("GOOGLE_API_KEY"))
        # No ValueError here — ADC may be available even without an explicit key.

    def run(self, input: str, options: dict | None = None) -> str:
        """
        Synthesise *input* text and save to an audio file.

        Parameters
        ----------
        input : str
            Text to synthesise.
        options : dict | None, optional
            ``voice``       — Google voice name, e.g. ``"en-US-Neural2-A"``.
                              When omitted the API picks the default for the
                              language.
            ``language``    — BCP-47 language code.  Default: ``"en-US"``.
            ``speed``       — speaking rate.  0.25–4.0, default 1.0.
            ``format``      — ``"mp3"`` (default), ``"wav"``, ``"opus"``,
                              ``"flac"``.
            ``output_path`` — destination path.  Auto-generated if omitted.

        Returns
        -------
        str
            Absolute path to the saved audio file.
        """
        texttospeech = require("google.cloud.texttospeech", package="google-cloud-texttospeech", who="ttsGoogle")

        opts        = options or {}
        voice_name  = opts.get("voice")
        language    = opts.get("language",  self._DEFAULT_LANGUAGE)
        speed       = float(opts.get("speed", _DEFAULT_SPEED))
        fmt         = opts.get("format",    _DEFAULT_FORMAT)
        output_path = opts.get("output_path")

        g_encoding, ext = self._FORMAT_MAP.get(fmt, ("MP3", "mp3"))
        if not output_path:
            output_path = _auto_path(ext)

        synthesis_input = texttospeech.SynthesisInput(text=input)

        voice_kwargs: dict = {"language_code": language}
        if voice_name:
            voice_kwargs["name"] = voice_name
        voice_params = texttospeech.VoiceSelectionParams(**voice_kwargs)

        audio_config = texttospeech.AudioConfig(
            audio_encoding = getattr(texttospeech.AudioEncoding, g_encoding),
            speaking_rate  = speed,
        )

        if self._api_key:
            client = texttospeech.TextToSpeechClient(
                client_options={"api_key": self._api_key}
            )
        else:
            client = texttospeech.TextToSpeechClient()

        response    = client.synthesize_speech(
            input        = synthesis_input,
            voice        = voice_params,
            audio_config = audio_config,
        )

        output_path = confine_output_path(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(response.audio_content)

        return output_path


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
            options={"output_path": "hello_xai.mp3"},
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
    Text-to-speech via the Alibaba DashScope CosyVoice / Qwen3-TTS API.

    Uses the native DashScope multimodal generation endpoint
    (``POST /api/v1/services/aigc/multimodal-generation/generation``).
    The API returns a temporary presigned audio URL (valid 24 h) which is
    downloaded and saved locally.

    Models
    ------
    ``cosyvoice-v2``           High-quality, multi-voice.  Default.
    ``qwen3-tts-flash``        Qwen3 TTS model — fast and natural.
    ``qwen3-tts-instruct-flash`` Instruction-following TTS.

    Voices (cosyvoice-v2 selection)
    --------------------------------
    longxiaochun · longxiaochun_v2 · loongstella · longhua · longshuo
    longfei · longxiang · longshu · longcheng · longmiao · loongbella
    See the DashScope docs for a full list of 40+ voices.

    Region
    ------
    The base URL is derived from the ``region`` setting, else
    ``DASHSCOPE_REGION`` (ap|us|cn|hk).  Default: ``"ap"``
    (``dashscope-intl.aliyuncs.com``).

    Parameters
    ----------
    api_key : str | None, optional
        DashScope API key.  Falls back to ``DASHSCOPE_API_KEY`` env var.
    region : str | None, optional
        Region selector: ``"ap"`` (default), ``"us"``, ``"cn"``, ``"hk"``.
        Overrides the ``DASHSCOPE_REGION`` env var.
    model : str | None, optional
        Default ``"cosyvoice-v2"``.  A setting of the tool.

    Examples
    --------
    ::

        from yait_aichain.tools import speech

        tool = speech("qwen", region="us")

        path = tool.run(
            input="Hello from Qwen TTS!",
            options={"voice": "longxiaochun", "output_path": "hello.wav"},
        )
    """

    name           = "ttsQwen"
    _DEFAULT_MODEL = "cosyvoice-v2"
    _DEFAULT_VOICE = "longxiaochun"
    _TTS_PATH      = "/api/v1/services/aigc/multimodal-generation/generation"

    _FORMAT_EXT: dict[str, str] = {
        "mp3":  "mp3",
        "wav":  "wav",
        "pcm":  "pcm",
        "flac": "flac",
    }

    # `language` and `speed` are dropped from the shared schema: this
    # endpoint reads neither, and an option a model can set and watch vanish
    # is worse than one it never saw. `region` is a setting of the tool, not
    # an option of a call, so the schema does not carry it either.
    parameters = _without_options(Speech.parameters, "language", "speed")

    def __init__(
        self,
        api_key: "str | None" = None,
        region:  "str | None" = None,
        model:   "str | None" = None,
    ) -> None:
        self.model = _model_setting(self, model)
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "No DashScope API key found.  "
                "Pass api_key= or set the DASHSCOPE_API_KEY environment variable."
            )
        self._region = region

        # Build HTTP client (no extra dependency — pure urllib3)
        self._http = make_http(
            timeout = urllib3.Timeout(connect=10.0, read=60.0),
            retries = urllib3.Retry(total=2, backoff_factor=0.5),
        )

    def _base_url(self) -> str:
        from ...clients._families.qwen import resolve_qwen_base_url
        return resolve_qwen_base_url(self._region)

    def run(self, input: str, options: dict | None = None) -> str:
        """
        Synthesise *input* text and save to an audio file.

        Parameters
        ----------
        input : str
            Text to synthesise.
        options : dict | None, optional
            ``voice``       — voice name.  Default: ``"longxiaochun"``.
            ``format``      — ``"mp3"`` (default), ``"wav"``, ``"pcm"``,
                              ``"flac"``.
            ``output_path`` — destination path.  Auto-generated if omitted.

        Returns
        -------
        str
            Absolute path to the saved audio file.
        """
        _no_model_option(self, options or {})
        opts        = options or {}
        voice       = opts.get("voice",       self._DEFAULT_VOICE)
        model       = self.model
        fmt         = opts.get("format",      _DEFAULT_FORMAT)
        output_path = opts.get("output_path") or _auto_path(fmt)
        if "region" in opts:
            raise ValueError(f"{self.name}: region is a setting of the tool, not of a call — "
                             "speech('qwen', region=…)")
        base_url = self._base_url()

        url = base_url + self._TTS_PATH

        body = {
            "model": model,
            "input": {"text": input},
            "parameters": {
                "voice":  voice,
                "format": fmt,
            },
        }

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
        assert_safe_url(audio_url)
        dl = request(self._http, "GET", audio_url, redirect=False, what="DashScope TTS audio download")

        output_path = confine_output_path(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(dl.data)

        return output_path
