"""
tools.convert.to_speech — convertToSpeech
==========================================

Base class and provider implementations for text-to-speech synthesis.

Supported providers
-------------------
  ttsOpenAI   OpenAI TTS (tts-1, tts-1-hd)
  ttsGoogle   Google Cloud Text-to-Speech
  ttsXAI      xAI TTS
  ttsQwen     Alibaba DashScope CosyVoice / Qwen3-TTS

All implement ``run(input, options=None) -> str`` where the return value
is the absolute path to the saved audio file — the same pattern as
``convertToPDF``.

A ``stream()`` placeholder is present on every class.  Real-time
streaming synthesis is reserved for a future release.

Installation
------------
  OpenAI  :  pip install openai
  Google  :  pip install google-cloud-texttospeech
  xAI     :  pip install openai   (uses the OpenAI-compatible xAI endpoint)
  Qwen    :  pip install urllib3  (pure HTTP; no extra SDK needed)

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

import datetime
import json
import os
import urllib3

from .._base import Tool


# ── Module-level defaults ─────────────────────────────────────────────────────

_DEFAULT_FORMAT = "mp3"
_DEFAULT_SPEED  = 1.0


def _auto_path(fmt: str) -> str:
    """Return a timestamped output path inside ``./audio/`` (relative to cwd)."""
    stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), "audio")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"speech_{stamp}.{fmt}")


# ── Base class ────────────────────────────────────────────────────────────────

class convertToSpeech(Tool):
    """
    Synthesise speech from text and save to an audio file.

    Subclasses implement the provider-specific ``run()`` logic.
    All share the same ``options`` schema.

    Returns
    -------
    str
        Absolute path to the saved audio file.
    """

    name        = "convertToSpeech"
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
                    "model": {
                        "type":        "string",
                        "description": "Model or quality tier (provider-specific).",
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
                            "Required for Google; optional for OpenAI and xAI."
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

    def run(self, input: str, options: dict | None = None) -> str:
        raise NotImplementedError(f"{type(self).__name__} must implement run()")

    def stream(self, input: str, options: dict | None = None):
        """Real-time streaming synthesis — reserved for a future release."""
        raise NotImplementedError("Streaming TTS is not yet implemented.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ── ttsOpenAI ─────────────────────────────────────────────────────────────────

class ttsOpenAI(convertToSpeech):
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

    Examples
    --------
    ::

        from tools.convert import ttsOpenAI

        tool = ttsOpenAI()

        # Call-style — returns ToolResult, never raises
        result = tool(
            input="Hello, world!",
            options={"voice": "nova", "model": "tts-1-hd", "output_path": "hello.mp3"},
        )
        if result:
            print("Saved to:", result.output)

        # Direct run — returns path str, raises on error
        path = tool.run(input="Hello!", options={"voice": "echo"})
    """

    name           = "ttsOpenAI"
    _DEFAULT_MODEL = "tts-1"
    _DEFAULT_VOICE = "alloy"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "No OpenAI API key found.  "
                "Pass api_key= or set the OPENAI_API_KEY environment variable."
            )

    def run(self, input: str, options: dict | None = None) -> str:
        """
        Synthesise *input* text and save to an audio file.

        Parameters
        ----------
        input : str
            Text to synthesise (max 4 096 characters).
        options : dict | None, optional
            ``voice``       — voice ID.  Default: ``"alloy"``.
            ``model``       — ``"tts-1"`` (default) or ``"tts-1-hd"``.
            ``speed``       — 0.25–4.0, default 1.0.
            ``format``      — ``"mp3"`` (default), ``"wav"``, ``"opus"``,
                              ``"flac"``, ``"pcm"``.
            ``output_path`` — destination path.  Auto-generated if omitted.

        Returns
        -------
        str
            Absolute path to the saved audio file.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for ttsOpenAI.  "
                "Install it with:  pip install openai"
            )

        opts        = options or {}
        voice       = opts.get("voice",       self._DEFAULT_VOICE)
        model       = opts.get("model",       self._DEFAULT_MODEL)
        speed       = float(opts.get("speed", _DEFAULT_SPEED))
        fmt         = opts.get("format",      _DEFAULT_FORMAT)
        output_path = opts.get("output_path") or _auto_path(fmt)

        client   = openai.OpenAI(api_key=self._api_key)
        response = client.audio.speech.create(
            model           = model,
            voice           = voice,
            input           = input,
            response_format = fmt,
            speed           = speed,
        )

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        response.stream_to_file(output_path)

        return output_path


# ── ttsGoogle ─────────────────────────────────────────────────────────────────

class ttsGoogle(convertToSpeech):
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

        from tools.convert import ttsGoogle

        tool = ttsGoogle()
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

    name              = "ttsGoogle"
    _DEFAULT_LANGUAGE = "en-US"

    # Maps common format names to Google AudioEncoding names and file extensions
    _FORMAT_MAP: dict[str, tuple[str, str]] = {
        "mp3":  ("MP3",      "mp3"),
        "wav":  ("LINEAR16", "wav"),
        "opus": ("OGG_OPUS", "opus"),
        "flac": ("FLAC",     "flac"),
    }

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
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
        try:
            from google.cloud import texttospeech
        except ImportError:
            raise ImportError(
                "google-cloud-texttospeech is required for ttsGoogle.  "
                "Install it with:  pip install google-cloud-texttospeech"
            )

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

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(response.audio_content)

        return output_path


# ── ttsXAI ────────────────────────────────────────────────────────────────────

class ttsXAI(convertToSpeech):
    """
    Text-to-speech via the xAI TTS API.

    xAI exposes an OpenAI-compatible audio endpoint.
    See https://docs.x.ai/developers/model-capabilities/audio/voice for
    available model names and voice IDs.

    Parameters
    ----------
    api_key : str | None, optional
        xAI API key.  Falls back to ``XAI_API_KEY`` env var.

    Examples
    --------
    ::

        from tools.convert import ttsXAI

        tool = ttsXAI()
        path = tool.run(
            input="Hello from xAI!",
            options={"output_path": "hello_xai.mp3"},
        )
    """

    name           = "ttsXAI"
    _BASE_URL      = "https://api.x.ai/v1"
    _DEFAULT_MODEL = "grok-tts"    # check docs.x.ai for current model names
    _DEFAULT_VOICE = "default"     # check docs.x.ai for available voice IDs

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "No xAI API key found.  "
                "Pass api_key= or set the XAI_API_KEY environment variable."
            )

    def run(self, input: str, options: dict | None = None) -> str:
        """
        Synthesise *input* text and save to an audio file.

        Parameters
        ----------
        input : str
            Text to synthesise.
        options : dict | None, optional
            ``voice``       — voice ID.  Default: ``"default"``.
            ``model``       — model name.  Default: ``"grok-tts"``.
            ``speed``       — 0.25–4.0, default 1.0.
            ``format``      — ``"mp3"`` (default), ``"wav"``, ``"opus"``.
            ``output_path`` — destination path.  Auto-generated if omitted.

        Returns
        -------
        str
            Absolute path to the saved audio file.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for ttsXAI.  "
                "Install it with:  pip install openai"
            )

        opts        = options or {}
        voice       = opts.get("voice",  self._DEFAULT_VOICE)
        model       = opts.get("model",  self._DEFAULT_MODEL)
        speed       = float(opts.get("speed", _DEFAULT_SPEED))
        fmt         = opts.get("format", _DEFAULT_FORMAT)
        output_path = opts.get("output_path") or _auto_path(fmt)

        client   = openai.OpenAI(api_key=self._api_key, base_url=self._BASE_URL)
        response = client.audio.speech.create(
            model           = model,
            voice           = voice,
            input           = input,
            response_format = fmt,
            speed           = speed,
        )

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        response.stream_to_file(output_path)

        return output_path


# ── ttsQwen ───────────────────────────────────────────────────────────────────

class ttsQwen(convertToSpeech):
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
    The base URL is derived from ``DASHSCOPE_REGION`` (ap|us|cn|hk)
    or the ``region`` option key.  Default: ``"ap"``
    (``dashscope-intl.aliyuncs.com``).

    Parameters
    ----------
    api_key : str | None, optional
        DashScope API key.  Falls back to ``DASHSCOPE_API_KEY`` env var.
    region : str | None, optional
        Region selector: ``"ap"`` (default), ``"us"``, ``"cn"``, ``"hk"``.
        Overrides the ``DASHSCOPE_REGION`` env var.

    Examples
    --------
    ::

        from tools.convert import ttsQwen

        tool = ttsQwen()

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

    def __init__(
        self,
        api_key: "str | None" = None,
        region:  "str | None" = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "No DashScope API key found.  "
                "Pass api_key= or set the DASHSCOPE_API_KEY environment variable."
            )
        self._region = region

        # Build HTTP client (no extra dependency — pure urllib3)
        self._http = urllib3.PoolManager(
            timeout = urllib3.Timeout(connect=10.0, read=60.0),
            retries = urllib3.Retry(total=2, backoff_factor=0.5),
        )

    def _base_url(self) -> str:
        from clients._qwen import resolve_qwen_base_url
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
            ``model``       — model name.  Default: ``"cosyvoice-v2"``.
            ``format``      — ``"mp3"`` (default), ``"wav"``, ``"pcm"``,
                              ``"flac"``.
            ``region``      — override region: ``"ap"``/``"us"``/``"cn"``/``"hk"``.
            ``output_path`` — destination path.  Auto-generated if omitted.

        Returns
        -------
        str
            Absolute path to the saved audio file.
        """
        opts        = options or {}
        voice       = opts.get("voice",       self._DEFAULT_VOICE)
        model       = opts.get("model",       self._DEFAULT_MODEL)
        fmt         = opts.get("format",      _DEFAULT_FORMAT)
        output_path = opts.get("output_path") or _auto_path(fmt)
        region      = opts.get("region",      self._region)

        # Override region from options if provided
        if region and region != self._region:
            from clients._qwen import resolve_qwen_base_url
            base_url = resolve_qwen_base_url(region)
        else:
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

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
            "X-DashScope-SSE": "disable",
        }

        # ── POST to DashScope ─────────────────────────────────────────
        resp = self._http.request(
            "POST", url,
            body    = json.dumps(body).encode("utf-8"),
            headers = headers,
        )
        raw = resp.data.decode("utf-8", errors="replace")
        if not (200 <= resp.status < 300):
            raise RuntimeError(
                f"ttsQwen: HTTP {resp.status} from DashScope TTS endpoint: {raw[:500]}"
            )

        data = json.loads(raw)

        # ── Extract audio URL from response ───────────────────────────
        # Shape: {"output": {"audio": {"url": "https://...", ...}}, ...}
        try:
            audio_url = data["output"]["audio"]["url"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"ttsQwen: unexpected response shape — could not find "
                f"output.audio.url: {raw[:300]}"
            ) from exc

        # ── Download audio bytes ──────────────────────────────────────
        dl = self._http.request("GET", audio_url)
        if not (200 <= dl.status < 300):
            raise RuntimeError(
                f"ttsQwen: failed to download audio from presigned URL "
                f"(HTTP {dl.status})"
            )

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(dl.data)

        return output_path
