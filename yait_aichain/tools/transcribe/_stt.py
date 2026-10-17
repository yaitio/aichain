"""
``Transcription``, the base of the speech-to-text tools, and its providers:
``sttOpenAI`` and ``sttGoogle``, both plain HTTP.

``run()`` takes the path of an audio file and returns the transcript, or a JSON
string when timestamps or ``format="json"`` are asked. A file over a provider's
limit is refused with the limit named; cutting it is the caller's — see "Long
audio" in docs/tools/media.md.
"""

from __future__ import annotations


import json
import os

import urllib3

from ...clients._base import make_http
from ...clients._http import request, request_json

from .._security import check_input_file
from .._base import Tool

#: Where a recording over a provider's limit is shown how to cut it.
_LONG_AUDIO = ("cut it into pieces and transcribe each — a wrapped tool that does it is "
               "under \"Long audio\" in docs/tools/media.md")


# ── Base class ────────────────────────────────────────────────────────────────



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

def _without_options(parameters: dict, *keys: str) -> dict:
    """A copy of *parameters* without option *keys* (the twin of the one in
    ``speech/_tts.py``): an option advertised and not read is one a model loses."""
    import copy
    out = copy.deepcopy(parameters)
    options = out.get("properties", {}).get("options", {}).get("properties", {})
    for key in keys:
        options.pop(key, None)
    return out


class Transcription(Tool):
    """
    Transcribe an audio file to text.

    Subclasses implement the provider-specific ``run()`` logic.
    All share the same ``options`` schema.

    Returns
    -------
    str
        Plain transcript string, or a JSON string when
        ``options={"timestamps": True}`` or ``options={"format": "json"}``
        is requested.
    """

    risk = "external"   # sends a local audio file to the provider
    priced = True

    name        = "transcription"
    description = (
        "Transcribe an audio file to text using an STT model.  "
        "Input is a file path; output is the transcript string."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "Path to the audio file to transcribe.",
            },
            "options": {
                "type":        "object",
                "description": "STT options.",
                "properties": {
                    "language": {
                        "type":        "string",
                        "description": (
                            "Language hint (ISO-639-1 or BCP-47), e.g. 'en', 'en-US'.  "
                            "Improves accuracy.  Auto-detected when omitted."
                        ),
                    },
                    "timestamps": {
                        "type":        "boolean",
                        "description": (
                            "Include word-level timestamps.  "
                            "When True the output is a JSON string.  Default: false."
                        ),
                    },
                    "prompt": {
                        "type":        "string",
                        "description": (
                            "Context hint — domain vocabulary or spelling guidance.  "
                            "Supported by OpenAI and xAI."
                        ),
                    },
                    "format": {
                        "type":        "string",
                        "enum":        ["text", "json"],
                        "description": (
                            "Output format.  "
                            "``text`` (default) — plain transcript.  "
                            "``json`` — includes timestamps and segment metadata."
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
        """Real-time streaming transcription — reserved for a future release."""
        raise NotImplementedError("Streaming STT is not yet implemented.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ── Internal base: OpenAI-compatible STT ─────────────────────────────────────

class _OpenAICompatSTT(Transcription):
    """The OpenAI transcriptions wire: one multipart POST to
    ``{_BASE_URL}/audio/transcriptions`` in plain HTTP. A provider on it sets the
    URL, default model, key variable and size limit as class variables."""

    _BASE_URL:        str          = "https://api.openai.com/v1"
    _DEFAULT_MODEL:   str          = "whisper-1"
    _ENV_KEY:         str          = "OPENAI_API_KEY"
    _MAX_FILE_MB:     float        = 25.0   # the API's own limit

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = _model_setting(self, model)
        self._api_key = api_key or os.environ.get(self._ENV_KEY)
        if not self._api_key:
            raise ValueError(
                f"No API key found for {type(self).__name__}.  "
                f"Pass api_key= or set the {self._ENV_KEY} environment variable."
            )
        self._http = make_http(timeout=urllib3.Timeout(connect=10.0, read=600.0),
                               retries=urllib3.Retry(total=2, backoff_factor=0.5))

    def run(self, input: str, options: dict | None = None) -> str:
        """Transcribe the audio file at *input* path.

        Parameters
        ----------
        input : str
            Path to the audio file: mp3, mp4, mpeg, mpga, m4a, wav or webm.
        options : dict | None, optional
            ``language`` — e.g. ``"en"``; ``timestamps`` — word-level, as JSON;
            ``prompt`` — a context hint; ``format`` — ``"text"`` (default) or ``"json"``.

        Returns
        -------
        str
            Plain transcript, or JSON string when timestamps/format="json".
        """
        _no_model_option(self, options or {})
        input = check_input_file(input, "audio")

        opts       = options or {}
        language   = opts.get("language")
        model      = self.model
        timestamps = bool(opts.get("timestamps", False))
        prompt     = opts.get("prompt")
        fmt        = opts.get("format", "text")

        if fmt not in ("text", "json"):
            raise ValueError(f"{type(self).__name__}: format is 'text' or 'json', not {fmt!r}")
        if timestamps and model.startswith("gpt-4o"):
            # verbose_json is whisper-1's alone; gpt-4o-transcribe answers 400.
            raise ValueError(f"{model} gives no word timestamps; use model='whisper-1' for them")
        file_mb = os.path.getsize(input) / (1024 * 1024)
        if file_mb > self._MAX_FILE_MB:
            raise ValueError(f"{type(self).__name__} takes a file up to {self._MAX_FILE_MB:.0f} MB, "
                             f"the provider's limit; this one is {file_mb:.1f} MB — {_LONG_AUDIO}")
        return self._transcribe_single(input, model, language, prompt, timestamps, fmt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transcribe_single(self, path: str, model: str, language: str | None,
                           prompt: str | None, timestamps: bool, fmt: str) -> str:
        """Send one file to the transcription endpoint and return the result."""
        verbose = timestamps or fmt == "json"
        # gpt-4o-transcribe takes `json`, never `verbose_json`.
        rich = verbose and not model.startswith("gpt-4o")
        with open(path, "rb") as fh:
            audio = fh.read()
        fields = [("model", model), ("file", (os.path.basename(path), audio)),
                  ("response_format", "verbose_json" if rich else "json" if verbose else "text")]
        fields += [(k, v) for k, v in (("language", language), ("prompt", prompt)) if v]
        if rich:
            fields.append(("timestamp_granularities[]", "word"))
        raw = request(self._http, "POST", f"{self._BASE_URL}/audio/transcriptions", fields=fields,
                      headers={"Authorization": f"Bearer {self._api_key}"},
                      what=f"{type(self).__name__} transcription API").data.decode("utf-8")
        if not verbose:
            return raw
        data = json.loads(raw)
        return json.dumps({
            "text":     data.get("text", ""),
            "words":    [{"word": w.get("word"), "start": w.get("start"), "end": w.get("end")}
                         for w in data.get("words") or []],
            "segments": [{"text": x.get("text"), "start": x.get("start"), "end": x.get("end")}
                         for x in data.get("segments") or []],
        }, ensure_ascii=False)


# ── sttOpenAI ─────────────────────────────────────────────────────────────────

class sttOpenAI(_OpenAICompatSTT):
    """
    Speech-to-text via the OpenAI Whisper API.

    A file over 25 MB, the API's limit, is refused with its size; cutting it
    is the caller's (see "Long audio" in docs/tools/media.md).

    Models
    ------
    ``whisper-1``              Standard Whisper model.  Default.
    ``gpt-4o-transcribe``      GPT-4o based — higher accuracy.
    ``gpt-4o-mini-transcribe`` Faster and cheaper GPT-4o variant.

    Parameters
    ----------
    api_key : str | None, optional
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    model : str | None, optional
        One of the models above.  A setting of the tool.

    Examples
    --------
    ::

        from yait_aichain.tools import transcribe

        tool = transcribe("openai")

        # Plain transcript
        text = tool.run(input="meeting.mp3", options={"language": "en"})

        # With word-level timestamps — returns JSON string
        import json
        data  = transcribe("openai", model="gpt-4o-transcribe").run(
            input="meeting.mp3",
            options={"timestamps": True},
        )
        words = json.loads(data)["words"]
    """

    name = "sttOpenAI"

    _BASE_URL        = "https://api.openai.com/v1"
    _DEFAULT_MODEL   = "whisper-1"
    _ENV_KEY         = "OPENAI_API_KEY"


# ── sttGoogle ─────────────────────────────────────────────────────────────────

class sttGoogle(Transcription):
    """
    Speech-to-text via Google Cloud Speech-to-Text, over its REST API.

    Audio up to a minute and 10 MB goes in one ``speech:recognize`` request —
    the synchronous API's limits; longer audio is refused, and cutting it is
    the caller's (see "Long audio" in docs/tools/media.md).

    One plain HTTPS request; no Google SDK. The key is an API key
    with the Cloud Speech-to-Text API enabled on its project.

    Parameters
    ----------
    api_key : str | None, optional
        Google Cloud API key.  Falls back to ``GOOGLE_CLOUD_API_KEY``, then
        ``GOOGLE_AI_API_KEY``, then ``GOOGLE_API_KEY``.
    model : str | None, optional
        Default ``"latest_long"``.  A setting of the tool.

    Examples
    --------
    ::

        from yait_aichain.tools import transcribe

        tool = transcribe("google", model="latest_long")
        text = tool.run(input="recording.wav", options={"language": "en-US"})
    """

    # Google's speech-to-text has no prompt: biasing is done with a speech
    # context, a different shape and not wired, so `prompt` is not offered.
    parameters = _without_options(Transcription.parameters, "prompt")

    name = "sttGoogle"

    _URL              = "https://speech.googleapis.com/v1/speech:recognize"
    # ── Chunking settings — advanced: adjust on the class before instantiating
    _DEFAULT_LANGUAGE = "en-US"
    _DEFAULT_MODEL    = "latest_long"
    _MAX_SYNC_SECONDS = 60.0    # the synchronous API's limits
    _MAX_SYNC_MB      = 10.0

    # Maps audio extensions to Google RecognitionConfig encoding names
    _ENCODING_MAP: dict = {
        ".wav":  "LINEAR16",
        ".flac": "FLAC",
        ".mp3":  "MP3",
        ".webm": "WEBM_OPUS",
        ".ogg":  "OGG_OPUS",
        ".opus": "OGG_OPUS",
    }

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = _model_setting(self, model)
        # GOOGLE_CLOUD_API_KEY first: speech is a Cloud API, and an AI Studio
        # key is often restricted to Gemini, which answers
        # API_KEY_SERVICE_BLOCKED here. GOOGLE_AI_API_KEY is the Google
        # provider's key and GOOGLE_API_KEY its older spelling.
        self._api_key = (api_key or os.environ.get("GOOGLE_CLOUD_API_KEY")
                         or os.environ.get("GOOGLE_AI_API_KEY")
                         or os.environ.get("GOOGLE_API_KEY"))
        if not self._api_key:
            raise ValueError("No Google API key found. Pass api_key= or set GOOGLE_CLOUD_API_KEY "
                             "(or GOOGLE_AI_API_KEY); the key's project needs the Speech-to-Text API.")
        self._http = make_http(timeout=urllib3.Timeout(connect=10.0, read=120.0),
                               retries=urllib3.Retry(total=2, backoff_factor=0.5))

    def run(self, input: str, options: dict | None = None) -> str:
        """Transcribe the audio file at *input* path.

        Parameters
        ----------
        input : str
            Path to the audio file.
        options : dict | None, optional
            ``language`` — BCP-47, default ``"en-US"``; ``timestamps`` — word-level,
            as JSON; ``format`` — ``"text"`` (default) or ``"json"``.

        Returns
        -------
        str
            Plain transcript, or JSON string when timestamps/format="json".
        """
        _no_model_option(self, options or {})
        input = check_input_file(input, "audio")

        opts       = {k: v for k, v in (options or {}).items() if v is not None}
        language   = opts.get("language", self._DEFAULT_LANGUAGE)
        timestamps = bool(opts.get("timestamps", False))
        fmt        = opts.get("format", "text")
        if fmt not in ("text", "json"):
            raise ValueError(f"sttGoogle: format is 'text' or 'json', not {fmt!r}")

        ext = os.path.splitext(input)[1].lower()
        if ext not in self._ENCODING_MAP:
            # The Speech API has no MP4 encoding; an unknown extension is
            # refused rather than sent under a guessed one.
            raise ValueError(f"sttGoogle reads {sorted(self._ENCODING_MAP)}; {ext or 'no extension'} "
                             "is not one — convert the file first")
        config = {"encoding": self._ENCODING_MAP[ext], "languageCode": language,
                  "model": self.model, "enableWordTimeOffsets": timestamps,
                  "enableAutomaticPunctuation": True}

        # The sync API is limited by duration, which only a WAV states without a
        # decoder, and by the request's size, which every file does.
        size_mb = os.path.getsize(input) / (1024 * 1024)
        seconds = _wav_seconds(input) if ext == ".wav" else None
        if size_mb > self._MAX_SYNC_MB or (seconds or 0) > self._MAX_SYNC_SECONDS:
            what = f"{seconds:.0f} s" if seconds else f"{size_mb:.1f} MB"
            raise ValueError(f"sttGoogle takes up to {self._MAX_SYNC_SECONDS:.0f} s and "
                             f"{self._MAX_SYNC_MB:.0f} MB of audio in one request; this file is "
                             f"{what} — {_LONG_AUDIO}")
        with open(input, "rb") as fh:
            results = self._recognize(config, fh.read())
        return self._format_results(results, timestamps, fmt)

    def _recognize(self, config: dict, audio: bytes) -> list:
        """One ``speech:recognize`` request; the results it answered with."""
        import base64
        data = request_json(self._http, "POST", self._URL,
                            json={"config": config,
                                  "audio": {"content": base64.b64encode(audio).decode("ascii")}},
                            headers={"x-goog-api-key": self._api_key},
                            what="Google Speech-to-Text API")
        return data.get("results") or []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seconds(value) -> float:
        """A protobuf Duration as JSON writes it — ``"1.500s"`` — in seconds."""
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value or "0").rstrip("s") or 0)

    @classmethod
    def _words(cls, results: list) -> list:
        words = []
        for r in results:
            alts = r.get("alternatives") or []
            for w in (alts[0].get("words") or []) if alts else []:
                words.append({"word": w.get("word", ""),
                              "start": cls._seconds(w.get("startTime")),
                              "end": cls._seconds(w.get("endTime"))})
        return words

    @staticmethod
    def _text(results: list) -> str:
        return " ".join(r["alternatives"][0].get("transcript", "")
                        for r in results if r.get("alternatives"))

    @classmethod
    def _format_results(cls, results: list, timestamps: bool, fmt: str) -> str:
        text = cls._text(results)
        if not timestamps and fmt == "text":
            return text
        return json.dumps({"text": text, "words": cls._words(results)}, ensure_ascii=False)


def _wav_seconds(path: str) -> float | None:
    """A WAV file's duration from its header, or None when it cannot be read."""
    import wave
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError, ZeroDivisionError):
        return None
