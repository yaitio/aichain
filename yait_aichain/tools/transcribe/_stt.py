"""
tools.transcribe._stt — Transcription
=====================================

Base class and provider implementations for speech-to-text transcription.

Supported providers
-------------------
  sttOpenAI   OpenAI Whisper (whisper-1, gpt-4o-transcribe, …)
  sttGoogle   Google Cloud Speech-to-Text
  sttXAI      xAI STT
  sttQwen     Alibaba DashScope Paraformer / Qwen3-ASR

All implement ``run(input, options=None) -> str`` where:
  input  — path to an audio file (string)
  output — transcript string (or JSON string when timestamps are requested)

Long audio files are automatically chunked, transcribed in pieces, and
stitched back into a single transcript.  The chunk size, overlap, and
size thresholds are class-level constants prefixed with ``_`` — advanced
users can adjust these directly on the class before instantiating.

``sttOpenAI``, ``sttXAI``, and ``sttQwen`` share the internal
``_OpenAICompatSTT`` base class; each simply points to a different
base URL.

Installation
------------
  OpenAI, xAI, Qwen :  nothing extra — plain HTTP
  Google            :  pip install google-cloud-speech
  Chunking          :  pip install pydub   (only when audio exceeds the size limit)

Environment variables
---------------------
  OpenAI  :  OPENAI_API_KEY
  Google  :  GOOGLE_APPLICATION_CREDENTIALS  or  GOOGLE_API_KEY
  xAI     :  XAI_API_KEY
  Qwen    :  DASHSCOPE_API_KEY
             DASHSCOPE_REGION  (optional: ap|us|cn|hk; default ap)
"""

from __future__ import annotations

from ..._require import require

import io
import json
import os
import tempfile

import urllib3

from ...clients._base import make_http
from ...clients._http import request

from .._base import Tool


# ── Base class ────────────────────────────────────────────────────────────────



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

def _without_options(parameters: dict, *keys: str) -> dict:
    """A copy of *parameters* with option *keys* removed — see the twin in
    `to_speech.py`. Declaring an option a provider has no control for
    advertises it to the model, which sets it and watches it vanish."""
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
#
# sttOpenAI and sttXAI both use the OpenAI Python SDK.
# xAI simply points to a different base URL.
# All shared logic lives here; subclasses set class-level variables only.

class _OpenAICompatSTT(Transcription):
    """
    Shared implementation for STT providers with an OpenAI-compatible
    transcriptions endpoint (OpenAI Whisper, xAI STT, DashScope) — one
    multipart POST to ``{_BASE_URL}/audio/transcriptions``, in plain HTTP.

    Subclasses configure behaviour through class-level variables:

    _BASE_URL        str         API base URL.
    _DEFAULT_MODEL   str         Default model name.
    _ENV_KEY         str         Environment variable that holds the API key.
    _CHUNK_SECONDS   int         Max audio duration (seconds) per API call.
    _OVERLAP_SECONDS int         Overlap between consecutive chunks.
    _MAX_FILE_MB     float       Files above this size (MB) are chunked.
    """

    _BASE_URL:        str          = "https://api.openai.com/v1"
    _DEFAULT_MODEL:   str          = "whisper-1"
    _ENV_KEY:         str          = "OPENAI_API_KEY"
    # ── Chunking — advanced: adjust on the class before instantiating ─────
    _CHUNK_SECONDS:   int          = 600    # 10 min per chunk
    _OVERLAP_SECONDS: int          = 3      # overlap to avoid clipping words
    _MAX_FILE_MB:     float        = 24.0   # stay under the 25 MB API limit

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
        """
        Transcribe the audio file at *input* path.

        Parameters
        ----------
        input : str
            Path to the audio file.  Supported: mp3, mp4, mpeg, mpga,
            m4a, wav, webm.
        options : dict | None, optional
            ``language``   — ISO-639-1 or BCP-47 code, e.g. ``"en"``.
            ``timestamps`` — ``True`` for word-level timestamps (JSON output).
            ``prompt``     — context hint for accuracy.
            ``format``     — ``"text"`` (default) or ``"json"``.

        Returns
        -------
        str
            Plain transcript, or JSON string when timestamps/format="json".
        """
        _no_model_option(self, options or {})
        if not os.path.isfile(input):
            raise FileNotFoundError(f"Audio file not found: {input!r}")

        opts       = options or {}
        language   = opts.get("language")
        model      = self.model
        timestamps = bool(opts.get("timestamps", False))
        prompt     = opts.get("prompt")
        fmt        = opts.get("format", "text")

        file_mb = os.path.getsize(input) / (1024 * 1024)
        if file_mb > self._MAX_FILE_MB:
            return self._transcribe_chunked(input, model, language, prompt, timestamps, fmt)
        return self._transcribe_single(input, model, language, prompt, timestamps, fmt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transcribe_single(self, path: str, model: str, language: str | None,
                           prompt: str | None, timestamps: bool, fmt: str) -> str:
        """Send one file to the transcription endpoint and return the result."""
        verbose = timestamps or fmt == "json"
        with open(path, "rb") as fh:
            audio = fh.read()
        fields = [("model", model), ("file", (os.path.basename(path), audio)),
                  ("response_format", "verbose_json" if verbose else "text")]
        fields += [(k, v) for k, v in (("language", language), ("prompt", prompt)) if v]
        if verbose:
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

    def _transcribe_chunked(
        self,
        path:       str,
        model:      str,
        language:   str | None,
        prompt:     str | None,
        timestamps: bool,
        fmt:        str,
    ) -> str:
        """Split oversized audio, transcribe each chunk, stitch results."""
        AudioSegment = require("pydub", package="pydub", who="transcribing a file over the size limit").AudioSegment

        audio      = AudioSegment.from_file(path)
        chunk_ms   = self._CHUNK_SECONDS * 1000
        parts: list[str] = []

        # Chunks are CONTIGUOUS (no overlap): this path joins plain transcript
        # text and has no word-timing data to de-duplicate an overlap region,
        # so overlapping audio would repeat a few seconds of speech at every
        # boundary. A clean cut at a 10-minute boundary risks splitting at most
        # one word — far less harmful than duplicated phrases.
        for start_ms in range(0, len(audio), chunk_ms):
            chunk = audio[start_ms:start_ms + chunk_ms]

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                chunk.export(tmp_path, format="mp3")
                part = self._transcribe_single(tmp_path, model, language, prompt,
                                               timestamps=False, fmt="text")
                parts.append(part.strip())
            finally:
                os.unlink(tmp_path)

        combined = " ".join(p for p in parts if p)
        if fmt == "json" or timestamps:
            return json.dumps({"text": combined}, ensure_ascii=False)
        return combined


# ── sttOpenAI ─────────────────────────────────────────────────────────────────

class sttOpenAI(_OpenAICompatSTT):
    """
    Speech-to-text via the OpenAI Whisper API.

    Audio files larger than 24 MB are automatically split into overlapping
    chunks, transcribed in sequence, and stitched into a single transcript.
    Requires ``pydub`` for chunking (``pip install pydub``).

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

    # ── Chunking settings — advanced: adjust on the class before instantiating
    _BASE_URL        = "https://api.openai.com/v1"
    _DEFAULT_MODEL   = "whisper-1"
    _ENV_KEY         = "OPENAI_API_KEY"
    _CHUNK_SECONDS   = 600
    _OVERLAP_SECONDS = 3
    _MAX_FILE_MB     = 24.0


# ── sttGoogle ─────────────────────────────────────────────────────────────────

class sttGoogle(Transcription):
    """
    Speech-to-text via Google Cloud Speech-to-Text.

    Files up to ~10 MB are transcribed with the synchronous ``recognize``
    call.  Larger files are automatically chunked using ``pydub`` and
    stitched into a single transcript.

    Authentication
    --------------
    Set ``GOOGLE_APPLICATION_CREDENTIALS`` to a service-account JSON path,
    or set ``GOOGLE_API_KEY`` for API-key authentication.
    When neither is set, Application Default Credentials (ADC) are used.

    Parameters
    ----------
    api_key : str | None, optional
        Google API key.  Falls back to ``GOOGLE_API_KEY`` env var.
        When absent ADC is used automatically.
    model : str | None, optional
        Default ``"latest_long"``.  A setting of the tool.

    Examples
    --------
    ::

        from yait_aichain.tools import transcribe

        tool = transcribe("google", model="latest_long")
        text = tool.run(input="recording.wav", options={"language": "en-US"})
    """

    # Google's speech-to-text has no prompt: biasing the transcription is
    # done with a speech context, which is a different shape and is not
    # wired. Declaring `prompt` offered the model something that fell on the
    # floor.
    parameters = _without_options(Transcription.parameters, "prompt")

    name = "sttGoogle"

    # ── Chunking settings — advanced: adjust on the class before instantiating
    _DEFAULT_LANGUAGE = "en-US"
    _DEFAULT_MODEL    = "latest_long"
    _CHUNK_SECONDS    = 55      # Google sync limit is 60 s; stay safely under
    _OVERLAP_SECONDS  = 2
    _MAX_SYNC_SECONDS = 55.0    # sync API hard limit is ~60 s of audio
    _MAX_SYNC_MB      = 10.0    # size fallback when duration can't be probed

    # Maps audio extensions to Google RecognitionConfig encoding names
    _ENCODING_MAP: dict = {
        ".wav":  "LINEAR16",
        ".flac": "FLAC",
        ".mp3":  "MP3",
        ".mp4":  "MP4",
        ".m4a":  "MP4",
        ".webm": "WEBM_OPUS",
        ".ogg":  "OGG_OPUS",
        ".opus": "OGG_OPUS",
    }

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = _model_setting(self, model)
        # GOOGLE_AI_API_KEY is the Google provider's key; GOOGLE_API_KEY is
        # accepted too. Embeddings accept both, and so does this.
        self._api_key = (api_key or os.environ.get("GOOGLE_AI_API_KEY")
                         or os.environ.get("GOOGLE_API_KEY"))
        # No ValueError — ADC may be available without an explicit key.

    def run(self, input: str, options: dict | None = None) -> str:
        """
        Transcribe the audio file at *input* path.

        Parameters
        ----------
        input : str
            Path to the audio file.
        options : dict | None, optional
            ``language``   — BCP-47 code, e.g. ``"en-US"``.  Default: ``"en-US"``.
            ``timestamps`` — ``True`` for word-level timestamps (JSON output).
            ``format``     — ``"text"`` (default) or ``"json"``.

        Returns
        -------
        str
            Plain transcript, or JSON string when timestamps/format="json".
        """
        _no_model_option(self, options or {})
        speech = require("google.cloud.speech", package="google-cloud-speech", who="sttGoogle")

        if not os.path.isfile(input):
            raise FileNotFoundError(f"Audio file not found: {input!r}")

        opts       = options or {}
        language   = opts.get("language", self._DEFAULT_LANGUAGE)
        model      = self.model
        timestamps = bool(opts.get("timestamps", False))
        fmt        = opts.get("format", "text")

        ext      = os.path.splitext(input)[1].lower()
        encoding = self._ENCODING_MAP.get(ext, "MP3")
        file_mb  = os.path.getsize(input) / (1024 * 1024)

        if self._api_key:
            client = speech.SpeechClient(
                client_options={"api_key": self._api_key}
            )
        else:
            client = speech.SpeechClient()

        config = speech.RecognitionConfig(
            encoding                     = getattr(
                speech.RecognitionConfig.AudioEncoding, encoding
            ),
            language_code                = language,
            model                        = model,
            enable_word_time_offsets     = timestamps,
            enable_automatic_punctuation = True,
        )

        # The sync recognize API is limited by audio *duration* (~60 s), not
        # file size: a 5 MB low-bitrate MP3 can hold half an hour of speech.
        # Route by real duration whenever pydub can probe it.
        duration_s = self._probe_duration_seconds(input)
        if duration_s is not None:
            use_sync = duration_s <= self._MAX_SYNC_SECONDS
        else:
            # pydub unavailable — fall back to the size heuristic.  This can
            # mis-route long low-bitrate files into the sync API; install
            # pydub for an exact duration check.
            use_sync = file_mb <= self._MAX_SYNC_MB

        if use_sync:
            with open(input, "rb") as fh:
                content = fh.read()
            audio    = speech.RecognitionAudio(content=content)
            response = client.recognize(config=config, audio=audio)
            return self._format_results(response.results, timestamps, fmt)

        return self._transcribe_chunked(client, config, input, timestamps, fmt)

    @staticmethod
    def _probe_duration_seconds(path: str) -> float | None:
        """Return the audio duration in seconds, or None if unprobeable."""
        try:
            from pydub import AudioSegment
        except ImportError:
            return None
        try:
            return len(AudioSegment.from_file(path)) / 1000.0
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(results, timestamps: bool, fmt: str) -> str:
        if not timestamps and fmt == "text":
            return " ".join(
                r.alternatives[0].transcript
                for r in results
                if r.alternatives
            )
        text  = " ".join(
            r.alternatives[0].transcript
            for r in results
            if r.alternatives
        )
        words = []
        for r in results:
            if not r.alternatives:
                continue
            for w in r.alternatives[0].words:
                words.append({
                    "word":  w.word,
                    "start": w.start_time.total_seconds(),
                    "end":   w.end_time.total_seconds(),
                })
        return json.dumps({"text": text, "words": words}, ensure_ascii=False)

    def _transcribe_chunked(
        self, client, config, path: str, timestamps: bool, fmt: str
    ) -> str:
        AudioSegment = require("pydub", package="pydub", who="transcribing a file over the size limit").AudioSegment

        from google.cloud import speech as _speech

        audio      = AudioSegment.from_file(path)
        chunk_ms   = self._CHUNK_SECONDS   * 1000
        overlap_ms = self._OVERLAP_SECONDS * 1000
        parts: list[str]  = []
        words: list[dict] = []

        # Each chunk starts an overlap before the last one ended, and the last
        # chunk is the one that reaches the end: a further chunk would be the
        # overlap alone, transcribed and billed a second time.
        for start_ms in range(0, len(audio), chunk_ms - overlap_ms):
            chunk = audio[start_ms:start_ms + chunk_ms]
            buffer = io.BytesIO()
            chunk.export(buffer, format="wav")
            content = buffer.getvalue()
            chunk_config = _speech.RecognitionConfig(
                encoding      = _speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code = config.language_code,
                model         = config.model,
                enable_word_time_offsets     = timestamps,
                enable_automatic_punctuation = True,
            )
            resp = client.recognize(
                config = chunk_config,
                audio  = _speech.RecognitionAudio(content=content),
            )
            part = " ".join(
                r.alternatives[0].transcript
                for r in resp.results
                if r.alternatives
            )
            parts.append(part.strip())

            if timestamps:
                offset_s = start_ms / 1000.0
                for r in resp.results:
                    if not r.alternatives:
                        continue
                    for w in r.alternatives[0].words:
                        local_start = w.start_time.total_seconds()
                        # Words inside the leading overlap of a non-first
                        # chunk were already captured by the previous
                        # chunk — drop them to avoid duplicates.
                        if start_ms > 0 and local_start < self._OVERLAP_SECONDS:
                            continue
                        words.append({
                            "word":  w.word,
                            "start": local_start + offset_s,
                            "end":   w.end_time.total_seconds() + offset_s,
                        })
            if start_ms + chunk_ms >= len(audio):
                break

        # NB: the plain-text join may duplicate a word or two at chunk
        # boundaries (the 2 s overlap is transcribed twice); the word list
        # above is deduplicated by timestamp and is the precise record.
        combined = " ".join(p for p in parts if p)
        if fmt == "json" or timestamps:
            payload: dict = {"text": combined}
            if timestamps:
                payload["words"] = words
            return json.dumps(payload, ensure_ascii=False)
        return combined


# ── sttXAI ────────────────────────────────────────────────────────────────────

class sttXAI(_OpenAICompatSTT):
    """
    Speech-to-text via the xAI STT API.

    xAI exposes an OpenAI-compatible transcriptions endpoint.
    See https://docs.x.ai/developers/model-capabilities/audio/voice for
    available model names.

    Audio files larger than 24 MB are automatically chunked.
    Requires ``pydub`` for chunking (``pip install pydub``).

    Parameters
    ----------
    api_key : str | None, optional
        xAI API key.  Falls back to ``XAI_API_KEY`` env var.

    Examples
    --------
    ::

        from yait_aichain.tools import transcribe

        tool = transcribe("xai")
        text = tool.run(input="recording.mp3", options={"language": "en"})
    """

    name = "sttXAI"

    # ── Chunking settings — advanced: adjust on the class before instantiating
    _BASE_URL        = "https://api.x.ai/v1"
    _DEFAULT_MODEL   = "grok-stt"    # check docs.x.ai for current model names
    _ENV_KEY         = "XAI_API_KEY"
    _CHUNK_SECONDS   = 600
    _OVERLAP_SECONDS = 3
    _MAX_FILE_MB     = 24.0


# ── sttQwen ───────────────────────────────────────────────────────────────────

class sttQwen(_OpenAICompatSTT):
    """
    Speech-to-text via the Alibaba DashScope ASR API.

    Uses the OpenAI-compatible transcriptions endpoint exposed at
    ``/compatible-mode/v1/audio/transcriptions``.

    Audio files larger than 24 MB are automatically chunked using ``pydub``
    and stitched into a single transcript (``pip install pydub``).

    Models
    ------
    ``paraformer-v2``   High-accuracy Paraformer ASR model.  Default.
    ``qwen3-asr``       Qwen3 ASR model — strong multilingual accuracy.

    Region
    ------
    The base URL is selected from ``DASHSCOPE_REGION`` (ap|us|cn|hk),
    defaulting to ``"ap"`` (international endpoint).

    Parameters
    ----------
    api_key : str | None, optional
        DashScope API key.  Falls back to ``DASHSCOPE_API_KEY`` env var.

    Examples
    --------
    ::

        from yait_aichain.tools import transcribe

        tool = transcribe("qwen", model="qwen3-asr")
        text = tool.run(input="meeting.mp3", options={"language": "zh"})
    """

    name = "sttQwen"

    # ── Class-level defaults ────────────────────────────────────────────────
    # The base URL is region-dependent; we resolve it lazily in __init__
    # by reading DASHSCOPE_REGION (or the default "ap").
    _BASE_URL        = None          # overwritten in __init__
    _DEFAULT_MODEL   = "paraformer-v2"
    _ENV_KEY         = "DASHSCOPE_API_KEY"
    _CHUNK_SECONDS   = 600
    _OVERLAP_SECONDS = 3
    _MAX_FILE_MB     = 24.0

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = _model_setting(self, model)
        # Resolve the region-aware base URL before calling the parent __init__
        # (which reads self._BASE_URL for its requests).
        from ...clients._families.qwen import resolve_qwen_base_url
        self._BASE_URL = resolve_qwen_base_url() + "/compatible-mode/v1"
        super().__init__(api_key=api_key, model=model)
