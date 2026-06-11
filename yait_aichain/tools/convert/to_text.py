"""
tools.convert.to_text — convertToText
======================================

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
  OpenAI  :  pip install openai
  Google  :  pip install google-cloud-speech
  xAI     :  pip install openai   (uses the OpenAI-compatible xAI endpoint)
  Qwen    :  pip install openai   (uses the DashScope compatible-mode endpoint)
  Chunking:  pip install pydub    (only needed when audio exceeds size limit)

Environment variables
---------------------
  OpenAI  :  OPENAI_API_KEY
  Google  :  GOOGLE_APPLICATION_CREDENTIALS  or  GOOGLE_API_KEY
  xAI     :  XAI_API_KEY
  Qwen    :  DASHSCOPE_API_KEY
             DASHSCOPE_REGION  (optional: ap|us|cn|hk; default ap)
"""

from __future__ import annotations

import json
import os
import tempfile

from .._base import Tool


# ── Base class ────────────────────────────────────────────────────────────────

class convertToText(Tool):
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

    name        = "convertToText"
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
                    "model": {
                        "type":        "string",
                        "description": "Model or tier to use (provider-specific).",
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

class _OpenAICompatSTT(convertToText):
    """
    Shared implementation for STT providers with an OpenAI-compatible
    transcriptions endpoint (OpenAI Whisper, xAI STT).

    Subclasses configure behaviour through class-level variables:

    _BASE_URL        str | None  API base URL.  None = official OpenAI endpoint.
    _DEFAULT_MODEL   str         Default model name.
    _ENV_KEY         str         Environment variable that holds the API key.
    _CHUNK_SECONDS   int         Max audio duration (seconds) per API call.
    _OVERLAP_SECONDS int         Overlap between consecutive chunks.
    _MAX_FILE_MB     float       Files above this size (MB) are chunked.
    """

    _BASE_URL:        "str | None" = None
    _DEFAULT_MODEL:   str          = "whisper-1"
    _ENV_KEY:         str          = "OPENAI_API_KEY"
    # ── Chunking — advanced: adjust on the class before instantiating ─────
    _CHUNK_SECONDS:   int          = 600    # 10 min per chunk
    _OVERLAP_SECONDS: int          = 3      # overlap to avoid clipping words
    _MAX_FILE_MB:     float        = 24.0   # stay under the 25 MB API limit

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get(self._ENV_KEY)
        if not self._api_key:
            raise ValueError(
                f"No API key found for {type(self).__name__}.  "
                f"Pass api_key= or set the {self._ENV_KEY} environment variable."
            )

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
            ``model``      — model name.  Uses class default if omitted.
            ``timestamps`` — ``True`` for word-level timestamps (JSON output).
            ``prompt``     — context hint for accuracy.
            ``format``     — ``"text"`` (default) or ``"json"``.

        Returns
        -------
        str
            Plain transcript, or JSON string when timestamps/format="json".
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for this STT tool.  "
                "Install it with:  pip install openai"
            )

        if not os.path.isfile(input):
            raise FileNotFoundError(f"Audio file not found: {input!r}")

        opts       = options or {}
        language   = opts.get("language")
        model      = opts.get("model",      self._DEFAULT_MODEL)
        timestamps = bool(opts.get("timestamps", False))
        prompt     = opts.get("prompt")
        fmt        = opts.get("format", "text")

        client_kwargs: dict = {"api_key": self._api_key}
        if self._BASE_URL:
            client_kwargs["base_url"] = self._BASE_URL
        import openai as _openai
        client = _openai.OpenAI(**client_kwargs)

        file_mb = os.path.getsize(input) / (1024 * 1024)
        if file_mb > self._MAX_FILE_MB:
            return self._transcribe_chunked(
                client, input, model, language, prompt, timestamps, fmt
            )
        return self._transcribe_single(
            client, input, model, language, prompt, timestamps, fmt
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transcribe_single(
        self,
        client,
        path:       str,
        model:      str,
        language:   "str | None",
        prompt:     "str | None",
        timestamps: bool,
        fmt:        str,
    ) -> str:
        """Send one file to the transcription endpoint and return the result."""
        api_kwargs: dict = {"model": model}
        if language:
            api_kwargs["language"] = language
        if prompt:
            api_kwargs["prompt"] = prompt

        with open(path, "rb") as audio_file:
            api_kwargs["file"] = audio_file

            if timestamps or fmt == "json":
                api_kwargs["response_format"]        = "verbose_json"
                api_kwargs["timestamp_granularities"] = ["word"]
                result = client.audio.transcriptions.create(**api_kwargs)
                return json.dumps(
                    {
                        "text":     result.text,
                        "words":    [
                            {
                                "word":  w.word,
                                "start": w.start,
                                "end":   w.end,
                            }
                            for w in (result.words or [])
                        ],
                        "segments": [
                            {
                                "text":  s.text,
                                "start": s.start,
                                "end":   s.end,
                            }
                            for s in (result.segments or [])
                        ],
                    },
                    ensure_ascii=False,
                )
            else:
                api_kwargs["response_format"] = "text"
                return client.audio.transcriptions.create(**api_kwargs)

    def _transcribe_chunked(
        self,
        client,
        path:       str,
        model:      str,
        language:   "str | None",
        prompt:     "str | None",
        timestamps: bool,
        fmt:        str,
    ) -> str:
        """Split oversized audio, transcribe each chunk, stitch results."""
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub is required to transcribe audio files larger than "
                f"{self._MAX_FILE_MB:.0f} MB.  "
                "Install it with:  pip install pydub"
            )

        audio      = AudioSegment.from_file(path)
        chunk_ms   = self._CHUNK_SECONDS   * 1000
        overlap_ms = self._OVERLAP_SECONDS * 1000
        parts: list[str] = []
        start_ms = 0

        while start_ms < len(audio):
            end_ms = min(start_ms + chunk_ms, len(audio))
            chunk  = audio[start_ms:end_ms]

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                chunk.export(tmp_path, format="mp3")
                part = self._transcribe_single(
                    client, tmp_path, model, language, prompt,
                    timestamps=False, fmt="text",
                )
                parts.append(part.strip())
            finally:
                os.unlink(tmp_path)

            next_start = end_ms - overlap_ms
            if next_start <= start_ms:
                break   # final chunk — guard against infinite loop
            start_ms = next_start

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

    Examples
    --------
    ::

        from tools.convert import sttOpenAI

        tool = sttOpenAI()

        # Plain transcript
        text = tool.run(input="meeting.mp3", options={"language": "en"})

        # With word-level timestamps — returns JSON string
        import json
        data  = tool.run(
            input="meeting.mp3",
            options={"model": "gpt-4o-transcribe", "timestamps": True},
        )
        words = json.loads(data)["words"]
    """

    name = "sttOpenAI"

    # ── Chunking settings — advanced: adjust on the class before instantiating
    _BASE_URL        = None
    _DEFAULT_MODEL   = "whisper-1"
    _ENV_KEY         = "OPENAI_API_KEY"
    _CHUNK_SECONDS   = 600
    _OVERLAP_SECONDS = 3
    _MAX_FILE_MB     = 24.0


# ── sttGoogle ─────────────────────────────────────────────────────────────────

class sttGoogle(convertToText):
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

    Examples
    --------
    ::

        from tools.convert import sttGoogle

        tool = sttGoogle()
        text = tool.run(
            input="recording.wav",
            options={"language": "en-US", "model": "latest_long"},
        )
    """

    name = "sttGoogle"

    # ── Chunking settings — advanced: adjust on the class before instantiating
    _DEFAULT_LANGUAGE = "en-US"
    _DEFAULT_MODEL    = "latest_long"
    _CHUNK_SECONDS    = 55      # Google sync limit is 60 s; stay safely under
    _OVERLAP_SECONDS  = 2
    _MAX_SYNC_MB      = 10.0    # files above this threshold are chunked

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

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
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
            ``model``      — default ``"latest_long"``.
            ``timestamps`` — ``True`` for word-level timestamps (JSON output).
            ``format``     — ``"text"`` (default) or ``"json"``.

        Returns
        -------
        str
            Plain transcript, or JSON string when timestamps/format="json".
        """
        try:
            from google.cloud import speech
        except ImportError:
            raise ImportError(
                "google-cloud-speech is required for sttGoogle.  "
                "Install it with:  pip install google-cloud-speech"
            )

        if not os.path.isfile(input):
            raise FileNotFoundError(f"Audio file not found: {input!r}")

        opts       = options or {}
        language   = opts.get("language", self._DEFAULT_LANGUAGE)
        model      = opts.get("model",    self._DEFAULT_MODEL)
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

        if file_mb <= self._MAX_SYNC_MB:
            with open(input, "rb") as fh:
                content = fh.read()
            audio    = speech.RecognitionAudio(content=content)
            response = client.recognize(config=config, audio=audio)
            return self._format_results(response.results, timestamps, fmt)

        return self._transcribe_chunked(client, config, input, timestamps, fmt)

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
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub is required to transcribe audio files larger than "
                f"{self._MAX_SYNC_MB:.0f} MB with Google STT.  "
                "Install it with:  pip install pydub"
            )

        from google.cloud import speech as _speech

        audio      = AudioSegment.from_file(path)
        chunk_ms   = self._CHUNK_SECONDS   * 1000
        overlap_ms = self._OVERLAP_SECONDS * 1000
        parts: list[str] = []
        start_ms = 0

        while start_ms < len(audio):
            end_ms = min(start_ms + chunk_ms, len(audio))
            chunk  = audio[start_ms:end_ms]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                chunk.export(tmp_path, format="wav")
                with open(tmp_path, "rb") as fh:
                    content = fh.read()
                chunk_config = _speech.RecognitionConfig(
                    encoding      = _speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    language_code = config.language_code,
                    model         = config.model,
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
            finally:
                os.unlink(tmp_path)

            next_start = end_ms - overlap_ms
            if next_start <= start_ms:
                break
            start_ms = next_start

        combined = " ".join(p for p in parts if p)
        if fmt == "json" or timestamps:
            return json.dumps({"text": combined}, ensure_ascii=False)
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

        from tools.convert import sttXAI

        tool = sttXAI()
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

        from tools.convert import sttQwen

        tool = sttQwen()
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

    def __init__(self, api_key: "str | None" = None) -> None:
        # Resolve the region-aware base URL before calling the parent __init__
        # (which reads self._BASE_URL to build the OpenAI client).
        from clients._qwen import resolve_qwen_base_url
        self._BASE_URL = resolve_qwen_base_url() + "/compatible-mode/v1"
        super().__init__(api_key=api_key)
