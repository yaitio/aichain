"""
``Transcription``, the base of the speech-to-text tools, and its providers:
``sttOpenAI`` and ``sttGoogle``, both plain HTTP.

``run()`` takes the path of an audio file and returns the transcript, or a JSON
string when timestamps or ``format="json"`` are asked. Audio over a provider's
limit is cut into pieces and the transcripts joined, which needs
``pip install "yait-aichain[audio]"``. See docs/tools/media.md.
"""

from __future__ import annotations

from ..._require import require

import io
import json
import os
import tempfile

import urllib3

from ...clients._base import make_http
from ...clients._http import request, request_json

from .._security import check_input_file
from .._base import Tool


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
    URL, default model, key variable and chunking constants as class variables."""

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
            if timestamps:
                # Chunks are joined as text, so the result's keys would
                # depend on the file's size.
                raise ValueError(f"{type(self).__name__}: word timestamps are not available for a "
                                 f"file over {self._MAX_FILE_MB:.0f} MB; split it first")
            return self._transcribe_chunked(input, model, language, prompt, timestamps, fmt)
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
        AudioSegment = require("pydub", package="pydub", who="transcribing a file over the size limit", extra="audio").AudioSegment

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
    Speech-to-text via Google Cloud Speech-to-Text, over its REST API.

    Audio up to about a minute goes in one ``speech:recognize`` request.
    Longer audio is cut into pieces of under a minute with ``pydub``
    (``pip install "yait-aichain[audio]"``) and the transcripts are joined.

    One plain HTTPS request per piece; no Google SDK. The key is an API key
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
    _CHUNK_SECONDS    = 55      # Google sync limit is 60 s; stay safely under
    _OVERLAP_SECONDS  = 2
    _MAX_SYNC_SECONDS = 55.0    # sync API hard limit is ~60 s of audio
    _MAX_SYNC_MB      = 10.0    # size fallback when duration can't be probed

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

        # The sync recognize API is limited by audio *duration* (~60 s), not
        # file size: a 5 MB low-bitrate MP3 can hold half an hour of speech.
        # Route by real duration whenever pydub can probe it.
        duration_s = self._probe_duration_seconds(input)
        if duration_s is not None:
            use_sync = duration_s <= self._MAX_SYNC_SECONDS
        else:
            use_sync = os.path.getsize(input) / (1024 * 1024) <= self._MAX_SYNC_MB

        if use_sync:
            with open(input, "rb") as fh:
                results = self._recognize(config, fh.read())
            return self._format_results(results, timestamps, fmt)
        return self._transcribe_chunked(config, input, timestamps, fmt)

    def _recognize(self, config: dict, audio: bytes) -> list:
        """One ``speech:recognize`` request; the results it answered with."""
        import base64
        data = request_json(self._http, "POST", self._URL,
                            json={"config": config,
                                  "audio": {"content": base64.b64encode(audio).decode("ascii")}},
                            headers={"x-goog-api-key": self._api_key},
                            what="Google Speech-to-Text API")
        return data.get("results") or []

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
    def _seconds(value) -> float:
        """A protobuf Duration as JSON writes it — ``"1.500s"`` — in seconds."""
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value or "0").rstrip("s") or 0)

    @classmethod
    def _words(cls, results: list, offset: float = 0.0, skip_before: float = 0.0) -> list:
        words = []
        for r in results:
            alts = r.get("alternatives") or []
            for w in (alts[0].get("words") or []) if alts else []:
                start = cls._seconds(w.get("startTime"))
                # Words inside the leading overlap of a later piece were
                # already heard in the piece before it.
                if start < skip_before:
                    continue
                words.append({"word": w.get("word", ""), "start": start + offset,
                              "end": cls._seconds(w.get("endTime")) + offset})
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

    def _transcribe_chunked(self, config: dict, path: str, timestamps: bool, fmt: str) -> str:
        AudioSegment = require("pydub", package="pydub", who="transcribing a file over the size limit",
                               extra="audio").AudioSegment

        audio      = AudioSegment.from_file(path)
        chunk_ms   = self._CHUNK_SECONDS   * 1000
        overlap_ms = self._OVERLAP_SECONDS * 1000
        parts: list[str]  = []
        words: list[dict] = []
        piece_config = {**config, "encoding": "LINEAR16"}

        # Each chunk starts an overlap before the last one ended, and the last
        # chunk is the one that reaches the end: a further chunk would be the
        # overlap alone, transcribed and billed a second time.
        for start_ms in range(0, len(audio), chunk_ms - overlap_ms):
            buffer = io.BytesIO()
            audio[start_ms:start_ms + chunk_ms].export(buffer, format="wav")
            results = self._recognize(piece_config, buffer.getvalue())
            parts.append(self._text(results).strip())
            if timestamps:
                words += self._words(results, offset=start_ms / 1000.0,
                                     skip_before=self._OVERLAP_SECONDS if start_ms else 0.0)
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
