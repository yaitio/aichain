# Changelog

## [Unreleased]

**Less code, the same behaviour.** The simplification half of the 4.0.0
audit: copies merged, dead paths removed, nothing a caller sees changed except
where a fix is named below.

### Changed

- **A tool's API error is typed.** Search, fetch, embeddings, reranking,
  decisions, vector stores and Qwen speech raised a bare `RuntimeError`
  carrying the whole response body, so `except RateLimitError` around a tool
  caught nothing and an HTML error page could reach a model's context. They go
  through one `clients/_http.request()` now, as the model clients do: a
  status raises `RateLimitError`, `InsufficientCreditsError`,
  `AuthenticationError` and the rest, with the API named and the body cut at
  2,000 characters. `APIError` is a `RuntimeError`, so code that caught that
  still does. The message reads `[HTTP 401] Brave Search API: …` where it read
  `Brave Search API error [401]: …`.

- **`transcribe("openai")`, `"xai"` and `"qwen"` need no `openai` package.**
  One multipart POST each, made directly. Verified live on OpenAI 2026-10-12:
  a sentence synthesised by `speech("openai")` came back word for word, as
  text and as JSON with word timings.
- **`speech("openai")` and `speech("xai")` need no `openai` package.** They
  called the OpenAI SDK for one JSON POST; they share one plain-HTTP class now,
  and the output path is checked before the paid request instead of after.
  Verified live on OpenAI 2026-10-12; the xAI account answered 403, so xAI is
  checked offline only.

- **A fallback is on the record.** When the first model of a chain fails
  and a later one answers, `last_adaptations` carries a `swapped` entry
  naming both and why, and an exhausted chain raises the last error with the
  earlier ones attached as notes (3.11+). The answer and its price came from
  another model, and nothing said so.
- **`Skill("gpt-4o")` is refused at construction** with the line to write
  instead, not on the first call as an `AttributeError`.
- **`Skill.stream()` retries the way `run()` does**: one rule for what is
  retried, the budget checked and `llm_call.started`/`ended` emitted per
  attempt, where it checked once and reported one call however many it made.

- **`MCP(filter=…)` names tools that exist.** A name the server does not
  have raises `KeyError` listing the ones it has, where it was skipped in
  silence; a string raises `TypeError`, where it matched substrings.

- **A record's metadata may not use `text` or `_aichain_id`.** The store
  keeps the record's text and original id under those names in the same
  payload, so such a key overwrote one or the other on Qdrant and Pinecone.
  It raises `ValueError` on upsert now.
- **`vector_query` takes `n` from 1 to 100**, declared in its schema, and
  the `filter` option of `vector_query` and `vector_delete` tells a model the
  backend's dialect — Chroma `where`, Qdrant `must`, Pinecone `$eq` — where it
  said "provider-native" and left the model to guess.
- **`VectorBackend` is an `abc.ABC`.** A backend missing a method fails at
  construction, not on its first call.

- **`Eval(verbose=True)` logs instead of printing**, to `yait_aichain.eval`
  with a console handler installed, as the agent's `verbose` does. The lines
  read the same and can now be routed or silenced.
- **`Eval(trials=0)` raises.** It returned an empty report, which reads as a
  run that went well.
- **A `Report` counts each `(arm, case, trial)` once.** A ledger holding the
  same attempt twice weighed that case double; the later line wins.

- **The package is typed (PEP 561).** `py.typed` ships, so mypy and an
  editor's language server read the annotations instead of treating the whole
  package as `Any`; `Agent`'s constructor annotates `model`, `team`,
  `planner_model`, `permissions`, `approve`, `store` and `max_cost`, and its
  internal delegation depth is no longer a parameter. Quoted annotations are
  unquoted throughout — every module has `from __future__ import annotations`.
- **`from yait_aichain import AgentResult`** works, and `import *` brings the
  public names in `__all__` rather than the subpackages loaded on the way.
- **`yait-aichain[mcp]` needs `fastmcp>=2.10,<5`**, `[yaml]` `pyyaml>=5.1`,
  and `[all]` is the three extras rather than a copy of their lists. The sdist
  no longer carries the test suite, which cannot run outside the repository.

- **A `Usage` is truthy when anything was counted**, a cost included: a
  provider's price for a picture with no tokens beside it read as falsy.
  `Usage + Usage` sums every count, so a counter added later cannot be
  dropped by the sum.

- **A client family's `build_request` takes `tools=None`**, as
  `BaseClient.build_request` now declares, and `supports_tools` is a field of
  the base (False). An override without the parameter raises `TypeError`
  when its class is defined — the shape of the 3.1.0 defect where Qwen
  dropped tools while inheriting `supports_tools = True`. A file-upload body
  is a `Multipart` instead of a dict carrying a `_multipart` key.

- **A fetch tool offers a model several URLs in one call.** `url` is an
  array of strings in the schema, so a model asking for three pages makes one
  call rather than three; `run()` still takes a single string.

- **A number for `max_cost` is a ceiling per run.** It was made into one
  `Budget` when the object was built and spent down for its whole life, so an
  agent or chain built once — at module level in a warm Lambda — refused every
  request after the first expensive ones, and a parked run resumed in a new
  process got the whole amount again. Each `run()` now starts a fresh one,
  and `resume()` starts it from what the run had spent. A `Budget` object is
  unchanged: shared, and spent down across every run that holds it.
  `Budget(limit, spent=…)` takes what is already spent.

- **Risk classes, declared where they were inherited.** `vector_delete` is
  `destructive`, the vector query and fetch tools `read`, upsert `write`,
  transcription `external` (it sends a local file out), `Wait` `read`. With a
  policy attached, transcription now asks and `vector_delete` is denied by
  default. `financial` means moving money, not incurring charges.
  `PermissionPolicy` refuses a key that is not a risk class.

### Deprecated

- **`AgentMemory`, `FileBackend`, `InMemoryBackend`, `MemoryBackend` and
  `AgentResult.memory`.** Nothing in the library reads or writes them; the
  agent's state is its conversation, and the field is always empty. A dict
  does the in-process half and `state.FileStore` the on-disk half. They warn
  now and go in 5.0.
- **`fetch*().fetch()`** — it is `run()`, kept from 3.2 when `run()`
  returned text; `search()` went in 4.0 and this goes in 5.0, so both
  families have one method. It warns now.
- **`speech(...).run(text, {"output_path": …})`** — the folder is the tool's
  setting, `output_dir=`, and files are named for you. It warns and goes in
  5.0.
- **`Embedder.batch()`** — it was `run()` with a type check, and `run()` takes
  a list. It warns now and goes in 5.0.

### Removed

- **`sttXAI` and `sttQwen`, which never transcribed anything.** Both
  answered 404, checked live 2026-10-13: xAI offers speech-to-text over a
  WebSocket stream, not the file endpoint the tool posted to, and DashScope's
  OpenAI-compatible path has no `audio/transcriptions`. `transcribe()` takes
  `"openai"` and `"google"`.

### Fixed

- **Small security hardening.** A model name that could walk or end a URL
  path (`gemini-x/../..`, `?`, `#`, spaces) and a collection name outside
  letters, digits, `_`, `-` and `.` are refused, where both went into request
  paths unchecked. `Skill`'s repr lists its variables' names, not their
  values, which went into logs. An MCP server's tool description is cut at
  2,000 characters, and `MCP(trust_annotations=False)` ignores the server's
  annotations so every tool is `privileged` unless `risk=` says otherwise. A
  fetched page's text is fenced in `<page>` tags it cannot close, where it
  could forge the "Note:" lines around it; a judged answer is fenced in
  `<answer>` tags likewise. A failed tool call is marked `is_error` on
  Anthropic and answered as an error on Gemini, where an "ERROR:" prefix was
  all that told it from output. `ttsQwen` downloads its audio over https and
  refuses a file over 50 MB.

- **Small gaps in state, stores and the journal.** `InMemoryStore` saves
  through JSON as every persistent store does, so a set or bytes fails in
  development as it would in production. `FileStore` serialises before it
  creates its temporary file, where a document it could not write left a
  descriptor open and a stray file, and expands `~`. `agent.resume(run_id)`
  with no signal is refused for a run waiting on an answer — it took the
  parked question, paused again with nobody to ask and deleted the document —
  while a run kept after a failure is still retried without one. A parked
  run's `awaiting` and `run.suspended` carry the tool and its arguments, so a
  person asked to approve sees what. An acceptance fingerprints the whole
  observation (`JournalEntry.observation_sha`), where a change past the
  240-character excerpt — "7 failed" — still held; a `Journal` is iterable,
  where `still_holds(journal)` raised; and the deprecated `FileBackend`
  refuses a damaged file instead of reading it as empty and overwriting it.

- **A budget takes only money it can count.** A limit is a finite positive
  number — `Budget(nan)` was never exhausted and `True` was a dollar — and a
  charge is finite and not negative, where `charge(-10)` gave money back. A
  cost a response reports that is NaN, negative or a bool is set aside with a
  warning and the call estimated instead, where a NaN from a proxy turned the
  budget off. The message shows six decimals, where a fraction of a cent read
  $0.0000. The docs now say plainly what a budget bounds under concurrency:
  up to one call per worker past the ceiling — a $0.10 budget was measured
  spending $4.32 across sixteen workers — because the call in flight cannot be
  made free.

- **A hook observes and does not steer.** `tool_call.started` and
  `tool_call.ended` carry copies of the arguments and the result, where a hook
  that masked a token in `payload["arguments"]` changed the argument the tool
  then ran with. A hook that raises is reported once, at WARNING with its
  traceback, where a broken tracer logged a DEBUG line and nobody learned that
  tracing had stopped.
- **`verbose` is the agent's own.** Its lines go to a logger of its own,
  `yait_aichain.console`, which has its handler and does not propagate, and
  only for an agent asked to speak: one `Agent(verbose=1)` added a handler to
  the package logger and lowered its level for good, after which every agent
  in the process printed its tool arguments and lines appeared twice. `Eval`
  progress goes there too. A step or item skipped by `on_error="skip"` is
  logged on `yait_aichain.chain` or `yait_aichain.pool` every time, where a
  warning showed once per process and pointed inside the library.

- **The adaptation channel tells the truth under `-W error`.** The warning was
  remembered before it was raised and the record stored after, so under
  `filterwarnings=error` the first call raised with an empty
  `last_adaptations` and every later call passed in silence. The record is
  stored first and the warning is remembered only once given. The check for
  options that never arrived no longer searches the conversation — a "1h" or a
  4096 in a tool's arguments made a declined option read as delivered — and
  stops before walking the body when nothing is left to look for; Anthropic,
  which writes the cache lifetime into the content, says so explicitly.

- **A model option's value is checked where it is written.** `temperature`,
  `top_p`, `top_k` and `max_tokens` are checked at construction — a number of
  the right kind, in range, not a bool, not `None` — where `"0.3"`, `-1` and
  `None` reached the provider and came back 400. A range a model declares is
  applied there too, and the clamp is reported as an adaptation: 1.7 on
  glm-5.x, declared 0–1, was sent as 1.7. `True` no longer passes a numeric
  range as 1. A build that raises no longer leaves the previous call's
  adaptations on `last_adaptations`. `fidelity="auto"` is no longer a
  shorthand for 0.5, which OpenAI reads as its expensive high. An image asked
  for by `modalities` alone has its options checked as an image's.

- **Google's speech tools are plain HTTP.** `speech("google")` and
  `transcribe("google")` post JSON to `text:synthesize` and
  `speech:recognize` with the key in `x-goog-api-key`, where they called
  google-cloud-texttospeech and google-cloud-speech — grpc and protobuf in
  the install, and errors that went past `APIError`. Every speech tool is now
  plain HTTP. They need an API key (`GOOGLE_AI_API_KEY` or `GOOGLE_API_KEY`)
  whose project has the API enabled; Application Default Credentials, which
  only the SDK read, are gone, and a missing key is said at construction.
  Checked live 2026-10-13 up to Google's answer — the test key's project has
  neither API enabled, so no audio came back.
- **What the remaining tools import is declared.** A new extra `audio`
  (`pydub`, with `audioop-lts` on 3.13+) for splitting a recording over a
  provider's size limit; `fetch` declares the `requests` it uses; `all`
  brings them all, and the ImportError names the extra. Not done here:
  `fetch` still brings `markitdown` and its `onnxruntime`; the installation
  page says how large that is.

- **Offline tests stay offline.** Fifteen `list_models` tests mocked
  `urllib3.PoolManager`, and under an `HTTPS_PROXY` the client built a
  `ProxyManager` instead, so the "offline" test asked the real API with the
  key "key" and failed on its 401. An offline test now runs with the proxy
  variables cleared and a connection to anything but this machine refused, by
  name. The `*InvalidAuth` classes, which call real APIs with a bad key on
  purpose, are `*InvalidAuthLive` and run with the nightly live suite. The
  offline suite passes on 3.10 and 3.14 with no failures.
- **Packaging.** `license = "MIT"` with `license-files = ["LICENSE"]` and
  `setuptools>=77`, so the wheel carries the licence text and a
  `License-Expression`, where the table form was deprecated.

- **The `step()` seam's state survives JSON.** Adaptations are kept in the
  state as plain dicts, where the first one put a dataclass there and
  `json.dumps` failed — and with `default=str` the dedupe never matched, so
  the list grew every step; `AgentResult.adaptations` still holds
  `Adaptation`s. `ToolCallRequest.from_turn(turn)` rebuilds the calls of a
  turn kept as JSON between invocations.

- **A run that cannot go on ends with a reason.** A request past the model's
  context window is `ContextWindowExceededError` (an `InvalidRequestError`,
  now exported) and stops an agent with `stopped_by="context_overflow"`, where
  it read as `llm_error` like a revoked key. A store that refuses a parked run
  — a DynamoDB item over 400 KB after twenty calls — ends it with
  `stopped_by="park_failed"`, history and cost kept, where `run()` raised and
  the paid conversation was lost. Arguments nested more than 64 levels are an
  error the model is told, where 450 raised `RecursionError` on the next turn.

- **OpenAI stores nothing it was not asked to.** A Responses API request sends
  `store: false`, where every conversation moved to that API was kept at
  OpenAI by default though nothing here reads `previous_response_id`;
  `extra={"store": True}` keeps it. `check_auth()` raises on a rate limit, an
  empty balance and a missing endpoint, where it answered False and sent
  people to rotate a key that worked. A billing word in a 5xx body is a server
  error, not an empty balance. An answer is read as an image by its shape, not
  by having a `data` key. Not done here: `list_models` pagination and the
  per-request timeout while polling.

- **An answer in several text blocks is the whole answer.** Claude splits an
  answer around citations and server tools, Gemini around code execution, and
  the Responses API may send several `output_text` parts; `run()` returned
  Claude's first block, Gemini's first part and the Responses API's last,
  while `stream()` gave all of them. They are joined now, without Gemini's
  thought parts.

- **A schema keeps its meaning on the way to each provider.** In OpenAI
  strict mode an optional `$ref`, `anyOf` or untyped enum is now "it, or null",
  and an optional enum may be null — they became required and not nullable,
  so the model had to invent a value. An object with no properties raises
  `SchemaNotExpressible` (now exported) instead of a 400. For Google, `anyOf`,
  `oneOf`, `allOf` and `$defs` are followed, so `additionalProperties` inside
  them no longer reaches the API (a 400); a local `$ref` is inlined; and a
  union of types is an `anyOf`, where it became its first type.

- **A part a provider cannot carry is left out and said.** Video, audio by
  URL and documents were dropped in silence on the OpenAI-compatible and
  Anthropic wires, and the model answered about an attachment it never saw;
  each is a `declined` adaptation now. A PDF goes to chat completions as a
  `file` part and to Claude as a `document`, where both dropped it. Audio's
  format is read from its MIME type — `audio/mpeg` went as `"mpeg"` — and
  Claude, which takes no audio, is no longer sent it as a document (a 400). A
  tool's media follow-up no longer holds `null` for a part it could not encode
  (a 400). An image edit given more images than it takes sends the newest
  and declines the rest, where Recraft took the first one, xAI the first
  three and BFL the first four — the oldest in the conversation — in
  silence; each provider's limit is `max_input_images` in its data.

- **A gpt-oss stream decides as its run does.** Where harmony parsing exists
  for — a private gpt-oss without a server-side parser — `Agent.stream()`
  lost the tool call and sent the raw analysis channel to a UI as
  `text.delta`. A harmony model's stream is the buffered reply parsed whole,
  in one piece, with a `declined` adaptation saying why.

- **Streamed tool calls run in the order they began, each its own.** The
  slots were sorted as strings, so in a turn of eleven calls or more slot 10
  ran before slot 2 and the side effects came in another order than `run()`
  gives. Google renumbers its parts in every event, so two whole calls from
  two events both took slot `"0:0"`: the first was lost and the second ran
  with `{}`. A whole call now takes a slot of its own.

- **A streamed request is built as a buffered one is.** `Model.stream()` had
  its own copy of the request's tail and dropped `extra` while
  `effective_options` said it was sent; both paths share one tail now, and one
  gate refuses tools on a family that has none. The stream docstring promised
  that `on_unsupported="requirements"` raises when a provider cannot stream;
  it is `"raise"` that does, and the docstring says so.

- **DeepSeek's reasoning swap is the chat/reasoner pair's.** `reasoning` on
  `deepseek-v4-pro` or `deepseek-flash` is declined and said, where even
  `"low"` sent `deepseek-chat` — another, retiring generation. A swapped call
  is priced as the model that answered, where the budget counted it at the
  rate of the one asked for (on every provider that swaps). A `json_schema`
  schema always goes into the prompt, where a system prompt that merely said
  "JSON" dropped it; a fenced JSON answer is unfenced before it is parsed.

- **Search and fetch read one value as one value, and keep what they paid
  for.** A string where a search option takes a list is a list of one, where
  `domains="hbr.org"` became `site:h OR site:b …` on all six providers, and
  `fetchLocal(robots_agents="ClaudeBot")` is one agent, not its letters. A
  robots.txt that could not be read is not remembered — one failure kept the
  host unreachable until restart — and reads are kept an hour, for at most 256
  hosts. A fetch batch that fails after others succeeded keeps their pages and
  marks its own, where it threw them all away; a page answered under another
  spelling of its URL is found, where a trailing slash read as "nothing for
  this URL". An unknown charset decodes as UTF-8 instead of losing the page,
  and `max_chars` below 1 is refused, where `-3` cut from the end.

- **A tool's key may be resolved per request, as a model's is.** Every tool
  and vector backend takes `api_key=` as a string or as a callable asked with
  the run's `RunContext` on each request, where a callable was sent as
  `Bearer <function …>` and answered 401 — a multi-tenant RAG had to build its
  tools per request or bill every tenant to one key. A resolver that returns
  nothing for a tenant is an error, never a fall-back to another key; a key
  that is neither a string nor a callable is refused at construction.

- **A decision tool fails closed.** A question the provider did not answer is
  an error naming it, where it was a probability of `None` in a run that
  succeeded. A provider that reports no usage gives a cost of `None`, "not
  reported", where it gave $0.00 with a basis claiming it was reported.
  Score levels that are not labels and choice options not named by strings
  are refused at construction (`levels: [1, 2, 3]` was an `AttributeError`, int
  option names answered nothing). One image path given as a string is one
  image, where it was read a character at a time; an image is at most 20 MB
  and its type is read from its bytes, where every byte string was labelled
  PNG.

- **The chunker keeps every line and joins only what belongs together.**
  Overlap joins text to text under the same headings, from the neighbour's own
  text: it glued a table's header to a data row and began a section with the
  end of the one before. A lone `_` or `*` no longer makes a document Markdown,
  where one `_` in a CSV header cut its rows mid-line. Plain text splits on
  line breaks and on `。！？`, where Chinese and Japanese were cut at fixed
  offsets; Windows line ends are normalised, so `\r\n\r\n` is a paragraph
  break. A code block closes on a bare fence of at least its own length, as in
  CommonMark, where a "```python" line inside one closed it and the text after
  was lost. Sizes that are not whole numbers in range and any option are
  refused, where a negative overlap broke the size contract.

- **Speech tools send what is set, in a format they write.** An option given
  as `None` is not sent, where `voice=None` went into the request and
  `speed=None` raised. Each speech tool names the formats it writes and
  refuses others, where Google turned an unknown one into MP3. `ttsQwen`
  defaults to `qwen3-tts-flash` with the voice `Cherry`, where its default,
  `cosyvoice-v2`, answered "Model not exist" (checked live 2026-10-13), and
  writes WAV, the one format that endpoint answers in — an MP3 asked for was
  WAV saved as `.mp3`. An unknown region is refused at construction.
- **Transcription keeps its shape.** `sttOpenAI` with `gpt-4o-transcribe`
  asks for `json`, where it sent `verbose_json` and got 400 (checked live), and
  refuses word timestamps, which only `whisper-1` gives. A file large enough to
  be chunked refuses timestamps instead of returning text alone, so the keys
  no longer depend on the file's size. `sttGoogle` refuses a file type it
  cannot read, where `.m4a` was an `AttributeError` and anything unknown was
  read as MP3.

- **A REST endpoint sends what its settings say.** `fields=` types an
  endpoint's fields — `{"limit": (int, "page size"), "tags": ("array", …)}` —
  where every field was a string described by its own name and an array could
  not be asked for. Each auth type has the keys it reads, and anything else
  is refused; a token that is not there — an unset variable, an empty field —
  is an error, where the request went anonymous and some APIs answer that
  with less data rather than 401. A new auth type `query` puts the key in the
  query string, and errors print the URL without it. Booleans go as `true`,
  lists repeat their key, and a form content type gets a form body, where it
  got JSON. Fields named `input` and `options` are sent (`run(input, options,
  /, **fields)`). `timeout=` keeps the connect timeout; retries spent end in
  the API's typed error, not `MaxRetryError`; a non-2xx answer is the typed
  `APIError` for its status. A URL placeholder with no path parameter, a
  field placed twice or badly named, and a credential in the URL's query are
  refused at construction. `REST(allow_private=True)` reaches one API on an
  internal network without the process-wide flag.

- **The local tools stay inside what they were given.** `local_run` starts
  each command in its own session with no stdin and kills the whole group on
  a timeout, keeping what it printed — only `/bin/sh` was killed, and a `&`
  child or a pipeline lived on; a model's timeout is capped at 600 seconds,
  where `10**9` was an `OverflowError`; the child no longer inherits names
  carrying KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL, AUTH or `AWS_`, which
  `env` printed to the model; and output is decoded with replacement, where
  one invalid byte raised after the command had run. `local_write` encodes
  before it opens, so text the encoding cannot hold no longer empties the
  file, and refuses a mode other than `w` or `a`. `local_browse` lists a
  dangling link instead of ending on it and no longer shows the host's
  absolute path. `local_read` with `end_line` of 0 or below reads to the end.

- **A secret a chain carries reaches its tool and nothing else.** `REST`
  recommends `token_field` from a chain variable; every pause wrote that token
  to the store, to `SuspendedResult.document` and to each history record, and
  an Agent step put it in the GIVEN block sent to the model's provider. A
  value wrapped in the new `state.Secret` is left out of all four, prints as
  `***`, refuses to be pickled, and is read by the tool with `reveal()` —
  `REST` does so for `token_field`. A run resumed in another process does not
  have it; pass it again in the signal.

- **A chain reads every setting of a step, or refuses it.** A Skill step's
  `input_map` is resolved as a tool's is — templates filled, a variable the
  chain lacks an error — where names it lacked were dropped. A tool with no
  declared properties refuses a mapping, where it took one and dropped it; an
  open schema takes it. An Agent step refuses an `input_map`, unknown options
  and an `output_field` that is not a field of `AgentResult` (a typo gave
  `None` and success); options on any other step are refused, since nothing
  read them. A store of one's own is kept even when it is empty and falsy. A
  step given as a list — `[runner, "key"]`, as YAML writes it — is a step,
  and a step that is not a runner is refused at construction, where it failed
  in `run()`. A function's positional-only parameters are passed
  positionally, so `len` works as a step.

- **A stream runs in the context it was made in.** `Skill.stream()` and
  `Agent.stream()` capture the context when they are called and advance every
  step inside it. A generator runs in whichever context advances it:
  Starlette's `iterate_in_threadpool` advances each step in a fresh copy, so
  an agent's beacon board was gone from its second step and every tool's
  beacon was dropped; a `skill.stream()` made inside `with using(ctx)` and
  read after it resolved its key with no tenant; and two streams advanced in
  turn mixed their boards, so tenant B's beacons reached A's result.

- **A Pool stops when it says it stops.** `on_error="raise"` raised only
  after the whole queue had run — a thousand paid calls after the first one
  failed — and so did Ctrl-C; the queue is cancelled now. Items a stop, a
  raise or an attention beacon never started are `CANCELLED` (new, `4`) with
  the reason in `error`, where they stayed `PENDING` and a loop polling for
  `DONE + FAILED` waited for ever. A worker checks for a stop before it
  starts an item, so one free between a blocker and the cancel no longer takes
  the next. A runner without `run()` (a plain function), items that are not
  dicts and a `max_flows` below 1 are refused at construction.

- **Tool names the model cannot tell apart are refused, and a rename
  renames.** A tool named `pool`, `delegate` or `write_plan` is refused —
  the loop's own tool answered it — and so are two workers of one name, which
  were one entry in the `delegate` enum. `Vectors(name=…)`, `Shell(prefix=…)`
  and `MCP(prefix=…)` name a utility's tools, so two of a kind fit in one
  agent; the error's own advice did not work for them before. An MCP tool
  calls the server by the server's name whatever it is called here, where a
  renamed tool sent its new name; names with `.` or `/` become `_`, where they
  were a 400 at OpenAI. A listed worker runs one level down, so the depth
  guard can fire; an `auto` worker gets no tool that parks a run; and a
  worker's `error` reaches the delegating model.

- **The approval gate opens on a yes and nothing else.** An approver answers
  `True`, `False` or `ApprovalDecision`; anything else is an error, where
  `granted = bool(answer)` approved a call on `"denied"` and on
  `{"approved": False}`. A policy answering a word that is not a decision is
  a refusal, where it meant "allowed". `PermissionPolicy.rules` is read-only
  after it is checked, and the approver is shown a deep copy of the arguments.
- **A caller's callback that raises ends the run one way.** A check, a
  ceiling, a nudge or an approver that raises stops the run with
  `stopped_by="condition_error"`, the error naming which, and a warning with
  the traceback — a check swallowed it at DEBUG, and the others raised out of
  `run()` with no result and no `run.finished`. A failed tool's
  `tool_call.ended` carries the exception's type and traceback.

- **Stop conditions hold on both seams.** A `stop_when` with no ceiling gets
  `step_count(30)` and a warning, where `stop_when=[check(...)]` dropped the
  default and the loop had none. `cost_budget` on a model with no price warns
  at construction, since its cost is never known and it never fires. Two
  nudges of one name each speak — the latch is keyed by place, not by name —
  and the latch survives a pause, where a resumed run repeated its nudge.
  `step()` counts a decision as `run()` does, so a driver's `step_count`
  moves, and `agent.stopped(state)` and `agent.nudges(state)` give a driver
  what `run()` asks after each turn. The journal records a call with a hash of
  its whole arguments, where two calls that differed past 200 characters read
  as one repeated and `repeating()` fired.

- **`Agent.stream()` hands each text delta over as it arrives.** The deltas
  were emitted inside a synchronous model call and drained when it returned,
  so hooks saw them at 0.31, 0.61 and 0.91 s and a `for ev in stream()`
  consumer saw all three at 0.91 s — the streaming R2 shipped did not reach the
  stream. The model turn is walked now, and a block opened is closed with
  `text.ended` even when the stream fails part way.

- **Four Skill traps closed.** `Skill(options=…)` is refused with the place
  options go — `Model(name, options={…})` — where it was stored and never
  applied, so `Skill(options={"temperature": 0})` ran at 1.0; a saved Skill
  carrying `options` is refused on load. A `schema` under format type `json`
  is refused with a pointer to `json_schema`, where it was never sent and the
  answer was still checked against it. Turns after the last generate marker
  are answered, where they were dropped and the caller got the draft before
  them. `format: {"type": "image"}` is refused as `modalities: ["image"]` is.

- **`Skill.stream()` parses structured output the way `run()` does.** Fences
  are stripped and the schema is checked; an answer that does not parse
  raises, where the stream handed back the text as the value (`last_result`
  still holds the text). A streamed Anthropic `json_schema` answer is the
  value, where `last_result` was the library's own synthetic tool call.
- **A paid call is recorded.** A stream the caller stops reading emits
  `llm_call.ended` with `error="abandoned"`, where it emitted nothing. A
  response that fails to parse after it was billed keeps its usage on
  `last_usage` and emits `llm_call.ended` with the error, where both were lost
  with the exception.

- **A placeholder is filled wherever it stands.** `{"summary": {text}}` was
  read as an escaped brace and the document never went in; the template is
  scanned left to right now, `{{` and `}}` taken as literals and anything else
  of the form `{name}` filled — `{θέμα}` included, which was not a
  placeholder. A placeholder a Skill could not fill, with variables given, is
  said once with the nearest name, where a typo sent `{topic}` to the model
  as written while a Chain raised on the same mistake. A plain dict or list
  goes into a prompt as JSON, not as its Python repr.

- **The vector backends agree.** A delete takes ids or a filter — exactly one,
  on every backend and in the store: both meant three things (Qdrant dropped
  the filter, Chroma took both), and a Pinecone delete with neither sent
  `deleteAll`. Qdrant writes and deletes wait until they are searchable,
  where a query straight after an upsert found nothing. A new Pinecone index
  is waited for until ready. A backend's settings are its named parameters
  only, so `indx=` and `regoin=` are a `TypeError` instead of being swallowed.
  A Qdrant id is an integer only in an integer's own spelling — `"007"` and
  `"7"` were one point, and `"²"` raised. A record with no id, no text or
  metadata that is not a dict is refused before anything is written, and
  metadata of `None` is empty.

- **A vector store's scores are read on the metric it was created with.**
  `create(4, metric="euclidean")` on Qdrant and Pinecone built a euclidean
  collection and scored it as cosine: an identical document scored 0.5 and a
  far one 1.0, and a `> 0.8` threshold kept the worst. The backend now keeps
  the metric it created, and on an existing collection or index reads the one
  it has — and refuses one of another dimension, where it took every write.
  Pinecone's and Chroma's euclidean is the squared distance and is scored as
  such. Chroma's ip score was off by one and its default space, l2, was read
  as cosine; its distances go through the same rule as the other backends'
  and come back as `raw_score`, which `to_dict()` now carries. An unknown
  metric is a `ValueError`, where it became cosine. Qdrant answers an
  existing collection as success, as the others do.

- **Embedding batches are sized by what a provider refuses.** A request was
  cut by count alone: 2,048 chunks of 400 tokens is 819k tokens against
  OpenAI's 300k, DashScope's `text-embedding-v3`/`v4` take ten rows where 25
  were sent, and 50 records of 3,072 floats are 3.4 MB against Pinecone's
  2 MB. A batch is now bounded by rows, estimated tokens (OpenAI 300k, Voyage
  120k) and bytes, and a vector store's write by records and about 1.5 MB. A
  provider that returns fewer vectors than texts is an error, where Google
  answering with none was a success.
- **An upsert keeps what it paid for.** Records are embedded and written a
  batch at a time, where every embedding was bought before the first write and
  a failure at batch k lost them all. A failed batch raises `PartialUpsert`
  with `written` and `total`, the cause chained; a bad record — reserved
  metadata keys, no vector and no embedder — is refused before anything is
  written, and `batch_size=0` is refused where it became 50.

- **A reranker and an embedder send text, and only text.** A document is a
  string, a mapping's `text` or an object's `.text`; anything else is a
  `TypeError` before the request. `rerank.run("hello world")` sent eleven
  one-character documents, a `VectorRecord` went as its repr (its fields are
  merged back into the result now), and a mapping with no `text` went whole,
  metadata included. A reranker refuses a missing query, where it sent `""`,
  reads `top_n` as a whole number of at least 1, and answers `[]` for no
  documents without a request. An embedder sends a `chunk()` dict's text, and
  refuses `None` and empty strings.

- **`tool(input, options)` calls `run` the way `run` declares.** Options
  went positionally to whatever came second — `Image.run(input, images,
  options)` took them as images — a `@tool` function with no parameters
  could not be called, and `wc("a a b", {"unique": True})` ran with the dict
  as `unique`. *input* is passed first, *options* by name, the rest by name,
  bound against `run`'s signature; a function with no `options` refuses them
  and says to pass the arguments by name. A failed call's error carries the
  exception's type, `KeyError: 'x'` where it was `'x'`. The safe call does not
  park a run; that is documented.

- **An MCP server's `input`, `options` and `name` arguments are sent.**
  `MCPTool.run(input, options, /, **arguments)` and `MCP.call(name, /,
  **arguments)` take those positionally, so a server argument of the same
  name goes to the server: `options` was refused, a dict `input` was spread
  into separate arguments, a list `input` was dropped and `call(name=…)` was a
  `TypeError`. Code that wrote `tool.run(input={...})` for the argument dict
  writes `tool.run({...})`.
- **An open schema admits its keys.** A tool whose schema has
  `additionalProperties: true`, or a schema for them, had every argument
  refused as unknown — an `MCPTool` built directly, which has that schema by
  default, refused all of them, and the agent spent its steps on it.

- **An MCP tool's image reaches the model as an image.** Images, audio and
  binary resources become the library's media parts, so a model is shown the
  picture rather than its base64 as text; text beside them is kept as strings.
  Structured results are the server's JSON as sent, where a typed schema gave
  fastmcp's generated dataclass and the model read `Root(code='LH1')`, and
  `{"result": …}` is unwrapped only where fastmcp says it wrapped it. Several
  text blocks are a list of strings, where they were `{"type": "text"}` dicts.

- **An MCP call ends.** `call_timeout` is 300 seconds by default, where a
  server that took a call and never answered held the agent forever; `None`
  still waits indefinitely, and a value of 0 or below is refused. From
  Python 3.11 a `TimeoutError` raised by the server's call was mistaken for
  the bridge's own wait and reported as a `TypeError`; the wait names the
  setting to raise, `call_timeout` or `connect_timeout`. A forked child starts
  its own event loop instead of waiting on the parent's, and a synchronous
  call from the loop's own thread raises instead of deadlocking.

- **The eval report counts what happened.** `Pass^k` used each arm's own
  number of trials, so an interrupted arm at Pass^2 ranked beside a finished
  one at Pass^3; k is one number for the report now. It is estimated from
  every trial, C(c, k) / C(n, k), not from the first k, and no data is NaN, not
  0. `reject()` thresholds are fractions (`max_empty=0.1`), where they were
  counts. `controls()` raises when an arm has no scored data, where a missing
  noise arm passed, and takes `oracle_arm`/`noise_arm` for reports that name
  their arms differently. `spend()` counts every attempt, abstained and
  rejected ones included, with `unpriced` for attempts with no cost. `ci()`,
  the Wilson interval, is new.
- **`Eval` runs what it says it ran.** A failed `AgentResult` (or any result
  with `success=False`) is recorded as an error, where its partial output was
  scored as an answer. A ledger is resumed only for attempts recorded with the
  same input, expectation and scorer, and an errored attempt is run again.
  `run(concurrency=…)` keeps the caller's context in each worker. A scorer
  that raises abstains.
- **Scorers.** `numeric()` reads the last number, and `1,234` as one thousand
  two hundred and thirty-four; it and the other text scorers refuse a case
  with no `expect`, which passed every answer. `all_of()` keeps what each
  scorer reported and an abstention by any of them, and refuses no scorers. A
  number is a pass only at 1.0 — `0.01` passed — and NaN abstains. A dict
  verdict must carry `ok`.
- **Records.** `adapt()` reads `"false"` as false and `"0"` as trial 0;
  `read()` keeps unknown keys in `meta` and warns about lines it skipped;
  `write()` repairs a line a crash left torn before appending, and a cost may
  be `None` rather than 0.

- **Cost estimates that were low, and a total that did not say it was
  partial.** A cache read was priced at a tenth of the input rate for every
  model, where gpt-4o reads at a half and gpt-4.1 at a quarter — an agent
  loop on them up to five times low; those rates are in the data now.
  Perplexity's per-request fee was left out (about eight times low); it is
  added at the low-context rate, the floor of a call. Past a long-context
  threshold the cache and image rates scale with the input rate. A price is
  looked up by provider, where `private/qwen3-32b` on one's own server was
  priced at DashScope's rate. And `Usage.unpriced_calls` counts the calls a
  total leaves out.
- **A chain reports what its steps spent.** An agent step added its tokens
  and not its cost, so `ChainResult.cost` was None or short; a Skill shared
  by a Pool's copies of one chain had its usage read after another thread
  overwrote it (35 of 400 items); a pause kept four of `Usage`'s seven
  fields, losing the cache and image counts. An agent step adds its whole
  usage, each Skill step runs on its own copy — the step the caller holds
  still shows the result — and the run document keeps every field.
- **A saved Skill, agent or tool comes back as it was, or saving says why
  not.** A Skill came back at temperature 1.0 from 0, with no fallback, no
  retries and no budget; an agent came back without its permission policy —
  a gate quietly lost — and its Shell tools moved to the loader's working
  directory; a Gate or an MCP tool was saved and then failed to load; and
  `!!python/tuple` in the file made it unloadable. `Skill.to_dict()` /
  `from_dict()` carry the model's options, the fallback chain, retries and a
  numeric budget; an agent's policy and its tools' settings are saved, and
  its approver and hooks are named as left behind; a Gate or an MCP tool is
  refused at save. Files are plain YAML, written atomically, and a file that
  is not a saved chain or skill is refused by name.
- **`NO_PROXY` is honoured, and each scheme gets its own proxy.** One proxy
  was chosen for the whole client, so a `private/` model on localhost, a
  Chroma or Qdrant in the VPC and an internal REST endpoint went through the
  corporate proxy, got 502 — read as the provider's server error — and
  started a fallback; and `HTTPS_PROXY` was used for `http://` too, where the
  proxy reads the key. The proxy is now chosen per request from the
  environment variables; an explicit `proxy=` still applies to everything.
- **A stream keeps its text whole, and a stream that fails says so.** Each
  network chunk was decoded alone, so a character split between two reads —
  any non-ASCII text — became two replacement marks; `splitlines()` cut an
  event at U+2028, legal inside a JSON string. An error event mid-stream
  (Anthropic's `overloaded`, an OpenAI-compatible `{"error": …}`) was taken
  for data, and the reply cut short there became the answer; it raises the
  typed error now (529 for overloaded). A timeout or a dropped connection
  while reading is a `NetworkError`, retried while nothing has been shown; a
  200 with a JSON body instead of a stream is read as one answer; and a
  stream that is abandoned or breaks is closed, so the provider stops
  generating for a reader that has gone.
- **Nothing a tool reads or returns is unbounded.** A 149 KB answer that
  MarkItDown's zip converter unpacked took 950 MB; `local_read` loaded a
  2 GB log whole for forty lines; `local_run` and REST buffered everything.
  `fetchLocal` reads at most 10 MB and converts only pages (text, markup,
  JSON, PDF); `local_read` streams and stops at 200,000 characters;
  `local_run` returns the head and tail of each stream, 50,000 characters;
  `local_browse` stops at 2,000 entries; REST refuses a response over 10 MB.
- **Security: credentials followed a redirect to another host.** urllib3
  drops `Authorization` on a cross-host redirect but sends `x-api-key`,
  `x-goog-api-key` and the body of a 307 on; REST forwarded its headers to
  wherever the server pointed, and `fetchLocal` applied the caller's session
  headers to every host. Provider and search clients no longer follow
  redirects (a 3xx raises); REST refuses a redirect to another host and says
  where it pointed; `fetchLocal` leaves the caller's headers off any host but
  the one it was asked for.
- **Security: image, transcription and Skill media read any local path.**
  `images=["/proc/self/environ"]` uploaded the environment — the AWS keys on
  Lambda — to an image provider, `transcribe("~/.aws/credentials")` sent the
  credentials, and `/dev/zero` as a Skill's media exhausted memory. A local
  media file must now be a regular file, within a size limit for its kind,
  and start the way that kind of file does; `AICHAIN_INPUT_ROOT` confines
  reads to a folder. `images` given as one string is one image, not a list of
  characters.
- **Security: an unclassified tool ran as `write`, which the policy
  allows.** Tools that declared no risk inherited the default, so
  `vector_delete` — a bulk delete by a filter the model writes — and every
  transcription tool ran unasked under the shipped policy, and
  `PermissionPolicy({"destructve": "deny"})` matched nothing and said
  nothing. Every tool in the package declares its class now, held by a test.
- **Every tool call has its own id, and every result finds its call.**
  harmony carried no id and Google used the function's name, so two calls to
  one function shared an id; the agent answered under `call.id or call.name`
  while a driver used `call.id`; `dangling_calls()` skipped empty ids and
  checked against a set, so a repeated id answered once hid a lost result;
  and a history made on Kimi (`functions.look:0`) sent to Anthropic — by a
  fallback, a planner model or a driver changing models — was HTTP 400. A
  `ToolCallRequest` now gives each call a unique, non-empty id; Google's
  results are named by the call they answer; Anthropic gets ids it accepts,
  mapped the same way on both sides; and `dangling_calls()` counts.
- **A chain is checkpointed after every step, and resumes only its own
  runs.** The documentation promised a write after each step; the document
  was saved only on a pause, so a process that died in the tail resumed from
  the start, repeating every side effect. A chain now saves after each done
  or skipped step (warning once, and carrying on, when a variable will not
  serialise), and `resume()` continues an unfinished document from the first
  step not done. A failure skipped before a pause is kept in the document and
  still fails the run after it. And `resume()` refuses — putting the document
  back — an agent's run or a chain whose steps differ from the ones it parked
  with, where it ran the wrong steps and deleted the run.
- **Claude is not sent combinations it refuses.** `reasoning` with a
  `json_schema`, `top_p` beside a temperature, `top_k` with thinking and a
  temperature above 1 each went out and came back HTTP 400 with nothing in
  `last_adaptations`. Reasoning with a schema is refused before the call;
  `top_p` replaces the temperature, thinking drops `top_k` and raises `top_p`
  to its floor of 0.95, and a temperature above 1 is clamped — each recorded.
- **Security: an API key could land in an error message.** A key read with
  `open().read()` ends in a newline; the header library raised with the
  whole `Bearer sk-…` in its text, and the transport's catch-all made that a
  `NetworkError` — retried, switching the fallback model, and written to
  events, logs and run documents. Keys and header values are stripped, a
  control character left inside is refused by naming the header and never
  the value, and the transport catches only what the network raises; a body
  that does not serialise raises as itself, before anything is sent.
- **A step that fails without a message has failed.** Success was read from
  the error's text, and `TimeoutError()`, a bare `assert` and
  `RuntimeError()` render as `""`: a Chain under `stop`, `skip` or `collect`
  returned `success=True, error=None` beside the failure's record, and a
  Pool item read as succeeded. An error with no message is recorded as its
  type, and a Chain reads failure from the failure record.
- **A pause is neither a failure nor a finished step.** `SuspendedResult` is
  falsy, so a Pool reported "Agent failed: unknown error" for an item whose
  agent had parked — losing the run id and orphaning its document — and a
  Chain counted a parked nested chain as done and passed the
  `SuspendedResult` on as a variable. A Pool now returns the
  `SuspendedResult` as the item's output, a Chain refuses a child's pause by
  name, and a `Suspend` from a function or a Skill step — re-run without its
  signal on resume, so it paused for ever — is refused: only a tool step is
  given the signal. `SuspendedResult` carries `success`, `suspended`,
  `error`, `usage`, `cost` and `tokens_used` like every result; reading
  `output` still says to resume.
- **A Chain's or a Pool's budget holds without editing the steps.** A Chain
  lent its budget by writing it onto each step's `max_cost` and restoring it
  afterwards, and a Pool did the same on the runner — but a Pool's copies of
  one Chain share its steps, so the writes raced: items ran with no budget,
  and a Skill kept the Pool's ceiling for good; an agent run by a Pool kept
  an exhausted overnight budget. The budget is now ambient for the run, as
  the tenant is, and a Skill checks and charges its own and every one it
  runs inside.
- **A call's usage and adaptations are its own, on a shared Model.**
  `last_stream_usage`, `last_stream_result` and `last_adaptations` were
  written to the Model and read back after the network round, and one Model
  serves a Pool's threads and every tenant: a stream read another's usage —
  1,000 tokens became 8, and `max_cost` undercounted — and adaptations were
  credited to another request. `Model.stream()` now returns a `StreamOutcome`
  when it ends, `Model.build()` returns the adaptations beside the request,
  and `from_response(..., tools_sent=)` is told whether tools were offered;
  the attributes remain as mirrors for a single caller. A buffered fallback
  keeps only the usage envelope, not the whole response with its images.
- **A retried call cannot bill a generation twice, or sleep for hours.** The
  transport repeated a POST whose read failed — a balancer cutting a long
  generation, a 600 s timeout — up to five more times, each paid, and slept
  through any `Retry-After`, six hours included; with `Skill(max_retries=2)`
  on top, one turn could be eighteen requests. A failed read is never resent
  now; the transport retries only the statuses returned before any work
  (429, 503 and Anthropic's 529 overloaded, which no layer retried before),
  three attempts with jitter, and a `Retry-After` past 60 s goes straight to
  the caller as `RateLimitError.retry_after`. A Skill's retry and its
  fallback share one definition of transient, its backoff waits at least the
  server's `Retry-After` and at most 60 s, and a negative `retry_delay` is
  refused. Embeddings, reranking and vector stores read for a minute, not
  ten.
- **What a provider attaches to a tool-calling turn comes back with it.**
  Gemini 3 signs each function call with a `thoughtSignature` and answers 400
  when the call returns without it; the library rebuilt calls from name and
  arguments and dropped it. A signed turn now goes back as its original
  parts. A turn's `reasoning_content` is kept with its tool calls for
  providers whose data sets `keep_reasoning = true` — MiMo, whose quick start
  asks for it, verified live 2026-10-13 on `mimo-v2.6-flash`. Kimi and
  DeepSeek's thinking modes are not flagged: not yet verified live.
- **Claude's structured answers are checked like every other family's.**
  With `json_schema`, a reply with no tool call — cut off or refused — came
  back as `{}`, and a partial object went out unchecked. It raises
  `TruncatedResponseError` at the token ceiling and
  `InvalidStructuredOutputError` otherwise, and an answer that does not match
  the schema raises with the problems named.
- **An approved action runs once, however often its signal arrives.** A
  resume loaded the parked run, ran the rest and deleted it at the end, so
  two deliveries of one signal — an SQS message seen twice, a webhook
  retried — both refunded. `StateStore.take()` now claims a run in one step
  (a lock in memory, a rename on disk; a shared store overrides it with its
  backend's conditional delete), and the second delivery gets `KeyError`.
  A resumed run that then fails on the model, or on a later chain step, is
  kept with what already happened — the approved action's result included —
  and `resume(run_id)` retries from there without repeating it, where the
  run used to be deleted and a person's approval lost to a transient error.
  Delivery is at most once for the action.
- **A reply cut off at the token ceiling is not executed, and not a
  success.** The stop reason was read only when parsing JSON: a tool call cut
  at `max_tokens` ran with `{}` or half its arguments, and an empty answer —
  common from reasoning models that spent the budget thinking — ended a run
  with `success=True`. A cut-off reply that carries a call, or no text, now
  raises `TruncatedResponseError` on every family, and an agent stops with
  `stopped_by="truncated"`; prose cut short is still returned. The Responses
  API's `incomplete` and `failed` statuses are read, Gemini's refusals
  (no candidate, `SAFETY` and the rest) raise instead of answering `""`, and
  Gemini's `maxOutputTokens` is raised above a thinking budget that would
  otherwise use all of it. Not done: one nudge for an empty answer that was
  not cut off.
- **Security: a model chose where speech was written.** `output_path` was in
  the schema a model sees and nothing confined it without
  `AICHAIN_OUTPUT_ROOT`, so an injected instruction replaced `~/.zshrc` with
  an MP3 — and a permission policy passed it as `write`. The folder is now a
  setting, `speech(..., output_dir=…)`, defaulting to the system's temporary
  directory, which Lambda can write where `./audio` it could not. Each file is
  created new and exclusively before the paid call, so parallel calls no
  longer write one file and all report it, and a call that failed leaves no
  file. The tools are `draft`.
- **Security: two runs of one agent crossed.** The run id, the parked
  document, the pending approval, the stream flag and the hook a stream
  collects through all lived on the `Agent`, and one agent serves many runs —
  a module-level agent in a warm Lambda, a `Pool`'s items. An approver was
  shown one run's call while another's executed (approved 5, executed 5000),
  one tenant's stream carried another's task, arguments and results, and
  events took another run's id. Every `run()`, `stream()`, `resume()` and
  `execute()` now works on its own copy of that state; the configuration —
  tools, model, policy, store, a shared `Budget` — stays shared. A stream
  advances in a context of its own, so two streams in one thread no longer
  restore each other's beacon board. A `Pool` hands its budget to a copy of
  an agent runner instead of leaving it on the caller's agent.
  `ApprovalRequest.run_id` says which run is asking.
- **An Anthropic agent with reasoning survives its second step.** With
  extended thinking on, the assistant turn before a tool result must open
  with its thinking blocks, unchanged; the library kept only the calls, so
  every such agent answered HTTP 400 on its second request.
  `ToolCallRequest.opaque` now carries what a provider needs back verbatim —
  Anthropic's `thinking` and `redacted_thinking` blocks — and the Anthropic
  builder puts them first. The streamed path does not carry them yet.
- **Parallel tool calls work on Gemini, and with pictures on OpenAI.** Gemini
  needs every function response of a turn in one content; each went in its
  own, so any turn with two calls answered HTTP 400. Chat Completions and the
  Responses API need a turn's tool results back to back; a tool that returned
  an image put a user message between them. A turn's results are now sent
  together and its media in one message after them, on all three.
- **Security: SSRF through a redirect, a parser disagreement or a shared
  address.** The URL check read the first URL and the HTTP library then
  followed redirects unchecked, so a public page answering `302` to
  `169.254.169.254` reached the metadata service — from `fetchLocal`, its
  `robots.txt` request and REST. Redirects are followed by the library now,
  at most five, each target checked first. A URL carrying credentials or a
  backslash is refused, the two shapes where this parser and the connecting
  one read different hosts. An address must be public by `is_global`, which
  covers the shared `100.64.0.0/10` range (Alibaba's metadata is
  `100.100.100.200`), and IPv4 inside IPv6 (`::ffff:`, NAT64) is checked as
  the address it is. Not covered: a DNS answer that changes between the check
  and the connection.
- **Security: quoted text became an executed tool call.** The harmony parser
  ran on any OpenAI-compatible reply containing `<|channel|>`, so a model —
  DeepSeek, Qwen, Grok, gpt-4o — quoting an example from a file or a page it
  had read had `to=functions.local_write …` executed. Harmony is parsed now
  only for the gpt-oss family, and only becomes a tool call when tools were
  sent; without tools the reply is its final text. A Skill without tools no
  longer returns a `ToolCallRequest` this way. The parser is one pass over
  the text, where its pattern took seconds on a long reply, and two calls to
  one tool get two ids.
- **Security: `Chain.load()` of an untrusted file could run shell commands.**
  The guard checked only that a class lived under `yait_aichain.`, and the
  package ships tools whose arguments say what they reach: `LocalRunTool`
  (the file's command, through its variables), `LocalWriteTool` (the file's
  folder), `MCPTool` (`server="sh -c …"`) and `RestApiTool` (the file's
  address). Agent tools in a saved chain were not checked at all. Now, unless
  `trusted=True`, a class loads only if it is a `Tool`, opts in with
  `_safe_for_untrusted_load` and builds as `read` or `draft`: search, fetch,
  embeddings, reranking, decisions, chunking and images without a folder.
  Shell, MCP, REST, speech and transcription need `trusted=True`, and both
  kinds of step go through the same check.
- **Security: a tool result could make the agent read any local file.** Any
  dict shaped as a media part with `"source": {"kind": "file", "path": …}` —
  from any tool, an MCP or REST server's raw JSON included — was carried as
  media, and on each later turn the path was read from disk with no limit and
  sent to the model: `/proc/self/environ` on Lambda holds the AWS keys,
  `/dev/zero` exhausts memory. Media in a tool result is now bytes or a URL
  only; a file-backed part stays text. `yait_aichain.tools.media_part(path)`
  reads a file once, up to 20 MB, into a self-contained part. `document`
  parts, which every family already renders, now pass `validate_input`.
- **The tenant in force stays in force.** `Chain.run()` with no `context=`
  set the ambient context to None for the run, so `with using(ctx):
  chain.run()`, a chain inside a chain step and a chain in a `Pool` all lost
  their tenant, and `Model(api_key=callable)` asked for the key of tenant
  None. `using(None)` now leaves the context alone, a chain takes the current
  one when given none, and an agent keeps it in a parked run's document and
  restores it on `resume()`, in whatever process the signal arrives.
- **Security: the agent's `pool` tool went around the permission policy.**
  The gate looked a call up by its own name, `pool` is in no tool map, and
  the fan-out then ran the named tool directly — so
  `pool(tool="local_run", items=[...])` ran a denied or privileged tool with
  no policy, no approver and no argument check. The policy now decides on the
  tool the pool names, asks once with the whole list, and every item is
  checked against that tool's schema before any of them starts.
- **Security: `Agent.execute()` applied no permission policy.** It is the
  seam a serverless driver runs tools through, and a denied tool ran there.
  It applies the same policy and approver as `run()`, and a refusal is
  `(None, reason)`.
- **Security: a `Gate` opened on any truthy value**, so the string `"false"`
  from a form or a query string ran the action. It opens on `True`, skips on
  `False`, and raises `TypeError` with the expected shape on anything else.
  It carries the risk class of the tool it wraps, so a policy that denies
  `destructive` sees `Gate(wipe_table)` as destructive, and a plain function
  is made a tool when the gate is built, not after a person approved it.
- **Security: a REST path parameter of `..` left its endpoint.** `quote`
  keeps dot segments and the HTTP layer collapses them, so
  `DELETE /projects/1/members/{mid}` with `mid=".."` deleted the project.
  `.`, `..` and the empty string are refused.
- **`pool(prompt=…)` spends inside the run.** Its calls ran without the
  agent's `max_cost`, hooks or retries, and their tokens and cost reached
  neither the run's counters nor the result. A pool also takes at most 200
  items and runs at most 8 at once; both numbers were the model's to choose.
- **The model reads what a `pool` call returned**: the list of results, a
  failed entry as `{"error": …}`, where it read `PoolResult(ok, items=3)`.
- **`local_run` runs `python x.py` as written.** Any command ending in `.py`
  had the interpreter put in front of it, so `python x.py`, `pytest t.py` and
  `cat m.py` all failed; only a command that starts with a script, or
  `python=True`, gets it now, quoted.
- **`Model(options=OPTS)` leaves `OPTS` alone.** It popped `extra` out of the
  caller's dict, so a second model built from the same options — the other
  arm of a comparison — lost its provider fields.
- **Small contracts in the agent and the run document.** An `Agent` told a
  user's own class named `Utility` or `Skill` it was the library's, because
  it compared class names; it checks types now. A plan given as one string
  was recorded one character per item. A run whose steps were all done or
  skipped reported `running`. `RunDocument.from_dict` on something else
  raised a bare `KeyError`; it raises `ValueError` saying so. A tool call's
  arguments are serialised for the log only when the log is on.
  `RunDocument.step()`, which nothing called, is gone.
- **A search result's date is a real day or none.** `2026-13-45` and
  `Feb 30, 2026` came back as dates; they are `""` now.
- **Google transcription no longer pays twice for the end of a file.** The
  chunk loop transcribed one more chunk made only of the last overlap; the
  chunk that reaches the end is the last. Chunks go to the API from memory
  instead of a temporary file.
- **`@tool` reads what a function says about itself.** A NumPy-style
  docstring — the form this library writes — gave each parameter the
  description `float`, the type after the colon; it now gives the indented
  text under it. A `functools.partial` was named `partial` and described by
  functools' docstring; it takes its function's. An unannotated `count=3`
  was a string in the schema; a default of a JSON type gives the type. An
  annotation naming something undefined raised a bare `NameError`; it raises
  `TypeError` naming the function. A name with a trailing newline passed the
  check.
- **The model layer reads its provider list from the data.** The keys
  accepted as a `provider/model` prefix and the prefixes named in the error
  for an unknown model were typed by hand, and the error's list had fallen
  behind (no `magistral-`, `devstral-`, `qwq`, `wan2`, `chatgpt-image-`).
  `ToolCall` hashes — its dict of arguments is left out of the hash — and
  `refresh("private")` asks a keyless server without demanding a key.
- **The library's exceptions pickle and copy.** `BudgetExceeded`, every
  `APIError`, `UnsupportedOption` and `Suspend` take their fields in
  `__init__`, so Python's way of rebuilding them — calling the class with the
  formatted message — raised `TypeError` or doubled the message. Through a
  process pool, Celery or Ray the caller got the unpickler's error instead of
  the one that happened. They are rebuilt from their message and fields now;
  `BudgetExceeded.what` is kept as a field as well.
- **An `Eval` can be built where the working directory is read-only.** The
  constructor created `./eval-runs`, which is an `OSError` on Lambda; the
  directory is made on the first write.
- **A vector record reads back as it was written.** Fetch now reads a
  `document` key as the text when `text` is absent, as query did; a Chroma
  document of None is empty text rather than a broken record; and a Chroma
  collection recreated under the same name is looked up again on a 404
  instead of failing for the life of the process. Each backend reads a
  payload in one place, and the four vector tools share one base.
- **MCP servers are reached as written.** `MCP("./server.py")` runs the
  script under the current interpreter (and `.js` under `node`) instead of
  executing it as a binary; a URL ending in `/sse` is SSE with a query string
  too; `headers: None` in a server dict is no headers rather than a crash;
  `MCP("")` and an unknown `transport=` are refused at construction; and a
  repr shows no query string, header value or env value, which is where keys
  live. The settings are one value, checked once, instead of five keywords
  passed through every signature.
- **A budget stop between turns leaves the paid turns readable.** A
  multi-turn skill stopped by `max_cost` before its last turn had `history`
  and `last_usage` at None, while the error told the caller to read them.
  They are published turn by turn now.
- **A stream no longer deletes a parked run.** `stream()` kept the document
  id of the agent's previous run, so a stream that finished after a `run()`
  had parked removed that parked run from the store, and its `resume()` then
  raised `KeyError`. `run()`, `stream()` and `resume()` start a run through
  one method now, which clears it.
- **`Shell("/")` reads.** The containment check compared string prefixes, so
  with the filesystem root as the folder every path had to start with `//`
  and nothing was inside. It compares path components now, in one `within()`
  that `Shell` and output-path confinement share.
- **A result's `cost` is read when no tokens were counted.** `ChainResult` and
  `PoolResult` read it through the truthiness of `Usage`, which is false with
  no tokens — so a cost the provider reported, or a price per picture, read as
  `None`. One `Spend` mixin serves both now.
- **Chain and Pool recognise an agent or a tool by type.** Both matched the
  class *name* in the MRO, so any class called `Agent` was run as an agent and
  another library's `Tool` as ours. One `_runners` module answers for both, by
  `isinstance`.
- **A chain with a `gate()` and no store warns,** as one with a `wait()` did:
  the check read the class name `Wait`, and now reads whether a step
  suspends.
- **The chunker keeps a space it used to drop** when a single token longer
  than `max_chars` was cut.

## [4.0.0] — 2026-10-12

**No compatibility layer.** Every name and path 3.x kept alive for older code
is removed, so the library has one way to do each thing. An old name does not
work quietly: it raises, and the error names what to write instead. Every
change, with its replacement: `docs/start/migrating-to-4.md`.

**More providers, described by data.** Each is a provider file on the OpenAI
family, written from the provider's own documentation read 2026-10-11. A
provider ships when its live calls pass — text, a JSON Schema, a tool call, a
stream and each reasoning level. Mistral, Xiaomi, Z.ai, MiniMax and StepFun
have passed on every model they register; Meta has no key, and answered
through OpenRouter.

The first release on PyPI since 3.1.0: 3.2.0 to 3.10.0 were tagged in this
file and never published, and everything they record is in 4.0.0.

### Changed

- **Shell, MCP and REST tools say what they risk.** Every one of them inherited the
  default class, `write`, which a policy allows — so with
  `permissions=PermissionPolicy()` on, an agent could run any shell command,
  or any tool a server declared, without being asked. `Shell.run` is now
  `privileged` (approval), `browse` and `read` are `read`. An MCP tool takes
  its class from the server's annotations — `readOnlyHint`,
  `destructiveHint`, `openWorldHint` — and is `privileged` where the server
  says nothing; `MCP(server, risk={...})` puts the caller's word over the
  server's. A REST endpoint's class follows its method — `GET` `read`,
  `POST`/`PUT`/`PATCH` `external`, `DELETE` `destructive` — with
  `endpoint(..., risk=…)` to override. An agent with no policy is unchanged.
  Found reading vercel/ai, whose MCP wrapper defaults an uncovered tool to
  approval for this reason.
- **A `Skill` with an image output raises**, naming `image(service, model=…)`.
  It warned since 3.9.0. The image tool makes its request through a `Skill`
  by a seam of its own, `_image=True`, so retries, the budget and the record
  of adaptations come with it.
- **Renamed:** `convertToSpeech` → `Speech`, `convertToText` → `Transcription`,
  `VectorChunkTool` → `Chunk`.
- **The tools package exports the tools.** Utilities' classes —
  `LocalBrowseTool` and the other three, `VectorQueryTool` and its siblings,
  `RestApiTool`, `MCPTool` — import from `yait_aichain.utilities.shell`,
  `.vectors`, `.rest` and `.mcp`, where they live.

### Removed

- **Old factories and aliases:** `Embedding(...)`, `Reranker(...)`, `TTS()`,
  `STT()`, `LocalTools()`, `localBrowse`/`localRead`/`localWrite`/`localRun`,
  `vectorChunk`/`vectorQuery`/`vectorUpsert`/`vectorFetch`/`vectorDelete`,
  `MCPTools` as a public name, `PerplexitySearchTool`, `BraveSearchTool`,
  `OpenAIEmbedder`, `CohereEmbedder`, `VoyageEmbedder`, `GoogleEmbedder`, and
  a search tool's `.search()`, the 3.2 name of `.run()`.
- **Old module paths** — `tools.vectordb`, `tools.embedding`,
  `tools.reranking`, `tools.local`, `tools.mcp`, `tools.rest_api`,
  `tools.convert` — and the import finder that kept them resolving.
- **Renamed image options' old names:** `output_compression`,
  `input_fidelity`, `reference_fidelity` are unknown keys now.
- **A `Hook`'s `step_started` / `step_ended`** for an agent's tool calls,
  which fired with a warning since 2.7.0.
- `ImageSkillWarning`, and the API page of older names.

### Added

- **An old name says what replaced it.** `from yait_aichain.tools import
  VectorDB` raises `ImportError` naming `Vectors`, and `import
  yait_aichain.tools.vectordb` names `yait_aichain.utilities`; the same for
  every name and path above. Not compatibility — nothing old runs — but the
  error an agent writing from its weights needs to fix itself in one retry.
  Held by `tests/test_old_names_say_what_to_write.py`.
- **`docs/start/migrating-to-4.md`.**

- **Z.ai (GLM)** — `glm-5.3`, `glm-5.2`, `glm-5.1`, `glm-5`, `glm-4.7`,
  `glm-5.3-flash`, `glm-5.3-flashx`; `ZAI_API_KEY`. Its chat path is
  `/api/paas/v4/chat/completions`; `reasoning` turns its `thinking` switch on
  and sets `reasoning_effort`, and `"none"` turns thinking off — except on
  GLM-5.3, 5.3 Flash and FlashX, which always think and answer 400 to it, so
  there `"none"` is declined with a notice (`reasoning_off = false` on the
  model). A JSON Schema is asked for as JSON with the schema in the prompt,
  and said so. Called live: all eleven models pass.
- **Meta (Muse)** — `muse-spark-1.3`, `1.2`, `1.1` and the cheaper
  `-contributor` models, which Meta trains on; `META_API_KEY`. Sent as
  `max_completion_tokens` and `reasoning_effort`; `reasoning="none"` is
  refused before the network, as the API refuses it.
- **Mistral** — `mistral-large-4-0`, `mistral-medium-3-5`,
  `mistral-small-2603`, `ministral-14b-2512`, `codestral-2508`;
  `MISTRAL_API_KEY`. Every id is in the model list a Mistral key returns.
  Called live: all five pass. Codestral and Ministral refuse a reasoning
  setting with HTTP 400, so it is declined for them, and said so.
- **Xiaomi (MiMo)** — `mimo-v2.6-pro`, `mimo-v2.6-flash`; `MIMO_API_KEY`. No
  reasoning control is documented, so none is claimed. Prices from the
  overseas rate card as third parties quote it. Called live: both pass.
- **MiniMax** — `MiniMax-M3`; `MINIMAX_API_KEY`. `reasoning_split` is sent on
  every request so thinking stays out of the answer; a level keeps adaptive
  thinking on, `"none"` turns it off. Its API accepts a JSON Schema and does
  not follow it — extra fields, and the JSON fenced in ```json — so the
  schema goes in the prompt and a fenced answer is read, for every provider
  that takes no schema. Called live: all eight models pass.
- **StepFun** — `step-5-preview`; `STEPFUN_API_KEY`. The API is from
  third-party documentation, StepFun's reference could not be fetched; the
  prices are StepFun's own. Called live: all four models pass.
- **The models each provider's key lists, registered** — Z.ai `glm-5-turbo`,
  `glm-4.6`, `glm-4.5`, `glm-4.5-air`; Mistral `mistral-large-2512` (Large 3),
  `ministral-8b-2512`, `ministral-3b-2512`; Xiaomi `mimo-v2.6-pro-ultraspeed`;
  MiniMax `MiniMax-M2.7`, `M2.5`, `M2.1`, `M2` and their `-highspeed`
  variants; StepFun `step-3.7-flash`, `step-3.5-flash`,
  `step-3.5-flash-2603`. Prices are each provider's own page, read
  2026-10-11, except `glm-5-turbo` and Xiaomi's, which are OpenRouter's. The
  new Mistral and Xiaomi models were called live and pass; the three Mistral
  ones refuse a reasoning setting, like Ministral 14B. Reading the rate cards
  corrected two existing entries: MiniMax-M3 doubles its price above 512k
  input tokens, and step-5-preview's cache read is $0.10, not $0.05, and it
  reads images.
- **NVIDIA is not a provider here**: its hosted catalogue's model ids and
  prices could not be confirmed; Nemotron 3 Ultra is reached through
  OpenRouter or a server of your own, by `private/`.
- **A provider the OpenAI family has no branch for is built from its data**:
  `[provider.reasoning]` names the reasoning field — or none, for a provider
  with a switch and no level — and what to merge to switch thinking on or off,
  `[provider.request] always` names fields sent on every request, and `[provider.options] json_schema = false` says a
  provider takes JSON but not a schema. A new provider in this dialect is a
  TOML file.
- **An empty balance answered with 429 fails at once.** Z.ai answers 429 for
  a rate limit and for an account with no money alike, and the transport
  retried both five times with backoff before the body was read — a minute a
  call, for an answer that could not change. The retry policy reads a 429's
  body first now: billing raises `InsufficientCreditsError` on the first
  answer (0.8 s, live), a rate limit is retried as before. The same policy
  covers model calls, streams, search, embeddings, reranking, vectors and
  REST endpoints.
- **A reply whose content is a list of parts is read.** Mistral's reasoning
  models — Large 4 — answer with `[{"type": "thinking", …}, {"type": "text",
  …}]` instead of a string. The JSON mode failed on it with `AttributeError`
  and an agent's next turn with `ValueError`; only the text parts are the
  answer now, in a reply and in a stream, and the thinking is kept out of it.
- **A cost the provider reports is the cost.** OpenRouter returns the call's
  price in `usage.cost`; it is read now and wins over the registry's estimate,
  so a model reached through `private/` with OpenRouter's URL reports what it
  cost instead of `None`. Verified live on two models.

### Fixed

- **`import yait_aichain` works on a bare Python 3.10.** Provider data is TOML,
  which the standard library reads only from 3.11; on 3.10 the package needs
  `tomli`, and since 1.2.3 nothing declared it — every install on 3.10,
  Lambda's python3.10 among them, failed at import. The test suite never saw
  it, because pytest brings `tomli` with it. It is a dependency now, for
  Python below 3.11 only.
- **A chain with a REST endpoint loads.** It saved, and nothing of the
  endpoint reached the file, so `Chain.load()` called its constructor with
  no arguments and raised `TypeError`. The URL, method, fields, headers,
  auth by name and risk class are written now; a token given as a value, or
  a header whose name says it carries a secret, makes `save()` refuse with
  the two sources that can be saved. Loading an endpoint that carries auth,
  or a risk class other than its method's, needs `trusted=True` — a
  tampered file could otherwise point a key from your environment at any
  address, or let a `DELETE` past a permission policy as `read`.

### Known issues

Found by an audit of this release, and to be fixed in 4.0.x. Each is a reason
to keep a pattern out of production until then, not a reason to wait for it.

- **A permission policy is checked in the agent's own loop only.** The
  built-in `pool` tool, `Agent.execute()`, `resume()`, and tools run as steps
  of a `Chain` or items of a `Pool` do not consult it: a tool the policy
  denies runs when the model calls it through `pool`.
- **`Chain.load()` without `trusted=True` still builds any tool class of the
  library from the file,** `Shell`'s `run` among them. Load only chain files
  you wrote.
- **An `Agent` holds the state of its run.** One instance must not serve
  concurrent runs or several tenants: approvals, events and the run id are
  shared. A numeric `max_cost` is spent over the object's life, not per run,
  and `pool(prompt=…)` spends outside it.
- **`gate()` approves any truthy decision,** the string `"false"` included.
- **A REST path parameter equal to `..`** reaches a different endpoint.
- **`Shell`'s `run` prefixes any command ending in `.py` with the
  interpreter,** so `python script.py` fails.
- **`Model(options=…)` removes `extra` from the dict it is given;** a second
  model built from the same dict goes without it.
- **Claude with `reasoning` and tools fails on the second step,** and Gemini
  rejects the results of parallel tool calls.

## [3.10.0] — 2026-10-10

**The registry, brought to what providers sell and charge on 2026-10-10.**
Every price was read from the provider's own pricing page that day, and each
file names its source. Every new text model answered a live call and reported
a cost, 35 of 35; six new image models drew a picture.

### Added

- **Models.** Anthropic: `claude-fable-5-1`, `claude-opus-5-5`,
  `claude-sonnet-5-5`, `claude-haiku-5-5`, and `claude-opus-4-5-20251101` and
  `claude-sonnet-4-5-20250929`, which were reachable and unlisted. OpenAI:
  `gpt-6.1-sol`, `gpt-5.2`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`.
  Google: `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`,
  `gemini-3.1-flash-lite`, and the image models `gemini-nano-banana-2.1` and
  `gemini-3.1-flash-lite-image`. xAI: `grok-4.7`, `grok-4.6`, `grok-4.5`,
  `grok-4.3`, `grok-4.20-0309-reasoning`, `grok-4.20-0309-non-reasoning`,
  `grok-build-0.1`, and `grok-imagine-image-2.0` and
  `grok-imagine-image-quality`. DeepSeek: `deepseek-flash`, `deepseek-v4-pro`.
  Kimi: `kimi-k2.7-code-highspeed`. Qwen: `qwen3.8-max`, `qwen3.7-max`,
  `qwen3.7-plus`, `qwen3.6-plus`, `qwen3.5-plus`, `qwen3.8-flash`,
  `qwen3.7-flash`, `qwen3.6-flash`, `qwen-flash`. BFL: `flux-3-image`,
  `flux-2-max`, `flux-2-flex`, `flux-2-klein-9b`, `flux-2-klein-4b`, each
  checked to be an endpoint. Recraft: `recraftv4_1_flash`.
- **Three price fields a model can carry.** `cache_read`, where a cached
  prefix is not a tenth of the input rate: Claude 5.5 reads at a twentieth and
  Fable 5.1 at a fortieth, so a cached loop on them was estimated two to four
  times too dear. `long_context = {above, input, output}`: Claude Haiku 5.5
  bills a whole call at five times the rate once the prompt passes 100K
  tokens. `per_image`, for a model priced by the picture rather than by the
  token; `image(...)` reports it as `cost` when the provider reports no usage,
  and says in `cost_basis` that it is the base size's rate. Twenty-five image
  models carry one; twenty were in the registry already, and `cost` was `None`
  on every one of them.

- **The plugin installs in Codex and Cursor**, beside Claude Code: one skill
  folder in the Agent Skills format, a manifest for each agent
  (`plugin/.claude-plugin/`, `plugin/plugin.json`, `plugin/.cursor-plugin/`)
  and a marketplace for each at the repository root, all generated by
  `scripts/skill.py` and checked in CI.

- **`SKILL.md` for 3.x**, and held-out set 3 to measure it: eleven tasks on
  what 3.x added, frozen before the rewrite. A mid-tier model passed all
  eleven in all three trials, before the rewrite as after — `llms.txt` had
  kept up with every 3.x release and `SKILL.md` had not — and Claude Code
  passed all eleven. The old `SKILL.md` taught a `Tool` subclass, `Wait`/`Gate`
  as classes and `tool(input=…, options=…)` as the only call, and said
  `agent.resume()` does not run, which is false since 3.1
  (`evals/agent_builds/HOLDOUT-RESULTS-3.md`).

- **The documentation in two layers.** Guides — `start/`, `pipelines/`,
  `agents/`, `tools/`, `utilities/`, `control/`, `models/`, and
  `with-your-agent.md` and `production.md` — say what each part is for, why
  it is built the way it is and how to use it well, with little code. The
  API — `docs/api/`, one page per area — is rendered from the code's
  signatures and docstrings, so it cannot drift from it; a public name with no
  page fails a test, and the number of parameters no docstring describes is
  held by a ratchet (361 when it was written, 140 now). Writing those
  docstrings found them describing 1.x and 2.x in `AgentResult`, `Chain`,
  `AgentMemory`, `Budget` and `Shell`, and two examples that could not run:
  `Vectors` upserted `chunk()` pieces, which have no `id`, and `Budget`
  passed a `budget=` no primitive takes. The README is the promises, a map of
  the guides and the examples; `design/` moved to the root; the six cookbook
  skeletons were deleted; `tests/test_doc_links.py` holds every relative link.

### Changed

- **Prices.** `claude-sonnet-5` 3.0/15.0 → 2.0/10.0: the introductory rate
  became the list price. The retired xAI names (`grok-4-0709`, the
  `grok-4-fast` and `grok-4-1-fast` models, `grok-3`) still answer and are
  billed at grok-4.3's 1.25/2.5, which they now carry. `chatgpt-image-latest`
  output 30 → 32. `image_input` added to `gpt-image-2`, `gpt-image-1.5`,
  `chatgpt-image-latest` (8.0), `gpt-image-1-mini` (2.5) and `gpt-image-1`,
  which had no price at all (5.0/40.0, image input 10.0).
  `qwen3-32b` priced. `gpt-5.6-sol` keeps its 5.0/30.0 standard rate against a
  promotional 4.0/20.0 with no end date, as the registry does with every
  introductory price.

### Removed

- **Models the provider retired.** `gemini-3.1-flash-lite-preview` (shut down
  2026-05-25; `gemini-3.1-flash-lite` is the same price). `kimi-k2-0905-preview`,
  `kimi-k2-turbo-preview`, `kimi-k2-thinking`, `kimi-k2-thinking-turbo`
  (discontinued 2026-05-25) and `kimi-k2.5` (2026-08-31). `sonar-reasoning`,
  which the API now refuses as deprecated. A name not in the registry still
  resolves by its prefix, so code naming one reaches the provider and hears its
  refusal.
- **The Reve provider.** Reve closed its API on 2026-08-14 and image
  generation in its app on 2026-09-27; on 2026-10-10 both API paths answered
  HTTP 405 and the API root redirected to the app. Gone: `ReveClient`,
  `reve.toml`, `reve-image`, the `reve-` prefix and `REVE_API_KEY`.
  `image("reve", …)` says why rather than "unknown service". The mechanism it
  brought — a universal value in a provider's own scale,
  `[provider.options.map.<option>]` — stays, pinned on a declared map.

### Not settled

- **Qwen**: `qwen3-max`, `qwen3-235b-a22b`, `qwen3-32b`, `qwen3-72b`,
  `QwQ-32B`, `qwen-vl-max`, `qwen-vl-plus` and the `qwen-image-edit` and
  `wan2.2` models answer "does not exist" to this project's key while the
  international catalogue lists them — the account's workspace, not the
  catalogue. Kept.
- **Not priced**: `flux-dev`, `flux-3-image` (tiered by resolution), the rest
  of the flux-2 family, the Recraft utility models, `grok-3-fast`. Perplexity's
  per-request search fee and DeepSeek's off-peak half price are outside the
  estimate. Per-image prices are the base size's; a larger picture on a
  tiered model costs more.
- **Deprecated, still listed**: `claude-sonnet-4-5-20250929` (retires
  2026-11-30), `gpt-image-1` (2026-10-23), `chatgpt-image-latest`,
  `gpt-image-1.5`, `gpt-image-1-mini` (2026-12-01), `gpt-5.4-nano`
  (2027-04-01), `gemini-2.5-flash-image`, `grok-imagine-image-quality`,
  `deepseek-chat` and `deepseek-reasoner`.

## [3.9.0] — 2026-10-10

**An image model is a tool.** A model is what can be an agent's mind — it
reads, reasons and calls tools; an image model does none of that, so it is
reached as a tool now: `image(service, model=…)`. Only the face changed: the
request is the one `Skill` and `Model` built before, from the same provider
data, with the same option vocabulary and the same notices.

### Added

- **`image(service, model=…, output_dir=…)`** — `openai`, `google`, `bfl`,
  `xai`, `qwen`, `recraft`, `reve` — and `Image`, the tool it makes.
  `.run(prompt, images=[…], options={…})` generates, or edits the images given
  (file paths or URLs). The options are the model layer's image vocabulary —
  `size`, `aspect_ratio`, `quality`, `background`, `output_format`,
  `compression`, `seed`, `fidelity` — converted where a provider spells it
  otherwise; an unknown one is an error. The result is a `Record`: the image
  (`base64`, `mime_type`, `url`, `revised_prompt`, and `path` when saved) for
  code, and one line for a model — what was made and where — never the bytes.
  `output_dir` is where images are saved, a setting the code chooses; without
  it the image goes back in the result. `cost` is the reported usage at the
  registry's rate where both exist (OpenAI) and `None` with the reason in
  `cost_basis` where they do not. Measured live 2026-10-10:
  gemini-2.5-flash-image at 16:9 returned 1344x768, and an edit of it kept the
  size; gpt-image-1-mini returned 1024x1024 for $0.0022.
- **A reference page for images**, `docs/tools-reference/image.md`.

### Deprecated

- **A `Skill` with an image model** — `Skill(Model("gpt-image-1.5"),
  output={"modalities": ["image"], …})` — warns with an `ImageSkillWarning`, a
  `FutureWarning`, pointing at `image(...)`, and raises in 4.0. It works as
  before until then. A `DeprecationWarning` was the first choice and was found
  hidden when the examples were run: Python shows one only when it is raised
  from `__main__`, and this one is raised in the library.

## [3.8.0] — 2026-10-10

**The convert tools leave the library.** `convertToMD`, `convertToHTML` and
`convertToPDF` were one project's case brought into a library, and they carried
more than they gave: bare MarkItDown pulls `onnxruntime` (80 MB) and `numpy`
(34 MB) to tell file types apart — 114 MB of a Lambda package's 250 — while its
PDF and DOCX converters need extras the `convert` extra never installed, so a
one-page PDF failed and a `.docx` came back as its raw bytes; `convertToHTML`
and `convertToPDF` needed `mistletoe` and `weasyprint`, which no extra declared,
WeasyPrint with system libraries pip cannot install. Their example lives in the
cookbooks now, using those libraries directly.

### Breaking

- `convertToMD`, `convertToHTML`, `convertToPDF` and their aliases
  (`MarkItDownTool`, `MistletoeTool`, `WeasyprintTool`) are gone, and so is the
  `convert` extra. Importing one — from `yait_aichain.tools`, from
  `yait_aichain.tools.convert`, or its old module, as a saved chain does — is an
  `ImportError` that says what replaced it: a page from its URL is
  `fetch("local")`; a local file, MarkItDown itself.

### Changed

- **`fetch("local")` keeps what it took from `convertToMD`** — the session with
  a timeout and the reading of a page's `<head>` for its title, dates and
  author — in `tools.fetch._page`. The `fetch` extra installs MarkItDown, as
  before.
- **`TTS(provider)` and `STT(provider)` live with `speech` and `transcribe`.**
  `yait_aichain.tools.convert`, `tools.convert.to_speech` and
  `tools.convert.to_text` still import, so speech code from before 3.5.0 and
  chains saved with those paths keep working.
- **The examples that read a page use `fetch("local")`** — 05 is now
  `05_tool_fetch.py`; 09, 12 and 14 fetch instead of converting, and the
  agent-as-chain-step guide saves its report with a `@tool` function instead
  of rendering a PDF.

## [3.7.0] — 2026-10-09

**A setting is accepted in exactly one place.** Eleven were accepted at
construction and again with a call, the call winning in silence. Each now lives
in the tool, and given to a call is an error that says where it belongs.

### Breaking

Each line is a setting a call can no longer give, and where it lives now:

- embeddings — `dimensions`, `batch_size`: `embed(service, model=…,
  dimensions=…, batch_size=…)`; `embed()` and `run()` take `input_type` only,
  and `Embedding(model, **defaults)` takes no other keyword.
- speech, transcription — `model`: `speech(service, model=…)`,
  `transcribe(service, model=…)`, or the class's `model=`; Qwen's `region`:
  `speech("qwen", region=…)`.
- vector tools — `collection`: the store's, `VectorDB(…, collection=…)`.
- chunking — `max_chars`, `overlap_chars`, `merge_peers`: `chunk(…)`.
- REST — `headers`, `timeout`: `RestApiTool(…, headers=…, timeout=…)` or
  `REST(…, headers=…, timeout=…)`.
- MCP — a call's `timeout`: `call_timeout=` on `MCPTool`, `MCPTools`, `MCP`;
  the connection's is `MCP(connect_timeout=…)`, still `timeout=` on `MCPTools`.

And:

- A REST or MCP argument given both in the input dict and as a keyword is an
  error; the keyword won.
- MCP `headers`, `env`, `cwd` or `transport` given both as an argument and in
  the server dict is an error; the dict won.
- A REST `auth` with more than one source of its token — `token`,
  `token_field`, `env` — is an error; the field won over the variable.
- `Reranker(model, …)` refuses any keyword it does not take; it accepted and
  dropped them — `Reranker("rerank-v3.5", top_n=3)` returned every document.
- An embedding `dimensions` on a model with a fixed length —
  `text-embedding-ada-002`, Qwen's `text-embedding-v2`, Google's older models —
  is refused when the embedder is made; it was sent nowhere and said nowhere.

### Added

- **A REST token a model never sees.** `auth={"token_field": …}` keeps the
  field in the tool's `parameters`, so code and a chain fill it — a token from a
  login step — and out of `schema()`, so a model is not asked for it; a model
  that passes it anyway is refused. `auth={"token": callable}` reads it when the
  call is made.
- **`REST(timeout=…)`, `RestApiTool(timeout=…)`; `MCPTool`/`MCPTools`/`MCP`
  `call_timeout=…`; `MCP(connect_timeout=…)`.**

### Fixed

- **`chunk`'s overlap scales with the piece.** It defaulted to 150 characters
  whatever `max_chars` was, so 780 characters at `max_chars=200` came back as 30
  near-identical windows; it is a tenth of `max_chars` now — 150 at the default
  1 500, as before — and 5 pieces there.
- **`yait_aichain.tools.convert.to_text` and the other moved modules are
  attributes again** without being imported first, as they were before 3.5.0.
  3.6.0's import finder made them attributes only once something had imported
  them, which a suite running every test in one process did not notice; a test
  now checks in a fresh process.

## [3.6.0] — 2026-10-09

**Utilities: what holds a connection or a place, and gives an agent the tools
it should have.** `Vectors`, `MCP`, `REST` and `Shell`, in
`yait_aichain.utilities`. Code calls a utility's methods; its tools are called
with `.run` and handed to an agent one by one — only a tool has `.run`, so the
code shows which is which.

### Added

- **`Vectors(backend, collection, description=…, embedder=…)`** — `create`,
  `upsert`, `fetch`, `delete`, `count` as methods, `query` as its tool:
  `Agent(tools=[kb.query])` searches and can do nothing else. A description of
  what the database holds is required, and the search tool carries it — "Search
  the product documentation — …" — so an agent with two knowledge bases can
  tell them apart.
- **`MCP(server)`** — `tools`, `names`, `tool(name)`, `call(name, **args)`;
  the server is asked for its tools on first use.
- **`REST(base_url, description=…, auth=…, headers=…)`** — `endpoint(name,
  method, path, description=…, …)` makes each tool, sharing the API's URL, auth
  and headers, set once; an endpoint is reachable as `api.<name>` once made. A
  description of the API is required, and every endpoint carries it before its
  own — "Restful Booker, a hotel booking API — One booking by its id." — since
  an endpoint alone does not tell a model which system it is in.
- **`Shell(folder)`** — `browse`, `read`, `write`, `run`, each a tool confined
  to the folder, each given to an agent on its own: `tools=[sh.browse,
  sh.read]` reads and cannot write.
- **A chain takes a function or a utility's method as a step** —
  `(kb.upsert, "count", {"records": "pieces"})` — its arguments filled by the
  names its signature takes, as a tool's are; a required one the chain cannot
  fill is an error naming it. Such a chain refuses to save: a method of a live
  object is not configuration.
- **`Agent(...)` refuses a utility, or one of its methods, by name**, and says
  which tools the utility has — `Vectors.upsert is a method of a utility, not a
  tool — code calls it. Its tools are query`.

### Changed

- **The vector store, MCP, REST and local tools moved to
  `yait_aichain.utilities`**: `tools.vectordb` → `utilities.vectors`,
  `tools.mcp` → `utilities.mcp`, `tools.rest_api` → `utilities.rest`,
  `tools.local` → `utilities.shell`. Their old names still import from
  `yait_aichain.tools` (`VectorDB`, `vectorQuery`, `MCPTools`, `RestApiTool`,
  `LocalTools`, …), and every old module path — including the vector store's
  providers — resolves to the moved module by every import form, `mock.patch`
  and `Chain.load`, on 3.10 as on 3.14. The 3.5.0 moves go through the same
  mechanism now: one map of old prefixes to new, longest first, rather than a
  list of modules.

## [3.5.0] — 2026-10-09

**Embeddings, reranking and chunking are tools; every tool is made by name.**
`embed`, `rerank`, `chunk`, `speech`, `transcribe`, `wait` and `gate` join
`search`, `fetch` and `decide`, and six families moved to folders of their own.
Every old path still imports.

### Breaking

- Each embedder and reranker is a `Tool` now, so calling one — `embedder(texts)`
  — is the safe call every tool has and returns a `ToolResult`. Before, it had
  no `__call__` and raised `TypeError`; `run()`, `embed()`, `rerank()` and
  `batch()` are unchanged.

### Added

- **`embed(service, model=…)`, `rerank(service, model=…)`** — `openai`,
  `cohere`, `voyage`, `google`, `qwen` and `cohere`, `voyage`, `qwen`. The model
  is a required setting: vectors from two models cannot be compared, so a store
  is filled by one. Both are `Tool`s with `risk = "read"`, named by provider
  (`embedOpenAI`, `rerankCohere`), and an agent accepts them — until now
  `Agent(tools=[Reranker(...)])` failed. An embedding reaches a model as a
  one-line summary — "2 vector(s) of 1536 dimensions from openai
  text-embedding-3-small, 7 tokens" — never as the vectors, a model can do
  nothing with 1 536 floats a text. `Embedding(model)` and `Reranker(model)`
  still work.
- **`speech(service)`, `transcribe(service)`** — `openai`, `google`, `xai`,
  `qwen`; `region` is `speech("qwen")`'s and a `TypeError` on any other.
  `TTS(provider)` and `STT(provider)` still work.
- **`chunk(max_chars=…, overlap_chars=…, merge_peers=…)`**, **`wait(…)`**,
  **`gate(tool, …)`** — `vectorChunk`, `Wait` and `Gate` by the convention's
  names; the old ones still work.
- **`name=` and `description=` on every factory**, and a saved chain keeps
  them for any tool, whatever its constructor takes.
- **Reference pages for embeddings and reranking**, which had none.

### Changed

- **Six families moved**: `tools.embedding` → `tools.embed`,
  `tools.reranking` → `tools.rerank`, `tools.convert.to_speech` →
  `tools.speech`, `tools.convert.to_text` → `tools.transcribe`,
  `tools.vectordb._chunk` → `tools.chunk`, `tools._wait` → `tools.general`.
  Each old path is the moved module itself, registered under its old name, so
  `import`, `from … import`, attribute access, `mock.patch` and `Chain.load` of
  a chain saved with an old class path all resolve — on 3.10 as on 3.14. A
  chain saved from now on writes the new path.

### Fixed

- **A saved chain keeps an embedder's model, a reranker's model and a
  chunker's sizes.** None was written: an embedder or a reranker did not load
  back at all (`missing 1 required positional argument: 'model'`), and a
  chunker came back with the default sizes — a store filled one way, queried
  with pieces cut another. Keys are never written.
- **The README's RAG chain runs.** It maps `{"input": "{candidates}",
  "options": {"query": "{question}", "top_n": 5}}` into a reranker; until 3.4.0
  the templates were dropped, and until now the reranker was not a tool.

## [3.4.0] — 2026-10-09

**Any function is a tool, and a chain fills a tool's arguments from a
template.** Plus what a saved chain lost of the 3.3.0 tools.

### Breaking

- A chain step that maps a tool's parameter to a variable the chain does not
  have, or maps a parameter the tool does not take, is an error naming it.
  Until now the first was dropped and the second ignored, both in silence.

### Added

- **`@tool`** — any function becomes a `Tool` that code, a chain and an agent
  call alike: `@tool(risk="read") def word_count(text: str) -> dict: …`. The
  function's name is the tool's name, the first line of its docstring the
  description, and its parameters the schema — `str`, `int`, `float`, `bool`,
  `list[...]`, `dict`, `Literal[...]` and `Optional[...]` to their JSON types,
  required where there is no default, each described by its `name: text` line
  in the docstring. `name=`, `description=` and `risk=` override. `risk`
  defaults to `"write"`, as on every tool: the library cannot tell what a
  function does, and assuming it only reads would let a permission policy pass
  it unasked. No docstring, `*args`, an annotation with no JSON type, a name a
  provider refuses or an unknown risk is refused when the tool is made, by
  name. Planned for 2.1.0 and never shipped. `docs/primitives/tools.md` and
  `llms.txt` show it before the subclass.
- **A chain fills a tool's argument from a template.** A step's mapping value
  is a variable's name, as before; a string with `{variables}`, filled as a
  Skill fills its prompt — `{"input": "Draw {topic} in watercolor"}`; one
  placeholder alone, `"{candidates}"`, passing the value itself so a list stays
  a list; or a dict resolved key by key, its templates filled and its numbers
  and plain strings kept — `{"options": {"query": "{question}", "top_n": 5}}`.
  Before, a template was looked up as a variable's name, found nothing and was
  dropped in silence, the tool then failing on the missing argument; a dict
  raised `TypeError: unhashable type: 'dict'` — the mapping the reranker's own
  docstring shows as its example.

### Fixed

- **A saved chain keeps what a search, fetch or decision tool is.** They wrote
  no settings, so `search("exa", name="web")` came back from `Chain.load` as
  `searchExa`, and a decision tool did not load at all (`missing 1 required
  positional argument: 'questions'`). They now write their questions, model,
  name and description, and `fetch("local")` its timeout and robots settings —
  never a key, and not its headers, which can carry cookies and tokens: that is
  a warning at save. A chain holding a `@tool` function refuses to save, by
  name: a function is code, and a file naming it could not bring it back.
- **`docs/primitives/chain.md` says a `Record` is kept under its key.** 3.3.0
  changed it and the table still said a dict output is merged.

## [3.3.0] — 2026-10-09

**Search, fetch and decide by one convention** — the first release of
`docs/design/tool-convention.md`. A tool is made by its provider's name —
`search("exa")`, `fetch("firecrawl")`, `decide("openai", questions=…)` — and
`run()` returns a `Record`: a dict for code, whose text is what a model reads.
Fetch is new, search has six providers, and the decision tools are a family.

### Breaking

Each is listed again below with its reason.

- `searchOpenAI` and `searchSerp` are removed, with their aliases.
- `run()` on a search or fetch tool returns a `Record`, not text: code that
  read the text reads `str(result)`. `search()` and `fetch()` on a tool are
  `run()` under their old names and return the same.
- `run()` on a fetch tool no longer raises when the page could not be fetched:
  the page carries its reason in `error`, and the model reads it.
- A chain keeps a search, fetch or decision result whole under its step's key
  instead of merging its fields (`results`, `answers`, `cost`) into the
  variables.
- `Agent(...)` refuses two tools with one name and anything that is not a
  tool, where the first was kept in silence and the second failed on the
  first turn.

### Added

- **`search(service)`, `fetch(service)`, `decide(service, questions=…)`** —
  a tool by its provider's name, the way the documentation now shows. Each
  takes `name=` and `description=`, which default to the provider's own;
  `api_key=` defaults to its environment variable; `decide` takes `model=`,
  `fetch("local", …)` its four local settings. Nothing else: every other
  setting of a search or a fetch is an option of the call, and one given to
  the factory is a `TypeError` that says so. An unknown service names the ones
  that exist. The classes (`searchExa`, `fetchFirecrawl`, `decisionOpenAI`, …)
  are the same tools by their own names, and take `name=` and `description=`
  as well. Each factory is its package, callable — `search` is
  `yait_aichain.tools.search` — so `from yait_aichain.tools.search.exa import …`,
  `import … as`, attribute access and `mock.patch` paths into it all resolve on
  every supported Python; a function by the same name hid the package and broke
  `mock.patch` on 3.10 and 3.11, which the first run on 3.10 showed.
- **`Record`** — what `run()` returns on search, fetch and decision tools: a
  dict for code, whose `str()` is the tool's `render()` — the numbered results,
  each page or why it could not be fetched, one line per answer. An agent shows
  the model that text rather than the dict as JSON; a Skill prompt that
  substitutes the result gets the same text; a chain keeps the record under its
  step's key, where a plain dict — a REST reply — is still merged. Copies keep
  the rendering; a pickled or stored record is the plain dict.
- **`cost_basis` on a decision** — where its `cost` came from: the input
  tokens the provider reported, at the published rate, with the source and the
  date it was read.

- **`fetch` — a web page's text from its URL, one record from five tools.**
  `fetchFirecrawl` (Scrape, one URL a call), `fetchTavily` (Extract, ≤ 20),
  `fetchExa` (Contents, ≤ 100) and `fetchParallel` (Extract, ≤ 20) use the
  same keys as their search tools; `fetchLocal` runs MarkItDown on this
  machine, costs nothing and comes only with the new `fetch` extra —
  `pip install "yait-aichain[fetch]"`; without it, construction raises with
  that line. `fetch(url_or_list, options)` returns `pages` — `url`,
  `final_url`, `title`, `published`, `author`, `text`, `error` — with
  `notices`, `cost`, `cost_basis` and `usage` as the search tools do; a URL
  that failed is a page with `error`, not a failed call. `run(url)` returns one
  page as text and raises with the reason. A list is split into each
  provider's batch size. One standard option, `max_chars` (default 20 000),
  and each provider's own: `only_main_content`, `extract_depth`,
  `max_age_hours`, `max_age_seconds`. `fetchLocal` reads `robots.txt` once per
  host and skips a page it disallows for `*` or the Claude crawlers, without
  requesting it (RFC 9309: 4xx allows, 5xx disallows); `respect_robots=False`
  turns that off. Verified live on all five, 2026-10-09.
- **`convertToMD` returns a web page's metadata and sends the caller's
  headers.** MarkItDown keeps only the body, and some sites state the
  publication date nowhere else — library.hbs.edu, checked 2026-10-07. After
  each call `last_metadata` holds the page's title, publication and
  modification dates, author, description, site name, canonical and final
  URL, read from JSON-LD and the common meta tags; a field the page lacks is
  left out. `options={"metadata": true}` puts it on top of the Markdown as
  front matter. The page is fetched once and read for both. `headers=` sends,
  say, a browser's `User-Agent`: blogs.lse.ac.uk answers 403 to the default
  agent and 200 to a browser's, and www.bcg.com does the reverse, so the
  default is unchanged.
- **`searchExa` — Exa's search, built for agents.** `EXA_API_KEY`. Each result
  comes with the passages of the page most relevant to the query (highlights,
  on by default), so an agent reads the answer without fetching the page;
  `text_chars` adds the page text, cut to a length. Highlights are capped at
  1 000 characters a result (`highlight_chars`; `0` turns them off): Exa sizes
  them to relevance, and uncapped they were measured at 5 000–8 000 characters
  for each result — most of every page, in the agent's context, at the same
  price. It takes the standard search options described below, and its own:
  `category`, `search_type` (`instant` to `deep-reasoning`), `exclude_domains`,
  `highlight_chars`, `text_chars`. Verified live, alone and inside a `gpt-6-luna`
  agent that searched, answered and cited a URL.
- **`searchFirecrawl` — Firecrawl's search (v2), a fifth provider in the common
  format.** `FIRECRAWL_API_KEY`. A scraper first: `text_chars` returns each
  page as Markdown in the same call, into `text`; its own `categories`
  (research, pdf, developer, gov) and `news` source are offered. Its domain
  filter takes host names only, so a domain with a path travels as `site:` and
  is reported; `language` has no field and is declined. **Its time filter is
  not enforced** — measured 2026-10-08, a 28.09–04.10 range returned exactly the
  unfiltered results — so it is sent and every such search says so in
  `notices`. Verified live: web, news and scraped results.
- **`searchTavily` — Tavily's search, a sixth provider in the common format.**
  `TAVILY_API_KEY`. Each result carries up to three passages chosen for the
  query, split into `snippets`; `text_chars` returns the page as Markdown.
  `country` is translated from an ISO code to the country name Tavily takes,
  and applied on the general topic only, said so otherwise; `recency` `hour`
  is widened to a day, said so. Dates are always asked for, and Tavily's
  `filter_by_published_date` is offered because without it undated pages pass
  a date filter. Verified live 2026-10-08: a week's window on `hbr.org` returned
  five pages, all dated inside it, with or without the strict filter.
- **`searchParallel` — Parallel's search, a seventh provider in the common
  format.** `PARALLEL_API_KEY`. Built for agents: it takes an objective beside
  short keyword queries, and returns the excerpts of each page that serve it.
  The query is both, and `search_queries` and `objective` add or replace
  them. Domains with paths and bare extensions such as `.edu` reach its source
  policy; `start_date` and `recency` become `after_date`, and `hour` is a day
  there, said so. Parallel filters from a start date only, so `end_date` is
  declined with a notice, as `language` is. Verified live 2026-10-08: the start
  date and a path prefix held; undated pages pass the date filter; the ceiling
  on results is 20, which the documentation does not state.
- **Decision tools — a decision with probabilities, not prose, from two
  providers in one format.** `decisionOpenAI` (OpenAI's Decisions API,
  `gpt-6-luna`, `OPENAI_API_KEY`) and `decisionTypeSafe` (TypeSafe's Jev,
  `TYPESAFE_API_KEY`) answer typed questions about a text: `predicate` (the
  probability a statement is true), `choice` (one option, a probability for
  each, `confidence`) and `score` (a value over ordered levels, `confidence`),
  all in one call. The two APIs differ in every wire detail — a map against a
  list, `noul` against `predicate`, probabilities keyed by index against
  objects — and the tools take and return one format, the way the search tools
  do; `Decision` is the base a third provider subclasses. The questions are
  fixed when the tool is made and checked there, and an agent passes only the
  text as `state`; the description the model sees is written from the
  questions. `run()` returns the answers, the version that answered and the
  cost, input only — $0.10 and $0.042 per million tokens. `decisionOpenAI`
  also takes images, as base64 data URLs, and `decisionTypeSafe` refuses them
  by name; a dict state reaches Jev as structure and OpenAI as JSON text. Both
  verified live with the same questions, and images on OpenAI; a `gpt-6-luna`
  agent called `triage(state=...)` by itself and routed by the number.

  A tool and not a `Model`, decided 2026-10-02 after trying Jev as one: every
  specialised model here that does not converse is a tool, as a model it bent
  ten places outside its own client, and an `Agent` could not use it at all.
  Built first as `convertToDecision` for Jev alone and replaced before release,
  when OpenAI shipped the same three question types on 2026-10-08.

### Changed

- **Search tools: one vocabulary of options, one record of results.** The base
  declared five standard options and one provider of five honoured all of them
  — Brave had neither `domains` nor `recency` — and results were
  text alone, in each provider's own layout. Now `searchPerplexity`,
  `searchBrave` and `searchExa` take `max_results`, `recency`,
  `start_date`, `end_date`, `country`, `language` and `domains`, each translated
  to the provider's own field or to `site:` in the query where there is none,
  and keep their own options beside them. An option a provider cannot honour,
  or honours differently, is reported as a warning, a `Note:` line the model
  reads and an entry in `notices` — Exa has no language filter, Brave's
  finest window is a day. `search()` returns `{"query", "provider", "results",
  "notices"}`, every result `{title, url, date, author, snippets, text}` from
  every provider, dates normalised to a day; `run()` returns the same results
  as one numbered layout. Brave's result date is the article's own rather than
  `page_age`, which is when Brave saw the page — 09-30 for an HBR article of
  09-01. Every call is priced: `search()` carries `cost` in USD, `cost_basis`
  — the provider's own figure, its credits or operation counts at the
  published rate, or the rate alone where the answer reports nothing — and
  `usage`, the provider's units; Tavily is now asked for its usage, which it
  otherwise leaves out. Risk is `read`. Verified live on Brave, Perplexity and Exa with one
  query and one week's window.
- The tool-page check in `tests/test_docs_generated.py` accepts a keyword the
  tool's schema declares. It allowed only `input`/`options`, which stopped being
  the whole truth in 3.1.0 when tools became callable by their own parameter
  names.

### Removed

- **`searchSerp` and its alias `SerpApiTool`.** SerpAPI resells scraped
  Google, Bing and other engines' result pages rather than an API those
  engines offer, which is not a footing the library will stand a tool on. It
  was also the one search provider never called live here. **A breaking change**: code importing either name
  fails.
- **`searchOpenAI` and its alias `OpenAIWebSearchTool`.** It answered with a
  paragraph the model wrote and its citations, not with results — a different
  contract under the same name, which the common search format could not hold.
  For an answer with sources, ask a model with web search; for results, use
  one of the search tools. **A breaking change**: code importing either
  name fails.

### Fixed

- **`Agent` refuses what it cannot use, by name.** An object without a name,
  `schema()` and `run()` — a reranker, an embedder, a `Skill` — passed
  construction and failed on the first turn with an `AttributeError`
  (`'RerankCohere' object has no attribute 'name'`); it is now a `TypeError`
  at `Agent(...)` naming the class, and for a `Skill` that it is a step for a
  Chain. Two tools with one name were kept as one, the second replacing the
  first in the agent's map while both went to the provider; that is now a
  `ValueError` naming them, and `name=` tells them apart.
- **MCP tools read their schema from `input_schema`.** MCP SDK v2 renamed
  `inputSchema`, and fastmcp 4 warned on every discovery; the old field is
  read only when the new one is absent, so a server on either SDK works.
- **`convertToMD` no longer waits forever on a URL.** MarkItDown fetches with
  `requests` and no timeout, so a server that held the connection open and
  never answered hung the call — measured 2026-10-06 on www.mckinsey.com, for
  hours. The tool now gives MarkItDown a session with a timeout, `(10, 30)`
  seconds (connect, read) by default; `timeout=` sets it and `None` restores
  the old behaviour. A timeout is a failed call, reported like any other.

## [3.2.0] — 2026-10-01

**What the yait agents tool evals needed from the library, five items.** Each
had a workaround in the product; each can now be removed.

1. **gpt-6 goes to the Responses API** — and so does every GPT generation after
   gpt-4, without an edit when the next one ships.
2. **gpt-6 has prices** — `gpt-6-luna`, `gpt-6-sol`, `gpt-6-astra`.
3. **Cached tokens are read from the Responses API.**
4. **`cache_control="last"`** — the Anthropic cache breakpoint an agent loop
   needs.
5. **Tools on Qwen** — shipped in 3.1.0; this release adds the live test the
   eval asked for.

### Added

- `gpt-6-luna` (0.10 / 0.50), `gpt-6-sol` (2 / 10) and `gpt-6-astra` (10 / 50),
  USD per 1M tokens, in `providers/openai.toml`; `estimate_cost` prices them.
  Their `caps` mirror gpt-5.6 — an assumption, not yet a measurement.
- `cache_control="last"` on Anthropic marks the newest block of the history,
  whatever its type, so a history ending in a `tool_result` is cached on that
  block and the next turn reads it back. Measured live on an agent-loop
  history: `"last"` wrote 3 921 tokens on the first tool turn and read all 3 921
  back on the next; `True` wrote nothing on the same turn. `True` and an
  integer index keep their meaning.
- `tests/clients/test_tools_live.py` — gpt-6-luna and qwen3.8-flash each call a
  function tool on their provider's own endpoint. Live-marked: the nightly job
  runs them, CI does not.
- `llms.txt` says which cache placement an agent loop wants.

### Fixed

- **gpt-6 was routed to Chat Completions**, which answers function tools
  together with reasoning with HTTP 400. Responses models were listed by
  prefix — `("gpt-5",)` — so every new generation arrived on the old endpoint
  by default. Now the exceptions are listed instead: the gpt-4 and gpt-3.5
  generations stay on Chat Completions, every other `gpt-<N>` model goes to
  `/v1/responses`. `gpt-image-*` and `gpt-oss-*` only share the prefix and are
  untouched. Verified live: gpt-6-luna calls a tool.
- **Responses API cached tokens were reported as full-price input.** The usage
  reader knew only the Chat Completions spelling,
  `prompt_tokens_details.cached_tokens`; the Responses API reports
  `input_tokens_details.cached_tokens`. Now both are read and subtracted from
  the input count — 1 000 input with 800 cached is `input_tokens=200`,
  `cache_read_tokens=800`. Measured live on gpt-6-luna: the second identical
  call reads 3 090 of 3 093 tokens from cache, which was billed as 3 093 fresh.

### Changed

- **`cache_control` refuses a string it does not know.** Any string went
  through `bool()` and became `True`, so a misspelt placement silently chose a
  different one; now only `"last"` is accepted, and anything else raises naming
  the three placements.

## [3.1.0] — 2026-10-01

**A run can wait for the world, and an agent can be the one who asks.** A
`Wait` or a `Gate` in an agent's tool list is a tool the model may call; the
invocation ends, the conversation goes to the store, and the signal that
answers it resumes the run. `arm()` is the same thing before the first step —
a run that exists and waits, which is what a subscription is. Plus the errors
the eval sets asked for, a Claude Code plugin, and two capabilities the library
claimed and did not have: a tool called by name, and tools on Qwen.

### Added

- **An agent can park its own run.** A `Wait` or a `Gate` in an agent's tool
  list is a tool the model may call, and calling it ends the invocation:
  `run()` returns a `SuspendedResult` with the run id and what is awaited, the
  conversation goes to the store, and `agent.resume(run_id, signal={...})`
  completes that tool with the signal — a `Gate` runs the guarded tool when
  approved — feeds its result back as the call's tool result and continues.
  Resuming a run that is no longer in the store raises `KeyError`, so a trigger
  delivering twice does not run the tail twice; a run that parks again keeps
  its run id. `Agent(store=...)` is new and is **required** when a tool can
  park: without it the agent is refused at construction.
- **`Chain.arm()` and `Agent.arm()` — a run that exists and has taken no
  step.** It returns the same `SuspendedResult` a pause does, with a `run_id`,
  a document in the store, `reason`, `resume_with` and the scheduler `hint`,
  so a subscription, a human form and a gate are one shape: whatever brings
  the signal, it arrives as `resume(run_id, signal)`. For a chain the signal
  seeds the variables; for an agent it is the task, as a string or under
  `"task"`, with the other keys seeding the conversation. `hint` is recorded
  and never acted on — the library parks runs and schedules nothing.
- `Tool.suspends` declares that a tool can park a run. `Wait` and `Gate` set it;
  it is what the refusal above reads.
- **An error names the call that works** for the names programs most often
  reach for, taken from what a model wrote in `evals/agent_builds`: keyword
  arguments that do not exist on `Model`, `Skill`, `Agent`, `Chain` and `Pool`
  (`max_steps=`, `orchestrator=`, `memory=`, `temperature=` outside the model,
  `system=` on a Skill, `on_error=` on a Chain, `concurrency=` on a Pool, …);
  messages written with `content`; a bare list passed as `input=`;
  `agent.chat()`, `agent.suspend()`, `skill.invoke()`; `SuspendedResult.variables`
  and `.output`; `result.text` and `result.content`; and imports of modules that
  do not exist (`yait_aichain.hooks`, `yait_aichain.tracing`, …). The wrong
  name is still refused.
- An Agent step whose `task` variable is missing says how to provide it.
- A Claude Code plugin, `yait-aichain`, in `plugin/`, with a marketplace
  `yait` in `.claude-plugin/marketplace.json`. Its skill carries `SKILL.md`,
  `llms.txt` and `recipes.md` — the twenty reference solutions of
  `evals/agent_builds/tasks.py` — generated by `scripts/skill.py`, whose
  `--check` runs in CI.
- A tool with its own parameters is called by name, `tool(email="a@b.c")`,
  and returns a `ToolResult` like any other call; a missing required parameter
  is named. It raised a bare `TypeError`.
- `llms.txt` adds what the second held-out set found missing: a system message
  and `variables=` defaults on a `Skill`, the risk classes, what a `ToolCall`
  and an event carry, `mode="waterfall"` as the only mode that takes
  `planner_model=`, how a worker is picked and bounded, what `Gate` returns on
  approval, `result.output` when the last step returns a dict, and
  `token_budget` as a `stopped_by` value.
- `llms.txt` shows what twenty held-out tasks found missing: an image part,
  `Skill.stream()` and `last_result`, what `agent.stream()` yields and where the
  answer is, `team=`, `Gate(tool)` and its signal, `"deny"`, a failed Pool
  item, the direction of a Chain rename, what a Tool step's output becomes,
  where a resume signal goes, the values of `stopped_by`, what `max_cost` does
  when it is reached, and how to point a `private/` model at a server.

### Fixed

- **Qwen takes tools.** DashScope's compatible mode speaks the chat-completions
  tool format, and the format was already implemented one class up — but
  `QwenClient.build_request` overrode it **without** the `tools` parameter,
  while `supports_tools` inherited `True` from the OpenAI family. So the
  library claimed native tool calling for Qwen and answered an `Agent` with
  tools with `TypeError: build_request() got an unexpected keyword argument
  'tools'`. Verified live on `qwen3-max` and `qwen3.8-flash`, buffered and
  streamed. The workaround this caused in `build_stream_request` — forwarding
  `tools` only when there were any — is gone with its cause. An image model
  (`wan*-t2i-*`, `qwen-image-edit*`) still refuses tools by name.
- **Recraft no longer claims native tool calling.** The same accident one door
  along: an image-only client inheriting `supports_tools = True` from
  `OpenAIClient`, exactly as it once inherited streaming. It now refuses at the
  gate, with the family named, instead of failing inside the builder.
- **A tool's request to pause is no longer flattened into an error.** Inside an
  Agent, `Suspend` was caught with every other exception and reached the model
  as the tool result `"Suspend: ..."`: the run did not park, the store was
  never written, and nothing outside the loop learned that an answer was owed.
- A `Pool` item that asks to pause fails with the two shapes that do work — the
  `Gate` around the Pool, or an `Agent` with a store parking its own run — 
  instead of a bare `Suspend: ...` beside the other items' results.

### Changed

- `Model(client_options=...)` raises on a key its client does not read, naming
  the ones it does (`url`, `timeout`, `retries`, `proxy`, and `region` for
  Qwen). `base_url` was accepted and dropped, so requests went to the default
  address.
- A message with `content` and no `parts` raises. An `assistant` turn written
  that way was read as the "generate here" marker.

## [3.0.0] — 2026-09-13

**Chain, Pool and Agent return one shape of result.** `.output`, `.success`,
`.error`, `.history`, `.usage`, `.tokens_used`, `.cost`; `bool(result)` is
`.success`. `skill.run()` still returns the value.

### Changed — breaking

- **`Chain.run()` and `Chain.resume()` return a `ChainResult`**, not the last
  step's output. `result.output` is that output; `result["key"]` is any step's
  output by its key; `result.success` is `False` when a step failed under
  `skip`, `collect` or `stop`.
- **`Pool.run()` returns a `PoolResult`**, not a `list`. Iterating, indexing and
  `len()` are unchanged; comparing with a list needs `result.output`.
- A Chain used as a step or as a Pool runner contributes its `.output`, and a
  chain that did not complete fails that step or item.

### Added

- `ChainResult` and `PoolResult`, exported from `yait_aichain`.
- `AgentResult.usage`.
- `Eval` records a result object's `.output`, tokens and cost.

**Tests chosen by what they catch.** Mutation testing on `Pool` and `Chain`:
1,447 one-token changes to the library, 62% caught by the suite at the start,
77% now.

### Fixed

- **A test deleted `OPENAI_API_KEY` for every test after it.** The suite passed
  only in the file order CI uses; run in another order, 27 Chain tests failed.
  The test uses `patch.dict` now, and `tests/conftest.py` restores the
  environment after every test.

### Removed

- `Chain`'s branch that resumed a paused agent step through `runner.resume` —
  a method the agent has not had since 2.0. Unreachable, and indistinguishable
  from deleting it under mutation testing.

### Tests

- `tests/chain/test_chain_round_trip.py`: a saved chain compared field by field
  with what went in. 225 changes to `save`/`load` survived before it; 81 now.
- `tests/chain/test_chain_mutation_gaps.py`: usage and `RunContext` carried
  across `resume`, a second `resume`, the step kind on events, `task_key`, and
  what `Pool` hands an agent runner.
- `[tool.mutmut]` in `pyproject.toml`: `mutmut run` reproduces the measurement.

## [2.19.0] — 2026-09-13

**Can an agent build on this library from two files? Measured: `Pass^3` 0.60.**
Stages 4 and 5 of `docs/design/cleanup-plan-2026-09-13.md`. Stage 4 changed
no library code; stage 5 adds public exports, hence the minor version.

### Added — stage 5

- **`llms.txt`** at the repository root: the five primitives and `Model`, one
  example each, generated constructor signatures, an imports block, and the
  rules agents get wrong. Under 300 lines. **`SKILL.md`**: when to use which
  primitive, the workflow, and a checklist.
- **The driver surface is public.** `ToolCall`, `ToolCallRequest`,
  `tool_result_turn` and `dangling_calls` are exported from
  `yait_aichain.models`. `Agent.step()`'s own error message told a caller to
  use `tool_result_turn`, whose only import path was the private
  `models._calls`.
- **`evals/agent_builds/`** — twenty tasks with automatic checks, run by a
  mid-tier model given only `SKILL.md` and `llms.txt`. Each program runs in a
  sandbox where every model call is scripted and everything the program built
  is recorded; the check reads that record, not what the program printed.
  Held offline by `tests/test_agent_builds_instrument.py`: every reference
  solution passes, a program that builds nothing fails, and deliberate
  mistakes in references are caught.

### Measured — `gpt-5.4-mini`, 20 tasks × 3 prompt levels × 3 trials, $0.66

`Pass^3` 0.65 supportive, **0.60 neutral**, 0.65 competing. Of 47 failed
attempts, **41 were things `llms.txt` did not say**: that `chain.run()`
returns the last step's output rather than a dict of steps (20), the message
shape — `parts`, not `content` (9), where `Tracer` is imported from (8), the
fields of a `SuspendedResult` (2), and where a tool option is declared (2).
The rules it did state held: 35 of 36 attempts on the tasks that exercise
them. Full
analysis, generated from the committed ledger:
`evals/agent_builds/RESULTS.md`.

### Changed — stage 5

- `llms.txt` covers every undocumented cause above. That change is **not**
  measured: re-scoring the same twenty tasks would score the edit against the
  questions it was written for. The next number needs a held-out task set.
- Generated signatures render `typing.Optional` as `dict | None` rather than
  the bare word `Union`, and `llms.txt` gets a compact form.

### Tests — stage 5

- Every `from yait_aichain… import …` shown in `docs/`, `README.md`,
  `examples/`, `llms.txt` and `SKILL.md` must resolve.
- The bind test reads `llms.txt` and `SKILL.md`.

### Fixed — stage 4

- **Two offline tests had never run in CI.** CI deselected the live suite with
  `pytest -k "not Live"`, and `-k` is a case-insensitive substring match:
  `test_nothing_claimed_goes_undelivered` and `test_a_refusal_is_not_a_delivery`
  ("de*live*red") went with it. Live tests are now a marker — a test class whose
  name ends in `Live` is marked `live` in `tests/conftest.py` — and every
  workflow selects with `-m`. Both tests pass.

### Added — stage 4

- **Coverage in CI, with a floor and a register.** 77.93% on 3.10 and 3.14;
  `fail_under = 77.9` in `pyproject.toml`, and it only rises. `COVERAGE.md`
  lists every module under 60% with a kind — external, optional-dependency,
  legacy, debt — and a reason; `scripts/coverage_register.py` fails CI on a gap
  with no entry, and on an entry whose module has climbed past the threshold.
  Eleven modules are marked `debt`: offline-testable and not yet tested.
- **A nightly live workflow** (`.github/workflows/live.yml`): the provider
  tests on a schedule and on demand, with a per-class summary on the job page.
  A run in which nothing passed fails — a green run that skipped every provider
  measured nothing. It needs the provider keys added as repository secrets
  before its first real run.
- **`Pool` and `Chain` contract tests** — 29 and 30 more, making 40 and 58 in
  their directories: every error policy including what `stop` leaves pending,
  history and status, per-item usage, agent runners and agent steps, context
  crossing into worker threads, beacons, a budget shared by reference and lent
  and given back, untrusted `Chain.load` refusing foreign and lookalike class
  paths, an agent step's save/load round trip, hooks at step boundaries.
  Each property was checked by breaking the library and watching its test fail.

## [2.18.1] — 2026-09-13

**The examples run, the docs are generated from the code, and two speech tools
construct.** Stages 1 and 2 of `docs/design/cleanup-plan-2026-09-13.md`.
The library changes are small and all fixes: `sttQwen`/`ttsQwen` imported a
package that has not existed since 2.0, and the Google speech tools ignored
the Google provider's key.

### Fixed — stage 1, examples

- **Three examples could not construct their agent for the whole 2.x line**
  and the index described all three as working. `16_debug.py` now uses the
  2.x constructor and prints the journal. `20_observability.py` shows what
  gates a risky call today, `approve=`, with an approved and a refused refund
  and the refusal's reason on the event channel; it no longer claims a
  suspend/resume that was removed in 2.0. `18_agent_external_trigger.py` is
  now **`18_chain_external_trigger.py`**: parking a run and resuming it from
  another process is a Chain capability, so the webhook story is told with a
  `Gate` step and a shared `FileStore`.
- `docs/agents/observability.md` said agent approval rides suspend/resume; it
  says what happens instead and where the cross-process pattern lives.
- The quick reference in `examples/README.md` constructed an `Agent` with
  `orchestrator=` and `max_steps=`.

### Changed

- `examples/23_scaffold.py` is **`24_scaffold.py`** — two examples shared 23.
- `examples/README.md` and the examples table in `README.md` are **generated**
  by `scripts/examples_index.py` from each script's docstring. Every example
  now carries a `Required env vars:` block, and one without it stops the
  generator.

### Fixed — stage 2, documentation generated from data

- **`sttQwen()` could not be constructed, and `ttsQwen` failed on a region
  override.** Both imported `clients._families.qwen` — the pre-2.0 top-level
  layout. The same spelling sat in some eighty docstring examples; all now
  import `yait_aichain.*`.
- `ttsGoogle` and `sttGoogle` read only `GOOGLE_API_KEY`; they now read
  `GOOGLE_AI_API_KEY` first, the Google provider's key, as the embeddings do.
- **Every tool page called its tool with keywords no tool accepts** —
  `tool(query=…, max_results=5)`, `run(source=…)`, `run(text=…, format=…)` —
  36 calls on seven pages, and Chain input maps naming parameters that do not
  exist. A Tool is called with `input` and `options`; the pages now say so.
- `docs/reference/model-registry.md` listed 67 of 90 models and a `REGISTRY`
  constant that does not exist; `environment-variables.md` and
  `installation.md` named keys the library does not read (`DEEPL_API_KEY`,
  `LATE_API_KEY`, `SERP_API_KEY`) and missed four it does; `docs/index.md`
  listed three image providers of seven and headlined "sectional generation"
  and "Professional and Expert pipelines", neither of which exists;
  `tools-reference` described `convertToText` as text extraction (it is
  speech-to-text).
- `concepts.md` taught a plan/act/reflect agent with `max_attempts`;
  `overview.md` called `waterfall` the default; `agent-as-chain-step.md`
  listed `executors` and `memory` among what `Chain.save` handles. All three
  describe the 2.x loop now.

### Changed — stage 2

- **`scripts/docs.py` generates every data table in the docs** between
  `<!-- g:NAME -->` markers: the provider/modality table, the whole model
  registry with prices, the provider and tool key tables, the install key
  table, the tools index, a page per tool (search, conversion, speech, local,
  vector store, REST, MCP, Wait/Gate — 30 tools, up from 7), the Chain and
  Pool error-policy tables, `Pool`'s signature, the model option vocabulary,
  and the headline counts. `--check` runs in CI.
- Provider data carries `label` and, where a source documents one, `keys_url`.
- `primitives/tools.md` gained the governance section it never had — schema
  checks on a call, `risk`, `PermissionPolicy`, `approve=`.
- `primitives/eval.md` documents `pairwise`, `abstain`, the `n/j` column and
  the abstention rejection.
- `docs/index.md` links the design notes.

### Tests

- The bind test reads `examples/*.py` and `examples/README.md`, and checks
  `Model(` beside the four primitives.
- `tests/test_docs_generated.py` holds every generated page, fails when an
  exported tool has no page, and checks that tool pages call tools with
  `input`/`options`, name only declared options, and map Chain inputs onto
  parameters that exist.
- `tests/test_no_flat_imports.py` — no source line imports the pre-2.0 layout,
  and every tool that needs only a key constructs.
- `tests/test_examples_index.py` holds both generated pages and executes the
  examples that need no key and no network (`18`, `20`), checking each for the
  thing it exists to show.

## [2.18.0] — 2026-09-13

**`nudge` — a stop condition that speaks instead of ending the run.** Stage 0
of *iterating toward a target*, first because it needs no scorer.

### Added

- **`nudge(predicate, message)`, `stalled(k)`, `repeating(k)`** in
  `stop_when`. A ceiling says "stop" where a stalled run needs "change
  approach", so a stuck agent either spun until a ceiling caught it or was cut
  short by a ceiling low enough to catch the spin. A nudge appends a visible
  user turn and the loop carries on; it fires **once per streak** and re-arms
  when the condition clears, and terminal conditions are checked first, so a
  nudge cannot talk the loop past its budget. `nudge.fired` goes on the event
  channel.

  `stalled` and `repeating` call `Journal.has_progress` and
  `Journal.is_repeating`, which existed, were documented, and were called
  from no library code at all. They read the journal, not the task, so they
  apply to every task and have nothing to overfit to.

### Fixed

- **The agent loop recorded the tool's name as the attempt's intent.** Every
  call to one tool therefore shared an "intent", and `is_repeating` read ten
  different searches as one search repeated ten times. Latent until
  `repeating()` called it, and found by its first test. The loop records no
  intent now — it has no plan step to name — and the action carries the tool.

- **`Journal.progress_summary()` crashed on any journal the 2.x agent wrote.**
  The loop stores the action as text and the renderer called `.get` on it.
  Unnoticed because nothing in the library calls the view. Both renderers now
  label an entry by its action when it has no intent, instead of printing
  "(no intent)" down the list.

### Changed — release process

- **Publishing is gated on every supported Python.** `publish.yml` already
  ran the suite before building — on 3.12 only, the one version where the
  line that broke 3.10/3.11 was legal, which is how eleven unimportable
  releases went through a gate that was standing. It now runs 3.10–3.14 and
  `publish` needs all of them. The `Tests` workflow covers the same five.


## [2.17.1] — 2026-09-13

### Fixed

- **The package could not be imported on Python 3.10 or 3.11 — from 2.7.0
  through 2.17.0.** `pyproject.toml` declares `requires-python = ">=3.10"`,
  and `_events.py` carried an f-string whose replacement field was broken
  across two lines. That is PEP 701 syntax, valid only from 3.12; on 3.10 and
  3.11 it is a `SyntaxError` at import, so `import yait_aichain` failed
  outright for anyone on those versions. Eleven releases were published to
  PyPI in that state.

  CI caught it on the first affected commit and on every one after — the
  `Tests` workflow went red at 2.7.0 and stayed red. It was not looked at,
  because every release was checked against a local test run on 3.14, where
  the line is legal, and "the suite passes" was reported on that basis. The
  local run was true and was not the check that mattered: the supported floor
  is what a release has to pass, not the interpreter that happens to be
  installed.

  If you are on 3.10 or 3.11 and pinned any of 2.7.0–2.17.0, upgrade to this
  release.


## [2.17.0] — 2026-09-13

**Swarm coordination.** The plan's row, and the reason it waited for `state/`
and `Pool` to settle: both mechanisms are a field and a comparison, not an
architecture.

### Added

- **`AgentResult.acceptance` — what a result rests on.** `checked` when a
  programmatic fact supports it, `claimed` when only a model's word does, and
  `unsupported` when nothing does. `_journal.py` has separated `CHECK` from
  `MODEL_CLAIM` since it was written and nothing ever read the field, so the
  distinction was decoration; an agent closed its run on its own word. It is
  reported, not enforced — making the kind *change* what the loop does is
  `nudge`, the next row.

  An acceptance can be **invalidated**. It fingerprints the checked evidence,
  so `still_holds(fresh)` tells a caller whether the result was accepted on
  facts that are still true, and `supersede(reason)` records that someone
  intervened. Only checked evidence is fingerprinted: a worker re-wording its
  report changes no fact, and folding claims in would make it look as if the
  facts had moved.

- **Beacons that interrupt a wait.** `beacon(kind, message)` from anywhere
  inside a run — no parameter, the board is ambient like `RunContext`.
  `blocker`, `question` and `contract_change` demand attention; `contract`,
  `decision` and `fact` are coordination and never interrupt. A `Pool`
  watching its items **stops starting new ones** on an attention beacon
  instead of spending the rest of the fan-out on work the blocker already made
  pointless; items in flight finish. Boards forward upward, so a delegated
  worker's blocker reaches the pool two levels above. They land on
  `result.beacons` and `pool.beacons`, and a beacon raised outside any run
  warns that nobody heard it rather than vanishing.

### Fixed

- **A delegated report entered the parent's journal as a verified fact.**
  `delegate` returned the worker's account marked `MODEL_CLAIM`, and the loop
  wrote the call as `CHECK` regardless — the call ran, after all. So the one
  place the two kinds were meant to be told apart was exactly where they were
  merged. It is recorded as `model_claim` now; a *failed* call stays `check`,
  because that it failed is verified.


## [2.16.0] — 2026-09-13

The last of the Consistency list. Two of its entries turned out to be wrong
about the code rather than the code being wrong about itself, and that is
recorded rather than quietly dropped: a stale plan item costs somebody an
afternoon rediscovering there is nothing to fix.

### Added

- **`Skill(model, prompt="...")`** — one user message, one text part, which
  is the commonest shape there is. `input=` remains the only way to say
  anything else, deliberately: a shortcut that grows options becomes a second
  input format. Passing both raises rather than merging, because which one
  the model should see would be a guess.

### Not done, and why

- **"Bring camelCase classes to one style"** — measured, and the plan was
  wrong. Every camelCase class name *is* its tool's `name`, the string the
  model sees; every PascalCase one differs and carries a snake_case wire
  name. The two styles encode two different things. Renaming the camelCase
  ones would either change the wire name — breaking every prompt, eval and
  saved chain that references it — or open exactly the gap the PascalCase
  ones have.

  There is a real inconsistency one layer down, and it is a decision rather
  than a sweep: the *wire* names mix `convertToMD` with `vector_query`, so a
  model sees both styles in one tool list. Recorded as such.

- **"`response_field` and the REST auth token are silently dropped"** —
  stale. Both are read and applied; so are the static headers. Pinned by a
  test so the entry cannot come back.


## [2.15.0] — 2026-09-13

Two more Consistency items: the escape hatch a closed vocabulary needs, and
one vocabulary where there were two.

### Added

- **`options={"extra": {...}}` — provider fields, verbatim.** The model
  options are a closed set on purpose: a typo in them is unambiguous and
  raises. A closed set with no valve is a cage, though, and Perplexity's
  `search_domain_filter` has no universal meaning and never will — nobody
  else has a search index. The answer to "how do I filter by domain?" was
  "not through this library", so a caller who needed it dropped the whole
  abstraction rather than one field.

  What goes in travels unchanged and **unchecked**: nothing translates it,
  nothing declines it, no warning is raised. The library says that once
  instead of pretending to supervise, and it wins over the library's own
  field on a clash — an escape hatch that loses to what it is escaping is
  not one.

  It still appears in `Model.effective_options`. No warning is not the same
  as no record: two arms differing only by an `extra` field must not look
  identical in a run's own account of itself, which is the failure the notice
  channel exists to prevent, arriving by the one door that channel does not
  watch.

### Changed

- **`Chain` and `Pool` share one error vocabulary.** `on_step_error` took
  {raise, stop, skip} and `on_error` took {raise, collect, skip}; only
  `raise` meant the same thing in both, so a caller who learned one primitive
  guessed wrong at the other. `skip` was the trap — in a chain "carry on,
  loudly", in a pool the silent option was `collect` while `skip` warned.

  Four words now, one meaning each, on both: `raise` propagates, `stop` ends
  and keeps what completed, `skip` carries on and warns, `collect` carries on
  silently. `Chain` gains `collect` and `Pool` gains `stop`; the defaults do
  not move. What each does with `stop` differs in the obvious way — a chain
  stops running steps, a pool starts no more items, and the ones in flight
  are let finish, because killing a call mid-request costs the tokens anyway
  and loses the answer.

- **A failure is recorded as more than `str(exc)`.** Both primitives stored
  the message alone, which is the least useful part: `KeyError('tenant')`
  renders as `'tenant'`, reading as a value rather than a fault, and a
  `TimeoutError` with no message renders as nothing at all. The record now
  carries the type and the traceback beside it.


## [2.14.0] — 2026-09-13

Consistency, measured rather than listed. Two items, and both are the same
defect in different layers: a claim the code does not keep.

### Fixed

- **`score` means one thing on every vector backend now.** Chroma normalised
  its distances to [0, 1]; Qdrant and Pinecone handed their raw numbers
  through. One field therefore carried a normalised score on one backend, a
  cosine similarity in [-1, 1] on another, and — for the euclidean metric — a
  **distance**, where lower is nearer. `query()` promised "highest score =
  most similar" and that was false for two backends on one metric, so a
  descending sort put the least similar first.

  Whoever this hurt was doing exactly what this library tells people to do:
  comparing retrievers, or tuning a threshold on one collection and moving it.
  Three scales read as one, with nothing saying so — the provider defaults
  again, one floor down.

  `raw_score` keeps what the backend returned, because a conversion that
  throws the original away cannot be checked. `dot` is passed through and says
  so: unbounded and scale-dependent, it cannot reach [0, 1] without the
  vectors' magnitudes, and a fabricated bound would look comparable when it is
  not.

  The metric is **declared**, not discovered. Asking the collection was tried
  and withdrawn within the hour: it puts a network round trip with retries in
  front of the first query, and on an unreachable host that is a minute of
  backoff before a search that would otherwise fail fast. The suite going from
  9 seconds to 68 is what showed it.

- **Five tools declared options nothing reads.** `ttsOpenAI` and `ttsXAI`
  offered `language`, `ttsQwen` offered `language` and `speed`, `ttsGoogle`
  offered `model`, `sttGoogle` offered `prompt` — a model could see each one,
  set it, and watch it fall on the floor. The shared schema even claimed
  `language` was "optional for OpenAI and xAI"; neither endpoint has the
  field. Each subclass now narrows the shared schema to what its own `run()`
  consumes.

- **The test that was supposed to catch that could only see half.** It read
  `inspect.getsource(cls)` — the class body alone — and most of these tools
  implement `run()` in a shared base, so a read in a base was invisible. It
  found `region` on `ttsQwen` in 2.9.3 only because that read happened to sit
  in the subclass. It walks the MRO now, and checks **both** directions:
  declared-and-unread as well as read-and-undeclared.


## [2.13.0] — 2026-09-12

**Per-request secrets.** The last of the serverless product's library
requirements. It was recorded as *"`RunContext` exists; whether it covers the
injection story end to end is unverified"* — measured, it did not cover it at
all: `RunContext` reached `Chain` alone, and its own docstring calls its
contents non-secret, correctly, because the run document is serialised and
`FileStore` writes it to disk.

### Added

- **`Model(api_key=callable)`** — asked **once per request** and handed
  whatever run is in flight, so one `Model` serves every tenant. Resolving at
  construction would bind the first tenant's credential to every later
  request.

  The secret is resolved *from* the context and never stored in it: the
  context carries the tenant's **name**, and putting a key there would write
  it to disk in plaintext with the rest of the document. That fact settled a
  design question rather than a preference.

  A resolver returning nothing **raises**. There is no fall-back to the
  process-wide key, deliberately: a fall-back would send one tenant's request
  under another's credential and the provider would answer normally.

- **`state.using(context)` and `state.current()`** — the run in flight, in a
  `ContextVar`. A `context=` on `Skill`, on `Pool`, on every `Tool` would be
  the environment leaking into the scenario, which `VISION.md` names as the
  signal that a design has gone wrong. `Pool` copies it into each worker
  thread explicitly, because `contextvars` do not cross a thread boundary on
  their own — a silent failure otherwise, reading as a missing key rather than
  a lost context.

### Fixed

- **Every family now resolves the key through one property.** Five clients
  read `self._api_key` straight out of the attribute in their own
  `_auth_headers`, so a resolver would have reached whichever family somebody
  remembered to change.

- **The key is read once per request.** OpenAI's `_auth_headers` read it twice
  — once to test it, once to format it — which for a resolver means doing a
  vault lookup twice for work the caller cannot see. Found by a test counting
  the calls, not by reading the code.


## [2.12.0] — 2026-09-12

**A ceiling in money**, on every primitive that spends it. The last of the
serverless product's library requirements bar one.

### Added

- **`max_cost` on `Skill`, `Chain`, `Pool` and `Agent`**, taking a number or a
  shared `Budget`. `max_tokens` bounds one reply, and nobody budgets in
  replies: a chain of ten steps, a pool over a thousand items and an agent
  that decides how many turns it needs were all unbounded in the only unit
  that appears on the invoice. The agent has had `cost_budget` since 2.0; the
  other three had nothing, so a runaway `Pool` was visible on the bill and
  nowhere else — the worst place for a product whose pitch is not having to
  administer it.

  **One object, shared.** A budget given to a `Chain` covers the `Agent`
  inside it and the `Skill` inside that, because they decrement the same
  thing. Copies would let every level spend the full amount, which is the
  failure the shape exists to avoid, and it is why `Pool` — whose items run
  concurrently — needs an object with a lock rather than a number.

  A `Chain` **lends** its ceiling to its steps and takes it back afterwards:
  the steps arrive already built, and a `Skill` reused in a second chain must
  not keep the first one's budget.

  **It bounds beginning, not exceeding.** A reply's length is not known before
  it is paid for, so the first call always runs and the ceiling refuses the
  next. Promising otherwise would be lying about the one number a caller
  checks — the caveat `cost_budget` already carried, now carried twice.

  Exhaustion **raises** `BudgetExceeded`. Returning what finished with a flag
  set is how a truncated run is read as a complete one; catch it and read
  `last_usage` for the partial result.

  Money is counted in `Skill` and nowhere else, because `Skill` is the one
  primitive that talks to a model — `Agent` builds them, `Chain` and `Pool`
  run them. A second ceiling elsewhere would be a second thing to keep in
  sync.


## [2.11.0] — 2026-09-12

**Judging.** Both items carried over from `2.3.0`, and the reason they matter
is that the first thing needing them cannot be built from what was here: every
number in the byheart study is recall of evidence, and nothing scores the
answer.

### Added

- **`pairwise(model)` — a comparison that is not measuring position.** LLM
  judges have a documented preference for whichever answer they read first, so
  a single-ordering comparison measures that preference alongside quality and
  produces a number that looks exactly like a quality score. The pair is put
  twice, A/B and B/A, and the candidate wins only by winning both.

  A disagreement between the orderings **is** the position bias showing
  itself, and it resolves to the champion: the burden is on the challenger,
  which is what keeps a best-so-far from drifting on noise. How often they
  disagreed is reported per verdict — that is the judge's own reliability,
  measured for free while grading, and a pair that disagrees half the time is
  a coin toss wearing a rubric.

- **`abstain(why)` — a verdict of no verdict.** `ok=False`, so nothing that
  ignores the flag can read it as a pass, and excluded from the denominator,
  so nothing reads it as a wrong answer. Those are different claims: "wrong"
  and "unreadable by the instrument" fold into one number only if you are
  willing to blame the arm for the judge.

### Changed

- **`judge()` abstains instead of raising** on an unparseable verdict. Raising
  was right about the important half — guessing "pass" raises every arm at
  once and reads as a good result — and wrong about the rest: it ended the
  whole run over one row a judge could not read.

- **`Report` counts and prints the exclusions** (`abstentions()`, an `n/j`
  column). A denominator that quietly shrinks is its own defect. And an arm
  the judge could not read for more than a fifth of its rows is **rejected**:
  excluding rows keeps the survivors honest, but past that share there are no
  survivors worth printing — 0.95 of the fifth it managed to read is not a
  result.


## [2.10.0] — 2026-09-12

**A tool call with no result never reaches a provider.** The history invariant
on the externally driven seam, outstanding in `PLAN.md` since the τ²-bench
study.

### Changed

- **`step()` refuses a malformed history**, naming the unanswered call ids and
  what to append. Its docstring already handed the obligation to the caller —
  *"the caller appends the reply and whatever the world answered, then calls
  again"* — and nothing verified that they did. That seam is what a benchmark
  harness and a serverless driver use, which is exactly where a result crosses
  a process boundary and can be lost.

  It matters more than it looks because the providers disagree in the worst
  possible way. OpenAI chat completions, the Responses API and Anthropic all
  reject the request; Google keys results by name and is looser; a self-hosted
  OpenAI-compatible server — vLLM, llama.cpp, mlx — validates **nothing** and
  templates whatever it was handed, so the model meets its own unanswered call
  and improvises: repeats it, apologises, or invents the result, differently
  each time. Provider interchangeability is this library's main promise and
  this is a place it did not hold.

  And the condition that produces a dangling call — a tool that raised, a
  result dropped across an invocation boundary — is itself intermittent, so it
  became unbounded behaviour on some trials and not others: run-to-run
  variance manufactured by us. The check does not turn a wrong answer into a
  right one; it turns quiet weirdness into the same named failure every time,
  locally, before the network, pointing at the caller rather than at a
  provider's wording. An instrument for reliability, not for accuracy — and on
  a benchmark a crash can rank below a model muddling through, which is worth
  knowing before reading the next score.

  `run()` is unaffected: it appends a result turn for every call in a reply,
  failed ones included, and always did. Turning the check on cost the existing
  suite nothing, which is the evidence for that.

### Added

- **`dangling_calls(messages)`** in `models._calls` — the invariant as a
  function, so a driver can ask before it commits rather than learn from the
  raise.

### Fixed

- **The documented external-driver example could not run.** It read
  `agent.execute(decision["action"], state)`; `step()` returns a
  `ToolCallRequest` or a string, never a dict. The paragraph after it
  described reflection assigning a `store_as` key into memory — gone since
  `2.0.0` — and the modes table still listed a third mode, `goal`. Replaced
  with the whole loop, including the appending the invariant now requires.


## [2.9.3] — 2026-09-12

**A tool's schema becomes authority over its options.** They already declared
them and nothing read them: `searchPerplexity` describes seven option keys,
and `options={"recencyy": "day"}` passed validation, reached the tool and was
dropped on the floor — the search ran unfiltered and answered confidently. A
typo cost a run and left no trace. The same shape as the model options fixed
in 2.3.0, one layer down.

### Added

- **`Tool.unknown_keys()`**, and both call paths use it:

  * a **model's** call gets a remediation message from `check_args` naming the
    wrong key and the accepted ones, which an Agent feeds back inside the
    step's attempt budget;
  * a **person's** call gets a warning from `_validate`, not a refusal. The
    asymmetry is deliberate: a subclass may legitimately read an option its
    schema does not advertise, and refusing someone's call over the library's
    reading of their own schema would be the library knowing better.

  One level into a declared object and no further. The top level already fails
  loudly — an unexpected keyword reaches `run(**kwargs)` and Python raises —
  and a nested object with no `properties` of its own is a free-form dict by
  declaration, left alone. A schema is authority only over what it describes.

  It reaches most of the surface: 28 of 32 tool classes describe their
  options, and the number is now asserted so a drop is visible.

### Fixed

- **`ttsQwen` read an option it did not declare.** The other direction, and
  the worse one: `region` is honoured by the tool and absent from the schema,
  so a **model could not discover it at all**, and a caller who used it was
  passing a key the schema calls unknown. Declared on `ttsQwen` alone —
  putting it on the shared TTS schema would advertise it on OpenAI, Google
  and xAI, where nothing reads it, which is the opposite defect and the one
  2.3.0 was about. A test now checks every tool for the same disagreement;
  this was the only one.


## [2.9.2] — 2026-09-12

Two silent divergences in what the providers were actually sent, both in the
handling of the system prompt, and both invisible to every example this
repository ships.

### Fixed

- **The Responses API path kept only the last system message.** `instructions`
  was **assigned** inside the loop over messages rather than accumulated, so a
  second system message replaced the first. Present since the very first
  commit (`0211386`, 1.0.0) and carried through the 1.2.3 migration verbatim.

  Scope, measured rather than estimated: `gpt-5.4` / `5.5` / `5.6` only, and
  only when the caller wrote two or more `system` messages — one is the common
  shape and always worked, and `Agent` emits exactly one. Nothing shipped in
  this repository could reach it: of 24 examples one carries a system prompt
  (a single one, on Anthropic) and one runs a Responses model (through an
  Agent). The prompt did not fail, it arrived **partially** — which is worse
  than failing, because the answer is plausible and the model gets the blame.

  The plan had this recorded as *"Responses overwrites, the rest
  concatenate"* — as a **property of the provider**. It was ours: the
  `instructions` field is one string, so joining several system messages is
  the library's job, and every other family does it. A defect filed as a
  provider difference is one nobody fixes, because provider differences are
  supposed to differ.

- **A mid-conversation system message meant two different things.** Anthropic,
  Google and the Responses API each carry the system prompt in a field of
  their own, so one written in the middle is hoisted out of the sequence and
  applies from the start; the OpenAI-compatible family left it in place, where
  it reads as an instruction that begins at that point. The same conversation,
  two meanings, nothing said.

  Three providers cannot express the positional reading at all, so global is
  the only portable one, and it is now global everywhere — hoisted on the
  OpenAI family too, and **reported** as an adaptation, because for that
  family it is a real change to what the model sees.

### Added

- **The conversations no example contains**, compared across all nine text
  providers: a system message in the middle, system messages at both ends, an
  assistant turn first and in the middle, two user turns in a row, multi-part
  messages, empty parts. What is compared is the **meaning** — is the system
  prompt global, what is the sequence of turns — not the wire, because the
  wire legitimately differs: Google says `model` where the others say
  `assistant`, and three providers carry the system prompt in a field.

  Both defects above lived in a shape our own corpus cannot produce. The
  examples had been doubling as the test set, and the test set never contained
  the case.


## [2.9.1] — 2026-09-12

The five instruments `PLAN.md` has been asking for since the June audit, and
the defect the third one found on its first run.

### Fixed

- **`format: {"type": "json"}` on Anthropic did nothing and said nothing.**
  The request went out **byte-identical** to a plain-text one: the caller
  asked for JSON, got prose, and `parse_response` was left digging an object
  out of whatever came back. Anthropic has no JSON-mode field and every other
  provider does, so by the universal-vs-local rule it is absorbed rather than
  refused — carried as a system instruction and reported as `adapted`.
  (`json_schema` was never affected: it is a forced tool call and works.)

  Silence about a format is worse than silence about a sampling knob. A
  dropped `temperature` gives a different answer; a dropped JSON mode gives
  the wrong *type*.

### Added — tests

- **A lightness-invariant test.** `VISION.md` calls the README hello world an
  architectural guarantee — no new mandatory parameter, ever — and says a
  `Skill.run()` that got more complex is the signal that the environment
  leaked into the scenario. Nothing watched for that signal, and the README's
  *agent* example meanwhile carried a `TypeError` on its first line through
  four minor versions. The hello world now runs against a fake transport with
  exactly the arguments the page shows, and every other fenced example is
  **bound** against the real signatures — required arguments supplied, arity
  right, which is the half the existing docs test could not see.

- **Lifecycle smokes** — create → save → load → run, for every primitive that
  serialises. Each individual piece of the round trip worked when it was
  broken last week; only running it in one breath showed that what came back
  was not what went in. Where a primitive deliberately does not serialise,
  that is asserted rather than skipped: a missing test and a deliberate
  absence look identical in a run, and only one is a defect.

- **One input across every text provider.** The instrument for the class where
  a request works on six providers and quietly does something else on the
  seventh. It asserts what must be true everywhere — the system prompt
  travels, JSON is asked for, the answer has a ceiling — and separately pins
  where they *legitimately* differ, so a check that had to be loosened until
  it passed everywhere would be visible as such.

- **The chunker's contract**, which is arithmetic and therefore the one part a
  reader cannot verify by looking at output and finding it plausible. Every
  assertion is a sentence already in the module docstring.

### Changed — tests

- **The legacy import bridge is gone.** `conftest.py` aliased every
  sub-package into `sys.modules` under its pre-2.0 top-level name; written as
  temporary, it lived three months and cost more than it looked. `tests/agent/`
  is itself a package named `agent`, so inside that directory the alias lost
  to the real one and an import resolved differently depending on where the
  file sat; a module imported under two names is two module objects with two
  copies of every module-level value; and nothing ever failed, so the
  migration it existed to enable never happened. All 37 files import
  `yait_aichain.*` now — the suite reaches the library the way its users do.

  Two things the mechanical rewrite got wrong, both caught: `patch()` takes
  its target as a **string**, so those were left behind (they raise, which is
  the one mercy), and one search string inside a docs test was rewritten into
  a module path, leaving a test that passed while looking for something no
  page could contain. A blind rewrite is exactly as blind as `getattr` with a
  default.


## [2.9.0] — 2026-09-11

**Approval travels on the same channel.** R5 of
[design/streaming-to-a-ui.md](design/streaming-to-a-ui.md), and the last
of the five. A UI can now present the prompt, return a decision and render the
outcome without reaching into the permission layer at all.

### Added

- **`approval.requested` · `approval.decided`**, carrying the call id, the
  tool, the risk class, the arguments, and then the verdict with its reason.
  Emitted **even when there is nobody to ask**: otherwise the call simply
  fails and nothing says it was a governance decision rather than a broken
  tool.

- **`ApprovalDecision(granted, reason)`** — return it instead of a bare
  `False` and the refusal says why. The reason travels into the event and
  into the denial the model is told about. "Not approved" and nothing else
  throws away the only part a person can act on, and it cannot be recovered
  afterwards: it was in the head of whoever clicked no.

- `ApprovalRequest` and `ApprovalDecision` are importable. The approver has
  been handed an `ApprovalRequest` since `2.6.0` and could not import the
  type to check against it.

### Changed

- **The permission gate moved from the tool call to the loop boundary.**
  This is the requirement that had to be built rather than described. Inside
  the call, the gate's events were drained at the next boundary — after
  `tool_call.ended` — so a streaming consumer received `approval.requested`
  only once the decision had been made. The record read correctly and was
  useless for the one thing R5 exists for. Now the request reaches the
  consumer **before** the approver is called, which is also the better place
  for it: a permission is a property of the turn, not of the invocation.

  The human is asked once — a second gate left inside the call would put the
  question to a person twice.

### Notes

**R5 did not reopen the `2.0.0` decision**, which is worth recording because
the opposite was expected when the requirements were reviewed.
Suspend/resume would be needed only if the library owned the *wait*; it does
not. The approver is an ordinary callable, so a UI's answer arrives by
whatever means the application already has, and the library carries the
question and the answer without parking a run. Across processes the wait is
the product's half, exactly as the document says of transport.

### Fixed

- **The second test double only answered the buffered call**, the same defect
  fixed in the agent's double in `2.8.0` — and it mattered here, because the
  tests that needed it are about streaming: the fake would have failed the
  feature rather than the code. Both doubles stand in for a provider in both
  of its modes now.


## [2.8.0] — 2026-09-11

**The agent streams the answer, not only its actions.** R2 of
[design/streaming-to-a-ui.md](design/streaming-to-a-ui.md).
`Agent.stream()` said why it did not: *a turn in an agent loop is usually a
tool call, not prose, so token deltas would be empty for most of a run*. True
of the middle of a run and false of its end — the last turn **is** the answer,
a reader is watching it, and it arrived in one piece after a silence as long
as the model takes.

### Added

- **`text.started` · `text.delta` · `text.ended`** on the same channel as
  everything else, all carrying one `payload["id"]` so a consumer can open a
  block of prose, append to it and close it. One ordered stream rather than
  two to reassemble, and interleaved correctly with the tool calls that
  preceded it. A turn that asks for a tool and says nothing produces none of
  the three: an empty open/close pair is something a consumer would have to
  filter.

  `run()` emits none of them. It has nobody to show pieces to, and buffering
  keeps the fallback chain.

### Changed

- **A stream retries while the attempt is still discardable.** `2.5.0`
  refused to retry at all, reasoning that a rule holding only before the
  first byte "holds sometimes". That was wrong — *before the first piece* is
  not a sometimes, it is a state the caller can see, because nothing has been
  yielded — and the agent is what made it matter: its loop retries every model
  call deliberately, since one that does not pays for each transient blip with
  a lost decision. Transient failures (429, 5xx, network) now retry until the
  first piece is out and are raised after it.

  The fallback chain stays out of a stream on purpose: a second model is a
  different answer, not a retry of this one, and swapping models mid-sentence
  is not a thing to do quietly. That cost is why `run()` does not stream.

### Fixed

- **The agent test double only answered the buffered call.** It mocked
  `_post` and not `_post_sse`, so the moment the agent streamed, every tool
  event vanished and the run ended in a network error — which reads as the
  feature being broken rather than the fake being half-written. A double that
  stands in for a provider has to stand in for both of its modes.


## [2.7.0] — 2026-09-11

**The event channel carries enough to reconstruct a turn.** R1, R3 and R4 of
[design/streaming-to-a-ui.md](design/streaming-to-a-ui.md). It carried a
description: `step.started` emitted the tool's name and nothing else, and a
call's arguments and its result reached only the message list — the model's
view — and the journal, where the result had already been rendered to prose
for the model, media reduced to `[returned 2 image]`. That is right for a
journal and wrong for the only channel a program can watch while a run is
alive.

### Changed

- **`step.*` → `tool_call.*` on the agent.** The documented names were never
  the emitted ones, but the deciding argument is a different one: **`Chain`
  emits `step.*` too**, for a chain step, so a hook attached to a chain
  containing an agent received both under one name and could tell them apart
  only by the shape of the payload. A `Hook` subclass with `step_started` /
  `step_ended` still fires, once, with a `DeprecationWarning` naming the new
  method — the rename is loud, and costs nobody a run.

- **`tool_call.*` carries the call.** `payload["id"]` (a model may ask for
  several in one turn and the agent honours all of them, so the id is how a
  result pairs with its arguments), `payload["arguments"]` on started, and the
  tool's **raw** result on ended.

  By value, and that is a decision about what an `Event` is. A rendering
  cannot be recovered downstream — units, column metadata, and the difference
  between *no rows* and *the tool declined* are gone once it is prose, and a
  consumer that cannot tell a refusal from an empty answer draws an empty
  chart for both. A handle instead would need a lifetime and somewhere to
  live, which is hostile to the serverless target: the process that issued it
  may be gone. Events are therefore no longer uniformly small — about 90 KB
  for a thousand-row result — and `Event.__repr__` omits the payload so a log
  stays readable.

### Fixed

- **`run_id` was a declared field the agent never filled.** Two concurrent
  invocations — the ordinary case for the serverless target — wrote into one
  stream that could not be demultiplexed afterwards. Every event of a run now
  carries it, **including the `llm_call.*` a `Skill` emits inside the run**: a
  Skill knows nothing of the run it is inside, so the agent stamps identity on
  what its children emit rather than asking them to know.

- **`step` on a tool event**, so a consumer can say which turn a call belongs
  to.

### Notes

R2 (text deltas from the agent's last turn) and R5 (approval as events) stay
open. R5's *mechanism* shipped in `2.6.0` — `approve` gates now — and its
*conversation* is the product's half; putting the request and the response on
the event stream reopens the deliberate `2.0.0` decision that the agent has no
suspend/resume, which should be said out loud rather than arrive inside a UI
feature.


## [2.6.1] — 2026-09-11

The documentation sweep 2.6.0 ratcheted, and the code defect it uncovered on
the way: **a chain containing an agent could not be saved and reloaded at
all.**

### Fixed

- **`Chain.save()` lost the agent and `Chain.load()` then crashed.** The
  serialiser read the agent through `getattr(runner, "...", default)` under
  the pre-`2.0.0` names — `orchestrator`, `max_steps`, `max_attempts`,
  `max_tokens`, `persona`. Every one had been renamed or removed, so every
  read fell back to its default: the file lost the model, the instructions,
  the stop conditions and the labels, and gained three invented budget
  numbers. Load then raised `TypeError` on the null model name.

  The defaults are what made it quiet, and the lesson generalises: `getattr`
  with a fallback turns a renamed attribute into a plausible value, and
  plausible values do not fail tests. This shipped through four minor
  versions with a documented "Persistence inside `Chain.save()`" section
  describing it.

  A ceiling now carries a `spec` naming itself and its number, so
  `step_count` / `token_budget` / `cost_budget` survive the trip. A `check`
  cannot — its predicate is the caller's own function — and **saving one
  warns** rather than dropping it silently. A file written by the broken
  versions is refused on load with what to do, instead of being rebuilt into
  a different agent.

### Fixed — documentation

**34 documented parameters that do not exist, down to 0.** All of them the
agent, all of them 2.0 renames that never reached the pages teaching the API:

| documented | actually |
|---|---|
| `orchestrator` (13 pages, README included) | `model` |
| `persona` | `instructions` |
| `max_steps`, `max_tokens`, `max_attempts`, `done_when` | entries in `stop_when` |
| `memory`, `store`, `executors` | gone |
| `allow_spawn` | `team` |
| `mode="goal"` | `"agile"` already is that |

Three pages were rewritten rather than patched, because each taught a
subsystem that is gone: `configuration.md` (the whole pre-2.0 parameter set),
`memory.md` (the agent has no memory — the word does not occur in its source,
and `AgentResult.memory` is never written to), and the agent halves of
`state.md` and `observability.md`, which documented crash recovery and a
cross-process serverless pattern through `Agent(store=...)` and
`agent.resume()` — removed in `2.0.0` when the agent's state became the
conversation.

Two further claims were false and are now stated plainly: a `Gate` tool inside
an agent does not pause it (the `Suspend` it raises is caught like any tool
failure and reported to the model as an error), and the two documented
automatic stop rules — "no progress" and repetition — are fired by **no
library code**: `Journal.has_progress()` and `Journal.is_repeating()` exist
and are called from nowhere.

Every decision behind every rename is recorded, in full and with reasons, in
`docs/design/default-agent.md`. The design record was kept; the pages were
not touched. That is the same failure as the provider defaults losing their
authorship in 1.2.3 and as the changelog living outside git — the reasoning
survives and does not reach the person who needs it.


## [2.6.0] — 2026-09-11

**A permission policy that actually permits.** `approve` was the decision the
shipped defaults give to `external`, `financial`, `privileged` and to any risk
class nobody classified — and the agent consulted the policy for `deny` and for
nothing else, so a gated tool ran. A policy attached in order to gate spending
gated nothing while reading as protection in every docstring.

### Changed

- **`approve` now gates.** The Agent asks its new `approve=` callable and runs
  the tool only on a yes. The callable is handed an `ApprovalRequest` carrying
  the tool's name, its risk class, **the arguments it would run with**, the
  call id and the asking agent's name — approving a name rather than a call is
  approving nothing, since the arguments are what separate a $5 refund from a
  $50,000 one.

  **With no approver attached the call is refused.** A decision whose entire
  content is "a human should see this first" cannot resolve to "go ahead"
  because no human was configured. This is breaking for anyone who attached a
  policy and relied on it doing nothing; the refusal names both ways out —
  attach an approver, or set that risk class to `allow`.

  A refusal comes back through the tool channel as a result, not as a crash:
  the model is told and can choose something else. A denied call it never
  hears about is one it will simply make again. A delegated worker inherits
  the approver along with the policy.

### Fixed — documentation that taught an API that is gone

The module's own docstring promised `approve` would "pause the run for an
external approval, reusing suspend/resume", and two pages showed the flow with
`Agent(store=...)` and `agent.resume(...)`. `2.0.0` removed the agent's
suspend/resume deliberately — its state **is** the conversation — so those
examples could not have run either. Corrected, including the cross-process
serverless pattern, which was written with an `Agent` on both sides and is a
`Chain` capability.

A `Gate` tool handed to an agent does not pause it: the `Suspend` it raises is
caught like any other tool failure and reported to the model as an error. Said
out loud now rather than implied.

### Added

- **`tests/test_docs_promise_what_exists.py`** parses every fenced example and
  checks the constructor parameters it names exist. It found **34** that do
  not, all of them the agent: `orchestrator` (renamed to `model`, in 13 pages
  including the README), `max_steps` / `max_tokens` / `max_attempts` /
  `done_when` (folded into `stop_when`), `memory`, `store`, `executors`
  (dropped), `persona`, `allow_spawn` (now `team`).

  Every one of those decisions is written down in
  `docs/design/default-agent.md`, in full, with reasons. The design record was
  kept and the pages that teach the API were never touched — so the agent
  documentation describes a library that has not existed since `2.0.0`, and a
  reader copying the README's agent example gets a `TypeError` on line one.

  Held by a ratchet at 34, the way the parameter matrix holds silent cells:
  lowered as pages are fixed, never raised, with a test that fails when it is
  left slack. The sweep itself is the next release, not this one — a security
  fix should not wait behind a documentation pass.


## [2.5.2] — 2026-09-10

**Tool calls stream.** 2.5.0 shipped streaming with this named as missing and
2.5.1 made the gap loud — a tool-calling turn was not streamed and said so.
This closes it. A patch rather than a minor because no public API changed and
the release that opened the gap is the one that documented it.

### Added

- **A call is reassembled from its fragments**, on all three families. It does
  not arrive as a call: an id and a name land once, the arguments trickle in
  as pieces of a JSON string, and two calls in one turn interleave. Three
  things make the difference between working and appearing to work, and each
  is a test:

  * **Keyed on the provider's index, never on arrival order.** OpenAI numbers
    its calls and Anthropic uses the content-block index precisely because
    order identifies nothing once there is more than one. Keying on order
    splices two calls' argument strings into one that does not parse — and
    unparseable arguments recover to an empty dict, so the tool runs, with
    nothing, and the run carries on.
  * **Concatenated, then parsed once.** Parsing on the way sees truncated
    JSON on every fragment but the last. Parsing is left to the same builder
    the buffered path uses, so a malformed argument string fails identically
    whether it was streamed or not.
  * **A later fragment does not erase the name.** The name comes once and the
    fragments after it carry arguments alone; writing each fragment's empty
    name over the stored one leaves a call nothing can route.

  Anthropic streams the arguments under `input_json_delta.partial_json`, which
  is why the text filter never saw them. Google fragments nothing — its
  `functionCall` arrives whole with `args` already an object — and is handled
  by the same assembler rather than a special case.

- **Prose beside a call** is yielded as it arrives and kept on the request's
  `text`, so it is neither shown twice nor lost. The call itself is never
  yielded: the pieces are what a caller prints, and a decision is not prose.

### Notes

- **Agent turns are still not streamed**, and not for want of the capability.
  `Agent` calls the model with two retries, on the finding that a loop which
  does not retry pays for each transient blip with a lost decision — and a
  stream cannot retry once the caller has seen the first piece. The event
  stream is what `Agent.stream()` offers; text is `Skill.stream()`'s.


## [2.5.1] — 2026-09-10

Three defects in 2.5.0's streaming, found by asking the code what it does
rather than by reading what the release notes said it did. All three returned
something plausible, which is why none of them failed a test.

### Fixed

- **Every Qwen stream crashed.** `QwenClient` and `RecraftClient` override
  `build_request` *without* a `tools` parameter, and the streaming builder
  forwarded one — `TypeError`, not the `NotImplementedError` the fallback
  catches, so it reached the caller. A keyword a sibling does not take is not
  a keyword this family can pass.

- **Recraft claimed it could stream.** It renders images and has nothing to
  deliver progressively; it inherits from the client that streams, and the
  capability came with it. That is the defect the option layer was cleared of
  in 2.3.0 — a claim not backed by effect — arriving through the class
  hierarchy instead of through the data. A test now builds a streaming
  request for every provider that declares the capability, so the flag has to
  be earned rather than inherited.

- **A streamed tool call vanished.** With tools declared, a reply's call
  arrives as deltas of a JSON argument string spread over several events, and
  nothing reassembled them: the stream yielded nothing and `last_result` was
  an empty string, which reads as "the model said nothing" — the most
  expensive wrong reading available. Reassembly is a feature of its own and
  is not in this patch; until it exists, a tool-calling turn is not streamed,
  the answer comes back whole with the call intact, and a `declined` says so.
  The object is kept rather than rendered: a tool call turned into text and
  parsed back is not a tool call.

  Its repr is not streamed either — `str(result)` would have put
  `ToolCallRequest(calls=(...))` in front of whoever was printing the pieces.

- **The non-streaming fallback read Google's usage as zero**, the same
  envelope-versus-block mistake fixed in the streaming path itself, in the
  one branch that had no test.


## [2.5.0] — 2026-09-10

**Streaming.** `run()` is untouched — this is a second way to spend the same
request, not a replacement. `2.4.0` is skipped over rather than cancelled;
judging and guardrails keep their number and their content.

### Added

- **`Skill.stream()`** yields the answer as the provider produces it. The
  whole text is assembled as well as yielded — `last_result`, parsed when the
  output format asks for JSON — so a caller does not have to choose between
  showing progress and having the value. Usage is asked for explicitly and is
  `None` when the provider reported none.

- **`Agent.stream()`** yields each event as the run produces it, and
  `last_result` holds the `AgentResult` at the end. It is the **same loop**
  walked instead of exhausted, not a second implementation, and the events are
  the ones already going to hooks — so a caller's own hooks still fire, in the
  same order, and `llm_call.*` (emitted by `Skill`, not by the loop) arrives
  without anything being wired for it.

  Events, not tokens, and deliberately: a turn in an agent loop is usually a
  tool call rather than prose, so token deltas would be empty for most of a
  run and interleave with decisions in no useful order. The granularity is the
  turn, because the loop is synchronous by design — no threads, no async, the
  target is Lambda.

- **Nine providers stream** — OpenAI chat completions and everything
  compatible with it (DeepSeek, Kimi, Qwen, xAI, Perplexity, self-hosted),
  plus Anthropic and Google.

### Fixed

Four defects found by the tests rather than by the author, each returning
something plausible:

- **Google streams priced themselves at zero.** `stream_usage` returned the
  bare usage block, which two families' shapes were understood from and
  Google's was not. It returns a response-shaped envelope now, so a streamed
  report reaches the same branch of `extract_usage` a buffered one does.
- **Nothing streamed through Google at all.** The image guard tested whether
  `modalities` was present rather than what was in it, and a `Skill` sets
  `modalities: ["text"]` on every text call — so every Gemini stream declined
  with "an image is not delivered progressively", a sentence plausible enough
  to be believed.
- **Anthropic streams under-billed by the whole prompt.** Usage arrives in two
  events — input tokens in `message_start`, output tokens in `message_delta` —
  so keeping the last report drops the input side and leaves a believable
  number behind. Reports are merged, not replaced.
- **A `declined` notice was warned about and then vanished** from the
  machine-readable half, which is the half a measurement run reads: on the
  non-streaming fallback path `_built` overwrote `last_adaptations`.

### Notes

- **A provider that cannot stream still answers.** Image endpoints have
  nothing to deliver progressively and two of OpenAI's own paths speak a
  different event vocabulary; rather than raise — a library that raises on a
  provider gap is one you cannot swap a provider under — the whole answer
  arrives in one piece and a `declined` is recorded. One chunk at the end
  looks like a working stream, so `Model(on_unsupported="requirements")`
  raises for a caller who needs the real thing.
- **Streamed calls do not retry and do not fall back.** Both work by
  discarding the attempt and starting again, which is impossible once the
  caller has seen the first piece.
- **Not in scope:** `Chain` and `Pool`. A stream through a Chain means a
  stream of *steps*, which is a different question; through a Pool it means
  merging several, which is meaningless without concurrency. Half of each
  would have been worse than neither.


## [2.3.0] — 2026-09-09

**Sampling and every other option, audited.** The library's promise is one
vocabulary across providers; what it was doing was dropping whatever a
provider could not take and saying nothing. Measured rather than assumed —
by building each request twice, with and without an option, and diffing the
bodies — **69 of 116 (option, model) pairs were being changed in silence.**
For a library whose own measurement rule is "one variable between arms", that
is the mechanism that invalidates its users' comparisons.

### Added

- **The library says what it had to change.** Providers disagree, so a
  universal option cannot always travel unchanged — `top_k` has no field on
  the OpenAI wire, a reasoner refuses `temperature`, and asking DeepSeek to
  think routes to a different model. Adapting is right; adapting in silence
  is not.

  Every option now meets one of five fates — translated, adapted, declined,
  swapped, refused — and the last four are reported twice: a warning once per
  (model, option), and a record that survives on `Skill.last_adaptations`,
  `Model.last_adaptations` and `AgentResult.adaptations`, so a run can write
  down what its arms actually sent. Of the 116 pairs, none are silent now;
  the matrix has since grown to 174 and holds at zero.

  Three were not silent but wrong — `cache_control` on Anthropic and `size`
  on BFL and Qwen were reported as "this provider has no such control" while
  being honoured under another name. The conformance test now requires a
  notice whose kind matches what happened, and which names the replacement
  when there is one.

- **What each provider accepts is declared in its data**, not scattered
  through `if provider == "..."` branches: `[provider.options] accepts` and
  `format_accepts` per provider, with the two vocabularies in
  `models/_options.py`. A typo and a genuine capability gap used to read
  identically; now an unknown name is refused with the real names listed,
  and a missing control is declined with what to reach for instead —
  `top_k` on OpenAI says "top_p narrows the sampling in a way every provider
  takes".

- **A level means the same thing on every provider's own scale.** `medium`
  becomes 10000 thinking tokens on Anthropic, 8192 on Google, `"medium"` on
  OpenAI, a boolean on Qwen, and a different model on DeepSeek. The maps live
  in the provider data (`[provider.options.map.<option>]`) and are printed on
  the docs page, because a universal name without its scale is half a promise.

- **A value this model does not take fails before the wire**, not after the
  round trip is spent — an enum raises with the allowed values named, a range
  clamps and says so.

- **`Model(on_unsupported=...)`.** An option that cannot be honoured is
  reported and the request still goes; that is right for `temperature`, whose
  loss gives a different answer, and wrong for `seed`, `size`, `background`,
  `fidelity`, `output_format` and `compression`, whose loss gives a result
  that *looks* correct and silently is not. The library cannot pick — the
  cost belongs to the caller — so: `"warn"` (default, providers stay
  swappable), `"requirements"` (raise for the invisible losses only) or
  `"raise"`. The record is written before the raise.

- **`Model.effective_options`** — what this model actually runs with, asked
  for or not. For a measurement run to record beside its results.

- **`docs/reference/parameters.md`**, generated by `scripts/parameters.py`
  together with the conformance snapshot, so the page cannot say one thing
  while the code does another. Four sections: what you get without asking
  (with the author of every default), what you can ask for, what each
  provider declares it takes, and what happens per (option, model) cell.

### Fixed

- **Four providers read no image shape control at all**, and it was a gap
  rather than a limitation in every case. Google and xAI now take a ratio
  (16:9 returns 1344x768 and 1280x720, verified by decoding the pixels);
  Recraft, Qwen and OpenAI took pixels and ignored a ratio. All six accept
  either spelling now and say which they converted.

  The rest of the cells were genuinely "cannot", each established **by
  effect** rather than by status code, because three of these APIs answer
  200 to parameters they ignore and one answers 200 to invented ones:
  `seed` works on BFL and nowhere else; `top_k` exists on Anthropic and
  Google only; `output_format` is not Recraft's; Recraft's edit path derives
  its size from the input image.

- **An image key on a text model now says so** instead of blaming the
  provider: a mistake that is a mistake everywhere is refused locally, while
  one that is only wrong here is absorbed and reported.

- **`strength`, `input_fidelity` and `reference_fidelity` were one axis under
  three names** — merged into `fidelity`, direction measured rather than
  assumed. Old names keep working and report the rename.
  `output_compression` is now `compression`.

- **A multipart body no longer reports a delivered option as lost.** A form
  field carries every value as text, so `strength=0.4` went out as `"0.4"`
  and was read back as declined.

### Changed

Two defaults were the library disagreeing with a provider quietly. Both are
withdrawn; callers who want the old behaviour pass it as an option.

- **`deepseek-*` now sends `temperature=1.0`** (was `0.0`). 1.0 is DeepSeek's
  own default; 0.0 is their advice for code and maths, and adopting it here
  made one provider deterministic while no other one was — the odd arm in any
  comparison that set nothing.
- **`qwen-max` now sends `max_tokens=8192`** (was `2048`). 2048 is a quarter
  of what the model produces and was the smallest in the table by four times,
  so long answers were being cut off by us and read as the model stopping.
- **An unknown model option raises.** `Model(options={"temperatur": 0.5})` was
  accepted, ignored, and cost a run to notice. Output-format keys stay a
  warning, because that vocabulary is open — providers read names of their own.
- **Provider defaults carry their author.** Every number sent on every call is
  now marked in `[provider.default_notes]` as the vendor's or ours, and a test
  fails when one arrives anonymous. The reasons existed in the per-provider
  modules and were lost by the 1.2.3 migration, which changed no behaviour and
  dropped every justification.

## [2.2.2] — 2026-09-09

### Fixed
- **Vision works on the Responses API.** The two OpenAI wire formats name
  content differently — chat takes `text`/`image_url` with the image as an
  object, the Responses API takes `input_text`/`input_image` with the URL as a
  bare string — and the chat encoder was reused for both. Any message with
  more than one part was refused: `Invalid value: 'text'. Supported values
  are: 'input_text', 'input_image', …`. It had never worked on those models;
  the tool-returns-an-image feature of 2.2.0 only made it visible.
- **A lone text part is no longer collapsed to a bare string.** That shortcut
  is why the above went unnoticed: single-part messages are the commonest
  shape by far, so every ordinary run passed and only multi-part ones failed.

## [2.2.1] — 2026-09-09

### Fixed
- **A parameter one model refuses no longer fails the request.**
  `input_fidelity` sent to `gpt-image-2` came back HTTP 400 and the library
  added nothing to the provider's refusal. Models that reject it now carry
  `rejects` in the provider data and it is dropped before sending, with one
  warning per model and parameter — dropping a setting the caller asked for
  is not something to do quietly.

  Which models refuse it was established by asking the API, not by reading
  the guide: the guide names only `gpt-image-2`, while both GPT Image 2.5
  models and `gpt-image-1-mini` refuse it too. `gpt-image-1.5`, `gpt-image-1`
  and `chatgpt-image-latest` accept it and still receive it.

## [2.2.0] — 2026-09-09

Providers disagree, and the library absorbs the difference instead of the
caller. Everything here was reported from building on 2.1.0.

### Fixed
- **`Agent` could not start on Google.** The synthetic `pool` tool declared an
  array parameter with no element schema, which Google rejects outright —
  before the first step, and along with every user tool in the same request.
  The Google sanitizer now fills in a permissive default for any array missing
  `items`, so a tool that works elsewhere is not the reason a run cannot begin.
- **A response cut off at the token ceiling is no longer a `JSONDecodeError`.**
  It arrives as HTTP 200 carrying JSON that stops, and only the OpenAI chat
  path recognised it; Google and the Responses API did not. All three now
  raise `TruncatedResponseError`, naming the reason and the token count that
  was hit — the number `max_tokens` has to be raised above.
- **Structured replies are checked against the schema.** Google strips
  `additionalProperties`, which inverts "no fields beyond these": models
  answered with fields the schema forbade and nothing noticed until two
  providers' outputs were diffed by hand. A mismatch now raises
  `InvalidStructuredOutputError` carrying the list of violations.

### Added
- **`portable_schema(schema, target)`** — the same schema in the form
  `"openai"` (strict) or `"google"` accepts. The two families demand opposite
  things of the same document, and every caller was deriving this by hand. A
  map with arbitrary keys has no strict equivalent and is refused by name
  rather than mangled.
- **`check_structure(value, schema)`** — a dependency-free structural check:
  declared types, required properties, closed objects, enums, array elements.
  Shallow on purpose; a full validator is a dependency and this package has
  one.
- **`TruncatedResponseError` and `InvalidStructuredOutputError`**, both also
  `ValueError` so existing handlers keep working.
- **A tool can hand back an image.** Media parts are carried through instead
  of serialised, so an agent that renders something can look at what it
  rendered — a vision loop no longer has to be lifted out of the agent into a
  separate Skill. Anthropic takes the image inside the tool result; OpenAI and
  Google receive it as the user message immediately after, captioned with the
  call it came from, and the result text says what follows.

  Two layers had to stop flattening it, and the second was only found by
  running it: `result_message` json-dumped every dict and list one level
  above the code that knows how to carry media, so the first version passed
  its unit tests and shipped an agent that sent base64 as prose. Measured on
  the real path, sighted against a blind arm that receives the same words
  without the picture, three trials each: gpt-4o-mini 3/6 → 6/6,
  gemini-2.5-flash 0/6 → 5/6, blind 0/6 throughout.
- **A warning when Google strips `additionalProperties`**, once per process,
  raised where the schema is written rather than where the wrong answer
  appears.

### Added, 2026-09-09
- **GPT Image 2.5** — `gpt-image-2.5-flare` and `gpt-image-2.5-sunburst`.
  Flare is the default choice for most work (better than `gpt-image-2` at half
  the latency); Sunburst is the premium end, for edits needing tight control.
  The docs read the other way round, so the distinction sits beside the
  entries.
- **`output_compression`** reaches the wire on generations and edits. It was
  documented and silently dropped.

### Fixed, 2026-09-09
- **Image tokens sent in are priced at their own rate.** $8 per million
  against text's $5 on the 2.5 family. One input rate and one input count
  billed an edit carrying reference images as if it were prose: measured on a
  live call, 37% under on the input line and 22% under across the call.
  `image_input` is an optional price field, `image_input_tokens` a separate
  count.
- **Google's thinking tokens are counted.** `thoughtsTokenCount` sits outside
  `candidatesTokenCount` and is billed — "response pricing is the sum of
  output tokens and thinking tokens". A live vision turn reported 1 output
  token beside 235 of thought.

### Changed
- **Media sources resolve on send, not at construction.** `{"kind": "file"}`
  used to be read into base64 inside `Skill.__init__`, and `substitute()` only
  ever reached text parts — together they pinned a vision Skill to one file
  for its lifetime. A media path may now carry a `{placeholder}`, so one Skill
  serves a whole `Pool` of images.
- **`Pool` records usage per item** and sums it on `pool.usage`. Reading
  `last_usage` off the shared runner raced: whichever thread finished next
  overwrote it.
- **A `Chain` with a `Wait` step and no store warns at construction.** The
  pause is answered in another process, usually a separate command and
  sometimes the next day; the in-memory default said so only at `resume()`
  time, as "No suspended run in the store".
- **`Tool` documents what a returned dict does.** The keys are spread into the
  run's variables and the step name is not used — deliberate, since one step
  can feed several named inputs, but undocumented until now and discovered
  only through a wrong result.

## [2.1.0] — 2026-08-06

Eval — the fifth and last of the planned mechanisms. Additive; nothing
existing changes.

### Added
- **`yait_aichain.eval`** — run the same cases through several arms, several
  times each. `Eval` executes arms × cases × trials to a durable JSON-lines
  ledger and resumes from it; `Report` counts.
- **Three columns, because they disagree.** `mean` (share of attempts passed),
  `pass^k` (share of cases passed on *every* trial) and `flips` (share of
  cases answered differently on different trials). On a regression run the
  reference agent led on mean and trailed on `Pass^3`; `flips` is the shape
  neither of the other two shows.
- **`Report.of(rows, arm=…, case=…, ok=…)`** reports data produced elsewhere,
  so a benchmark that owns its own loop still gets `Pass^k`, stratified
  tables, paired tests and validity guards without being rewritten.
- **Scorers** — `exact`, `contains` (with partial credit), `regex`, `numeric`,
  `all_of`, and `judge` for an LLM verdict. A judge reply that will not parse
  raises rather than defaulting to a pass.
- **Controls** — `Report.controls(oracle, noise)`: feed the harness
  known-correct and known-wrong answers; if the oracle does not score ≈1.0 the
  metric is broken rather than the system.
- **Validity guards** — `Report.reject()` drops a cell with too many empty
  outputs or errors. A rejected cell is *absent* from every figure, not zero.
- **`Report.paired(a, b)`** — an exact sign test on discordant cases, and it
  says when there are too few to conclude anything. `se(p, n)` beside it.

### Fixed
- **harmony replies no longer lose their tool call.** gpt-oss models answer in
  OpenAI's channel format; a server that does not implement it — mlx_lm,
  llama.cpp, vLLM without a tool-call parser — passes the raw text through, so
  the call arrived as a string and a string is a final answer by our own
  contract. The `analysis` channel is dropped rather than surfaced as the
  answer, `commentary to=functions.NAME` becomes a typed call, `final` is the
  text. A body cut by `max_tokens` yields empty arguments instead of an
  exception.
- **`scripts/release.sh` works again.** It read `project.version` from
  `pyproject.toml`, which has not existed since the version became dynamic in
  1.4.1 — the script failed with a `KeyError` on its first step.

## [2.0.0] — 2026-08-03

The agent is rebuilt as a conversation that can act, on native tool
calling. Breaking across the agent surface; the model layer gains a tool
channel without changing existing call sites.

### Changed
- **`Agent` is one loop.** `Agent(model, tools, instructions, mode, team,
  stop_when)`; a text reply is the answer, requested calls execute and
  append, the conversation only grows (stable cacheable prefix).
  `orchestrator=` is now the first positional `model`.
- **Native tool calling.** Schemas ride the provider `tools` field; replies
  parse to typed `ToolCall`s. Four wire shapes golden-tested: chat
  completions, Responses API, Anthropic blocks, Google functionCall.
- **`stop_when` replaces `max_steps`/`max_attempts`/`max_tokens`/`done_when`.**
  `check()` that passes is success; a ceiling reached is failure;
  `AgentResult.stopped_by` names which fired.
- **Reasoning levels validate against the provider's `reasoning_map`**;
  openai gains `"none"`. Providers without a map refuse the option loudly.
- **`Pool` refuses a fan-out it cannot feed** — pre-flight, in the caller's
  thread, instead of returning a silent list of `None`.

### Added
- `planner_model` (waterfall): think while planning, act without
  deliberating — measured 0.85 vs 0.70/0.65 for always/on-demand reasoning
  on τ²-bench retail with everything else leveled.
- Synthetic tools `pool`, `delegate`, `write_plan` on the native channel;
  `team=None | [workers] | "auto"` gates delegation.
- Externally driven mode: `opening()` / `step()` / `execute()` — one
  decision per call, for harnesses and serverless drivers.
- `AgentResult.cost` and `stopped_by`.

### Removed
- `mode="goal"` (the loop without a plan is now `mode="agile"`, the
  default), `done_when`, `max_attempts`, `allow_spawn`, `executors`,
  agent suspend/resume, and the agent-memory preview layer — state lives
  in the conversation.

---

## Before 2.0, incompletely

This file records twelve releases; the repository carries twenty-eight tags.
Everything from `1.2.5` to `1.6.0` — seventeen releases — shipped without an
entry, and the 1.6.0 text below was recovered from the README, which was
carrying it alone.

The gap is written down rather than filled: reconstructing seventeen summaries
from commit history would produce a record of what the log says, not of what
was released, and a changelog nobody can trust is worse than one that names
its own holes. What is enforced instead is that it stops here — a release
whose version has no entry fails `tests/test_changelog.py`.

## [1.6.0] — 2026-07-27

**Goal mode** — a third agent mode, for tasks where no plan can be written in
advance because each step depends on what the last one returned.

```python
agent = Agent(orchestrator=Model("claude-sonnet-4-6"), tools=[...],
              mode="goal", done_when=lambda memory: "answer" in memory)
```

- No planning phase. Each iteration the orchestrator sees the objective, the
  observation trail, and what has been ruled out — and picks one action.
- `done_when` is **required** and is best given as a callable: a callable lets
  the run finish on a `check` the harness performed, a string only ever on the
  model's own claim. An unearned `final_answer` is recorded as `refuted` and the
  loop continues.
- Stop rules an open-ended loop needs: `done_when` met, iteration cap (50),
  token budget (250 000), and **no progress** — five consecutive failed attempts
  end the run instead of spending the rest of the budget proving it again.
- Suspend/resume, permissions, checkpoints and the journal all work unchanged.

**Journal entries now carry the observation.** A record of what was *attempted*,
without what it *returned*, cannot drive a next decision — an agent that cannot
read its own feedback re-issues the same action forever. Entries keep a bounded
excerpt of the result (the full value stays in memory), and the trail rendered
into the prompt includes failures, since a failed probe still returned
information. Applies to all modes.

**`memory_read` — a built-in tool on every agent.** Memory previews in the
prompt are truncated so one large value cannot crowd out everything else, but
until now there was no way past that limit: an agent could store a document,
see its first 500 characters on every subsequent turn, and never reach the
rest. Measured on a research task, that failure is total — the agent searched
15 times, stored 48 sources and never wrote an answer, because it could not
read what it had gathered. `memory_read(key, offset, length)` pages through a
stored value, and a truncated preview now says so and names the tool. Prompt
stays bounded; access does not.

**Three fixes found by running the agent, not by reading it.**

- Spawning from a goal-mode agent raised `ValueError`. `spawn()` forwarded
  `mode` but not `done_when`, and goal mode requires one — so `allow_spawn=True`
  crashed the moment the orchestrator delegated. A child now falls back to the
  plan-driven mode: `done_when` is a predicate over *this* agent's objective and
  memory, and neither transfers to a scoped sub-task.
- A spawned child could read its **parent's** memory. `spawn()` forwards the
  parent's tool list, which now contains an agent-bound `memory_read`; the child
  prepended its own and ended up with two tools of the same name, one pointed at
  the wrong memory. A foreign reader is dropped on construction.
- Reading memory wrote it back. The result of a `memory_read` was eligible for
  `store_as` like any other, so looking at a stored document copied it under a
  second key — measured on a research run as 9 of 13 reads going to the agent's
  own bookkeeping rather than to sources. A memory view is no longer stored.

**Repetition detector** — `has_progress()` catches a run that is failing; it
cannot catch one that is succeeding pointlessly. `Journal.is_repeating(k)`
reports a window of attempts that were all the same move, and goal mode writes
that into the next action prompt. Surfaced, never enforced: an agent circling a
hard sub-problem must not be cut off, so the model decides. Only exact
repetition counts — a similarity-based rule was measured against real runs and
did not separate a stuck agent from a healthy search.

**Fix — `Journal.has_progress()` judged on a partial window,** reporting "no
progress" after a single early failure. It now requires a full window, so one
bad attempt cannot end a run that was about to recover.

See [`examples/22_goal_mode.py`](examples/22_goal_mode.py) and
[docs/agents/overview.md](docs/agents/overview.md#goal-mode).

### 1.5.2

**The attempt journal — an append-only record of what the agent did.** Separate
from `AgentMemory` (which holds the data the agent works *with*), the journal
records every attempt with a typed outcome and the evidence behind it, and is
preserved across suspend/resume.

```python
res = agent.run("…")
for e in res.journal:
    print(e["outcome"], e["evidence"]["kind"], e["intent"])
# failed  check        call the API      ← we KNOW it failed: the tool raised
# done    model_claim  fallback path     ← the model says so; nothing verified it
```

- **Outcomes:** `done` · `failed` · `refuted` · `skipped`.
- **Evidence is typed** — `check` (a programmatic fact) vs `model_claim` (asserted,
  not verified). A run whose `done` entries are all `model_claim` has proven
  nothing, and the journal shows that instead of hiding it.
- **Do not redo** — `refuted` entries are fed back into the next action prompt as
  an `ALREADY RULED OUT` block, so a long run stops re-attempting dead ends.
- **Stuck detection** — `Journal.has_progress(k)` answers "did anything actually
  move lately", the stop rule an open-ended loop needs (a budget alone lets an
  agent spin until the tokens run out).

**Crash recovery.** The run is now checkpointed to the `Store` after **every
committed step**, not only when it suspends — so an unplanned death (crash, OOM,
function timeout) is recoverable: a fresh `Agent` sharing the store picks the run
up with `resume(run_id)`, restoring memory, cursor and journal. Suspend/resume
becomes the special case of the same mechanism. (Recovery retries the step that
was in flight — gate side-effecting tools with a `PermissionPolicy` or make them
idempotent; committed steps are never re-run.)

**Fix — a run that suspended twice became unresumable.** Since 1.4.4 the parked
`run_id` is kept aligned with the event-stream id, but `resume()` still deleted
that id unconditionally afterwards — so a second suspend parked a document and
immediately dropped it. Any approval → resume → approval flow lost the run.
`resume()` now keeps the document when the loop re-suspended under the same id.

**Fix — honest success could be bypassed.** The 1.3.4 guarantee ("a step that
ended with an execution error fails the run") was only enforced on the
run-to-completion path: an orchestrator emitting `final_answer` skipped the check
and the run reported `success=True` despite an earlier failed step. The same rule
now applies on every exit path.

### 1.5.1

**Recraft raster → vector (vectorize).** `Model("recraft-vectorize")` traces an
existing raster image into an SVG — no prompt, no generation, just a format
conversion (distinct from `imageToImage`, which is a content *variation*). Same
`Model` + `Skill` ergonomics; the model name selects the `/v1/images/vectorize`
endpoint. Verified live (returns `{"mime_type": "image/svg+xml", ...}`).

```python
Skill(model=Model("recraft-vectorize"),
      input={"messages": [{"role": "user", "parts": [
          {"type": "image", "source": {"kind": "file", "path": "logo.png"}}]}]},
      output={"modalities": ["image"], "format": {"type": "image"}}).run()
```

**11 providers / 82 models.**

- Anthropic json-mode robustness: when the model wraps its JSON in prose (leading
  or trailing commentary), the first balanced top-level object/array is now
  recovered instead of failing to parse (`_extract_first_json`).

### 1.5.0

**Directed multi-turn reasoning in one `Skill`.** Some tasks are a guided
sequence of refinements — ask, refine, refine, finish — where *you* know the
steps (unlike an `Agent`, which decides them). Express it inside a single Skill:
put several `user` turns in the message template, separated by **"generate here"
markers** — an `assistant` turn with no `parts`. Each marker is one model call;
its reply is appended to the running context, so later turns see what earlier
ones produced. No `Chain`, no manual output threading.

```python
Skill(model=Model("claude-sonnet-4-6"), input={"messages": [
    {"role": "system",    "parts": ["Be concise."]},
    {"role": "user",      "parts": ["Write 10 quotes about {topic}."]},
    {"role": "assistant"},                                   # ← generate, keep in context
    {"role": "user",      "parts": ["Drop the 5 most clichéd, write 5 fresh."]},
    {"role": "assistant"},                                   # ← generate, keep in context
    {"role": "user",      "parts": ["Translate the result into {language}."]},
]}).run(variables={"topic": "perseverance", "language": "French"})
```

- An `assistant` turn **with** `parts` is a fixed/seed turn (few-shot, a default
  answer) — no model call. **Without** `parts` it's the generate marker.
- The final turn uses the skill's `output` format (text / json); intermediate
  turns are plain text (they only feed the context).
- `skill.history` holds each generated turn (`history[-1]` is the return);
  `skill.last_usage` is the sum across turns.
- **Backward compatible:** no markers → one call, exactly as before.
- Structure is validated at construction (`ValueError`): at least one `user`
  turn, no leading `assistant`, no two `assistant` turns in a row.

### 1.4.6

**Honest billing errors + registry hygiene.**

- New **`InsufficientCreditsError`** (subclass of `APIError`). An out-of-credits
  state is a *billing* problem, not a credentials problem, but providers signal
  it inconsistently — xAI as `403 permission-denied "used all available credits"`,
  Reve as `402 PARTNER_API_BUDGET_EXHAUSTED`, OpenAI as `429 insufficient_quota`,
  Anthropic as `400 credit balance is too low`. These used to surface as
  `AuthenticationError` / `RateLimitError` / `InvalidRequestError`, sending you to
  re-check the key. Now they are detected from the response body (checked before
  the status-code mapping) and raised as `InsufficientCreditsError`, which is
  **never retried** (retrying the same account won't add funds).
- Registry hygiene (model names verified live against each API): removed the
  phantom `grok-imagine-image-pro` (xAI rejects it) and added the live-verified
  Recraft **V4.1 variants** — `recraftv4_1_utility` (controlled raster for
  product/icon/e-commerce), `recraftv4_1_utility_pro`, `recraftv4_1_pro`,
  `recraftv4_1_utility_vector`. **11 providers / 81 models.**
- `registry.refresh("perplexity")` now works: Perplexity exposes a public
  `/v1/models` router catalog, so `list_models()` reads it live (falling back to
  the curated static list only if the call fails) instead of always returning a
  hard-coded list.

### 1.4.5

**+1 image provider: Reve** (`api.reve.com`). `Model("reve-image")` generates,
edits, and remixes images behind the same `Skill`. Reve has its own (non-OpenAI)
wire shape, so it is its own client family: the operation is chosen by the input —
no image → `POST /v1/image/create`; one input image → `/v1/image/edit`
(`edit_instruction` + `reference_image`); two or more → `/v1/image/remix`
(`reference_images[]`, with `<img>0</img>` tokens in the prompt). Bearer auth
(`REVE_API_KEY`), `aspect_ratio` and `version` (default `latest`) forwarded.
**Now 11 providers / 78 models.**

```python
Skill(model=Model("reve-image"),
      input={"messages": [{"role": "user", "parts": ["A stack of golden pancakes"]}]},
      output={"modalities": ["image"], "format": {"type": "image", "aspect_ratio": "16:9"}}).run()
```

### 1.4.4

**The step boundary — observability & control.** Make every step of an Agent,
Chain, or Skill legible and governable from *outside* the model, without
touching `run()`.

```python
from yait_aichain import Tracer, PermissionPolicy
from yait_aichain.tools import Tool, FINANCIAL

class IssueRefund(Tool):
    name = "issue_refund"; risk = FINANCIAL          # ← risk class as data
    ...

tracer = Tracer()
agent = Agent(
    orchestrator = Model("gpt-4o-mini"),
    tools        = [IssueRefund()],
    hooks        = [tracer],                          # structured event stream
    permissions  = PermissionPolicy({"financial": "approve"}),
)
result = agent.run("Refund order #123")              # pauses for approval
result = agent.resume(result.run_id, signal={"approved": True})
for e in tracer.events: print(e)                     # run/step/llm_call/tool_call
```

- **Logging instead of `print()`.** The library emits through named
  `logging` loggers (`yait_aichain.*`) with a `NullHandler`; the application
  decides the sink. `verbose=` still prints to the console (now routed through
  the logger). Nothing is printed by default.
- **Lifecycle hooks + events.** `Agent`, `Chain`, and `Skill` accept
  `hooks=[...]` — callables that receive a structured `Event` at every boundary
  (`run.*`, `step.*`, `llm_call.*`, `tool_call.*`) carrying `run_id`, `step`,
  token `usage`, `cost`, `duration`, `error`. Ships `Hook`, `Tracer`, and
  `LoggingTracer` conveniences. A buggy hook can never crash the run.
- **Permission matrix.** A tool declares a `risk` class (`read`/`draft`/`write`/
  `external`/`financial`/`destructive`/`privileged`); a `PermissionPolicy` maps
  it to `allow` / `approve` / `deny` enforced *before* the tool runs, outside the
  model. `approve` pauses for an external approval (reusing suspend/resume —
  no manual `Gate`); `deny` still returns a result. Opt-in: no policy → no gating.
- **Tool-call repair.** Tool arguments are validated locally against the schema
  before execution; a malformed call returns a model-readable remediation so the
  agent corrects it within the step's attempt budget.
- **Invariant:** every tool call returns a result — even on denial, validation
  failure, or error.

### 1.4.1

Version is now a **single source of truth** in `yait_aichain/__init__.py`
(`__version__`); `pyproject.toml` reads it dynamically (`[tool.setuptools.dynamic]`)
and the publish workflow checks the same file — no more drift between the two
(which silently blocked the 1.4.0 publish until both were bumped).

### 1.4.0

**Multimodal: image-to-image (image editing).** Restyle / recompose / edit an
existing image across four providers behind the same `Skill` — an input image
part plus an image-output model *is* an edit; swap the provider by changing the
model name, nothing else.

```python
Skill(
    model  = Model("gpt-image-1.5"),     # or gemini-3.1-flash-image / grok-imagine-image / qwen-image-edit
    input  = {"messages": [{"role": "user", "parts": [
        {"type": "image", "source": {"kind": "file", "path": "product.png"}},
        "Place this product on a marble kitchen counter, soft morning light",
    ]}]},
    output = {"modalities": ["image"], "format": {"type": "image"}},
).run()
```

- **Six image providers** edit: OpenAI (`gpt-image-*`, `chatgpt-image-latest`,
  multipart `/v1/images/edits`), Google (Gemini image, conversational edit), xAI
  (`grok-imagine-*`, JSON edits), Qwen (`qwen-image-edit` series, synchronous
  multimodal-generation), plus two dedicated image houses — **Recraft**
  (`recraftv3`, multipart imageToImage) and **Black Forest Labs / FLUX**
  (`flux-kontext-*`, async submit→poll→download). New `image-to-image` task in
  the registry.
- **+2 providers, now 10 / 77 models.** Recraft (`RECRAFT_API_TOKEN`) and BFL
  (`BFL_API_KEY`) join as image specialists — `Model("flux-kontext-pro")` /
  `Model("recraftv3")` resolve and edit like any other.
- **Multiple reference images for edits.** Pass several image parts to compose /
  restyle with references — OpenAI (`image[]`, ≤16), xAI (`images`, ≤3), Qwen and
  Gemini all accept multi-image input.
- **Empty image responses raise a clear error.** When a provider returns no image
  (blocked / refused / text-instead-of-image), the call raises a descriptive
  `ValueError` instead of silently returning `base64=None` (which crashed callers
  on decode).
- **Local files as input.** A media source `{"kind": "file", "path": "..."}` is
  read and base64-encoded automatically (MIME inferred), so you can pass a path
  straight into any vision or edit call.

### 1.3.6

Added **Kimi K2.7 Code** (`kimi-k2.7-code`) — Moonshot's coding-focused model
($0.95 / $4.00 per 1M input/output). It runs with Thinking enabled by default
on the Kimi API; the client never disables it, so it works out of the box. For
non-coding tasks, `kimi-k2.6` stays the recommendation.

### 1.3.5

Model registry refresh — Google image models.

- Gemini image generation is now GA: `gemini-3.1-flash-image` and
  `gemini-3-pro-image` (the `-preview` suffixes are gone), plus the new
  `gemini-2.5-flash-image`. These are the migration target for Google's
  discontinued Imagen 4 endpoints (`imagen-4.0-*`), which this library never
  referenced.

### 1.3.4

Agent engine, token accounting, and transport fixes from the audit.

- **Token budget** is enforced *within* a step (after the action and execution
  calls, not only between steps), so one step can no longer overshoot
  `max_tokens`; agile replans are capped; tokens from an unparseable
  orchestrator reply are still counted.
- **Honest success**: a run fails if *any* executed step ended in an execution
  error (not only the last), and the final output is the last executed step's
  output (even `None`) rather than a stale earlier value.
- **Agent LLM calls go through the `send()` seam** (consistent with Skill;
  async providers work).
- **Usage**: `Skill.last_usage` resets each run (no stale value after a
  failure); `chain.last_usage` includes Agent-step tokens; `NetworkError` is
  retried within a model when `max_retries > 0`.
- **Robustness**: token extraction tolerates `usage: null` / non-numeric; the
  DeepSeek-reasoner gate matches the name exactly; Google embeddings accept
  `GOOGLE_API_KEY` or `GOOGLE_AI_API_KEY`.
- **Unified HTTP transport**: one `make_http()` factory builds the urllib3
  manager for both model and tool clients and honours `HTTPS_PROXY` /
  `HTTP_PROXY` — a proxy now applies to tool traffic (search, fetch, REST, …),
  not just LLM calls.

### 1.3.3

Durable-run (suspend/resume) correctness fixes from the audit.

- Resume no longer re-runs already-attempted steps: skipped/failed steps get a
  terminal status and the resume cursor skips past them; a terminal error during
  a resumed run clears the parked document, so a duplicate trigger is a no-op
  (no double side effects).
- A nested `Agent` that pauses (`Wait`/`Gate`) inside a `Chain` now propagates
  the suspension up — `chain.run()` returns a `SuspendedResult` instead of
  reporting a failed step, and `chain.resume()` continues the child agent run.
- `RunContext` is now real: exposed as `chain.context` / `agent.context` during
  a run, persisted in the run document, and restored on `resume`
  (`Agent.run` / `Agent.resume` accept `context=` too).
- `FileStore`: a non-JSON-serialisable variable/output at a suspend point raises
  a clear error instead of a raw `TypeError`.

### 1.3.2

Correctness & safety fixes from a code audit.

- Qwen image: a terminal task failure (FAILED / timeout / no result) now raises
  a non-retryable `TaskFailedError`, so a retry or model fallback can't silently
  submit a second billable generation.
- SSRF hardening: `RestApiTool` now blocks private / loopback / metadata
  targets (opt out with `AICHAIN_ALLOW_PRIVATE_URLS`); the Qwen TTS audio
  download is guarded and no longer follows redirects.
- VectorDB: `ChromaBackend.delete` refuses an empty `ids`+`filter` (which would
  wipe the whole collection), matching Qdrant.
- `FileStore`: `fsync` before the atomic rename (durable parked runs); a corrupt
  run file raises a clear error instead of a raw `JSONDecodeError`.
- `convertToHTML` writes through the confined output path, consistent with the
  other convert tools.

### 1.3.1

Image-generation fixes and additions.

- Qwen text-to-image now works. DashScope serves the `wan` models only through
  its native *asynchronous* task API, so a new request lifecycle (submit → poll
  → download) runs behind a `send()` seam on the client and returns the image in
  the same shape the synchronous path does. Registry updated to the available
  `wan2.2-t2i-flash` / `wan2.2-t2i-plus` (the old `wanx2.1-*` ids 404'd).
- New OpenAI image models: `chatgpt-image-latest` (the always-current model used
  by ChatGPT — fast, recommended default) and `gpt-image-2`.
- Transparent backgrounds for `gpt-image-*` / `chatgpt-image-*`: `background`
  and `output_format` are forwarded from the output format (`background:
  "transparent"`, `output_format: "png"`), so these models can emit real PNG
  alpha. The `response_format` param is correctly omitted for the
  `chatgpt-image-*` family (it returns base64 natively and rejects the param).

### 1.3.0

Durable, resumable runs — the serverless core. A Chain or Agent can **pause**
until an external signal arrives (a human, a webhook, a cron tick) and
**resume** later, even in a different process. All additive; existing programs
are unchanged.

- `Wait` / `Gate` suspend tools — pause a run until a signal arrives; on resume
  the signal drives the step. `Wait` is a leaf (its output is the signal);
  `Gate` wraps any tool behind an approval decision.
- `Chain.resume(run_id, signal)` / `Agent.resume(run_id, signal)` — continue a
  suspended run from where it paused; completed steps are not re-run.
- Self-contained run documents in a pluggable `StateStore`: `InMemoryStore`
  (default, process-local) and `FileStore` (survives restart); subclass for
  S3/DynamoDB/any KV. The store holds only suspended runs and a resume is
  idempotent — a duplicate trigger is a no-op.
- `run()` returns a falsy `SuspendedResult` (carrying `run_id` and `awaiting`)
  when paused instead of a final result. `RunContext` (tenant, metadata) can be
  passed to `run()` for per-request context.

### 1.2.6

- Security hardening: SSRF guard on outbound URL tools (private/loopback ranges
  blocked; opt out with `AICHAIN_ALLOW_PRIVATE_URLS=1`), URL-scheme allow-list,
  and output-path confinement (`AICHAIN_OUTPUT_ROOT`). Safe-class checks on
  tool instantiation during chain load.

### 1.2.5

- Correctness fixes across the model and tooling layer: response parsing,
  search/MCP timeouts, table chunking when a header exceeds the chunk size,
  and assorted edge cases surfaced by analysis.
- Refreshed the model registry to current provider catalogues and June 2026
  pricing.

### 1.2.3

Two-tier model layer: **format is code (by API family), provider is data.**
Changing provider — one word in `Model("…")` — changes only data, never the
request format or any model code. Behaviour is byte-for-byte unchanged
(guarded by characterisation tests).

- `Model` is a single, thin, data-driven class: it resolves the provider from
  the name and delegates the wire format to the matching family client. The
  provider is exposed as `model._provider`.
- Provider settings, model capabilities and prices live in data — one file per
  provider under `models/providers/*.toml`.
- The protocol layer (`clients/`) owns the wire format. Five family clients
  cover all eight providers: `OpenAIClient` (openai, xai, kimi, deepseek),
  `PerplexityClient`, `QwenClient`, `AnthropicClient`, `GoogleClient`.
- Removed the per-provider `Model` subclasses and client classes; use
  `Model("name")` (or an explicit `provider/` prefix for custom names).
- Fixed Qwen region endpoints (`us`, `hk`) and base URL.

### 1.2.1

- Removed leaked application code from the library (`skills/summarise.py`,
  `tools/section_context.py`) — an app-specific pipeline that did not belong in
  the general-purpose library.

### 1.2.0

Mechanism 1 — "LLM layer as data". All additive; the minimal program is
unchanged.

- `Usage` on every result — normalised `input_tokens` / `output_tokens` /
  `total_tokens` across providers; additive; `skill.last_usage` and
  `chain.last_usage` sum across steps.
- Cost estimation from a per-model price table (`usage.cost`; `None` when a
  model is unpriced).
- Exception hierarchy under `APIError` (`RateLimitError`,
  `AuthenticationError`, `InvalidRequestError`, `NotFoundError`,
  `ServerError`, `NetworkError`).
- `provider/model` routing — `Model("openai/gpt-4o")`; unlocks custom /
  fine-tuned names.
- Model fallback chain — `Skill(model=[primary, backup, …])` advances on a
  transient failure; a non-transient failure propagates immediately.
- `registry.refresh(provider)` — diffs the registry against the provider's
  live `list_models()`.

### 1.1.0

Foundation repaired: five features that never worked in the installed package
are revived, plus fragility closed across the library. 69 regression tests.

- Fixed `Chain.load()`, Agent inside `Chain`, the Qwen embedder/reranker, and
  Agent persistent memory — each previously crashed or was dead code.
- Honest `success=False` on token-budget exhaustion and exhausted step retries.
- Hardened LLM-response and agent-JSON parsing; timeouts on all search tools
  and the MCP bridge; `delete(ids=[])` no longer wipes a collection.
- Chunker contract fixes; thread-safe `Chain.run()`; fixed
  `from yait_aichain.tools import *`.
- POST retry policy (429/503); Anthropic auto-raises `max_tokens` above the
  thinking budget; gpt-5 / o-series parameter correctness; Chroma v2; Qdrant
  string IDs; batched VectorDB `upsert`; safe prompt templating.
- Security: the Google API key moved from the query string to the
  `x-goog-api-key` header.

### 1.0.0

Initial release — `Skill`, `Chain`, `Pool`, `Agent`, `Tool` / `MCPTools`,
`VectorDB`, `Reranker`, and 8 providers (Anthropic, OpenAI, Google, xAI,
Perplexity, Kimi, DeepSeek, Qwen).

## [1.2.3] — 2026-06-14

Two-tier model layer: **format is code (by API family), provider is data.**
Changing provider — one word in `Model("…")` — changes only data, never our
request format or any model code. Behaviour is byte-for-byte unchanged
(guarded by golden-master + family-equivalence characterisation tests).

### Changed
- **`Model` is now a single, thin, data-driven class.** It resolves the
  provider from the name, merges that provider's defaults from data, and
  delegates `to_request`/`from_response` to the family client that owns the
  wire format. The provider is exposed as `model._provider`.
- **Provider settings, model capabilities and prices live in data** —
  one file per provider under `models/providers/*.toml`.
- **The protocol layer (`clients/`) owns the wire format.** Five family
  clients cover all eight providers: `OpenAIClient` (openai, xai, kimi,
  deepseek), `PerplexityClient`, `QwenClient`, `AnthropicClient`,
  `GoogleClient`. Each holds both format (`build_request`/`parse_response`)
  and transport.
- Registry query (`registry.models/providers/tasks/is_supported/refresh`)
  and cost (`estimate_cost`/`attach_cost`) now read the provider data.

### Removed
- The 8 per-provider `Model` subclasses (`OpenAIModel`, `AnthropicModel`, …).
  Use `Model("name")`; for an unrecognised name use an explicit `provider/`
  prefix. The provider is available as `model._provider`.
- The 8 per-provider client classes — dissolved into the five family clients.
  `clients` now exports the family classes; the per-provider `*Client` names
  (`GoogleAIClient`, `XAIClient`, `KimiClient`, `DeepSeekClient`) are gone.
- `models/_registry.py` and `models/_pricing.py` — their data moved into the
  provider files; query folded into `models/_base`, cost into `models/_usage`.

### Fixed
- Qwen region endpoints (`us`, `hk`) and base URL — restored to the correct
  per-region hosts.

## [1.2.1] — 2026-06-14

Remove leaked application code from the library.

### Removed
- `skills/summarise.py` (`make_summarise_skill`) and `tools/section_context.py`
  (`SectionContextTool`) — two halves of one app-specific pipeline ("sectional
  reports / rolling-context"). Neither was used inside the package or covered
  by tests; `summarise.py` had broken imports (`from models import …`) and a
  docstring reference to a non-existent `run.py`. App-specific code does not
  belong in the general-purpose library — moved out. `SectionContextTool`
  dropped from `tools` exports.

## [1.2.0] — 2026-06-14

Mechanism 1 — "LLM layer as data": token accounting, cost, explicit routing,
a typed error hierarchy, model fallback, and live model-list refresh. All
additive — the minimal program is unchanged (lightness invariant).

### Added
- **`Usage`** on every result — normalised `input_tokens` / `output_tokens`
  / `total_tokens` across all providers (OpenAI / Anthropic / Google shapes
  flattened).  `skill.last_usage`; `chain.last_usage` sums across steps.
  `Usage` is additive (`a + b`).
- **Cost estimation** from a per-model price table (`models/_pricing.py`,
  USD per Mtok).  `usage.cost` is filled in; an unpriced model yields
  `None` (honest unknown).  Reference snapshot, trivial to update.
- **Exception hierarchy** under `APIError`: `RateLimitError` (carries
  `retry_after` from the header), `AuthenticationError`, `InvalidRequestError`,
  `NotFoundError`, `ServerError`, `NetworkError`.  `except APIError` still
  catches everything.  Exported at the top level.
- **`provider/model` routing** — `Model("openai/gpt-4o")` selects the
  provider by prefix and unlocks custom / fine-tuned names the auto-detector
  can't recognise (`Model("openai/ft:gpt-4o:org:abc")`).  Bare names work
  exactly as before; the prefix never reaches the API.
- **Model fallback chain** — `Skill(model=[primary, backup, …])` advances to
  the next model on a transient failure (rate limit / server / network); a
  non-transient failure (auth / bad request) propagates immediately.
- **`registry.refresh(provider)`** — diffs the static registry against the
  provider's live `list_models()` → `{new, removed, live, registered}`;
  surfaces roster drift without mutating the registry.

## [1.1.0] — 2026-06-14

The "foundation repaired" release: the audit surfaced five features that
**never worked** in the installed package — they are now revived, plus
fragility was closed across the whole library. 69 regression tests.

### Fixed
- `Chain.load()` — broken absolute imports raised `ModuleNotFoundError` on
  every call; the method now works.
- Agent inside `Chain` — detection via MRO (previously the agent step failed
  with `TypeError`; the entire agent branch of Chain was dead code).
- Qwen embedder and reranker — a broken `resolve_qwen_base_url` import crashed
  the very first `embed()`/`rerank()`.
- Agent persistent memory — `Agent.run()` wiped the state loaded from file
  before the first step; memory now persists across runs.
- The agent returns an honest `success=False` on token-budget exhaustion and
  when all retries of a step fail (previously a false `success=True`).
- LLM response parsing (shared across 6 OpenAI-compatible providers): clear
  errors on an empty `choices`, `refusal`, or truncated JSON instead of a bare
  `KeyError`/`JSONDecodeError`.
- Agent JSON parsing: non-dict results are rejected; one corrective retry on an
  invalid orchestrator response; crashes in plan logging eliminated.
- Timeouts on all search tools and in the MCP bridge (previously a hung server
  blocked the pipeline forever).
- `delete(ids=[])` in VectorDB no longer wipes the entire collection — an empty
  list/filter is rejected with an error.
- Chunker: the `chars <= max_chars` contract is honoured for code and tables;
  separators (`. `) are no longer lost at boundaries; overlap is actually
  applied.
- `Chain.run()` is thread-safe — concurrent runs (the Pool pattern) no longer
  interleave history; the stale `accumulated` after an error is eliminated.
- `from yait_aichain.tools import *` no longer fails — 7 non-existent names were
  removed from `__all__`.

### Changed
- The retry policy now covers POST requests (on 429/503 — safe pre-inference
  statuses); previously retry on 429/5xx fired for no generative call at all.
- Anthropic: with `reasoning=medium/high`, `max_tokens` is automatically raised
  above `budget_tokens` (previously a guaranteed HTTP 400).
- OpenAI: gpt-5 and the o-series no longer receive `temperature`/`top_p`, which
  these models reject; `reasoning` for gpt-5 now reaches the Responses API.
- Chroma: migrated from the remote REST API v1 to v2 (tenant/database, env-vars
  `CHROMA_TENANT`/`CHROMA_DATABASE`, idempotent `get_or_create`).
- Qdrant: string IDs (`"doc_1"`) are accepted — mapped to UUIDv5 while
  preserving the original; previously an HTTP 400.
- VectorDB `upsert` is batched (50 records at a time) — large uploads no longer
  fail on provider limits.
- `Skill` templating survives literal curly braces in prompts (for example,
  JSON samples); previously a `ValueError`.
- sttGoogle: sync/chunked is selected by audio duration rather than file size
  (previously a long low-bitrate file < 10 MB was guaranteed to fail).

### Security
- The Google API key is no longer passed in the query-string URL (it leaked
  into proxy/server logs) — moved to the `x-goog-api-key` header (client,
  model, embedder).

## [1.0.2] — 2026-05-17
### Added
- ...

## [1.0.0]
### Added
- `Skill` — call any LLM through a single interface
- `Chain` — sequential composition of steps
- `Pool` — parallel execution of steps
- `Agent` — autonomous multi-step reasoning
- `Tool` / `MCPTools` — tools and MCP servers
- `VectorDB` — vector search and RAG
- `Reranker` — result reranking
- 8 providers: Anthropic, OpenAI, Google, xAI, Perplexity, Kimi, DeepSeek, Qwen
