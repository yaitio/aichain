"""
The SSE reader keeps text whole, and a stream that fails says so.

Each chunk was decoded alone, so a character split across two network reads
became two replacement marks; `splitlines()` cut an event at U+2028, legal
inside a JSON string; an error event mid-stream was taken for data and the
reply cut short there became the answer; a timeout while iterating escaped
as a raw urllib3 error, past the retry; a 200 with JSON instead of a stream
read as empty; and an abandoned stream returned its socket to the pool while
the provider went on generating, and billing.
"""

import json
import unittest
from unittest.mock import MagicMock

from urllib3.exceptions import ReadTimeoutError

from yait_aichain import NetworkError
from yait_aichain.clients._base import BaseClient
from yait_aichain.clients._errors import ServerError


class _Resp:
    status = 200

    def __init__(self, raw: bytes, pieces=None, headers=None, fail=None):
        self.raw, self.headers, self.fail = raw, headers or {"content-type": "text/event-stream"}, fail
        self.pieces = pieces or [raw[i:i + 1] for i in range(len(raw))]   # one byte at a time
        self.data = raw
        self.closed = self.released = False

    def stream(self, amt=None, decode_content=True):
        yield from self.pieces
        if self.fail:
            raise self.fail

    def release_conn(self):
        self.released = True

    def close(self):
        self.closed = True


def _client(resp):
    c = object.__new__(BaseClient)
    c._base_url, c._http = "https://x", MagicMock()
    c._http.request.return_value = resp
    return c


def _events(resp):
    return list(_client(resp)._post_sse("/s", {}, {}))


class TestTheReader(unittest.TestCase):

    def test_text_split_mid_character_and_a_line_separator_survive(self):
        text = "Γειά κόσμε"
        raw = f"data: {json.dumps({'t': text}, ensure_ascii=False)}\r\n\r\n".encode()
        self.assertEqual(_events(_Resp(raw)), [{"t": text}])

    def test_an_error_event_raises(self):
        raw = b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}\n\n'
        with self.assertRaises(ServerError) as caught:
            _events(_Resp(raw))
        self.assertEqual(caught.exception.status, 529)

    def test_an_openai_style_error_object_raises(self):
        with self.assertRaises(ServerError):
            _events(_Resp(b'data: {"error": {"message": "upstream"}}\n\n'))

    def test_a_timeout_mid_stream_is_a_network_error_and_closes(self):
        resp = _Resp(b'data: {"a": 1}\n\n', fail=ReadTimeoutError(None, "/s", "slow"))
        with self.assertRaises(NetworkError):
            _events(resp)
        self.assertTrue(resp.closed)

    def test_an_abandoned_stream_is_closed(self):
        resp = _Resp(b'data: {"a": 1}\n\ndata: {"a": 2}\n\n')
        gen = _client(resp)._post_sse("/s", {}, {})
        next(gen)
        gen.close()
        self.assertTrue(resp.closed)

    def test_json_instead_of_a_stream_is_one_event(self):
        resp = _Resp(b'{"choices": []}', headers={"content-type": "application/json"})
        self.assertEqual(_events(resp), [{"choices": []}])


if __name__ == "__main__":
    unittest.main()
