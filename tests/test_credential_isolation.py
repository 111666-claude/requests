"""Credential isolation tests across proxy switches, environment variables,
redirects and authentication retries.

These tests use only local HTTP servers and a small in-process forward proxy
(no network access). They assert that ``Authorization`` and
``Proxy-Authorization`` headers never cross a host/proxy/scheme boundary,
that same-origin security headers are retained, and that non-replayable
request bodies are never silently re-sent.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from urllib.parse import urlsplit

import pytest

import requests
from requests.exceptions import UnrewindableBodyError

from .testserver.server import Server, consume_socket_content
from .utils import override_environ

CRLF = b"\r\n"

# RFC 7230 section 6.1.2 hop-by-hop headers, plus the proxy-specific ones.
# Transfer-Encoding is omitted because this test proxy forwards body bytes
# verbatim and must keep their framing description consistent.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "upgrade",
    }
)


def _parse_raw_request(raw: bytes) -> tuple[bytes, dict[str, str], bytes]:
    """Split a raw HTTP/1.1 request into request line, headers and body."""
    head, _, body = raw.partition(CRLF * 2)
    lines = head.split(CRLF)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if b":" in line:
            name, _, value = line.partition(b":")
            headers[name.decode("latin1").strip().lower()] = value.decode(
                "latin1"
            ).strip()
    return lines[0], headers, body


def _read_full_request(sock) -> bytes:
    return consume_socket_content(sock, timeout=0.5)


class RecordingServer(Server):
    """Server whose handler keeps every raw request it receives."""

    def __init__(self, handler, requests_to_handle=6, **kwargs):
        self.recorded: list[bytes] = []
        self._user_handler = handler

        def recording_handler(sock):
            raw = _read_full_request(sock)
            self.recorded.append(raw)
            self._user_handler(sock, len(self.recorded) - 1, raw)
            return raw

        super().__init__(
            recording_handler,
            requests_to_handle=requests_to_handle,
            **kwargs,
        )

    @property
    def parsed_requests(self):
        return [_parse_raw_request(raw) for raw in self.recorded]


@contextlib.contextmanager
def recording_server(handler, requests_to_handle=6, **kwargs):
    close = threading.Event()
    server = RecordingServer(
        handler,
        requests_to_handle=requests_to_handle,
        wait_to_close_event=close,
        **kwargs,
    )
    with server as (host, port):
        try:
            yield server, host, port
        finally:
            close.set()


def redirect_response(location: str, status: int = 302) -> bytes:
    return (
        f"HTTP/1.1 {status} REDIRECT\r\n"
        f"Content-Length: 0\r\n"
        f"Location: {location}\r\n\r\n"
    ).encode("latin1")


def ok_response(text: bytes = b"ok") -> bytes:
    return b"HTTP/1.1 200 OK\r\n" + b"Content-Length: %d\r\n\r\n" % len(text) + text


class ForwardProxy(Server):
    """Minimal HTTP forward proxy (absolute-form, no CONNECT) that records
    requests and forwards them to the origin.
    """

    def __init__(self, requests_to_handle=6, **kwargs):
        self.recorded: list[bytes] = []
        super().__init__(
            self._handle,
            requests_to_handle=requests_to_handle,
            **kwargs,
        )

    def _handle(self, sock):
        raw = _read_full_request(sock)
        self.recorded.append(raw)
        request_line, headers, body = _parse_raw_request(raw)
        method, target, version = request_line.split(b" ")

        parts = urlsplit(target.decode("latin1"))
        origin_port = parts.port or 80
        assert parts.hostname is not None
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query

        upstream = socket.create_connection((parts.hostname, origin_port), timeout=5)
        # RFC 7230 section 6.1.2: a forward proxy must remove hop-by-hop
        # headers before forwarding the message.
        headers = {
            name: value
            for name, value in headers.items()
            if name not in _HOP_BY_HOP_HEADERS
        }
        headers["host"] = parts.netloc
        rebuilt = b" ".join((method, path.encode("latin1"), version)) + CRLF
        for name, value in headers.items():
            rebuilt += f"{name}: {value}".encode("latin1") + CRLF
        rebuilt += CRLF + body
        upstream.sendall(rebuilt)

        response = b""
        while True:
            chunk = upstream.recv(65536)
            if not chunk:
                break
            response += chunk
        upstream.close()
        sock.sendall(response)
        return raw

    @property
    def parsed_requests(self):
        return [_parse_raw_request(raw) for raw in self.recorded]


@contextlib.contextmanager
def forward_proxy(requests_to_handle=6):
    close = threading.Event()
    proxy = ForwardProxy(
        requests_to_handle=requests_to_handle,
        wait_to_close_event=close,
    )
    with proxy as (host, port):
        try:
            yield proxy, host, port
        finally:
            close.set()


def _header(requests_seen, index, name):
    _, headers, _ = requests_seen.parsed_requests[index]
    return headers.get(name.lower())


# ---------------------------------------------------------------------------
# 1. Environment configuration: proxy env vars and NO_PROXY boundaries
# ---------------------------------------------------------------------------


def test_http_proxy_env_var_routes_through_proxy():
    def origin_handler(sock, index, raw):
        sock.sendall(ok_response())

    with (
        recording_server(origin_handler) as (origin, oh, op),
        forward_proxy() as (
            proxy,
            ph,
            pp,
        ),
    ):
        origin_url = f"http://{oh}:{op}/resource"
        proxy_url = f"http://{ph}:{pp}"
        with override_environ(http_proxy=proxy_url):
            r = requests.get(origin_url, timeout=5)
        assert r.status_code == 200
        # The proxy saw the request in absolute form...
        assert len(proxy.recorded) == 1
        request_line, _, _ = proxy.parsed_requests[0]
        assert request_line.startswith(b"GET http://")
        # ...and the origin received exactly one forwarded request.
        assert len(origin.recorded) == 1


def test_no_proxy_bypasses_proxy_and_drops_proxy_authorization():
    def origin_handler(sock, index, raw):
        sock.sendall(ok_response())

    with (
        recording_server(origin_handler) as (origin, oh, op),
        forward_proxy() as (
            proxy,
            ph,
            pp,
        ),
    ):
        origin_url = f"http://{oh}:{op}/direct"
        proxy_url = f"http://{ph}:{pp}"
        session = requests.Session()
        session.headers["Proxy-Authorization"] = "Bearer env-secret"
        with override_environ(http_proxy=proxy_url, no_proxy=oh):
            r = session.get(origin_url, timeout=5)
        assert r.status_code == 200
        # NO_PROXY forces a direct connection: the proxy never sees anything
        assert proxy.recorded == []
        # ...and the proxy credential never reaches the origin either.
        assert _header(origin, 0, "Proxy-Authorization") is None


def test_no_proxy_boundary_inside_redirect_chain_recalculates_proxy():
    # The first host goes through the proxy. It redirects to a second host
    # name covered by NO_PROXY, so the second hop must be direct and lose
    # the proxy credentials.
    def first_handler(sock, index, raw):
        sock.sendall(redirect_response(f"http://127.0.0.1:{port2}/bypassed"))

    def second_handler(sock, index, raw):
        sock.sendall(ok_response())

    with (
        recording_server(first_handler, host="127.0.0.1") as (
            s1,
            _,
            port1,
        ),
        recording_server(second_handler, host="127.0.0.1") as (
            s2,
            _,
            port2,
        ),
        forward_proxy() as (proxy, ph, pp),
    ):
        start_url = f"http://localhost:{port1}/start"
        proxy_url = f"http://{ph}:{pp}"
        session = requests.Session()
        session.headers["Proxy-Authorization"] = "Bearer chain-secret"
        with override_environ(http_proxy=proxy_url, no_proxy="127.0.0.1"):
            r = session.get(start_url, timeout=5)
        assert r.status_code == 200
        assert len(r.history) == 1
        # Only the first request was proxied.
        assert len(proxy.recorded) == 1
        assert proxy.parsed_requests[0][0].startswith(b"GET http://")
        # The direct second request carries no Proxy-Authorization.
        assert _header(s2, 0, "Proxy-Authorization") is None
        # And it reached the second server directly, not via the proxy.
        assert len(s2.recorded) == 1


def test_explicit_proxies_argument_overrides_environment():
    def origin_handler(sock, index, raw):
        sock.sendall(ok_response())

    with (
        recording_server(origin_handler) as (origin, oh, op),
        forward_proxy() as (
            explicit_proxy,
            eph,
            epp,
        ),
        forward_proxy() as (env_proxy, nph, npp),
    ):
        origin_url = f"http://{oh}:{op}/x"
        r = requests.get(
            origin_url,
            proxies={"http": f"http://{eph}:{epp}"},
            timeout=5,
        )
        assert r.status_code == 200
        assert len(explicit_proxy.recorded) == 1
        assert env_proxy.recorded == []


# ---------------------------------------------------------------------------
# 2. Redirects: Authorization isolation and same-origin header retention
# ---------------------------------------------------------------------------


def test_authorization_stripped_on_cross_host_redirect():
    # Two IPv4 loopback servers reached under different host names
    # ("localhost" vs "127.0.0.1"): should_strip_auth sees a host change.
    def first_handler(sock, index, raw):
        sock.sendall(redirect_response(f"http://127.0.0.1:{port2}/other"))

    def second_handler(sock, index, raw):
        sock.sendall(ok_response())

    with (
        recording_server(first_handler, host="127.0.0.1") as (
            first,
            _,
            port1,
        ),
        recording_server(second_handler, host="127.0.0.1") as (
            second,
            _,
            port2,
        ),
    ):
        r = requests.get(
            f"http://localhost:{port1}/start",
            auth=("user", "pass"),
            timeout=5,
        )
        assert r.status_code == 200
        assert len(r.history) == 1
        # First request carried the credential...
        assert _header(first, 0, "Authorization") is not None
        # ...the cross-host second request did not.
        assert _header(second, 0, "Authorization") is None


def test_authorization_retained_on_same_host_redirect():
    def handler(sock, index, raw):
        if index == 0:
            sock.sendall(redirect_response(f"http://{host}:{port}/next"))
        else:
            sock.sendall(ok_response())

    with recording_server(handler) as (server, host, port):
        r = requests.get(
            f"http://{host}:{port}/start",
            auth=("user", "pass"),
            timeout=5,
        )
        assert r.status_code == 200
        assert len(r.history) == 1
        first_auth = _header(server, 0, "Authorization")
        second_auth = _header(server, 1, "Authorization")
        assert first_auth is not None
        assert first_auth == second_auth


def test_same_origin_security_headers_survive_redirect():
    def handler(sock, index, raw):
        if index == 0:
            sock.sendall(redirect_response(f"http://{host}:{port}/next"))
        else:
            sock.sendall(ok_response())

    with recording_server(handler) as (server, host, port):
        custom = {
            "X-Request-Id": "abc-123",
            "X-Api-Key": "keep-me",
        }
        r = requests.get(
            f"http://{host}:{port}/start",
            headers=custom,
            auth=("user", "pass"),
            timeout=5,
        )
        assert r.status_code == 200
        for index in (0, 1):
            assert _header(server, index, "X-Request-Id") == "abc-123"
            assert _header(server, index, "X-Api-Key") == "keep-me"
            assert _header(server, index, "Authorization") is not None


def test_cross_host_redirect_keeps_non_credential_headers_drops_authorization():
    def first_handler(sock, index, raw):
        sock.sendall(redirect_response(f"http://127.0.0.1:{port2}/land"))

    def second_handler(sock, index, raw):
        sock.sendall(ok_response())

    with (
        recording_server(first_handler, host="127.0.0.1") as (
            s1,
            _,
            port1,
        ),
        recording_server(second_handler, host="127.0.0.1") as (
            s2,
            _,
            port2,
        ),
    ):
        r = requests.get(
            f"http://localhost:{port1}/start",
            headers={
                "X-Request-Id": "trace-1",
                "Authorization": "Bearer origin",
            },
            timeout=5,
        )
        assert r.status_code == 200
        assert _header(s1, 0, "Authorization") == "Bearer origin"
        assert _header(s2, 0, "Authorization") is None
        # Non-credential headers still follow the redirect.
        assert _header(s2, 0, "X-Request-Id") == "trace-1"


# ---------------------------------------------------------------------------
# 3. Explicit proxies and proxy-switch credential isolation
# ---------------------------------------------------------------------------


def test_proxy_authorization_sent_to_proxy_not_origin():
    def origin_handler(sock, index, raw):
        sock.sendall(ok_response())

    with (
        recording_server(origin_handler) as (origin, oh, op),
        forward_proxy() as (
            proxy,
            ph,
            pp,
        ),
    ):
        url = f"http://{oh}:{op}/v"
        r = requests.get(
            url,
            headers={"Proxy-Authorization": "Bearer proxy-secret"},
            proxies={"http": f"http://{ph}:{pp}"},
            timeout=5,
        )
        assert r.status_code == 200
        # The proxy receives the hop-by-hop credential...
        assert _header(proxy, 0, "Proxy-Authorization") == "Bearer proxy-secret"
        # ...the origin must not.
        assert _header(origin, 0, "Proxy-Authorization") is None


def test_proxy_switch_does_not_forward_old_proxy_credential():
    # Origin redirects between two hostnames; each hostname is pinned to a
    # different explicit proxy, forcing a proxy switch mid-redirect.
    def origin_handler(sock, index, raw):
        if index == 0:
            sock.sendall(redirect_response(f"http://127.0.0.1:{oport}/next"))
        else:
            sock.sendall(ok_response())

    with (
        recording_server(origin_handler, host="127.0.0.1") as (
            origin,
            _,
            oport,
        ),
        forward_proxy() as (proxy_a, ah, ap),
        forward_proxy() as (
            proxy_b,
            bh,
            bp,
        ),
    ):
        session = requests.Session()
        session.headers["Proxy-Authorization"] = "Bearer only-for-A"
        proxies = {
            "http://localhost": f"http://{ah}:{ap}",
            "http://127.0.0.1": f"http://{bh}:{bp}",
        }
        r = session.get(
            f"http://localhost:{oport}/start",
            proxies=proxies,
            timeout=5,
        )
        assert r.status_code == 200
        assert len(r.history) == 1
        # Proxy A (first hop) saw the credential...
        assert len(proxy_a.recorded) == 1
        assert _header(proxy_a, 0, "Proxy-Authorization") == "Bearer only-for-A"
        # ...Proxy B (second hop) never received proxy A's credential.
        assert len(proxy_b.recorded) == 1
        assert _header(proxy_b, 0, "Proxy-Authorization") is None


def test_same_proxy_keeps_proxy_authorization_across_redirect():
    def origin_handler(sock, index, raw):
        if index == 0:
            sock.sendall(redirect_response(f"http://{host}:{port}/next"))
        else:
            sock.sendall(ok_response())

    with (
        recording_server(origin_handler) as (origin, host, port),
        forward_proxy() as (
            proxy,
            ph,
            pp,
        ),
    ):
        r = requests.get(
            f"http://{host}:{port}/start",
            headers={"Proxy-Authorization": "Bearer stable"},
            proxies={"http": f"http://{ph}:{pp}"},
            timeout=5,
        )
        assert r.status_code == 200
        assert len(r.history) == 1
        assert len(proxy.recorded) == 2
        for index in (0, 1):
            assert _header(proxy, index, "Proxy-Authorization") == "Bearer stable"


def test_direct_request_never_sends_proxy_authorization_to_origin():
    def origin_handler(sock, index, raw):
        sock.sendall(ok_response())

    with recording_server(origin_handler) as (origin, host, port):
        session = requests.Session()
        session.trust_env = False
        r = session.get(
            f"http://{host}:{port}/plain",
            headers={"Proxy-Authorization": "Bearer leaked?"},
            timeout=5,
        )
        assert r.status_code == 200
        assert _header(origin, 0, "Proxy-Authorization") is None


# ---------------------------------------------------------------------------
# 4. Authentication challenges and retries
# ---------------------------------------------------------------------------


def test_basic_auth_401_does_not_replay_consumed_body():
    # HTTPBasicAuth has no response retry hook: a 401 is returned to the
    # caller. Assert the streamed body is sent exactly once, never replayed.
    def handler(sock, index, raw):
        challenge = (
            b"HTTP/1.1 401 UNAUTHORIZED\r\n"
            b'WWW-Authenticate: Basic realm="t"\r\n'
            b"Content-Length: 0\r\n\r\n"
        )
        sock.sendall(challenge)

    def body():
        yield b"one"
        yield b"two"

    with recording_server(handler, requests_to_handle=2) as (server, host, port):
        r = requests.post(
            f"http://{host}:{port}/",
            data=body(),
            auth=("user", "pass"),
            timeout=5,
        )
        assert r.status_code == 401
        # Exactly one request reached the server; the body was not re-sent.
        assert len(server.recorded) == 1
        _, _, sent_body = _parse_raw_request(server.recorded[0])
        assert b"one" in sent_body and b"two" in sent_body


def test_digest_401_retry_rewinds_seekable_body():
    text_401 = (
        b"HTTP/1.1 401 UNAUTHORIZED\r\n"
        b"Content-Length: 0\r\n"
        b'WWW-Authenticate: Digest realm="r", nonce="n123", qop=auth\r\n\r\n'
    )
    text_200 = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

    def handler(sock, index, raw):
        if index == 0:
            sock.sendall(text_401)
        else:
            _, headers, body = _parse_raw_request(raw)
            assert headers.get("authorization", "").startswith("Digest ")
            # The seekable body must be re-sent intact.
            assert body == b"seekable-payload"
            sock.sendall(text_200)

    import io

    with recording_server(handler, requests_to_handle=3) as (server, host, port):
        r = requests.post(
            f"http://{host}:{port}/",
            data=io.BytesIO(b"seekable-payload"),
            auth=requests.auth.HTTPDigestAuth("user", "pass"),
            timeout=5,
        )
        assert r.status_code == 200
        assert len(server.recorded) == 2


def test_digest_401_retry_rejects_non_rewindable_stream():
    text_401 = (
        b"HTTP/1.1 401 UNAUTHORIZED\r\n"
        b"Content-Length: 0\r\n"
        b'WWW-Authenticate: Digest realm="r", nonce="n123", qop=auth\r\n\r\n'
    )

    def handler(sock, index, raw):
        sock.sendall(text_401)

    def body():
        yield b"abc"

    with recording_server(handler, requests_to_handle=2) as (server, host, port):
        with pytest.raises(UnrewindableBodyError):
            requests.post(
                f"http://{host}:{port}/",
                data=body(),
                auth=requests.auth.HTTPDigestAuth("user", "pass"),
                timeout=5,
            )


def test_digest_401_resets_and_does_not_leak_authorization_across_host():
    # 401 challenge, then a redirect to a second host: the Authorization
    # header built for the first host must not follow the redirect.
    def first_handler(sock, index, raw):
        if index == 0:
            sock.sendall(
                b"HTTP/1.1 401 UNAUTHORIZED\r\n"
                b"Content-Length: 0\r\n"
                b'WWW-Authenticate: Digest realm="r", nonce="n", qop=auth\r\n\r\n'
            )
        else:
            sock.sendall(redirect_response(f"http://127.0.0.1:{port2}/land"))

    def second_handler(sock, index, raw):
        sock.sendall(ok_response())

    with (
        recording_server(first_handler, host="127.0.0.1", requests_to_handle=4) as (
            s1,
            _,
            port1,
        ),
        recording_server(second_handler, host="127.0.0.1") as (
            s2,
            _,
            port2,
        ),
    ):
        r = requests.get(
            f"http://localhost:{port1}/start",
            auth=requests.auth.HTTPDigestAuth("user", "pass"),
            timeout=5,
        )
        assert r.status_code == 200
        assert _header(s2, 0, "Authorization") is None


# ---------------------------------------------------------------------------
# 5. Non-replayable request bodies on 307/308 redirects
# ---------------------------------------------------------------------------


def test_307_redirect_with_generator_body_raises_instead_of_truncating():
    def handler(sock, index, raw):
        sock.sendall(redirect_response(f"http://{host}:{port}/next", status=307))

    def body():
        yield b"chunk-a"
        yield b"chunk-b"

    with recording_server(handler, requests_to_handle=2) as (server, host, port):
        with pytest.raises(UnrewindableBodyError):
            requests.post(
                f"http://{host}:{port}/start",
                data=body(),
                timeout=5,
            )
        # The original body was delivered once; no truncated second request.
        assert len(server.recorded) == 1


def test_308_redirect_with_generator_body_raises():
    def handler(sock, index, raw):
        sock.sendall(redirect_response(f"http://{host}:{port}/next", status=308))

    with recording_server(handler, requests_to_handle=2) as (server, host, port):
        with pytest.raises(UnrewindableBodyError):
            requests.post(
                f"http://{host}:{port}/start",
                data=(b"x" for _ in range(1)),
                timeout=5,
            )
        assert len(server.recorded) == 1


def test_307_redirect_replays_bytes_body_intact():
    def handler(sock, index, raw):
        if index == 0:
            sock.sendall(redirect_response(f"http://{host}:{port}/next", status=307))
        else:
            sock.sendall(ok_response())

    with recording_server(handler, requests_to_handle=3) as (server, host, port):
        r = requests.post(
            f"http://{host}:{port}/start",
            data=b"fully-buffered-body",
            timeout=5,
        )
        assert r.status_code == 200
        assert len(server.recorded) == 2
        # Both requests are POST and carry the identical body.
        for index in (0, 1):
            request_line, _, body = _parse_raw_request(server.recorded[index])
            assert request_line.startswith(b"POST ")
            assert body == b"fully-buffered-body"


def test_307_redirect_rewinds_seekable_stream_body():
    import io

    def handler(sock, index, raw):
        if index == 0:
            sock.sendall(redirect_response(f"http://{host}:{port}/next", status=307))
        else:
            sock.sendall(ok_response())

    with recording_server(handler, requests_to_handle=3) as (server, host, port):
        r = requests.post(
            f"http://{host}:{port}/start",
            data=io.BytesIO(b"seek-me"),
            timeout=5,
        )
        assert r.status_code == 200
        assert len(server.recorded) == 2
        for index in (0, 1):
            _, _, body = _parse_raw_request(server.recorded[index])
            assert body == b"seek-me"


def test_303_redirect_drops_generator_body_and_uses_get():
    # 303 See Other changes the method to GET and discards the body, so an
    # exhausted generator is not a problem and must not raise.
    def handler(sock, index, raw):
        if index == 0:
            sock.sendall(redirect_response(f"http://{host}:{port}/next", status=303))
        else:
            sock.sendall(ok_response())

    def body():
        yield b"gone"

    with recording_server(handler, requests_to_handle=3) as (server, host, port):
        r = requests.post(
            f"http://{host}:{port}/start",
            data=body(),
            timeout=5,
        )
        assert r.status_code == 200
        assert len(server.recorded) == 2
        line1, headers1, body1 = _parse_raw_request(server.recorded[1])
        assert line1.startswith(b"GET ")
        assert body1 == b""
        assert "content-length" not in headers1


# ---------------------------------------------------------------------------
# 6. Timeouts
# ---------------------------------------------------------------------------


def test_read_timeout_raises_without_automatic_replay():
    import time
    from unittest import mock

    def slow_handler(sock, index, raw):
        # Never respond: the client's read timeout fires while waiting.
        try:
            time.sleep(1)
        except OSError:
            pass

    with recording_server(slow_handler, requests_to_handle=1) as (
        server,
        host,
        port,
    ):
        session = requests.Session()
        adapter = session.get_adapter(f"http://{host}:{port}/")
        # Each call to adapter.send is one attempt. Requests defaults to no
        # retries, so a read timeout must surface after a single send.
        with mock.patch.object(adapter, "send", wraps=adapter.send) as send_mock:
            with pytest.raises(requests.exceptions.ReadTimeout):
                session.get(
                    f"http://{host}:{port}/slow",
                    auth=("user", "pass"),
                    headers={"Proxy-Authorization": "Bearer p"},
                    timeout=(5, 0.3),
                )
            assert send_mock.call_count == 1


def test_connect_timeout_is_surfaced_as_connect_timeout(monkeypatch):
    # Point at a closed local port: connection refused is surfaced as a
    # Requests connection error, never a silent retry loop.
    import contextlib

    with contextlib.closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]

    session = requests.Session()
    session.trust_env = False
    with pytest.raises(requests.exceptions.ConnectionError):
        session.get(f"http://127.0.0.1:{closed_port}/", timeout=1)
