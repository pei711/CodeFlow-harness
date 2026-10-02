"""Tests for the send-error classifiers used by Channel adapters."""

import httpx
from websockets.exceptions import WebSocketException

from codeflow.channels.errors import retryable_http, transient_network


def test_transient_network_builtins():
    assert transient_network(TimeoutError()) is True
    assert transient_network(ConnectionError()) is True
    assert transient_network(ConnectionResetError()) is True


def test_transient_network_websockets():
    assert transient_network(WebSocketException()) is True


def test_transient_network_rejects_business_errors():
    assert transient_network(RuntimeError("errcode=1")) is False
    assert transient_network(ValueError("bad payload")) is False


def test_retryable_http_timeouts_and_transport():
    assert retryable_http(httpx.ConnectTimeout("t")) is True
    assert retryable_http(httpx.ConnectError("boom")) is True


def _status_error(code: int) -> httpx.HTTPStatusError:
    resp = httpx.Response(code, request=httpx.Request("GET", "http://x"))
    return httpx.HTTPStatusError("e", request=resp.request, response=resp)


def test_retryable_http_5xx_yes_4xx_no():
    assert retryable_http(_status_error(503)) is True
    assert retryable_http(_status_error(404)) is False
    assert retryable_http(RuntimeError()) is False
