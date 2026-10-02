"""WebFetchTool validates both requested and Jina-reported final URLs.

DNS and external HTTP are controlled at their system seams. Initial private
targets must be rejected before HTTP, while unsafe final targets must be
rejected without returning Jina's fetched content to the agent.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json

import httpx
import pytest

from codeflow.agent.tools.web import WebFetchTool


def _resolve_to(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    def fake_getaddrinfo(host, *_a, **_k):
        try:
            resolved = str(ipaddress.ip_address(host))
        except ValueError:
            resolved = ip
        return [(0, 0, 0, "", (resolved, 0))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)


def _mock_http(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return async_client(*args, **kwargs)

    monkeypatch.setattr("codeflow.agent.tools.web.httpx.AsyncClient", client_factory)


async def test_rejects_url_resolving_to_private_ip(monkeypatch):
    _resolve_to(monkeypatch, "169.254.169.254")

    def _boom(*_a, **_k):
        raise AssertionError("HTTP client must not be constructed for a blocked URL")

    monkeypatch.setattr("httpx.AsyncClient", _boom)

    out = await WebFetchTool().execute(url="http://totally-public.example.com/x")
    parsed = json.loads(out)
    assert "validation failed" in parsed["error"]
    assert "private/internal" in parsed["error"]


async def test_rejects_loopback(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("HTTP client must not be constructed for a blocked URL")

    monkeypatch.setattr("httpx.AsyncClient", _boom)

    out = await WebFetchTool().execute(url="http://127.0.0.1/admin")
    parsed = json.loads(out)
    assert "validation failed" in parsed["error"]


async def test_rejects_non_http_scheme(monkeypatch):
    out = await WebFetchTool().execute(url="file:///etc/passwd")
    parsed = json.loads(out)
    assert "validation failed" in parsed["error"]


@pytest.mark.parametrize(
    "final_url",
    [
        "http://127.0.0.1/admin",
        "http://10.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data/",
    ],
)
async def test_rejects_private_jina_reported_final_url_before_returning_content(monkeypatch, final_url):
    _resolve_to(monkeypatch, "93.184.216.34")
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if request.url.host == "r.jina.ai":
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "data": {
                        "url": final_url,
                        "content": "internal secret",
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    _mock_http(monkeypatch, handler)

    out = await WebFetchTool().execute(url="https://public.example/start")

    assert getattr(out, "failed", False) is True
    assert "validation failed" in json.loads(out)["error"]
    assert "internal secret" not in out
    assert [httpx.URL(requested).host for requested in requested_urls] == ["r.jina.ai"]


async def test_accepts_public_jina_reported_final_url_and_truncates_content(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34")
    requests: list[httpx.Request] = []
    content = "x" * 150

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "code": 200,
                "data": {
                    "url": "https://final.example/article",
                    "content": content,
                },
            },
        )

    _mock_http(monkeypatch, handler)

    out = await WebFetchTool().execute(url="https://public.example/start", maxChars=100)

    assert getattr(out, "failed", False) is False
    parsed = json.loads(out)
    assert parsed["finalUrl"] == "https://final.example/article"
    assert parsed["text"] == content[:100]
    assert parsed["length"] == 100
    assert parsed["truncated"] is True
    assert requests[0].headers["accept"] == "application/json"
    assert requests[0].headers["x-base"] == "final"


@pytest.mark.parametrize(
    "reader_payload",
    [
        {"code": 200, "data": {"content": "internal secret"}},
        {"code": 200, "data": {"url": 123, "content": "internal secret"}},
    ],
)
async def test_rejects_invalid_jina_reported_final_url_without_returning_content(monkeypatch, reader_payload):
    _resolve_to(monkeypatch, "93.184.216.34")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reader_payload)

    _mock_http(monkeypatch, handler)

    out = await WebFetchTool().execute(url="https://public.example/start")

    assert getattr(out, "failed", False) is True
    assert "Invalid Jina Reader response" in json.loads(out)["error"]
    assert "internal secret" not in out


async def test_reader_timeout_returns_failed_tool_result(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("reader timed out", request=request)

    _mock_http(monkeypatch, handler)

    out = await WebFetchTool().execute(url="https://public.example/start")

    assert getattr(out, "failed", False) is True
    assert "reader timed out" in json.loads(out)["error"]


async def test_reader_cancellation_propagates(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34")

    async def handler(_request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    _mock_http(monkeypatch, handler)

    with pytest.raises(asyncio.CancelledError):
        await WebFetchTool().execute(url="https://public.example/start")
