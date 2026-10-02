"""Unit tests for the Gate0 subject-endpoint probe (benchmarks.appworld.evolve.precheck).

The probe is what stands between `run`/`check` and burning trials against a
dead or degraded endpoint; each failure mode must map to its own actionable
message (unreachable vs unhealthy vs degraded), because the operator acts on
that text.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

httpx = pytest.importorskip("httpx")

from benchmarks.appworld.evolve.adapter import AppWorldConfig  # noqa: E402
from benchmarks.appworld.evolve.precheck import (  # noqa: E402
    _endpoint_problem,
    _subject_endpoint,
    make_appworld_precheck,
)


class TestSubjectEndpoint:
    def test_reads_api_base_model_and_auth_headers(self, tmp_path):
        cfg = tmp_path / "subject.json"
        cfg.write_text(
            '{"providers": {"custom": {"api_base": "http://h/v1", "api_key": "secret-token",'
            '"extra_headers": {"X-Subject": "fixture"}}},'
            '"agents": {"defaults": {"provider": "custom", "model": "m1"}}}'
        )
        assert _subject_endpoint(cfg) == (
            "http://h/v1",
            "m1",
            {
                "X-Subject": "fixture",
                "Authorization": "Bearer secret-token",
            },
            None,
        )

    def test_reads_native_camel_case_provider_config(self, tmp_path):
        cfg = tmp_path / "subject.json"
        cfg.write_text(
            '{"skillForge": {"enabled": false},'
            '"providers": {"custom": {"apiBase": "http://h/v1", "apiKey": "secret-token",'
            '"extraHeaders": {"X-Subject": "fixture"}}},'
            '"agents": {"defaults": {"provider": "custom", "model": "m1"}}}'
        )

        api_base, model, headers, problem = _subject_endpoint(cfg)

        assert (api_base, model, problem) == ("http://h/v1", "m1", None)
        assert headers == {
            "X-Subject": "fixture",
            "Authorization": "Bearer secret-token",
        }

    def test_missing_model_is_a_problem(self, tmp_path):
        cfg = tmp_path / "subject.json"
        cfg.write_text(
            '{"providers": {"custom": {"api_base": "http://h/v1"}},"agents": {"defaults": {"provider": "custom"}}}'
        )
        _, _, _, problem = _subject_endpoint(cfg)
        assert "missing provider api_base/model" in problem

    def test_unreadable_config_is_a_problem(self, tmp_path):
        cfg = tmp_path / "subject.json"
        cfg.write_text("{not json")
        _, _, _, problem = _subject_endpoint(cfg)
        assert "unreadable" in problem


def _response(status: int = 200, tokens: int = 300):
    return SimpleNamespace(
        status_code=status,
        text="body",
        json=lambda: {"usage": {"completion_tokens": tokens}},
    )


def _probe(
    monkeypatch,
    *,
    post=None,
    headers: dict[str, str] | None = None,
    seconds_per_call: float = 1.0,
    min_tok_per_s: float = 12.0,
):
    clock = {"t": 0.0}

    def monotonic():
        clock["t"] += seconds_per_call
        return clock["t"]

    monkeypatch.setattr("time.monotonic", monotonic)
    monkeypatch.setattr(httpx, "post", post)
    return _endpoint_problem("http://h/v1", "m1", headers or {}, 60.0, min_tok_per_s)


class TestEndpointProblem:
    def test_healthy_endpoint_is_none(self, monkeypatch):
        assert _probe(monkeypatch, post=lambda *a, **k: _response()) is None

    def test_authenticated_endpoint_receives_bearer_header(self, monkeypatch):
        def post(*args, **kwargs):
            assert kwargs["headers"] == {
                "Authorization": "Bearer secret-token",
                "X-Subject": "fixture",
            }
            return _response()

        assert (
            _probe(
                monkeypatch,
                post=post,
                headers={
                    "Authorization": "Bearer secret-token",
                    "X-Subject": "fixture",
                },
            )
            is None
        )

    def test_endpoint_without_key_omits_authorization_header(self, monkeypatch):
        def post(*args, **kwargs):
            assert "headers" not in kwargs
            return _response()

        assert _probe(monkeypatch, post=post) is None

    def test_timeout_is_degraded(self, monkeypatch):
        def post(*a, **k):
            raise httpx.TimeoutException("slow")

        problem = _probe(monkeypatch, post=post)
        assert "degraded" in problem and "no 300-token completion" in problem

    def test_connection_error_is_unreachable(self, monkeypatch):
        def post(*a, **k):
            raise httpx.ConnectError("nodename nor servname")

        problem = _probe(monkeypatch, post=post)
        assert "unreachable" in problem and "ConnectError" in problem

    def test_http_error_status_is_unhealthy(self, monkeypatch):
        problem = _probe(monkeypatch, post=lambda *a, **k: _response(status=503))
        assert "unhealthy" in problem and "HTTP 503" in problem

    def test_empty_generation_is_unhealthy(self, monkeypatch):
        problem = _probe(monkeypatch, post=lambda *a, **k: _response(tokens=0))
        assert "empty generation" in problem

    def test_slow_decode_trips_the_throughput_floor(self, monkeypatch):

        problem = _probe(monkeypatch, post=lambda *a, **k: _response(), seconds_per_call=30.0)
        assert "degraded" in problem and "tok/s floor" in problem


class TestAppWorldRuntime:
    def _config(self, tmp_path, *, python_script: str = "#!/bin/sh\nexit 0\n"):
        data_root = tmp_path / "appworld"
        (data_root / "data").mkdir(parents=True)
        appworld_bin = data_root / "bin/appworld"
        appworld_python = data_root / "bin/python"
        appworld_bin.parent.mkdir()
        appworld_bin.write_text("#!/bin/sh\nexit 0\n")
        appworld_python.write_text(python_script)
        appworld_bin.chmod(0o755)
        appworld_python.chmod(0o755)
        config_path = tmp_path / "subject.json"
        config_path.write_text("{}")
        return AppWorldConfig(
            appworld_root=tmp_path / "subject",
            data_root=data_root,
            appworld_bin=appworld_bin,
            appworld_python=appworld_python,
            python_exe=sys.executable,
            config_path=config_path,
            out_dir_root=tmp_path / "runs",
        )

    def test_non_executable_runtime_fails_gate_zero(self, tmp_path):
        aw = self._config(tmp_path)
        aw.appworld_python.chmod(0o644)

        with pytest.raises(RuntimeError, match="python not executable"):
            make_appworld_precheck(aw, check_endpoint=False)()

    def test_broken_appworld_import_fails_gate_zero(self, tmp_path):
        aw = self._config(tmp_path, python_script="#!/bin/sh\nexit 1\n")

        with pytest.raises(RuntimeError, match="cannot import appworld"):
            make_appworld_precheck(aw, check_endpoint=False)()
