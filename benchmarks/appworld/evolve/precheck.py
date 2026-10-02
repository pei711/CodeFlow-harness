"""Gate0 pre-scoring environment health check for the AppWorld line (SOP §0).

Wired as ``EvalBackend.precheck``, the loop runs it at the start of every round
BEFORE any scoring. A dirty environment makes every score of the round invalid
(the TB2 debian-apt / proxy-hijack lesson), so this raises with everything it
found wrong — fix the box, then resume — instead of letting a batch run against
a broken setup and record garbage.

Checks, cheapest first:

- the appworld install ``batch.py`` will spawn env servers from (venv binary +
  data dir) is present;
- the subject config exists and names a provider endpoint;
- no orphan env servers hold the ports this run's batches will bind (the
  classic killed-run leftover that 500s every task);
- the subject endpoint sustains a REAL generation (~300 tokens) at healthy
  throughput. A tiny ping is not enough: a congested shared backend answers a
  16-token probe in seconds while real tasks blow the runner timeout (observed:
  3.5s ping alongside 68% of trials timing out at 900s), so the probe must
  measure decode throughput, not reachability.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Optional

from benchmarks.appworld.evolve.adapter import AppWorldConfig
from codeflow.config.loader import load_config
from codeflow.evolver.orchestrator.scoring import PrecheckFn


def _port_bound(port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _subject_endpoint(
    config_path: Path,
) -> tuple[Optional[str], Optional[str], dict[str, str], Optional[str]]:
    """Return the resolved OpenAI-compatible endpoint without logging secrets."""
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        return None, None, {}, f"subject config unreadable ({config_path}): {exc}"
    defaults = raw.get("agents", {}).get("defaults", {}) if isinstance(raw, dict) else {}
    if not (isinstance(defaults, dict) and defaults.get("provider") and defaults.get("model")):
        return None, None, {}, f"subject config missing provider api_base/model: {config_path}"
    try:
        runtime = load_config(config_path)
    except ValueError:
        return None, None, {}, f"subject config fails the CodeFlow runtime schema: {config_path}"
    model = runtime.agents.defaults.model
    provider = runtime.get_provider(model)
    api_base = runtime.get_api_base(model)
    if not (api_base and model):
        return None, None, {}, f"subject config missing provider api_base/model: {config_path}"
    headers = dict(provider.extra_headers or {}) if provider else {}
    api_key = provider.api_key if provider else ""
    if api_key:
        if runtime.get_provider_name(model) == "azure_openai":
            headers.setdefault("api-key", api_key)
        else:
            headers.setdefault("Authorization", f"Bearer {api_key}")
    return api_base, model, headers, None


def _endpoint_problem(
    api_base: str,
    model: str,
    headers: dict[str, str],
    timeout: float,
    min_tok_per_s: float,
) -> Optional[str]:
    import time

    import httpx  # 延迟导入，与驱动传输层一致。

    url = api_base.rstrip("/") + "/chat/completions"
    # 真实生成探针：请求约 300 token 的补全并判断解码吞吐量。推理模型可能用思考
    # token 填满预算，这没有问题，usage.completion_tokens 仍能衡量解码速度。
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Explain how TCP congestion control works, in about 300 words."}],
        "max_tokens": 300,
        "temperature": 0,
    }
    t0 = time.monotonic()
    try:
        kwargs = {"json": body, "timeout": timeout}
        if headers:
            kwargs["headers"] = headers
        resp = httpx.post(url, **kwargs)
    except httpx.TimeoutException:
        return f"subject endpoint degraded ({url}): no 300-token completion within {timeout:.0f}s"
    except httpx.HTTPError as e:
        return f"subject endpoint unreachable ({url}): {type(e).__name__}: {e}"
    elapsed = max(time.monotonic() - t0, 1e-6)
    if resp.status_code != 200:
        return f"subject endpoint unhealthy ({url}): HTTP {resp.status_code}"
    try:
        ntok = int(resp.json().get("usage", {}).get("completion_tokens") or 0)
    except ValueError:
        return f"subject endpoint unhealthy ({url}): non-JSON response"
    if ntok == 0:
        return f"subject endpoint unhealthy ({url}): empty generation (0 completion tokens)"
    tps = ntok / elapsed
    if tps < min_tok_per_s:
        return (
            f"subject endpoint degraded ({url}): {ntok} tokens in {elapsed:.1f}s "
            f"= {tps:.1f} tok/s (< {min_tok_per_s:.1f} tok/s floor)"
        )
    return None


def make_appworld_precheck(
    aw: AppWorldConfig,
    *,
    check_endpoint: bool = True,
    # SOP 健康线为 25 秒内生成 300 token，即 12 tok/s；共享目标后端观察到的健康
    # 区间为 12 到 33 tok/s。
    endpoint_timeout: float = 60.0,
    min_tok_per_s: float = 12.0,
) -> PrecheckFn:
    """Build the per-round Gate0 precheck for one AppWorld configuration."""

    def precheck() -> None:
        problems: list[str] = []
        if not aw.appworld_bin.exists():
            problems.append(f"appworld venv binary missing: {aw.appworld_bin}")
        elif not os.access(aw.appworld_bin, os.X_OK):
            problems.append(f"appworld venv binary not executable: {aw.appworld_bin}")
        if not aw.appworld_python.exists():
            problems.append(f"appworld venv python missing: {aw.appworld_python}")
        elif not os.access(aw.appworld_python, os.X_OK):
            problems.append(f"appworld venv python not executable: {aw.appworld_python}")
        else:
            try:
                probe = subprocess.run(
                    [str(aw.appworld_python), "-c", "import appworld"],
                    env={**os.environ, "APPWORLD_ROOT": str(aw.data_root)},
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                problems.append(f"appworld venv python probe failed: {type(exc).__name__}")
            else:
                if probe.returncode != 0:
                    problems.append(f"appworld venv python cannot import appworld: {aw.appworld_python}")
        data_dir = aw.data_root / "data"
        if not data_dir.exists():
            problems.append(f"appworld data dir missing: {data_dir}")

        base = aw.base_port if aw.base_port is not None else 8100
        busy = [p for p in range(base, base + aw.conc) if _port_bound(p)]
        if busy:
            problems.append(
                f"env-server ports already bound (orphan servers from a killed run?): {busy}"
                " — pkill -9 -f 'serve environment' and re-run"
            )

        if not aw.config_path.exists():
            problems.append(f"subject config missing: {aw.config_path}")
        elif check_endpoint:
            api_base, model, headers, cfg_problem = _subject_endpoint(aw.config_path)
            if cfg_problem:
                problems.append(cfg_problem)
            else:
                ep_problem = _endpoint_problem(
                    api_base,
                    model,
                    headers,
                    endpoint_timeout,
                    min_tok_per_s,
                )
                if ep_problem:
                    problems.append(ep_problem)

        if problems:
            raise RuntimeError("Gate0 precheck failed: " + " | ".join(problems))

    return precheck


__all__ = ["make_appworld_precheck"]
