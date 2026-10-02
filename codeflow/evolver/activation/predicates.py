"""集中定义 mechanism trigger predicate 的 single source of truth。

Runtime hook 与 Evolver activation-spec evaluator 都 import THESE functions，使 chamber
preflight prediction 和 live hook behavior 不会漂移；这是 round-1 incident C1/C3 后建立的
约束。每个 predicate 接收 normalized record：包含 ``content`` string、``tool_calls`` list，
可选 ``role``。hook 通过 ``normalize_response()`` 转换 live response object，corpus evaluator
直接传入同 shape 的 logged Session record。

predicate 命中只表示 trigger condition 成立，不证明 hook 副作用成功、task 完成或 candidate
有效；修改任一规则会同时改变离线可达性预测与线上触发行为。
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def normalize_response(response: Any) -> dict:
    """把 dict 或 provider response object 归一化为 predicate record。

    返回始终包含 string ``content`` 与 list-like ``tool_calls``；缺失/None 转为空值。函数不
    验证 Tool call schema，也不复制嵌套对象，只提供 trigger predicate 所需最小 shape。
    """
    if isinstance(response, dict):
        return {"content": str(response.get("content") or ""), "tool_calls": response.get("tool_calls") or []}
    return {
        "content": str(getattr(response, "content", None) or ""),
        "tool_calls": getattr(response, "tool_calls", None) or [],
    }


def is_empty_response(rec: dict) -> bool:
    """判断是否为真正 dead iteration：无 visible content AND 无 Tool call。

    纯空白 content 视为空；只要存在任一 Tool call 就不是 empty，即使没有 prose。返回值只
    描述本次 response shape，不判断 Tool 是否成功。
    """
    return not rec.get("content", "").strip() and not rec.get("tool_calls")


def visible_reasoning_len(rec: dict) -> int:
    """计算去除 ``<think>...</think>`` 并折叠 whitespace 后的 content 长度。

    长度按 Python character 计数，用于 trigger threshold，不代表 token 数或推理质量。
    """
    stripped = _THINK_RE.sub("", rec.get("content", ""))
    return len(" ".join(stripped.split()))


def is_short_toolcall_iteration(rec: dict, max_chars: int = 80) -> bool:
    """判断 Tool-call iteration 的 visible reasoning 是否短于 ``max_chars``。

    必须至少存在一个 Tool call；随后复用 ``visible_reasoning_len``。默认阈值 80 character，
    严格使用 ``<``，等于阈值不命中。
    """
    return bool(rec.get("tool_calls")) and visible_reasoning_len(rec) < max_chars


def command_head(rec: dict) -> str | None:
    """提取 actual shell command 的 head token，无法取得时回退到 assistant prose。

    exec-style Tool 把 command 放在 ``tool_calls[0].function.arguments``，arguments 可为 JSON
    string 或 dict，key 为 ``command``。这里读取 command head，而不是 prose head，使
    ``repeated_failure_run`` 与 forced-replan family detection 按真实 command family 分组，
    不会被 ``Let`` 之类 prose opener 误导；这是 C3 round-2 improvement。

    Tool call 存在但 arguments 无法解析时，当前实现仍回退到 content head；两者都为空返回
    ``None``。返回 token 只用于分类，不执行命令。
    """
    tcs = rec.get("tool_calls") or []
    if tcs:
        tc = tcs[0]
        fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
        args = None
        if isinstance(fn, dict):
            args = fn.get("arguments")
        elif fn is not None:
            args = getattr(fn, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = None
        if isinstance(args, dict):
            cmd = str(args.get("command") or "").strip()
            if cmd:
                return cmd.split()[0]
    c = rec.get("content", "").strip()
    return c.split()[0] if c else None
