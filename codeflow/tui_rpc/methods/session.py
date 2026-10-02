"""实现 TUI Session 的 lifecycle 与 management RPC：``session.*``。

Session 是一段可恢复对话的持久化边界，不等同于一次 Turn。主要 method 的职责是：

* ``session.create`` 每次生成新的 ``tui:<chat_id>`` key，保持 lazy，第一次 save 前不写文件；
* ``session.resume`` 为已知 ``session_id`` 从磁盘加载 stored transcript，unknown/absent id
  明确返回 not-found；
* ``session.close`` flush 尚未持久化的消息，并丢弃该 Session 未提交的附件；
* ``session.list`` 按 ``updated_at`` desc 返回 tui-channel sessions；
* ``session.delete`` 删除 session file 并使 cache 失效；
* ``session.most_recent`` 包装 ``find_most_recent_chat_id("tui")``；
* ``session.title`` 读写 metadata title，lazy Session 要等下一次 metadata save 才落盘；
* ``session.clear``、``session.undo``、``session.branch`` 改写或分叉 history；
* ``session.export`` 生成并校验 portable artifact。

``session.create``/``session.resume`` 的 ``info`` 是
``ui-tui/src/components/branding.tsx`` 中 ``SessionPanel`` 消费的 init bundle，必须包含
``info.skills``、``info.tools``、``info.model``；line 138 的
``Object.entries(info.skills)`` 在字段缺失时会抛错。``agent_loop=None`` 时 graceful
fallback 为 empty tools/skills、zero usage、``lazy=True``，与 ``turn.py`` 的
factory-exception guard 一致。

本模块严格区分状态：create 成功只生成 key；save/flush 成功才表示持久化；resume 成功只
表示 transcript 已读取；这些都不代表 Agent 任务完成或回复已交付。

"""

from __future__ import annotations

import copy
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from codeflow.call_efficiency.pricing import resolve_context_window
from codeflow.config.loader import load_config
from codeflow.session.export import default_export_path, verify_export, write_portable_export
from codeflow.session.manager import SessionManager, new_chat_id
from codeflow.tui_rpc.errors import SessionNotFoundError, TurnInProgressError
from codeflow.tui_rpc.methods import turn as turn_module
from codeflow.tui_rpc.methods.image import clear_pending_images
from codeflow.tui_rpc.methods.system import _codeflow_version

if TYPE_CHECKING:
    from codeflow.agent.loop.main import AgentLoop
    from codeflow.config.schema import Config
    from codeflow.tui_rpc.confirm_broker import ConfirmBroker
    from codeflow.tui_rpc.dispatcher import Dispatcher


AgentLoopFactory = Callable[[], "AgentLoop | None"]


# 模块加载时只缓存一次包版本，因为 importlib.metadata.version 每次调用都会扫描
# site-packages 的 dist-info。system._codeflow_version() 已兼容源码检出环境中的 PackageNotFoundError。
_CODEFLOW_VERSION = _codeflow_version()


def _safe_invoke_factory(
    factory: "AgentLoopFactory | None",
) -> "AgentLoop | None":
    """使用与 ``turn.py`` 相同的 try/except guard 调用 ``factory()``。

    ``factory`` 为 ``None`` 时直接返回 ``None``。Boot race、transient construction
    failure 或其他 factory-raises 路径都会记录异常并降级为 ``agent_loop=None``，从而返回
    lazy bundle，而不是让 banner 崩溃。此行为对应 ``turn.py::turn_send`` lines 103-109。

    ``None`` 只表示当前无法取得 loop，不证明 Runtime 永久不可用；调用方必须使用降级字段。
    """
    if factory is None:
        return None
    try:
        return factory()
    except Exception:
        logger.exception("session.*: agent_loop_factory raised")
        return None


def _enumerate_tools(agent_loop: "AgentLoop | None") -> dict[str, list[str]]:
    """构造 banner 的 ``info.tools``，按 handoff §3.4 使用单一 ``"builtin"`` bucket。

    ``agent_loop=None`` 时返回空 dict；否则对 ``agent_loop.tools.tool_names`` 排序，使 wire
    输出稳定。该清单只表示 Tool 已注册，不证明依赖可用或某次 Tool call 会成功。
    """
    if agent_loop is None:
        return {}
    return {"builtin": sorted(agent_loop.tools.tool_names)}


def _enumerate_skills(agent_loop: "AgentLoop | None") -> dict[str, list[str]]:
    """构造 banner 的 ``info.skills``，按 ``source`` 分组并排序。

    ``agent_loop=None`` 时返回空 dict。``LocalSkillCatalog.list_skills(filter_unavailable=True)``
    返回 legacy drop-in shape ``list[dict[str, str]]``，元素为
    ``{name, path, source}``，而不是 :class:`SkillMeta` instance。输出只包含当前 catalog
    判定 available 的 Skill；出现名称不表示 Skill 已注入某个 Turn 或执行成功。
    """
    if agent_loop is None:
        return {}
    skills = agent_loop.context.skills.list_skills(filter_unavailable=True)
    grouped: dict[str, list[str]] = {}
    for skill in skills:
        grouped.setdefault(skill["source"], []).append(skill["name"])
    return {source: sorted(names) for source, names in grouped.items()}


def _baseline_usage(
    agent_loop: "AgentLoop | None",
    config: "Config",
) -> dict[str, Any]:
    """构造 banner ``info.usage`` 的 boot baseline，此时尚无 Turn 执行。

    ``session.create`` 的所有 counter 都是 zero，因为 fresh session_key 没有 prior LLM
    call；每个 Turn 的 ``message.complete`` event 在 post-turn 更新它们。``context_max``
    优先取 provider table 中 model 的 real window，用于 LiteLLM 信息滞后场景（例如
    OpenRouter），查不到时回退到 config default。

    Resume 也复用 zero baseline，counter 会在下一次 Turn 刷新；因此这里的零不表示历史
    Session 从未调用模型，只表示当前 init bundle 不重建历史 usage。
    """
    context_max = config.agents.defaults.context_window_tokens
    model = getattr(agent_loop, "model", None)
    if model:
        live_window = resolve_context_window(model)
        if live_window:
            context_max = live_window
    return {
        "input": 0,
        "output": 0,
        "cost_usd": 0.0,
        "calls": 0,
        "context_max": context_max,
        "context_used": 0,
        "context_percent": 0,
    }


def _memory_status(agent_loop: "AgentLoop | None") -> str:
    if agent_loop is not None and hasattr(agent_loop, "backend"):
        return "enabled" if agent_loop.backend is not None else "disabled"

    try:
        from codeflow.config.loader import get_config_path, read_raw_or_raise

        raw = read_raw_or_raise(get_config_path())
        memory = raw.get("memory")
        backend = memory.get("backend", "myna") if isinstance(memory, dict) else "myna"
        return "enabled" if backend else "disabled"
    except Exception:
        return "unknown"


def _default_session_info(
    agent_loop: "AgentLoop | None",
    config: "Config",
) -> dict[str, Any]:
    """构造 ``session.create`` / ``session.resume`` 返回的 init bundle。

    bundle 包含 model、provider、memory 状态、context window、Skill、Tool、usage、version、
    cwd 与 ``mcp_servers``。``agent_loop=None`` 触发 graceful fallback：``tools={}``、
    ``skills={}``、zero usage、``lazy=True``；version 始终使用模块加载时缓存的真实包版本。
    memory 探测失败返回 ``unknown``，不会阻止 Session 打开。
    """
    model_id = config.agents.defaults.model
    return {
        "model": model_id,
        "provider": config.agents.defaults.provider,
        "memory": _memory_status(agent_loop),
        "context_window": config.agents.defaults.context_window_tokens,
        "lazy": agent_loop is None,
        "skills": _enumerate_skills(agent_loop),
        "tools": _enumerate_tools(agent_loop),
        "usage": _baseline_usage(agent_loop, config),
        "version": _CODEFLOW_VERSION,
        "cwd": os.getcwd(),
        "mcp_servers": [],
    }


def _get_or_build_manager(config: "Config") -> SessionManager:
    """为 active foreground state 创建 ``SessionManager``。

    state path 由 ``resolve_foreground_paths(config)`` 决定。函数保留在 module level，使
    test 可以 monkeypatch 注入 pre-populated manager 而不访问 filesystem；这是与
    ``load_config`` 相同的 seam。每次 fallback 调用都可能创建新 manager，但它们仍指向
    同一 foreground state。
    """
    from codeflow.config.paths import resolve_foreground_paths

    return SessionManager(resolve_foreground_paths(config).state)


def _manager_for(agent_loop: "AgentLoop | None", config: "Config") -> SessionManager:
    """优先复用 AgentLoop 的 shared manager，否则创建 foreground manager。

    只有 ``agent_loop.sessions`` 确实是 ``SessionManager`` 时才复用；其他对象或缺失属性
    都回退到 ``_get_or_build_manager(config)``。复用可保持 live cache 与 Turn writer 一致。
    """
    if agent_loop is not None:
        mgr = getattr(agent_loop, "sessions", None)
        if isinstance(mgr, SessionManager):
            return mgr
    return _get_or_build_manager(config)


def _map_to_wire(messages: list[dict[str, Any]], session_key: str) -> list[dict[str, Any]]:
    """把 stored Session messages 转成 ``GatewayTranscriptMessage`` wire shape。

    TS 端 ``gatewayTypes.ts:23`` 需要 ``{role, text?, context?, name?}``；磁盘 message 使用
    ``content`` 而不是 ``text``，所以这里重命名字段。所有 well-formed message 都保留，
    即 N stored -> N wire，不应用 consolidation filter。non-dict 或缺少 role 的 entry 会记录
    warning 并跳过，避免单条 corrupt line 让整个 Session 无法 resume。

    Multimodal user message 的 content 是 text/image block LIST；函数连接所有
    ``type == "text"`` block 的 ``text``，丢弃 non-text block。``role="tool"`` entry 保留
    name/context；已知 degradation 是 TS renderer 会把它折叠为附在下一条 assistant message
    上的 generic tool trail line。转换成功只证明 wire 可渲染，不表示原始媒体完整恢复。
    """
    out = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            logger.warning("session.resume: skipping malformed stored message in {}", session_key)
            continue
        entry: dict[str, Any] = {"role": m["role"]}
        content = m.get("content", "")
        if isinstance(content, list):
            entry["text"] = " ".join(
                blk.get("text", "") for blk in content if isinstance(blk, dict) and blk.get("type") == "text"
            )
        elif isinstance(content, str):
            entry["text"] = content
        elif content is not None:
            entry["text"] = str(content)
        for extra_key in ("context", "name"):
            if extra_key in m:
                entry[extra_key] = m[extra_key]
        out.append(entry)
    return out


async def session_create(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """执行 ``session.create``：安全取得 AgentLoop 并构造 init bundle。

    Zero-factory 调用 ``session_create({})`` 是 test/demo 路径，会降级为
    ``agent_loop=None`` fallback bundle；production 通过
    :func:`register_session_methods` 注入 ``agent_loop_factory``。每次调用都生成 fresh
    ``tui:<chat_id>`` key，并保持 lazy，第一次 Session save 前不写文件。

    可选 ``title`` param 在这里被接受但忽略，client 应调用 ``session.title``。返回包含
    ``session_id`` 与 ``info``；成功只表示身份已生成，不表示 Session 已持久化。
    """
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    session_id = f"tui:{new_chat_id()}"
    return {
        "session_id": session_id,
        "info": _default_session_info(agent_loop, load_config()),
    }


async def session_close(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """执行 ``session.close``：flush messages，并丢弃未发送附件。

    per-turn save 下 Session 通常已经 fully persisted；这里补足 last save 后又新增 message
    的 edge case。切换离开 Session 会放弃从未提交的 pending image，因此 flush 成功后一并
    清理。absent ``session_id`` silently ignored 并返回 ``ok=True``；unknown key 或 flush
    failure 返回 ``ok=False``。

    Active Turn 会在 flush 和附件清理前抛出 ``TurnInProgressError``，防止与 writer 竞争。
    ``ok=True`` 表示 flush/cleanup 完成，不表示 Agent 任务或外部交付完成。
    """
    session_key = params.get("session_id")
    if not session_key:
        return {"ok": True}
    if turn_module.is_turn_active(session_key):
        raise TurnInProgressError(
            f"session {session_key!r} has an active turn; interrupt it before closing",
            data={"session_key": session_key},
        )
    config = load_config()
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    mgr = _manager_for(agent_loop, config)
    try:
        flushed = mgr.flush(session_key)
    except Exception:
        logger.warning("session.close: failed to flush {}", session_key)
        return {"ok": False}
    if not flushed:
        return {"ok": False}
    clear_pending_images(session_key)
    return {"ok": True}


async def session_resume(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """执行 ``session.resume``：加载 stored messages 并返回恢复后的 Session key。

    ``session_id`` 缺失或 unknown 时抛出 ``SessionNotFoundError``。读取使用
    ``manager.peek()``：先查 cache，再查 disk，且不会缓存 unknown key；storage failure
    保留自身 diagnostics。AgentLoop factory 失败不会阻止 transcript 恢复，只会让 info
    使用 lazy fallback。

    wire 使用 raw ``session.messages``，所以 N stored -> N wire；不用会切片并丢弃开头
    non-user message 的 ``get_history()``。成功表示 Session snapshot 已读取，不代表新的
    Turn 已开始。
    """
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    info = _default_session_info(agent_loop, config)
    session_key = params.get("session_id")

    if not session_key:
        raise SessionNotFoundError("session_id is required")
    mgr = _manager_for(agent_loop, config)
    raw = mgr.peek(session_key)
    if raw is None:
        raise SessionNotFoundError(
            f"session {session_key!r} does not exist",
            data={"session_key": session_key},
        )
    return {
        "session_id": session_key,
        "info": info,
        "messages": _map_to_wire(raw.messages, session_key),
    }


def _session_to_list_item(info: dict[str, Any]) -> dict[str, Any]:
    """把 ``list_sessions`` entry 转成 ``SessionListItem`` wire shape。

    TS ``SessionListItem``（``gatewayTypes.ts:130``）要求 ``id``、``message_count``、
    ``preview``、Unix timestamp ``started_at`` 与 ``title``。``started_at`` 从
    ``created_at`` ISO string 转换；格式无效时保持 ``0.0``。v0.1 的 ``preview`` 永远为空，
    因为 metadata 不含 message content，TS picker 会回退到 title 或 ``"(untitled)"``。
    """
    key = info.get("key", "")
    created_at_str = info.get("created_at") or ""
    started_at: float = 0.0
    if created_at_str:
        try:
            started_at = datetime.fromisoformat(created_at_str).timestamp()
        except ValueError:
            pass
    meta = info.get("metadata") or {}
    title = meta.get("title") or ""
    return {
        "id": key,
        "message_count": info.get("message_count", 0),
        "preview": "",
        "source": "tui",
        "started_at": started_at,
        "title": title,
    }


async def session_list(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """执行 ``session.list``：按 ``updated_at`` desc 列出 tui-channel Sessions。

    RPC 只面向 TUI surface，因此固定过滤 ``channel="tui"``。可选 ``limit`` 只有在它是
    positive integer 且不是 bool 时才生效，并在 manager 排序后截断，所以保留 newest
    sessions；zero、negative 或 non-integer limit 被忽略。

    返回 ``SessionListResponse`` shape ``{sessions: SessionListItem[]}``。该列表来自 metadata
    snapshot，不读取完整 transcript，也不证明 Session 仍可成功 resume。
    """
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    entries = mgr.list_sessions(channel="tui")
    limit = params.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        entries = entries[:limit]
    return {"sessions": [_session_to_list_item(e) for e in entries]}


async def session_delete(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
    confirm_broker: "ConfirmBroker | None" = None,
) -> dict:
    """执行 ``session.delete``：使指定 logical Session generation 失效。

    缺失、unknown 或 generation 已变化时返回 ``{deleted: null}``。匹配的 current
    generation 删除成功时返回 ``{deleted: session_id}``，包括尚未写文件的 lazy cached
    Session。Active Turn 在任何删除前抛出 ``TurnInProgressError``。

    Production TUI 在提供 ``confirm_broker`` 时必须先完成 Confirm Round-Trip，default 为
    ``False``；用户拒绝会额外返回 ``cancelled=True``，避免 frontend 报告 Session missing。
    确认后再次检查 active Turn，并以 ``storage_epoch`` 与 file existence 做 compare-and-delete，
    防止等待期间删除了新 generation。删除成功还会清理 pending image；操作不可撤销。
    """
    session_key = params.get("session_id", "")
    if not session_key:
        return {"deleted": None}
    if turn_module.is_turn_active(session_key):
        raise TurnInProgressError(
            f"session {session_key!r} has an active turn; interrupt it before deleting",
            data={"session_key": session_key},
        )

    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    session = mgr.peek(session_key)
    if session is None:
        return {"deleted": None}
    file_existed = mgr.exists(session_key)
    storage_epoch = session._storage_epoch

    if confirm_broker is not None:
        confirmed = await confirm_broker.await_confirm(
            f"Delete session {session_key}? This cannot be undone.",
            default=False,
        )
        if not confirmed:
            return {"deleted": None, "cancelled": True}
        if turn_module.is_turn_active(session_key):
            raise TurnInProgressError(
                f"session {session_key!r} has an active turn; interrupt it before deleting",
                data={"session_key": session_key},
            )

    deleted = mgr.delete(
        session_key,
        allow_cached_missing=True,
        expected_epoch=storage_epoch,
        expected_exists=file_existed,
    )
    if deleted:
        clear_pending_images(session_key)
    return {"deleted": session_key if deleted else None}


async def session_most_recent(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """执行 ``session.most_recent``：返回最近更新的 tui Session key。

    方法包装 ``find_most_recent_chat_id("tui")``，找到 chat id 时加上 ``tui:`` 前缀；
    无 Session 时返回 ``{"session_id": None}``。wire 符合
    ``SessionMostRecentResponse`` shape ``{session_id?: string | null, ...}``；TS caller
    ``createGatewayEventHandler.ts:242`` 读取 ``r?.session_id``，允许 null。

    返回 key 只来自索引排序，不会同时加载或验证完整 Session file。
    """
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    chat_id = mgr.find_most_recent_chat_id("tui")
    session_id = f"tui:{chat_id}" if chat_id else None
    return {"session_id": session_id}


async def session_title(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """执行 ``session.title``：读取或设置 Session metadata title。

    Set path 在 ``title`` param present 时调用 ``get_or_create`` 并更新 metadata。若 Session
    file 已存在，立即做 metadata-only save，成功时 ``pending=False``；保存异常返回原 title
    且 ``pending=True``。never-saved lazy Session 的 title 只留在 memory，``pending=True``，
    等第一次 save 一并落盘，以保留 lazy mint。

    Get path 从 cached 或 disk-loaded Session 返回 current title；unknown key 得到 ``None``。
    wire 遵循 ``SessionTitleResponse``（``gatewayTypes.ts:154``）：
    ``{title?: string, session_key: string, pending: bool}``。``pending=False`` 才能证明 title
    已无需后续落盘，但仍不表示任何 Turn 完成。
    """
    session_key = params.get("session_id", "")
    if not session_key:
        return {"title": None, "session_key": "", "pending": False}
    title = params.get("title")
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)

    if title is not None:
        session = mgr.get_or_create(session_key)
        session.metadata["title"] = title
        if mgr.exists(session_key):
            try:
                mgr.save(session)
            except Exception:
                logger.warning("session.title: failed to persist title for {}", session_key)
                return {"title": title, "session_key": session_key, "pending": True}
            return {"title": title, "session_key": session_key, "pending": False}
        return {"title": title, "session_key": session_key, "pending": True}

    raw = mgr.peek(session_key)
    current_title = None
    if raw is not None:
        current_title = (raw.metadata or {}).get("title")
    return {"title": current_title, "session_key": session_key, "pending": False}


async def session_clear(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """执行 ``session.clear``：原地清空 messages，但保留 Session id。

    与生成 new id 的 ``session.create`` 不同，clear 保留 ``session_key``，使现有 script 和
    bookmark 继续有效。Active Turn 期间抛出 ``TurnInProgressError``，因为运行中的 writer
    与 history rewrite 会竞争。

    方法在内存中备份 messages、consolidation cursor、timestamp、metadata 与 pending
    clarification，调用 ``session.clear()`` 后通过 ``commit_history_rewrite`` 持久化；失败时
    恢复全部备份并返回 ``cleared=False``。``cleared=True`` 表示 rewrite 已提交，不代表外部
    备份或 export 被删除。
    """
    session_key = params.get("session_id", "")
    if not session_key:
        return {"session_id": "", "cleared": False}
    if turn_module.is_turn_active(session_key):
        raise TurnInProgressError(
            f"session {session_key!r} has an active turn; interrupt it before clearing",
            data={"session_key": session_key},
        )
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    session = mgr.get_or_create(session_key)
    previous_messages = session.messages
    previous_last_consolidated = session.last_consolidated
    previous_updated_at = session.updated_at
    previous_metadata = copy.deepcopy(session.metadata)
    previous_pending_clarification = copy.deepcopy(session.pending_clarification)
    session.clear()
    try:
        mgr.commit_history_rewrite(session)
    except Exception:
        logger.warning("session.clear: failed to persist cleared {}", session_key)
        session.messages = previous_messages
        session.last_consolidated = previous_last_consolidated
        session.updated_at = previous_updated_at
        session.metadata = previous_metadata
        session.pending_clarification = previous_pending_clarification
        return {"session_id": session_key, "cleared": False}
    return {"session_id": session_key, "cleared": True}


async def session_undo(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """执行 ``session.undo``：原地删除最后 ``n`` 个 Turn，默认 1。

    Turn boundary 由 ``role=="user"`` 确定。Active Turn 期间拒绝，避免与 writer 竞争。
    ``n`` 为 forward-compat 保留；ui-tui 的 ``/undo`` 与 ``/retry`` 不发送 ``n``，因此走
    default 1。实际裁剪委托给 ``session.undo_last_turn(n)``。

    有删除时通过 ``commit_history_rewrite`` 持久化；失败则恢复 messages、timestamp、
    metadata 与 pending clarification，并返回 ``removed=0``。返回正数表示历史 rewrite
    成功，不表示模型响应已重新生成；``/retry`` 仍需另行发起 Turn。
    """
    session_key = params.get("session_id", "")
    if not session_key:
        return {"removed": 0}
    if turn_module.is_turn_active(session_key):
        raise TurnInProgressError(
            f"session {session_key!r} has an active turn; interrupt it before undo",
            data={"session_key": session_key},
        )
    n = params.get("n", 1)
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    session = mgr.get_or_create(session_key)
    previous_messages = session.messages
    previous_updated_at = session.updated_at
    previous_metadata = copy.deepcopy(session.metadata)
    previous_pending_clarification = copy.deepcopy(session.pending_clarification)
    removed = session.undo_last_turn(n)
    if removed:
        try:
            mgr.commit_history_rewrite(session)
        except Exception:
            logger.warning("session.undo: failed to persist undo for {}", session_key)
            session.messages = previous_messages
            session.updated_at = previous_updated_at
            session.metadata = previous_metadata
            session.pending_clarification = previous_pending_clarification
            return {"removed": 0}
    return {"removed": removed}


async def session_branch(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """执行 ``session.branch``：把指定 Session fork 为可独立演化的 child。

    方法在 source head 处通过 ``SessionManager.fork`` 做 full-copy，并返回 TUI 消费的
    ``SessionBranchResponse`` shape ``{session_id, title, message_count}``；TUI 会把 ``sid``
    切换到 child，并报告带走的 ``message_count``。可选 non-empty ``name`` 成为 child title。

    Active Turn 期间拒绝。unknown 或 zero-message source 返回 ``session_id=None``，让 TUI
    guard 当作 no-op；持久化异常也返回 no-op。成功后清理 source 的 pending image，避免
    未提交附件被误带入分支。branch 成功表示 child 已创建，不代表其后续 Turn 成功。
    """
    session_key = params.get("session_id", "")
    if not session_key:
        return {"session_id": None, "title": None}
    if turn_module.is_turn_active(session_key):
        raise TurnInProgressError(
            f"session {session_key!r} has an active turn; interrupt it before branching",
            data={"session_key": session_key},
        )
    name = params.get("name")
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    try:
        child = mgr.fork(session_key, title=(name or None))
    except Exception:
        logger.warning("session.branch: failed to persist child for {}", session_key)
        return {"session_id": None, "title": None}
    if child is None:
        return {"session_id": None, "title": None}
    clear_pending_images(session_key)
    return {
        "session_id": child.key,
        "title": child.metadata.get("title"),
        "message_count": len(child.messages),
    }


async def session_export(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """执行 ``session.export``：写出完整且可验证的 portable Session artifact。

    相对 clear/undo，这是 read-only 操作，所以没有 busy-guard。``session_id`` 通过 shared
    cross-channel resolver 解析：unresolved 返回 ``not_found``，ambiguous 返回 candidate
    keys，这两种情况都不写文件。resolved Session 使用 ``default_export_path`` 定位
    Workspace exports directory，再由 ``write_portable_export`` 生成 JSON envelope。

    写入后必须通过 ``verify_export`` 才返回 ``exported=True`` 与 absolute path；write error
    和 verification failure 分别返回明确 reason。验证通过证明 artifact 自洽，不证明其中
    对话对应的任务真实完成或正向结论可用。
    """
    value = params.get("session_id", "")
    if not value:
        return {"exported": False, "path": None, "reason": "not_found"}
    agent_loop = _safe_invoke_factory(agent_loop_factory)
    config = load_config()
    mgr = _manager_for(agent_loop, config)
    res = mgr.resolve_key(value)
    if res.status == "ambiguous":
        return {
            "exported": False,
            "path": None,
            "reason": "ambiguous",
            "candidates": list(res.candidates),
        }
    session = mgr.peek(res.key) if res.status == "resolved" else None
    if session is None:
        return {"exported": False, "path": None, "reason": "not_found"}
    dest = default_export_path(mgr.workspace, res.key)
    try:
        written = write_portable_export(session, dest)
    except OSError:
        logger.warning("session.export: failed to write export for {}", res.key)
        return {"exported": False, "path": None, "reason": "write_failed"}
    if not verify_export(written):
        logger.warning("session.export: verification failed for {}", res.key)
        return {"exported": False, "path": str(written), "reason": "verification_failed"}
    return {"exported": True, "path": str(written)}


def register_session_methods(
    dispatcher: "Dispatcher",
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
    confirm_broker: "ConfirmBroker | None" = None,
) -> None:
    """在 Dispatcher 上注册 11 个 ``session.*`` handler。

    该实现镜像 :func:`codeflow.tui_rpc.methods.turn.register_turn_methods`：用单参数 closure
    包装 module-level handler，预绑定 ``agent_loop_factory`` 与删除流程使用的
    ``confirm_broker``，从而满足 Dispatcher 的 ``params -> dict`` contract。

    注册包括 create、close、resume、list、delete、most_recent、title、clear、undo、branch
    与 export。函数不创建或加载 Session；重复注册由 Dispatcher 抛出 ``ValueError``。
    """

    async def _create(params: dict) -> dict:
        return await session_create(params, agent_loop_factory=agent_loop_factory)

    async def _close(params: dict) -> dict:
        return await session_close(params, agent_loop_factory=agent_loop_factory)

    async def _resume(params: dict) -> dict:
        return await session_resume(params, agent_loop_factory=agent_loop_factory)

    async def _list(params: dict) -> dict:
        return await session_list(params, agent_loop_factory=agent_loop_factory)

    async def _delete(params: dict) -> dict:
        return await session_delete(
            params,
            agent_loop_factory=agent_loop_factory,
            confirm_broker=confirm_broker,
        )

    async def _most_recent(params: dict) -> dict:
        return await session_most_recent(params, agent_loop_factory=agent_loop_factory)

    async def _title(params: dict) -> dict:
        return await session_title(params, agent_loop_factory=agent_loop_factory)

    async def _clear(params: dict) -> dict:
        return await session_clear(params, agent_loop_factory=agent_loop_factory)

    async def _undo(params: dict) -> dict:
        return await session_undo(params, agent_loop_factory=agent_loop_factory)

    async def _branch(params: dict) -> dict:
        return await session_branch(params, agent_loop_factory=agent_loop_factory)

    async def _export(params: dict) -> dict:
        return await session_export(params, agent_loop_factory=agent_loop_factory)

    dispatcher.register("session.create", _create)
    dispatcher.register("session.close", _close)
    dispatcher.register("session.resume", _resume)
    dispatcher.register("session.list", _list)
    dispatcher.register("session.delete", _delete)
    dispatcher.register("session.most_recent", _most_recent)
    dispatcher.register("session.title", _title)
    dispatcher.register("session.clear", _clear)
    dispatcher.register("session.undo", _undo)
    dispatcher.register("session.branch", _branch)
    dispatcher.register("session.export", _export)


__all__ = [
    "AgentLoopFactory",
    "session_create",
    "session_close",
    "session_resume",
    "session_list",
    "session_delete",
    "session_most_recent",
    "session_title",
    "session_clear",
    "session_undo",
    "session_branch",
    "session_export",
    "register_session_methods",
]
