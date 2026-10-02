"""实现受 PAHF 启发的四阶段 Personalization Flow。

Step 1 Request Triage 由 ``classify()`` 判断是否绝对需要偏好；Step 2 Pre-action Interaction 用
``generate_question()`` 只问一个问题，并由 ``extract_and_store_preference()`` 从答案学习可复用
事实；Step 3 Execution 完全由 AgentLoop 处理；Step 4 ``post_learn()`` 在完成 Turn 后被动提取新
Signal。所有 LLM Failure 都收敛为 Neutral Default，Personalization 不能阻塞 Main Agent。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

from codeflow.tracing import semconv, trace

if TYPE_CHECKING:
    from codeflow.memory_engine.consolidate.consolidator import MemoryStore
    from codeflow.providers.base import LLMProvider


# 第 1 步：请求分流，判断是否需要澄清
_CLASSIFY_PROMPT = """\
Classify whether this user request ABSOLUTELY CANNOT be answered without \
knowing a user preference. Default to needs_clarification=false.

## Memory (known preferences)
{memory}

## Recent Conversation
{history}

## User Request
{message}

Reply with JSON only:
{{"needs_clarification": true/false, "domain": "short domain e.g. programming_language"}}

Return needs_clarification=true ONLY when ALL of these are true:
- The request is genuinely ambiguous with no reasonable default
- The missing preference would lead to a COMPLETELY DIFFERENT answer
- Memory does not already contain a relevant preference
- Recent conversation does not already clarify the intent

Return needs_clarification=false if:
- The request is fully specified or has a reasonable default
- Memory already covers the relevant preference
- The preference would not meaningfully change the outcome
- The request is a simple factual, technical, or follow-up question
- The user is responding to or continuing a previous conversation topic

When in doubt, return false — make a reasonable assumption rather than asking."""

# 第 2a 步：行动前交互，生成澄清问题
_QUESTION_PROMPT = """\
Generate ONE short clarifying question for this request.

Request: {message}
Preference domain: {domain}
Known memory: {memory}

Rules:
- Ask only the single most important missing preference
- List concrete options when possible
- Be brief and natural

Output the question line only, in this exact format:
Quick question: [question]? Options: [A] / [B] / [C]"""

# 第 2b 步：行动前交互，从用户回答中提取偏好
_EXTRACT_PROMPT = """\
Extract reusable preference facts from this clarification Q&A.

Original request: {original_message}
Question asked: {question}
User answer: {answer}

Output JSON:
{{"facts": ["User always uses Python for scripts"], "section": "Preferences"}}

Rules:
- State facts as general reusable rules, NOT tied to this specific task
- Only include facts useful for future interactions
- 1-3 facts maximum
- If the answer reveals no reusable preference, return {{"facts": [], "section": "Preferences"}}"""

# 第 4 步：行动后学习，从已完成交互中被动提取信号
_POST_LEARN_PROMPT = """\
Analyze this completed interaction for new preference signals.

Request: {message}
Response summary: {response_summary}

## Current Memory
{memory}

Did this interaction reveal NEW preferences not already in memory?

Output JSON:
{{"has_new_preference": false, "new_facts": [{{"text": "...", "category": "preference"}}]}}

Each fact object has:
- text: the reusable rule statement about the user
- category: "preference" for general preferences such as languages, tools,
  communication style, and topics

Examples of "preference" category:
  - "User prefers Python for scripting tasks"
  - "User prefers concise responses without preamble"

Only set has_new_preference=true for facts that are:
- Genuinely new (not already in memory)
- Reusable (will help future interactions)
- Expressible as general rules about the user

Legacy format (flat list of strings with a `section` field) is still accepted
for backwards compatibility — category defaults to "preference" in that case."""


class Personalizer:
    """协调四阶段 PAHF-inspired Personalization，并把可复用偏好写入 MemoryStore。

    每个 Method 可独立调用：Classifier/Question/Extractor/Post-learn 都自行构造 Prompt、调用同一
    Provider/Model，并在异常时 Log 后返回 False、空 String 或不澄清 Default，使 Main Agent Loop
    never blocked。它只学习 General Reusable Preferences，不把单次 Task Result 或推测写成 User
    Fact；Memory 更新通过锁保护的 Read-modify-write 完成。
    """

    def __init__(self, memory: MemoryStore, provider: LLMProvider, model: str):
        self.memory = memory
        self.provider = provider
        self.model = model

    @trace.instrument("personalize.classify", kind="memory", extract=semconv.personalize)
    async def classify(self, message: str, history: list[dict] | None = None) -> dict:
        """判断当前 User ``message`` 是否必须先询问 Personalization Preference。

        可选 ``history`` 提供最近 2–4 条 Conversation Context，Long-term Memory 也进入 Classifier。
        Prompt 强制 Default False：只有 Request 真正 Ambiguous、无 Reasonable Default、缺失偏好会
        产生 Completely Different Answer，且 Memory/History 都未说明时才 True。

        返回 ``{"needs_clarification": bool, "domain": str}``。JSON Parse 或 Provider 任意 Error 回退
        ``{"needs_clarification": False, "domain": ""}``，让 Agent 做合理假设而不因 Personalizer
        阻塞。
        """
        current_memory = self.memory.read_long_term()

        history_text = self._format_history(history) if history else "(no prior context)"

        prompt = _CLASSIFY_PROMPT.format(
            memory=current_memory or "(empty)",
            history=history_text,
            message=message,
        )

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,  # 分类需要确定性输出
                max_tokens=100,  # JSON 很短，限制令牌数以节省成本
            )
            result = self._parse_json(
                response.content or "",
                fallback={"needs_clarification": False, "domain": ""},
            )
            logger.debug("Personalizer.classify: {}", result)
            return result
        except Exception:
            logger.exception("Personalizer.classify failed, skipping clarification")
            return {"needs_clarification": False, "domain": ""}

    @trace.instrument("personalize.question", kind="memory", extract=semconv.personalize)
    async def generate_question(self, message: str, domain: str) -> str:
        """为 Request 与 Preference ``domain`` 生成一个聚焦的 Clarifying Question。

        Prompt 带 Current Memory，并要求只问最重要缺失偏好、尽量列 Concrete Options。成功返回
        例如 ``Quick question: Which language? Options: Python / Go`` 的单行 String；Provider Failure
        返回 ``""``，Caller 可 Gracefully Skip Clarification。方法不发送问题或保存答案。
        """
        current_memory = self.memory.read_long_term()

        prompt = _QUESTION_PROMPT.format(
            message=message,
            domain=domain,
            memory=current_memory or "(empty)",
        )

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,  # 少量随机性让问题更自然
                max_tokens=120,
            )
            question = (response.content or "").strip()
            logger.debug("Personalizer.generate_question: {}", question)
            return question
        except Exception:
            logger.exception("Personalizer.generate_question failed, skipping clarification")
            return ""

    @trace.instrument("personalize.extract", kind="memory", extract=semconv.personalize)
    async def extract_and_store_preference(self, original_message: str, question: str, answer: str) -> bool:
        """从 Clarification Q&A 提取 Reusable Preference，并持久化到 MEMORY.md。

        ``original_message``、实际 ``question`` 与 User ``answer`` 一起交给低温度 LLM，最多提取
        1–3 条 General Rule 与目标 Section。空 Fact 返回 ``False``；有 Fact 则通过
        `_append_to_memory_section` 加锁写入，并返回 ``True``。JSON/Provider/Storage Failure 记录后
        返回 False，不把本次特定 Task Detail 强行推广为偏好。
        """
        prompt = _EXTRACT_PROMPT.format(
            original_message=original_message,
            question=question,
            answer=answer,
        )

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=200,
            )
            result = self._parse_json(
                response.content or "",
                fallback={"facts": [], "section": "Preferences"},
            )

            facts: list[str] = result.get("facts", [])
            section: str = result.get("section", "Preferences")

            if not facts:
                logger.debug("Personalizer.extract: no reusable preference found in Q&A")
                return False

            self._append_to_memory_section(section, facts)
            logger.info("Personalizer.extract: stored {} fact(s) → {}", len(facts), facts)
            return True

        except Exception:
            logger.exception("Personalizer.extract_and_store_preference failed")
            return False

    @trace.instrument("personalize.postlearn", kind="memory", extract=semconv.personalize)
    async def post_learn(self, message: str, response_summary: str) -> bool:
        """从 Completed Interaction 被动提取 New Preference Signal。

        本方法 Intended 作为 Background asyncio Task，never blocks Response；Response Summary 截到
        600 字符，Current Memory 用于排除已有事实。只有 LLM 明确 ``has_new_preference`` 且归一化
        后仍有 Fact 才分 Section 写入，返回是否实际存储。

        同时接受新 Per-fact ``{text, category}`` Schema 与 Legacy Flat-list Format；Legacy Fact 的
        Category Default 为 ``"preference"``，Sibling section 仍可指定 Header。任意 Failure 返回
        False，不影响已交付 Reply。
        """
        current_memory = self.memory.read_long_term()

        prompt = _POST_LEARN_PROMPT.format(
            message=message,
            response_summary=response_summary[:600],  # 避免提示无限增长
            memory=current_memory or "(empty)",
        )

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=250,
            )
            result = self._parse_json(
                response.content or "",
                fallback={"has_new_preference": False, "new_facts": []},
            )

            if not result.get("has_new_preference"):
                return False

            grouped = self._group_facts_by_category(
                result.get("new_facts", []),
                legacy_section=result.get("section", "Preferences"),
            )
            if not grouped:
                return False

            total = 0
            for section, facts in grouped.items():
                self._append_to_memory_section(section, facts)
                total += len(facts)
                logger.info(
                    "Personalizer.post_learn: stored {} fact(s) → section={} facts={}",
                    len(facts),
                    section,
                    facts,
                )
            return total > 0

        except Exception:
            logger.exception("Personalizer.post_learn failed")
            return False

    _CATEGORY_SECTION_MAP = {
        "preference": "Preferences",
    }

    @classmethod
    def _group_facts_by_category(cls, new_facts: list, legacy_section: str = "Preferences") -> dict[str, list[str]]:
        """把两种 ``new_facts`` Shape 归一为 ``{section_header: [fact_text, ...]}``。

        New Shape 是 ``[{"text": str, "category": str}]``，Category 通过 Class Map 选择 Section；
        Legacy Shape 是 ``["fact text", "fact text"]``，使用 Result Sibling ``section`` 传入的
        ``legacy_section``。非 Dict/String、空 Text 跳过；返回值保留输入顺序，不去重 Fact。
        """
        grouped: dict[str, list[str]] = {}
        for item in new_facts or []:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                category = str(item.get("category", "preference")).strip().lower()
                section = cls._CATEGORY_SECTION_MAP.get(category, legacy_section)
            elif isinstance(item, str):
                text = item.strip()
                section = legacy_section
            else:
                continue
            if not text:
                continue
            grouped.setdefault(section, []).append(text)
        return grouped

    @staticmethod
    def _format_history(history: list[dict], max_messages: int = 4) -> str:
        """把最近 Conversation History 压缩为 Classifier Prompt 可读 String。

        只读取最后 ``max_messages`` 条，Role 转大写，非空 String Content 最多保留 200 字符并加
        Ellipsis；非 String/Empty Message 忽略。没有可用内容时返回 ``(no prior context)``，函数
        不修改 History，也不把 Tool Structured Payload 展开。
        """
        recent = history[-max_messages:]
        lines = []
        for m in recent:
            role = m.get("role", "unknown").upper()
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                text = content[:200] + "..." if len(content) > 200 else content
                lines.append(f"{role}: {text}")
        return "\n".join(lines) if lines else "(no prior context)"

    def _parse_json(self, text: str, fallback: dict) -> dict:
        """从 LLM Text 中截取最外层首尾 Brace 范围并解析第一个 JSON Object。

        这让模型在 JSON 外包 Extra Prose 时仍可恢复；找不到有效 ``{...}`` 或 `json.loads` 失败时
        原样返回 ``fallback``。函数不修复 Invalid JSON、不执行 Code Fence，也不验证具体 Schema，
        Caller 仍用 Default 读取 Field。
        """
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return fallback
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return fallback

    def _append_to_memory_section(self, section: str, facts: list[str]) -> None:
        """把 New Facts 追加到 MEMORY.md 的指定 Section Header 下。

        Existing Section 在 Header 后立即插入 Bullet；Missing Section 在 File End 新建。整个
        Read-modify-write 经 ``MemoryStore.locked()`` 使用 fcntl Lock，使 Another Process 的
        Concurrent MemoryConsolidator Writer 不会 Clobber Update。方法不做 Semantic Deduplication，
        Caller 应先确保 Facts genuinely new。
        """
        header = f"## {section}"
        fact_lines = "\n".join(f"- {f}" for f in facts)

        with self.memory.locked():
            current = self.memory.read_long_term()
            if header in current:
                updated = current.replace(header, f"{header}\n{fact_lines}", 1)
            else:
                updated = current.rstrip() + f"\n\n{header}\n{fact_lines}\n"
            self.memory.write_long_term(updated)
