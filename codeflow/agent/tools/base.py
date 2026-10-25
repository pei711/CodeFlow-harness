"""定义 Agent Tool 的基础契约、参数类型处理和 OpenAI function schema 边界。

Tool 是模型与文件、Shell、Web、Channel 等环境能力之间的受控入口。具体实现声明稳定名称、
人类可读描述与 JSON Schema，再实现异步执行；`ToolRegistry` 统一负责类型转换、校验、超时、
失败归一化与 tracing。基础类只提供协议和纯参数逻辑，不决定某个 Tool 是否允许当前操作。
"""

from abc import ABC, abstractmethod
from typing import Any

from codeflow.agent.tools.execution import ToolCapability, ToolExecutionContext, ToolInvocation


class ToolResult(str):
    """在字符串输出上附加明确执行状态的 Tool 结果。

    它继承 `str`，所以旧调用方仍可直接显示、拼接或写入 LLM History；额外 ``failed`` 字段让
    Registry 与 Agent Loop 不必从自然语言猜测成功。默认 ``failed=False``；当失败文本不遵循
    ``Error:`` 约定时，Tool 应显式返回 ``ToolResult(..., failed=True)``。这个状态表示 Tool
    调用失败，不证明整个用户任务成功或失败。
    """

    failed: bool

    def __new__(cls, value: object = "", *, failed: bool = False):
        result = super().__new__(cls, str(value))
        result.failed = failed
        return result


class Tool(ABC):
    """所有 Agent Tool 的抽象基类与统一运行契约。

    Tool 是 Agent 与外部环境交互的 capability，例如读取文件、执行命令或发送消息。子类必须
    定义 ``name``、``description``、``parameters`` 与 `execute`；Registry 会把前三者暴露给 LLM，
    对模型参数先 `cast_params` 再 `validate_params`，最后在超时和异常边界内调用执行方法。

    ``capability`` 声明 READ/WRITE/EXECUTE/EXTERNAL effect 与并发安全性；只有 READ 且
    concurrency_safe 的调用可并行。``timeout_seconds`` 可覆盖 Registry 宽松上限，
    ``blocking_interaction=True`` 则表示 Ask User 等 Tool 有意等待人类，不应被统一计时器中断。
    子类仍负责自己的业务权限、路径和副作用边界。
    """

    # 注册表通过 asyncio.wait_for 强制的硬上限（秒），避免未设置自身超时的工具
    # 卡死整个 Agent Loop。None 使用注册表默认值；exec、spawn 等合法长时任务会提高它。
    timeout_seconds: float | None = None

    # 有意阻塞等待人类的工具（ask_user、request_permissions 及未来的人工审批门禁）
    # 将此值设为 True，使注册表不为它们包装超时。它们自行管理自动解决，
    # 而不是在等待中途被终止。
    blocking_interaction: bool = False
    capability = ToolCapability()

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @property
    @abstractmethod
    def name(self) -> str:
        """返回 LLM function call 和 Registry dispatch 使用的稳定 Tool 名称。

        名称必须与模型看到的 schema、ToolEvent 和执行查找一致；注册表用它检测重复项。该属性
        不应随 Session 或参数变化，否则模型生成的调用可能无法解析到同一实现。
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """返回提供给 LLM 的 Tool 能力、适用场景与关键边界说明。

        模型会依据这段文本选择是否调用 Tool，因此描述应说明可见行为而不是实现细节，也不能
        承诺代码没有保证的成功结果。它与 `parameters` 一起进入 OpenAI function schema。
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """返回 Tool 参数的 JSON Schema。

        顶层通常必须是 ``type=object``，properties 定义字段类型、范围与 enum，required 指定
        必填项。Registry 根据同一 Schema 做安全转换和递归校验，因此实现不能只把它当模型
        提示；非 object 顶层会在 `validate_params` 明确失败。
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """使用已经转换并校验的 Tool-specific ``**kwargs`` 执行一次调用。

        参数来自子类 `parameters` 声明，Registry 在进入本方法前完成 schema-driven cast 与
        validation。返回值是写回模型 History 的 String result；失败文本若不遵循 Registry 的
        ``Error:`` convention，必须使用 ``ToolResult(..., failed=True)``，避免被误计为成功。

        实现抛出的异常会由 Registry 捕获并转成失败结果；取消和 blocking interaction 的具体
        生命周期仍由外层执行边界控制。方法只表示一次 Tool call 结束，不表示用户目标完成。
        """
        pass

    async def execute_with_context(self, context: ToolExecutionContext, **kwargs: Any) -> str:
        return await self.execute(**kwargs)

    def resolve_invocation(self, invocation: ToolInvocation) -> ToolInvocation:
        return invocation

    def resource_key(self, invocation: ToolInvocation) -> str | None:
        """Return the shared resource touched by this call, if it has one.

        Tools without a resource key do not participate in the registry's
        resource-level read/write locking.
        """
        return None

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """在正式校验前按 JSON Schema 做有限且安全的参数类型转换。

        模型可能把 integer、number 或 boolean 以字符串输出；本方法递归处理 object/array，把
        可明确解释的值转换成 Schema 类型，例如 ``"3"`` 到 int、``"true"`` 到 bool。无法
        转换的值保持原样，交给 `validate_params` 产生具体错误；未知字段也保留，不能借转换
        静默删除模型输入。顶层 Schema 不是 object 时直接返回原 params。

        转换只规范表示，不扩大权限、不填补 required 字段，也不执行 Python 任意解析。
        ``None`` 转 string 时特意保留 None，使校验准确拒绝而不是伪造 ``"None"``。
        """
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params

        return self._cast_object(params, schema)

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        """按 object Schema 递归转换字典中已声明的字段。

        ``obj`` 不是 dict 时原样返回，让后续 validator 报类型错误。properties 中存在的 key 交给
        `_cast_value`，未知 key 原样复制；返回新字典，不修改调用方参数。该方法只处理表示层
        转换，不检查 required、enum 或数值范围。
        """
        if not isinstance(obj, dict):
            return obj

        props = schema.get("properties", {})
        result = {}

        for key, value in obj.items():
            if key in props:
                result[key] = self._cast_value(value, props[key])
            else:
                result[key] = value

        return result

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        """根据单字段 Schema 转换一个值，并在含糊时保持原值。

        已经符合目标类型的值直接返回，同时特别阻止 bool 被当作 integer/number。字符串只在
        int/float 解析成功，或命中 boolean 的 ``true/1/yes``、``false/0/no`` 集合时转换；array
        按 items 逐项递归，object 委托 `_cast_object`。没有可靠规则时返回 ``val``，由校验器
        决定错误，而不是猜测模型意图。
        """
        target_type = schema.get("type")

        if target_type == "boolean" and isinstance(val, bool):
            return val
        if target_type == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if target_type in self._TYPE_MAP and target_type not in ("boolean", "integer", "array", "object"):
            expected = self._TYPE_MAP[target_type]
            if isinstance(val, expected):
                return val

        if target_type == "integer" and isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return val

        if target_type == "number" and isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val

        if target_type == "string":
            return val if val is None else str(val)

        if target_type == "boolean" and isinstance(val, str):
            val_lower = val.lower()
            if val_lower in ("true", "1", "yes"):
                return True
            if val_lower in ("false", "0", "no"):
                return False
            return val

        if target_type == "array" and isinstance(val, list):
            item_schema = schema.get("items")
            return [self._cast_value(item, item_schema) for item in val] if item_schema else val

        if target_type == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)

        return val

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """对 JSON Schema 验证 Tool 参数，并返回全部可解释错误。

        参数不是 dict 时返回顶层 object 类型错误；Tool Schema 本身不是 object 则抛 `ValueError`，
        因为这是实现配置缺陷而非模型输入问题。合法顶层交给 `_validate` 递归检查 type、required、
        enum、数值上下限、字符串长度和 array items。返回空 list 表示通过，但不执行 Tool，也不
        替代业务层权限检查。
        """
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, "")

    def _validate(self, val: Any, schema: dict[str, Any], path: str) -> list[str]:
        t, label = schema.get("type"), path or "parameter"
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} should be integer"]
        if t == "number" and (not isinstance(val, self._TYPE_MAP[t]) or isinstance(val, bool)):
            return [f"{label} should be number"]
        if t in self._TYPE_MAP and t not in ("integer", "number") and not isinstance(val, self._TYPE_MAP[t]):
            return [f"{label} should be {t}"]

        errors = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if t == "object":
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in val:
                    errors.append(f"missing required {path + '.' + k if path else k}")
            for k, v in val.items():
                if k in props:
                    errors.extend(self._validate(v, props[k], path + "." + k if path else k))
        if t == "array" and "items" in schema:
            for i, item in enumerate(val):
                errors.extend(self._validate(item, schema["items"], f"{path}[{i}]" if path else f"[{i}]"))
        return errors

    def to_schema(self) -> dict[str, Any]:
        """把 Tool 契约转换为 OpenAI function schema 形状。

        返回顶层 ``type="function"``，内部 function 携带稳定 name、description 与 parameters。
        Registry 用该列表随 LLM 请求发送，模型返回的 function name 再映射回同一 Tool。函数
        不缓存结果，因此动态属性会在每次定义读取时反映，但名称仍应保持稳定。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
