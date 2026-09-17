"""任务上下文：步骤之间传递变量，并提供 ${path} 插值。

变量来源：
- 任务 YAML 的 params（启动参数）；
- 运行期步骤产出（如 read_route 查表得到的目标点/目标 NPC）。

支持点路径取值：${route.target} → vars["route"]["target"]。
"""

from __future__ import annotations

import re
from typing import Any

_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


class ContextError(ValueError):
    """变量缺失或插值失败。"""


class QuestContext:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.vars: dict[str, Any] = dict(params or {})
        # 最近一次读到的对话全文，供 read_route / dialog_until 使用
        self.vars.setdefault("dialog", "")

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.vars
        for part in path.strip().split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def require(self, path: str) -> Any:
        value = self.get(path)
        if value is None:
            raise ContextError(f"上下文变量不存在: ${{{path}}}")
        return value

    def set(self, path: str, value: Any) -> None:
        parts = path.strip().split(".")
        node = self.vars
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def interpolate(self, template: str) -> str:
        """把字符串里的 ${path} 全部替换为变量值。"""
        if not isinstance(template, str) or "${" not in template:
            return template

        def _sub(match: re.Match) -> str:
            return str(self.require(match.group(1)))

        return _INTERP_RE.sub(_sub, template)

    def resolve_point(self, value: Any) -> list[float] | list[int]:
        """解析坐标参数：允许 ["${route.target}"] 或 ["100","200"] 等写法。"""
        if isinstance(value, (list, tuple)) and len(value) == 2:
            if isinstance(value[0], str) and "${" in value[0]:
                resolved = self.require(_INTERP_RE.search(value[0]).group(1))
                return list(resolved)
            return [value[0], value[1]]
        if isinstance(value, str) and "${" in value:
            resolved = self.require(_INTERP_RE.search(value).group(1))
            return list(resolved)
        raise ContextError(f"无法解析坐标参数: {value!r}")
