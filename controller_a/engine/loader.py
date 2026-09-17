"""任务 YAML 加载与校验。

任务文件放在 tasks/ 目录，一个文件可包含多个任务（顶层 tasks: 映射），
也可以直接是单个任务结构。新增任务 = 新增配置，不改代码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

VALID_ON_FAIL = ("abort", "restart", "notify")
VALID_STEP_KEYS = {
    "do", "npc", "match", "mode", "point", "point_frac", "route_map",
    "store_as", "scene", "roi", "timeout", "retry", "secs", "text",
    "continue_option", "fallback_frac", "max_rounds", "interval",
    "settle", "confidence", "regex", "desc",
    # 保镖任务：组合键 / 固定区域 / 两级地图 / NPC 视觉
    "keys", "times", "region", "button", "map", "level", "place",
}


class TaskConfigError(ValueError):
    """任务配置非法。"""


@dataclass
class TaskDef:
    key: str
    name: str
    steps: list[dict[str, Any]]
    on_fail: str = "notify"
    max_restarts: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    regions_file: str = ""


def load_task_file(path: str) -> list[TaskDef]:
    """加载一个 YAML 文件，返回其中的全部任务定义。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        raise TaskConfigError(f"{path} 内容为空")

    if isinstance(data, dict) and "steps" in data:
        raw_tasks = {data.get("key") or os.path.splitext(
            os.path.basename(path))[0]: data}
    elif isinstance(data, dict) and "tasks" in data:
        raw_tasks = data["tasks"]
        if not isinstance(raw_tasks, dict):
            raise TaskConfigError(f"{path}: tasks 必须是 key:任务定义 的映射")
    else:
        raise TaskConfigError(
            f"{path}: 顶层必须是单个任务（含 steps）或含 tasks 映射")

    out: list[TaskDef] = []
    for key, raw in raw_tasks.items():
        out.append(_parse_one(str(key), raw, path))
    return out


def _parse_one(key: str, raw: Any, source: str) -> TaskDef:
    if not isinstance(raw, dict):
        raise TaskConfigError(f"{source}: 任务 {key} 必须是字典")
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        raise TaskConfigError(f"{source}: 任务 {key} 的 steps 必须是非空列表")

    on_fail = raw.get("on_fail", "notify")
    if on_fail not in VALID_ON_FAIL:
        raise TaskConfigError(
            f"{source}: 任务 {key} 的 on_fail 必须是 {VALID_ON_FAIL} 之一")

    max_restarts = int(raw.get("max_restarts", 0))
    if max_restarts < 0 or max_restarts > 5:
        raise TaskConfigError(f"{source}: max_restarts 必须在 0~5")

    parsed_steps: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "do" not in step:
            raise TaskConfigError(
                f"{source}: 任务 {key} 第 {i + 1} 步必须是含 do 的字典")
        unknown = set(step) - VALID_STEP_KEYS
        if unknown:
            raise TaskConfigError(
                f"{source}: 任务 {key} 第 {i + 1} 步存在未知字段 {sorted(unknown)}")
        timeout = step.get("timeout", 30)
        retry = step.get("retry", 0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise TaskConfigError("timeout 必须是正数（秒）")
        if not isinstance(retry, int) or retry < 0 or retry > 10:
            raise TaskConfigError("retry 必须是 0~10 的整数")
        parsed_steps.append(dict(step))

    params = raw.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TaskConfigError(f"{source}: params 必须是字典")

    regions_file = raw.get("regions_file", "")
    if regions_file is not None and not isinstance(regions_file, str):
        raise TaskConfigError(f"{source}: regions_file 必须是字符串路径")

    return TaskDef(
        key=key,
        name=str(raw.get("name", key)),
        steps=parsed_steps,
        on_fail=on_fail,
        max_restarts=max_restarts,
        params=params,
        source_file=source,
        regions_file=regions_file or "",
    )


def load_all_tasks(tasks_dir: str) -> list[TaskDef]:
    """扫描 tasks 目录下全部 *.yaml/*.yml。"""
    tasks: list[TaskDef] = []
    if not os.path.isdir(tasks_dir):
        return tasks
    for filename in sorted(os.listdir(tasks_dir)):
        if filename.endswith((".yaml", ".yml")):
            tasks.extend(load_task_file(os.path.join(tasks_dir, filename)))
    return tasks
