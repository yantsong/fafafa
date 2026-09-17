"""任务调度引擎：逐步执行 TaskDef，处理重试、超时、失败恢复与停止。

执行语义
--------
- 步骤顺序执行；每步失败按其 retry 重试（重试时整个步骤从头跑）；
- 单步耗尽重试后按任务 on_fail 处理：
    abort   —— 立即结束，报告失败；
    restart —— 从第 1 步重跑整个任务（最多 max_restarts 次）；
    notify  —— 同 abort，但要求外部通知用户（默认策略，绝不盲点）；
- 连续失败看门狗：任何异常都即时上报，不吞错。

引擎只编排，具体动作在 steps.py；在后台线程运行，通过 log 回调输出。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from engine.context import QuestContext
from engine.kit import Kit, StepStopped
from engine.loader import TaskDef
from engine.steps import HANDLERS, StepFailed


@dataclass
class QuestResult:
    success: bool
    message: str
    elapsed: float
    restarts: int = 0
    failed_step: int = -1      # 1-based，-1 表示无
    failed_action: str = ""
    stopped: bool = False


class QuestEngine:
    def __init__(self, task: TaskDef, kit: Kit,
                 log: Callable[[str, str], None] | None = None) -> None:
        self.task = task
        self.kit = kit
        self._log = log or (lambda msg, level="": None)
        self.stop_event = kit.stop_event

    def log(self, msg: str, level: str = "") -> None:
        self._log(msg, level)

    def run(self) -> QuestResult:
        t0 = time.monotonic()
        restarts = 0
        while True:
            ctx = QuestContext(dict(self.task.params))
            ctx.set("_task_dir",
                    self.task.source_file.rsplit("/", 1)[0]
                    if "/" in self.task.source_file else ".")
            try:
                self._run_once(ctx)
                return QuestResult(
                    True, f"任务「{self.task.name}」完成",
                    time.monotonic() - t0, restarts)
            except StepStopped:
                return QuestResult(
                    False, "任务已被用户停止", time.monotonic() - t0,
                    restarts, stopped=True)
            except StepFailed as exc:
                idx, action = getattr(exc, "_step_pos", (-1, ""))
                if self.task.on_fail == "restart" and restarts < self.task.max_restarts:
                    restarts += 1
                    self.log(f"第 {idx} 步（{action}）失败：{exc}；"
                             f"第 {restarts}/{self.task.max_restarts} 次重跑整个任务",
                             "fail")
                    continue
                msg = (f"任务在第 {idx} 步（{action}）失败：{exc}")
                if self.task.on_fail == "notify":
                    msg += " —— 已停止，请人工处理（不会继续操作）"
                return QuestResult(False, msg, time.monotonic() - t0,
                                   restarts, idx, action)

    def _run_once(self, ctx: QuestContext) -> None:
        total = len(self.task.steps)
        for i, step in enumerate(self.task.steps, start=1):
            self.kit.check_stop()
            action = step["do"]
            desc = step.get("desc") or _describe(action, step)
            self.log(f"[{i}/{total}] {desc}", "info")
            handler = HANDLERS.get(action)
            if handler is None:
                err = StepFailed(f"未知步骤类型 {action!r}")
                raise self._mark(err, i, action)

            retries = int(step.get("retry", 0))
            attempts = retries + 1
            last_exc: StepFailed | None = None
            for attempt in range(1, attempts + 1):
                self.kit.check_stop()
                try:
                    handler(step, ctx, self.kit)
                    last_exc = None
                    break
                except StepStopped:
                    raise
                except StepFailed as exc:
                    last_exc = exc
                    if attempt < attempts:
                        self.log(f"  第 {attempt} 次失败：{exc}；重试…", "fail")
                        self.kit.sleep(0.5)
                except Exception as exc:  # 意料外错误不静默
                    last_exc = StepFailed(
                        f"{type(exc).__name__}: {exc}")
                    if attempt < attempts:
                        self.log(f"  第 {attempt} 次异常：{last_exc}；重试…",
                                 "fail")
                        self.kit.sleep(0.5)
            if last_exc is not None:
                raise self._mark(last_exc, i, action)

    @staticmethod
    def _mark(exc: StepFailed, i: int, action: str) -> StepFailed:
        exc._step_pos = (i, action)  # type: ignore[attr-defined]
        return exc


def _describe(action: str, step: dict) -> str:
    if action == "click_npc":
        return f"点击 NPC「{step.get('npc')}」"
    if action == "wait_dialog":
        return f"等待对话出现「{step.get('match', '任意文字')}」"
    if action == "wait_text":
        return f"等待文字「{step.get('match')}」"
    if action == "choose_option":
        return f"选择对话选项「{step.get('match')}」"
    if action == "read_route":
        return f"读取对话并查路线表 {step.get('route_map')}"
    if action == "click_point":
        return f"点击坐标 {step.get('point') or step.get('point_frac')}"
    if action == "wait_arrive":
        return f"等待到达「{step.get('scene')}」"
    if action == "dialog_until":
        return f"推进对话直到「{step.get('match')}」"
    if action == "hotkey":
        times = step.get("times", 1)
        suffix = f" ×{times}" if times != 1 else ""
        return f"发送组合键 {step.get('keys')}{suffix}"
    if action == "click_region":
        button = step.get("button", "left")
        side = "右键" if button == "right" else "左键"
        return f"{side}点击固定区域「{step.get('region')}」"
    if action == "map_click":
        level = step.get("level", "big")
        where = "大地图地名" if level == "big" else "小地图寻路点"
        npc = step.get("npc")
        target = f"{step.get('place')}/{npc}" if npc else step.get("place")
        return f"点击{where}「{target}」"
    if action == "wait_npc":
        return f"等待 NPC「{step.get('npc')}」出现（OCR/模板）"
    if action == "click_quest_npc":
        return f"点击目标 NPC「{step.get('npc')}」"
    if action == "sleep":
        return f"等待 {step.get('secs')}s"
    return action


def run_in_thread(engine: QuestEngine,
                  on_done: Callable[[QuestResult], None]) -> threading.Thread:
    """在后台线程跑引擎，结束后回调 on_done（调用方负责切回 UI 线程）。"""
    def _target() -> None:
        result = engine.run()
        on_done(result)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread
