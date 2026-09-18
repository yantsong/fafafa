"""左上角 HUD 状态解析：坐标行 + 场景名行。

真实格式（固定两行）：
    第一行：X:286  Y:291
    第二行：长安          （纯中文场景名）

设计约束（参考坐标 OCR 的踩坑经验）：
- 坐标只做严格正则提取，不做“字母→数字”之类的启发式纠错；
  OCR 个位抖动交给步骤层 coord_tol 容差吸收。
- 场景名与坐标解耦：只取 CJK 片段中最长的一段，避免吞进坐标旁噪声。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vision.npc_detector import normalize

# X:286  Y:291 / x：286 y 291 / X286 Y291 都兼容（分隔符宽松，数字 1~4 位）
_COORD_PAIR_RE = re.compile(
    r"[Xx][:：]?\s*(\d{1,4})\D{0,8}?[Yy][:：]?\s*(\d{1,4})")
# X/Y 字母被 OCR 弄丢时的兜底：取整行前两个独立数字
_DIGITS_RE = re.compile(r"\d{1,4}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass
class HudState:
    x: int
    y: int
    scene: str


def parse_coords(text: str) -> tuple[int, int] | None:
    """从拼接文本中提取 (x, y)；严格失败再兜底取前两个数字。"""
    if not text:
        return None
    m = _COORD_PAIR_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    nums = _DIGITS_RE.findall(text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return None


def parse_scene(text: str) -> str:
    """提取场景名：最长 CJK 连续段；没有中文返回空串。"""
    runs = _CJK_RE.findall(text or "")
    return max(runs, key=len) if runs else ""


def parse_hud(text: str) -> HudState | None:
    """整段 OCR 文本 → HudState；坐标或场景任一缺失都返回 None。

    坐标与场景分属两行，OCR 结果用空格拼接后一并解析。
    """
    coords = parse_coords(text)
    scene = parse_scene(text)
    if coords is None or not scene:
        return None
    return HudState(coords[0], coords[1], scene)


def scene_matches(scene: str, target: str) -> bool:
    """场景名模糊一致（双向包含，去空白标点）。"""
    s, t = normalize(scene), normalize(target)
    return bool(s) and bool(t) and (s in t or t in s)


def coord_close(a: tuple[int, int], b: tuple[int, int], tol: int) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
