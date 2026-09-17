"""多尺度模板匹配：在游戏画面里找「没有名字、只有模型」的 NPC / 物品图标。

为什么多尺度：A 端截图是 Retina 2x 物理像素，UU 窗口缩放比例变化时，
模板与实时帧中目标的像素尺度可能不完全一致，按 0.8~1.2 倍逐档缩放模板
（TM_CCOEFF_NORMED 归一化相关系数）取最高分，可容忍 ±20% 尺度差。

调用方应先用 game_window 的归一化 ROI 把搜索范围裁到游戏画面内
（排除黑边/UU 边框/控制台窗口），本模块只负责在给定搜索图上滑窗匹配。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_SCALES = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
DEFAULT_THRESHOLD = 0.72


@dataclass
class TemplateHit:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


def load_template(path: str) -> np.ndarray | None:
    """读取模板图（BGR）。文件不存在/损坏返回 None，由调用方决定如何报错。"""
    templ = cv2.imread(path, cv2.IMREAD_COLOR)
    if templ is None or templ.size == 0:
        return None
    return templ


def match_template(search_bgr: np.ndarray,
                   templ_bgr: np.ndarray,
                   threshold: float = DEFAULT_THRESHOLD,
                   scales=DEFAULT_SCALES) -> TemplateHit | None:
    """在 search_bgr 上做多尺度模板匹配，返回超过阈值的最高分命中。

    返回的 x/y/w/h 是相对 search_bgr 的坐标；模板比搜索图大的尺度自动跳过。
    """
    if search_bgr is None or search_bgr.size == 0:
        return None
    search_gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(templ_bgr, cv2.COLOR_BGR2GRAY)
    sh, sw = search_gray.shape[:2]

    best: TemplateHit | None = None
    for scale in scales:
        if abs(scale - 1.0) < 1e-6:
            scaled = templ_gray
        else:
            scaled = cv2.resize(
                templ_gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1.0
                else cv2.INTER_LINEAR)
        th, tw = scaled.shape[:2]
        if th >= sh or tw >= sw:
            continue
        result = cv2.matchTemplate(search_gray, scaled,
                                   cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best.score:
            best = TemplateHit(int(max_loc[0]), int(max_loc[1]),
                               int(tw), int(th), float(max_val), float(scale))

    if best is not None and best.score >= threshold:
        return best
    return None
