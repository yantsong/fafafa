"""截图坐标 ↔ 电脑 B 屏幕坐标 的映射、配置读写与校准验证。

映射模型（线性归一化）：
    前提：截图区域恰好完整覆盖电脑 B 的整个屏幕内容（校准时框选）。
        x_B = u * B_w / img_w
        y_B = v * B_h / img_h
    只依赖截图的实际像素尺寸，不依赖 uu 窗口在 A 屏上的位置，
    天然兼容非 1:1 缩放和 Retina 高分屏。

校准验证（无需模板）：
    通过 CH9329 把 B 的鼠标移到若干已知 B 屏坐标，对移动前后两帧做
    差分定位光标，把光标位置反算回 B 屏坐标，统计与目标点的误差。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable

import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULT_CONFIG: dict = {
    "agent": {"host": "10.219.18.34", "port": 5093},
    "b_screen": {"width": 1920, "height": 1080},
    "capture": {"left": 0, "top": 0, "width": 0, "height": 0},
}


def load_config(path: str = CONFIG_PATH) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for section, values in data.items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    return cfg


def save_config(cfg: dict, path: str = CONFIG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


class CoordMapper:
    """截图像素坐标 (u, v) 与 B 屏坐标 (x, y) 的双向换算。"""

    def __init__(self, b_width: int, b_height: int) -> None:
        self.b_width = int(b_width)
        self.b_height = int(b_height)
        if self.b_width <= 0 or self.b_height <= 0:
            raise ValueError("B 屏分辨率必须为正数")

    def to_b(self, u: float, v: float, img_w: int, img_h: int) -> tuple[int, int]:
        """截图坐标 → B 屏坐标（裁剪到屏幕范围内）。"""
        x = round(u * self.b_width / img_w)
        y = round(v * self.b_height / img_h)
        x = max(0, min(self.b_width - 1, x))
        y = max(0, min(self.b_height - 1, y))
        return x, y

    def to_capture(self, x: float, y: float, img_w: int, img_h: int) -> tuple[float, float]:
        """B 屏坐标 → 截图坐标（用于把识别结果画回预览图）。"""
        return x * img_w / self.b_width, y * img_h / self.b_height

    def check_aspect(self, img_w: int, img_h: int, tol: float = 0.03) -> bool:
        """检查截图宽高比与 B 分辨率是否一致。

        偏差超过 tol 说明框选区域没有精确对齐 B 的屏幕内容
        （框大了含边框、框小了裁掉边缘），映射会有系统性偏差。
        """
        if img_w <= 0 or img_h <= 0:
            return False
        img_ratio = img_w / img_h
        b_ratio = self.b_width / self.b_height
        return abs(img_ratio - b_ratio) / b_ratio <= tol


def locate_cursor_by_diff(
    img_before: np.ndarray,
    img_after: np.ndarray,
    expected_uv: tuple[float, float] | None = None,
    diff_thresh: int = 30,
    min_area: float = 4.0,
    max_search_px: float = 250.0,
) -> tuple[tuple[float, float] | None, list[tuple[float, float]]]:
    """帧差法定位光标：比较移动前后两帧，找发生变化的部分。

    expected_uv: 光标理论位置（截图坐标）。给出时从候选中取最近的，
    且距离超过 max_search_px 视为定位失败（避免游戏动画干扰）。
    返回 (最优光标位置或 None, 全部候选位置列表)。
    """
    import cv2

    if img_before.shape != img_after.shape:
        return None, []

    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_before, gray_after)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    # 闭运算填平碎片缝隙但基本不改变外包围盒（dilate 会让左上角偏小引入偏差）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        # 光标热点在箭头尖端（包围盒左上角），用左上角而非质心可消除形状偏差
        bx, by, _bw, _bh = cv2.boundingRect(contour)
        candidates.append((float(bx), float(by)))

    if not candidates:
        return None, []

    if expected_uv is None:
        return candidates[0], candidates

    candidates.sort(key=lambda p: (p[0] - expected_uv[0]) ** 2 + (p[1] - expected_uv[1]) ** 2)
    best = candidates[0]
    if math.hypot(best[0] - expected_uv[0], best[1] - expected_uv[1]) > max_search_px:
        return None, candidates
    return best, candidates


def verify_mapping(
    capture,  # vision.capture.ScreenCapture
    mapper: CoordMapper,
    move_fn: Callable[[int, int], None],
    fractions: tuple[tuple[float, float], ...] = ((0.25, 0.25), (0.75, 0.5), (0.5, 0.75)),
    settle: float = 0.5,
    log: Callable[[str], None] | None = None,
) -> list[dict]:
    """自动校准验证：移动 B 鼠标到多个已知点，帧差定位光标评估映射误差。

    move_fn(x, y): 通过 CH9329 把 B 的鼠标移到 B 屏坐标 (x, y)。
    返回每个测试点的记录：目标、定位结果、误差（B 屏像素）。
    """
    import time

    results: list[dict] = []
    for fx, fy in fractions:
        tx = int(mapper.b_width * fx)
        ty = int(mapper.b_height * fy)

        img_before = capture.grab()
        move_fn(tx, ty)
        time.sleep(settle)
        img_after = capture.grab()

        h, w = img_after.shape[:2]
        expected_uv = mapper.to_capture(tx, ty, w, h)
        found_uv, _cands = locate_cursor_by_diff(img_before, img_after, expected_uv)

        record: dict = {
            "target_b": (tx, ty),
            "expected_uv": expected_uv,
            "found_uv": found_uv,
        }
        if found_uv is not None:
            found_b = mapper.to_b(found_uv[0], found_uv[1], w, h)
            record["found_b"] = found_b
            record["error_px"] = math.hypot(found_b[0] - tx, found_b[1] - ty)

        results.append(record)
        if log is not None:
            found = record.get("found_b", "未定位到")
            error = record.get("error_px")
            error_text = f"{error:.1f}px" if error is not None else "-"
            log(f"  目标B=({tx},{ty}) 定位={found} 误差={error_text}")

    return results
