"""固定区域坐标簿：regions JSON 的加载与三层坐标换算。

坐标分层：
    L1 游戏窗口坐标（配置文件里的全部值，原点 = 游戏窗口左上角）
        │  + game_window 在 B 屏上的偏移
        ▼
    L2 B 屏坐标（点击用）
        │  ÷ B 屏尺寸 → 归一化 ROI
        ▼
    L3 截图坐标（OCR / 模板匹配用，步骤层拿归一化 ROI 自己裁帧）

regions JSON 结构见 regions/baobiao.json；以下划线开头的键（_说明/_用途）
一律忽略，方便在数据文件里写中文注释。
"""

from __future__ import annotations

import json
import random


class RegionConfigError(ValueError):
    """区域配置非法。"""


class RegionBook:
    def __init__(self, game_window: dict, regions: dict,
                 jitter_default: float = 0.6) -> None:
        self.gx = int(game_window["x"])
        self.gy = int(game_window["y"])
        self.gw = int(game_window["w"])
        self.gh = int(game_window["h"])
        if min(self.gw, self.gh) <= 0:
            raise RegionConfigError("game_window 的 w/h 必须为正数")
        self.regions = regions
        self.jitter_default = float(jitter_default)

    # ── 加载 ──────────────────────────────────────────────

    @classmethod
    def load(cls, path: str) -> "RegionBook":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        gw = data.get("game_window")
        if not isinstance(gw, dict) or any(
                int(gw.get(k, -1)) < 0 for k in ("x", "y", "w", "h")):
            raise RegionConfigError(
                f"{path}: game_window 缺失或仍含 -1 占位值，请先测量填写")
        raw_regions = data.get("regions", {})
        regions: dict[str, dict] = {}
        for name, raw in raw_regions.items():
            if name.startswith("_") or not isinstance(raw, dict):
                continue
            box = [int(raw[k]) for k in ("x", "y", "w", "h")]
            if any(v < 0 for v in box):
                continue  # 仍是 -1 占位的区域不注册，用时给明确报错
            jitter = raw.get("jitter")
            screen = bool(raw.get("screen", False))
            regions[name] = {
                "box": box,
                "jitter": float(jitter) if jitter is not None else None,
                "screen": screen,
            }
        defaults = data.get("defaults", {}) or {}
        return cls(gw, regions,
                   jitter_default=float(defaults.get("click_jitter_ratio", 0.6)))

    # ── 换算：L1 → L2（点击）──────────────────────────────

    def game_point_to_b(self, x: float, y: float,
                        jitter_px: float = 0.0) -> tuple[int, int]:
        """游戏窗口内单点 → B 屏坐标，可叠加 ±jitter_px 随机偏移。"""
        bx = self.gx + x + random.uniform(-jitter_px, jitter_px)
        by = self.gy + y + random.uniform(-jitter_px, jitter_px)
        return int(bx), int(by)

    def game_box_click_b(self, box, jitter: float | None = None,
                         screen: bool = False) -> tuple[int, int]:
        """[x,y,w,h] 框 → B 屏随机点击点。

        screen=False（默认）：坐标相对游戏窗口，加 game_window 偏移。
        screen=True：坐标已是 B 屏绝对坐标，直接使用（用于游戏窗口外的区域）。
        """
        x, y, w, h = (float(v) for v in box)
        j = self.jitter_default if jitter is None else float(jitter)
        j = max(0.0, min(1.0, j))
        cx, cy = x + w / 2.0, y + h / 2.0
        rx, ry = w * j / 2.0, h * j / 2.0
        jx = cx + random.uniform(-rx, rx)
        jy = cy + random.uniform(-ry, ry)
        if screen:
            return int(jx), int(jy)
        return self.game_point_to_b(jx, jy)

    def region_click_b(self, name: str) -> tuple[int, int]:
        """命名区域 → B 屏随机点击点。"""
        region = self.regions.get(name)
        if region is None:
            raise RegionConfigError(
                f"区域「{name}」未配置或仍是占位值（-1），请在 regions 文件中填写")
        return self.game_box_click_b(region["box"], region["jitter"],
                                     region.get("screen", False))

    # ── 换算：L1/L2 → 归一化（截图 OCR/模板用）────────────

    def game_roi(self, b_width: int, b_height: int):
        """整个游戏窗口在截图中的归一化 ROI [x1,y1,x2,y2]。"""
        return (self.gx / b_width,
                self.gy / b_height,
                (self.gx + self.gw) / b_width,
                (self.gy + self.gh) / b_height)

    def game_box_roi(self, box, b_width: int, b_height: int,
                     screen: bool = False):
        """[x,y,w,h] 框 → 截图归一化 ROI（OCR 子区域用）。

        screen=False（默认）：坐标相对游戏窗口，加 game_window 偏移。
        screen=True：坐标已是 B 屏绝对坐标，直接换算（用于游戏窗口外的区域）。
        """
        x, y, w, h = (float(v) for v in box)
        if screen:
            ox, oy = 0.0, 0.0
        else:
            ox, oy = self.gx, self.gy
        return ((ox + x) / b_width,
                (oy + y) / b_height,
                (ox + x + w) / b_width,
                (oy + y + h) / b_height)

    def region_roi(self, name: str, b_width: int, b_height: int):
        """命名区域 → 截图归一化 ROI。"""
        region = self.regions.get(name)
        if region is None:
            raise RegionConfigError(
                f"区域「{name}」未配置或仍是占位值（-1），请在 regions 文件中填写")
        return self.game_box_roi(region["box"], b_width, b_height,
                                 region.get("screen", False))


def load_map_file(path: str) -> dict:
    """加载 MAP JSON，剔除注释键。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}
