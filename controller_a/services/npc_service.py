"""NPC 交互服务：OCR 找名字 → 移动到名字上方的人物身体 → hover 变色确认 → 点击。

核心机制
--------
大话西游中：
- NPC 名字在人物脚下，人物身体在名字框上方（垂直距离因 NPC 体型而异）；
- 鼠标悬停到人物上时，名字会整体变色。

因此候选点无法一次算准，本服务在名字上方用「高度档 × 左右档」网格搜索，
每移到一个候选点就比对名字区域颜色：
- 真实 hover：整块文字变色，差异像素占比高，且连续两帧稳定；
- 鼠标光标掠过：只有少量像素变化，占比低；
- 背景/人物动画：差异不稳定，连续两帧确认可滤除。

坐标约定：内部计算全部使用截图像素坐标（左上原点），
只有下发动作时才经 CoordMapper 换算为 B 屏绝对坐标。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from action_client import ActionClient, ActionError
from coord_mapper import CoordMapper
from vision.capture import ScreenCapture
from vision.npc_detector import NameMatch, find_all_npcs, find_npc
from vision.ocr_engine import OcrEngine, OcrResult, get_engine
from vision.ocr_rapid import merge_results

# 候选点网格（相对名字框尺寸的比例）
# 第一轮先用正中央各高度，第二轮再向左右扩展
HEIGHT_MULTS = (2.5, 1.8, 3.2, 1.2)
X_SHIFTS_ROUND2 = (-0.7, 0.7, -1.3, 1.3)

# hover 判定默认参数
DIFF_THRESHOLD = 30.0        # 单像素 RGB 平均色差超过该值算"变化"
CHANGED_RATIO = 0.08         # 变化像素占名字框比例超过该值算 hover
CONFIRM_FRAMES = 2           # 连续确认帧数（滤背景动画）


@dataclass
class NpcServiceConfig:
    height_mults: tuple[float, ...] = HEIGHT_MULTS
    x_shifts_round2: tuple[float, ...] = X_SHIFTS_ROUND2
    diff_threshold: float = DIFF_THRESHOLD
    changed_ratio: float = CHANGED_RATIO
    confirm_frames: int = CONFIRM_FRAMES
    settle_sec: float = 0.35       # 移动后等 hover 生效 + uu 画面传输
    frame_gap_sec: float = 0.12    # 连续确认帧之间的间隔
    safe_margin: int = 8           # 安全点距屏幕边缘像素（B 屏坐标）


@dataclass
class NpcResult:
    success: bool
    stage: str                          # not_found / no_hover / located / clicked
    message: str = ""
    name_box: tuple[int, int, int, int] | None = None  # 截图坐标 x,y,w,h
    point_capture: tuple[int, int] | None = None       # 命中点（截图坐标）
    point_screen_b: tuple[int, int] | None = None      # 命中点（B 屏坐标）
    attempts: int = 0
    best_ratio: float = 0.0
    debug: dict = field(default_factory=dict)


class NpcService:
    def __init__(self, capture: ScreenCapture, mapper: CoordMapper,
                 client: ActionClient, ocr: OcrEngine | None = None,
                 config: NpcServiceConfig | None = None,
                 log=None) -> None:
        self.capture = capture
        self.mapper = mapper
        self.client = client
        self.ocr = ocr or get_engine("vision")
        self.cfg = config or NpcServiceConfig()
        self._log = log

    def _emit(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    # ── 纯查找（不移动不点击）──────────────────────────────

    def locate(self, npc_name: str) -> NameMatch | None:
        frame, results = self.recognize_robust(npc_name)
        return find_npc(results, npc_name)

    def recognize_robust(self, target_name: str = "",
                         log=None):
        """稳健 OCR：首轮识别 → 未命中目标则换帧重试 → 仍未命中则 1.5x 放大。

        所有轮次结果做并集（同位置取高置信文本），对抗 uu 压缩逐帧抖动。
        target_name 为空时只跑首轮（诊断/全量识别场景用）。
        返回 (最后一帧图像, 合并后的 OcrResult 列表)。
        """
        emit = log or self._emit

        frame = self.capture.grab()
        results = self.ocr.recognize(frame)

        if target_name and find_npc(results, target_name) is None:
            emit("   首轮未命中，换一帧重试（对抗 uu 画面抖动）...")
            time.sleep(0.08)
            frame = self.capture.grab()
            second = self.ocr.recognize(frame)
            results = merge_results([results, second])

            if find_npc(results, target_name) is None and hasattr(
                    self.ocr, "recognize_upscaled"):
                emit("   仍未命中，1.5x 放大后再识别...")
                upscaled = self.ocr.recognize_upscaled(frame, 1.5)
                results = merge_results([results, upscaled])

        return frame, results

    # ── 完整流程 ─────────────────────────────────────────

    def find_and_click(self, npc_name: str, click: bool = True) -> NpcResult:
        frame, results = self.recognize_robust(npc_name)
        img_h, img_w = frame.shape[:2]
        matches = find_all_npcs(results, npc_name)
        if not matches:
            return NpcResult(False, "not_found", f"画面中未找到名字「{npc_name}」")

        self._emit(f"OCR 找到 {len(matches)} 个「{npc_name}」候选，"
                   f"开始 hover 搜索（名字框示例 {matches[0].w}x{matches[0].h}）")

        # 先把指针移到安全点，让所有名字恢复常态，再采基线
        sx, sy = self.mapper.to_b(
            img_w - self.cfg.safe_margin, img_h - self.cfg.safe_margin,
            img_w, img_h,
        )
        self.client.move_to(sx, sy, self.mapper.b_width, self.mapper.b_height)
        time.sleep(self.cfg.settle_sec)

        base_frame = self.capture.grab()
        attempts = 0
        best_ratio = 0.0

        for idx, match in enumerate(matches):
            baseline = self._crop(base_frame, match)
            if baseline.size == 0:
                continue
            for u, v in self._candidate_points(match):
                attempts += 1
                bx, by = self.mapper.to_b(u, v, img_w, img_h)
                try:
                    self.client.move_to(
                        bx, by, self.mapper.b_width, self.mapper.b_height
                    )
                except ActionError as exc:
                    return NpcResult(False, "error", f"移动指令失败: {exc}",
                                     attempts=attempts, best_ratio=best_ratio)
                time.sleep(self.cfg.settle_sec)

                ratio = self._measure_hover_ratio(match, baseline)
                best_ratio = max(best_ratio, ratio)
                if ratio < self.cfg.changed_ratio:
                    continue

                # 连续帧确认
                confirmed = True
                for _ in range(self.cfg.confirm_frames - 1):
                    time.sleep(self.cfg.frame_gap_sec)
                    ratio2 = self._measure_hover_ratio(match, baseline)
                    best_ratio = max(best_ratio, ratio2)
                    if ratio2 < self.cfg.changed_ratio:
                        confirmed = False
                        break
                if not confirmed:
                    continue

                self._emit(f"hover 确认：第 {attempts} 个候选点命中 "
                           f"(截图 {u},{v} / B屏 {bx},{by}，变色占比 {ratio:.2f})")
                if click:
                    # 当前指针已在目标上；click 会重新走极短轨迹（距离≈0）后点击
                    self.client.click_at(
                        bx, by, self.mapper.b_width, self.mapper.b_height
                    )
                    stage = "clicked"
                    message = f"已点击「{npc_name}」，尝试 {attempts} 次"
                else:
                    stage = "located"
                    message = f"已悬停「{npc_name}」，未点击"
                return NpcResult(
                    True, stage, message,
                    name_box=(match.x, match.y, match.w, match.h),
                    point_capture=(u, v), point_screen_b=(bx, by),
                    attempts=attempts, best_ratio=best_ratio,
                )

            if idx + 1 < len(matches):
                self._emit(f"第 {idx + 1} 个同名 NPC 未触发 hover，尝试下一个...")
                # 重新采基线，避免指针停在上一个身体上影响后续比较
                nx, ny = self.mapper.to_b(
                    img_w - self.cfg.safe_margin, img_h - self.cfg.safe_margin,
                    img_w, img_h,
                )
                self.client.move_to(nx, ny, self.mapper.b_width, self.mapper.b_height)
                time.sleep(self.cfg.settle_sec)
                base_frame = self.capture.grab()

        return NpcResult(
            False, "no_hover",
            f"找到名字但 {attempts} 个候选点均未触发 hover（最大变色占比 "
            f"{best_ratio:.2f}，阈值 {self.cfg.changed_ratio:.2f}），"
            f"可调高度档或阈值",
            name_box=(matches[0].x, matches[0].y, matches[0].w, matches[0].h),
            attempts=attempts, best_ratio=best_ratio,
        )

    # ── 内部工具 ─────────────────────────────────────────

    def _candidate_points(self, m: NameMatch):
        """生成候选点（截图像素坐标）：先中央各高度，再左右扩展。"""
        cx = m.x + m.w // 2
        points: list[tuple[int, int]] = []
        for mult in self.cfg.height_mults:
            points.append((cx, m.y - int(mult * m.h)))
        for shift in self.cfg.x_shifts_round2:
            ux = cx + int(shift * m.w)
            # 第二轮用最可能的前两个高度
            for mult in self.cfg.height_mults[:2]:
                points.append((ux, m.y - int(mult * m.h)))
        return points

    @staticmethod
    def _crop(frame: np.ndarray, m: NameMatch) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, m.x)
        y1 = max(0, m.y)
        x2 = min(w, m.x + m.w)
        y2 = min(h, m.y + m.h)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        # 名字框四周扩边，避免 OCR 框切边导致变色像素落在框外；
        # 边距随框高缩放（2x Retina 物理像素下固定 2px 太小）
        pad = max(2, m.h // 8)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return frame[y1:y2, x1:x2]

    def _measure_hover_ratio(self, m: NameMatch,
                             baseline: np.ndarray) -> float:
        """重新截图并与该名字框的常态基线比较，返回变化像素占比。"""
        frame = self.capture.grab()
        now = self._crop(frame, m)
        if now.size == 0 or now.shape != baseline.shape:
            return 0.0
        diff = np.abs(
            now.astype(np.int16) - baseline.astype(np.int16)
        ).mean(axis=2)
        return float((diff > self.cfg.diff_threshold).mean())
