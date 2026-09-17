"""对话框与屏幕文字定位。

对话/选项在游戏画面中的位置不固定，但都能被 OCR 检出。
本模块提供「按关键词找一段文字 → 返回其屏幕位置」的通用能力，
步骤层据此点击对话选项、判断对话内容、判断是否到达目标场景。

ROI 一律用截图宽高的归一化分数 [x1, y1, x2, y2]（0~1），
与分辨率/缩放无关；不指定则全图搜索。
"""

from __future__ import annotations

import re

from vision.npc_detector import normalize
from vision.ocr_engine import OcrEngine, OcrResult

ROI = tuple[float, float, float, float]


def validate_roi(roi) -> ROI | None:
    if roi is None:
        return None
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError(f"roi 必须是 [x1,y1,x2,y2] 四个分数值，收到: {roi!r}")
    x1, y1, x2, y2 = (float(v) for v in roi)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"roi 分数必须满足 0<=x1<x2<=1, 0<=y1<y2<=1: {roi!r}")
    return (x1, y1, x2, y2)


def crop_roi(frame, roi: ROI | None):
    """按归一化 ROI 裁剪，返回 (子图, 偏移 px, py)。"""
    if roi is None:
        return frame, 0, 0
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    return frame[py1:py2, px1:px2], px1, py1


def recognize_in_roi(ocr: OcrEngine, frame, roi: ROI | None) -> list[OcrResult]:
    """识别 ROI 内文字，坐标换算回全图。"""
    sub, ox, oy = crop_roi(frame, roi)
    if sub.size == 0:
        return []
    results = ocr.recognize(sub)
    if ox == 0 and oy == 0:
        return results
    return [
        OcrResult(r.text, r.x + ox, r.y + oy, r.w, r.h, r.confidence)
        for r in results
    ]


def text_matches(got: str, match, use_regex: bool = False) -> bool:
    """match 支持单个字符串或字符串列表（任一命中）。"""
    if isinstance(match, (list, tuple)):
        return any(text_matches(got, m, use_regex) for m in match)
    if use_regex:
        return re.search(match, got) is not None
    return normalize(match) in normalize(got)


def find_text(results: list[OcrResult], match, use_regex: bool = False,
              min_confidence: float = 0.3) -> OcrResult | None:
    """在识别结果中找包含关键词的文字块，置信度高者优先。"""
    best: OcrResult | None = None
    for r in results:
        if r.confidence < min_confidence:
            continue
        if text_matches(r.text, match, use_regex):
            if best is None or r.confidence > best.confidence:
                best = r
    return best


def join_text(results: list[OcrResult]) -> str:
    """把 OCR 结果拼成一整段（供关键词路由/内容判断）。"""
    return " ".join(r.text for r in results)
