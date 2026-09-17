"""OCR 统一适配层。

业务层只消费统一结构 OcrResult，不关心底层引擎（macOS Vision / PaddleOCR）。
坐标全部基于输入图像的像素坐标系，原点左上角，x 向右、y 向下。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OcrResult:
    """一段识别文本及其包围盒（像素坐标，左上原点）。"""

    text: str
    x: int
    y: int       # 顶边
    w: int
    h: int
    confidence: float = 1.0

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


class OcrEngine:
    """OCR 引擎接口：输入 BGR numpy 图像，返回 OcrResult 列表。"""

    def recognize(self, image_bgr) -> list[OcrResult]:
        raise NotImplementedError


def get_engine(name: str = "rapid") -> OcrEngine:
    """工厂：name in {'rapid', 'vision'}。

    rapid  = RapidOCR（PP-OCRv5），游戏描边小字首选，默认
    vision = macOS Vision（系统自带、无额外依赖，适合高对比常规字）
    """
    if name in ("rapid", "paddle"):
        from vision.ocr_rapid import RapidOcr
        return RapidOcr()
    if name == "vision":
        from vision.ocr_vision import VisionOcr
        return VisionOcr()
    raise ValueError(f"未知 OCR 引擎: {name!r}")


def merge_results(sets: list[list[OcrResult]],
                  iou_thresh: float = 0.3) -> list[OcrResult]:
    """合并多帧/多尺度 OCR 结果：位置高度重叠的框视为同一个文字，
    保留置信度更高的那条。用于对抗 uu 压缩逐帧抖动造成的偶发漏检。
    """
    merged: list[OcrResult] = []
    for items in sets:
        for r in items:
            best_idx = -1
            best_iou = iou_thresh
            for i, m in enumerate(merged):
                iou = _box_iou(r, m)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0:
                if r.confidence > merged[best_idx].confidence:
                    merged[best_idx] = r
            else:
                merged.append(r)
    return merged


def _box_iou(a: OcrResult, b: OcrResult) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0
