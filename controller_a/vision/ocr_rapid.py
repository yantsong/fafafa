"""RapidOCR 后端（PP-OCRv5 中文模型，mainline rapidocr 包）。

选型依据（在真实大话西游 2x Retina 截图上实测对比）：
- 对黄/粉描边艺术字、复杂石板背景的检测能力远强于 macOS Vision；
- PP-OCRv5 中文字符识别准确率显著优于旧 rapidocr-onnxruntime(v3)，
  如「百舸争流@2B青年」v3 读成「百前争流@28青年」，v5 置信度 0.97 正确。

检测参数相对默认值的调整：
- box_thresh 0.5→0.35：检出更暗的描边小字，实测「赛天成」13/13 帧稳定检出；
- unclip_ratio 1.6→1.9：描边字笔画外扩，避免框切边；
- text_score 0.5→0.35：保留低置信候选，由业务层做名字匹配，不漏字。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from vision.ocr_engine import OcrEngine, OcrResult, merge_results

# rapidocr 默认把模型加载信息打到 INFO，业务运行时静默到 WARNING
logging.getLogger("RapidOCR").setLevel(logging.WARNING)

__all__ = ["RapidOcr", "merge_results"]

# 检测/识别调优参数（实测于真实游戏帧）
BOX_THRESH = 0.35
UNCLIP_RATIO = 1.9
TEXT_SCORE = 0.35


class RapidOcr(OcrEngine):
    def __init__(self) -> None:
        from rapidocr import RapidOCR

        # mainline 的 box_thresh/unclip_ratio/text_score 是按次调用参数，
        # 不能在构造函数传（见 _run）
        self._engine = RapidOCR()

    def _run(self, image_bgr, scale: float = 1.0) -> list[OcrResult]:
        img = image_bgr
        if scale != 1.0:
            img = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LANCZOS4)
        out = self._engine(
            img,
            box_thresh=BOX_THRESH,
            unclip_ratio=UNCLIP_RATIO,
            text_score=TEXT_SCORE,
        )
        if out is None or out.boxes is None:
            return []
        results: list[OcrResult] = []
        for box, text, score in zip(out.boxes, out.txts, out.scores):
            box = np.asarray(box, dtype=np.float64)
            x1, y1 = box.min(axis=0)
            x2, y2 = box.max(axis=0)
            results.append(OcrResult(
                text=text,
                x=int(round(x1 / scale)),
                y=int(round(y1 / scale)),
                w=int(round((x2 - x1) / scale)),
                h=int(round((y2 - y1) / scale)),
                confidence=float(score),
            ))
        return results

    def recognize(self, image_bgr) -> list[OcrResult]:
        return self._run(image_bgr, scale=1.0)

    def recognize_upscaled(self, image_bgr, scale: float = 1.5) -> list[OcrResult]:
        """放大后再识别：对更小/更暗的文字的补救通道（耗时约翻倍）。"""
        return self._run(image_bgr, scale=scale)
