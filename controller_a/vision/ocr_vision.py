"""macOS Vision 框架 OCR 后端（简体中文 + 英文）。

- 无需下载模型，系统自带，速度快
- Vision bbox 为归一化坐标、原点在左下角，这里统一转成左上原图像素坐标
- recognize() 同步执行，可在后台线程调用
"""

from __future__ import annotations

import numpy as np
import Quartz
from Vision import (
    VNImageRequestHandler,
    VNRecognizeTextRequest,
)

# VNRequestTextRecognitionLevel: accurate=1, fast=0
_RECOGNITION_LEVEL_ACCURATE = 1

from vision.ocr_engine import OcrEngine, OcrResult


class VisionOcr(OcrEngine):
    def __init__(self, languages: tuple[str, ...] = ("zh-Hans", "en"),
                 accurate: bool = True) -> None:
        self.languages = list(languages)
        self.accurate = accurate
        # macOS 15：accurate 级别仅支持拉丁语言，中文只有 fast 级别支持，
        # 因此按"支持全部请求语言的最高级别"自动选择。
        self._level = self._resolve_level()

    def _supports(self, level: int) -> bool:
        probe = VNRecognizeTextRequest.alloc().init()
        probe.setRecognitionLevel_(level)
        supported, _err = probe.supportedRecognitionLanguagesAndReturnError_(None)
        if not supported:
            return False
        supported = list(supported)

        def lang_ok(req: str) -> bool:
            # 请求语言按前缀匹配（zh-Hans 精确，en 匹配 en-US）
            return any(s == req or s.split("-")[0] == req for s in supported)

        return all(lang_ok(lang) for lang in self.languages)

    def _resolve_level(self) -> int:
        if self.accurate and self._supports(_RECOGNITION_LEVEL_ACCURATE):
            return _RECOGNITION_LEVEL_ACCURATE
        return 0  # fast

    def recognize(self, image_bgr: np.ndarray) -> list[OcrResult]:
        h, w = image_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        cg_image = self._bgr_to_cgimage(image_bgr)
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLanguages_(self.languages)
        request.setRecognitionLevel_(self._level)
        request.setUsesLanguageCorrection_(True)

        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        ok, err = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f"Vision OCR 执行失败: {err}")

        results: list[OcrResult] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            text = str(candidate.string())
            confidence = float(candidate.confidence())
            box = observation.boundingBox()  # 归一化，左下原点
            bx = float(box.origin.x)
            by = float(box.origin.y)
            bw = float(box.size.width)
            bh = float(box.size.height)

            x = int(round(bx * w))
            width = int(round(bw * w))
            # 左下原点 -> 左上原点：顶边 y = (1 - by - bh) * h
            top = int(round((1.0 - by - bh) * h))
            height = int(round(bh * h))
            if width <= 1 or height <= 1:
                continue
            results.append(OcrResult(
                text=text,
                x=max(0, x), y=max(0, top),
                w=min(width, w - x), h=min(height, h - top),
                confidence=confidence,
            ))
        return results

    @staticmethod
    def _bgr_to_cgimage(image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        # BGRA（Little-Endian RGBA），供 kCGImageAlphaNoneSkipLast
        bgra = np.dstack([
            image_bgr,
            np.full((h, w), 255, dtype=np.uint8),
        ])
        # 保证连续内存
        bgra = np.ascontiguousarray(bgra)
        bytes_per_row = w * 4
        provider = Quartz.CGDataProviderCreateWithData(
            None, bgra, bytes_per_row * h, None
        )
        colorspace = Quartz.CGColorSpaceCreateDeviceRGB()
        return Quartz.CGImageCreate(
            w, h, 8, 32, bytes_per_row, colorspace,
            Quartz.kCGImageAlphaNoneSkipLast,
            provider, None, False,
            Quartz.kCGRenderingIntentDefault,
        )
