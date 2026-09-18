"""基于 HSV 色彩的文字区域检测。

用途：
  - dialog_choose：对话框选项是绿色文字（RGB 12,244,72），白色是描述文字。
    用绿色像素验证 OCR 命中的文字确实是可点选项，排除白色描述误匹配。
  - check_quest_accepted：接取成功后对话框区和任务追踪栏出现红色文字
    （RGB 255,0,0），用红色像素计数做双区域确认。

不依赖 OCR——只做色彩统计，速度快（<1ms/小块裁剪）。
"""

from __future__ import annotations

import cv2
import numpy as np

# ── HSV 阈值 ──────────────────────────────────────────────

# 绿色选项文字 RGB(12,244,72) → HSV H≈68 S≈242 V≈242
# 放宽范围以兼容抗锯齿/压缩伪影
_GREEN_LOW = np.array([55, 80, 80], np.uint8)
_GREEN_HIGH = np.array([85, 255, 255], np.uint8)

# 红色文字 RGB(255,0,0) → HSV H=0/180 S=255 V=255
# 红色在 HSV 环绕 0°，需两段
_RED_LOW1 = np.array([0, 80, 80], np.uint8)
_RED_HIGH1 = np.array([10, 255, 255], np.uint8)
_RED_LOW2 = np.array([170, 80, 80], np.uint8)
_RED_HIGH2 = np.array([180, 255, 255], np.uint8)


def _hsv_mask(bgr_crop: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """BGR 裁剪图 → HSV → inRange 二值掩码。"""
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, low, high)


def count_green_pixels(bgr_crop: np.ndarray) -> int:
    """统计绿色像素数（选项文字色）。"""
    if bgr_crop.size == 0:
        return 0
    return int(np.count_nonzero(_hsv_mask(bgr_crop, _GREEN_LOW, _GREEN_HIGH)))


def count_red_pixels(bgr_crop: np.ndarray) -> int:
    """统计红色像素数（接取成功确认色，两段 HSV 合并）。"""
    if bgr_crop.size == 0:
        return 0
    m1 = _hsv_mask(bgr_crop, _RED_LOW1, _RED_HIGH1)
    m2 = _hsv_mask(bgr_crop, _RED_LOW2, _RED_HIGH2)
    return int(np.count_nonzero(m1 | m2))


def has_green_text(bgr_crop: np.ndarray, threshold: int = 5) -> bool:
    """该裁剪图中是否有绿色文字（像素数 > threshold）。"""
    return count_green_pixels(bgr_crop) > threshold


def has_red_text(bgr_crop: np.ndarray, threshold: int = 5) -> bool:
    """该裁剪图中是否有红色文字（像素数 > threshold）。"""
    return count_red_pixels(bgr_crop) > threshold
