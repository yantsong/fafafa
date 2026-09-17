"""NPC 名字查找：在截图中通过 OCR 文本匹配定位 NPC 名字。

大话西游中 NPC 名字显示在人物脚下（名字底边贴近人物脚底），
人物身体位于名字框上方。本模块只负责找到名字框，
"名字 -> 人物身体" 的偏移由 npc_service 负责尝试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vision.ocr_engine import OcrResult

# OCR 常把游戏名字识别夹带的符号/空格
_PUNCT_RE = re.compile(r"[\s\u3000\[\]【】()（）<>《》*·.,，。:：;；!！?？'\"`|/\\_-]+")


def normalize(text: str) -> str:
    """归一化文本：去空白与常见标点，便于宽松匹配。"""
    return _PUNCT_RE.sub("", text or "")


@dataclass
class NameMatch:
    """匹配到的 NPC 名字（坐标均为截图像素坐标，左上原点）。"""

    name: str       # 目标名字
    text: str       # OCR 原始文本
    x: int
    y: int          # 名字框顶边
    w: int
    h: int
    confidence: float
    exact: bool     # 是否精确匹配

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def crop(self, image_bgr):
        return image_bgr[self.y:self.y + self.h, self.x:self.x + self.w]


def find_npc(results: list[OcrResult], target_name: str,
             min_confidence: float = 0.3) -> NameMatch | None:
    """在 OCR 结果中查找目标 NPC 名字。

    匹配优先级：
    1. 归一化后精确相等（最可靠）
    2. 双向包含且长度差不超过 2（容忍 OCR 多/漏一个字符）
    多个候选时：精确优先，其次置信度高、长度更接近目标。
    """
    target = normalize(target_name)
    if not target:
        return None

    candidates: list[NameMatch] = []
    for r in results:
        if r.confidence < min_confidence:
            continue
        got = normalize(r.text)
        if not got:
            continue
        if got == target:
            exact = True
        elif (target in got or got in target) and abs(len(got) - len(target)) <= 2:
            exact = False
        else:
            continue
        candidates.append(NameMatch(
            name=target_name, text=r.text,
            x=r.x, y=r.y, w=r.w, h=r.h,
            confidence=r.confidence, exact=exact,
        ))

    if not candidates:
        return None
    candidates.sort(key=lambda m: (
        not m.exact,                         # 精确匹配优先
        abs(len(normalize(m.text)) - len(target)),  # 长度更接近优先
        -m.confidence,                       # 置信度高优先
    ))
    return candidates[0]


def find_all_npcs(results: list[OcrResult], target_name: str,
                  min_confidence: float = 0.3) -> list[NameMatch]:
    """返回全部匹配（屏幕上存在同名 NPC 时使用，逐个尝试）。"""
    target = normalize(target_name)
    if not target:
        return []
    out: list[NameMatch] = []
    for r in results:
        if r.confidence < min_confidence:
            continue
        got = normalize(r.text)
        if got == target or (
            (target in got or got in target) and abs(len(got) - len(target)) <= 2
        ):
            out.append(NameMatch(
                name=target_name, text=r.text,
                x=r.x, y=r.y, w=r.w, h=r.h,
                confidence=r.confidence, exact=got == target,
            ))
    return out
