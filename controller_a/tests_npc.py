"""NPC 识别链路测试（合成帧 + 假 capture/client，不依赖屏幕与网络）。

覆盖：
1. npc_detector：精确/标点容忍/包含匹配、多候选择优、低置信度过滤
2. 候选点序列顺序（先中央各高度，再左右扩展）
3. find_and_click：基线对比 → 网格搜索 → 连续帧确认 → 点击，坐标换算正确
4. 光标级小干扰（低变化占比）不触发点击
5. 找不到名字 / 全部候选失败 的返回阶段
"""

from __future__ import annotations

import sys
import unittest

import numpy as np

sys.path.insert(0, "controller_a")

from coord_mapper import CoordMapper  # noqa: E402
from services.npc_service import NpcResult, NpcService, NpcServiceConfig  # noqa: E402
from vision.npc_detector import find_all_npcs, find_npc, normalize  # noqa: E402
from vision.ocr_engine import OcrResult  # noqa: E402

# 合成场景：帧 400x300（w x h），名字框
IMG_W, IMG_H = 400, 300
BOX = dict(x=100, y=200, w=80, h=20)
B_W, B_H = 1920, 1080
NORMAL_BGR = (200, 200, 200)
HOVER_BGR = (0, 255, 255)  # 黄色（BGR），与常态差异显著


class FakeCapture:
    """按脚本逐帧返回；grab 次数超过脚本长度后重复最后一帧。"""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.idx = 0

    def grab(self) -> np.ndarray:
        frame = self.frames[min(self.idx, len(self.frames) - 1)]
        self.idx += 1
        return frame.copy()


class FakeClient:
    def __init__(self) -> None:
        self.moves: list[tuple] = []
        self.clicks: list[tuple] = []

    def move_to(self, x, y, sw, sh) -> None:
        self.moves.append((x, y, sw, sh))

    def click_at(self, x, y, sw, sh, button="LE") -> None:
        self.clicks.append((x, y, sw, sh, button))


class FakeOcr:
    def __init__(self, results: list[OcrResult]) -> None:
        self._results = results

    def recognize(self, _frame) -> list[OcrResult]:
        return list(self._results)


def make_frame(box_state: str | None = "normal") -> np.ndarray:
    """box_state: None=无名字（全黑）, 'normal'=常态色, 'hover'=高亮色。"""
    frame = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    if box_state is not None:
        color = NORMAL_BGR if box_state == "normal" else HOVER_BGR
        frame[BOX["y"]:BOX["y"] + BOX["h"],
              BOX["x"]:BOX["x"] + BOX["w"]] = color
    return frame


def ocr_results(name: str = "帮派总管", conf: float = 0.9) -> list[OcrResult]:
    return [OcrResult(text=name, x=BOX["x"], y=BOX["y"],
                      w=BOX["w"], h=BOX["h"], confidence=conf)]


def make_service(frames, ocr_results_list) -> tuple[NpcService, FakeClient]:
    cfg = NpcServiceConfig(settle_sec=0.0, frame_gap_sec=0.0)
    client = FakeClient()
    svc = NpcService(
        FakeCapture(frames), CoordMapper(B_W, B_H), client,
        ocr=FakeOcr(ocr_results_list), config=cfg,
    )
    return svc, client


class DetectorTest(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(normalize(" 帮派 总管 "), "帮派总管")
        self.assertEqual(normalize("【帮派总管】"), "帮派总管")

    def test_exact_match(self) -> None:
        m = find_npc(ocr_results(), "帮派总管")
        self.assertIsNotNone(m)
        self.assertTrue(m.exact)
        self.assertEqual(m.center, (140, 210))

    def test_punctuation_tolerant(self) -> None:
        m = find_npc(ocr_results("【帮派总管】"), "帮派总管")
        self.assertIsNotNone(m)
        self.assertTrue(m.exact)  # 归一化后完全相等

    def test_one_char_diff_contains(self) -> None:
        m = find_npc(ocr_results("帮派总管大人"), "帮派总管")
        self.assertIsNotNone(m)
        self.assertFalse(m.exact)

    def test_low_confidence_filtered(self) -> None:
        self.assertIsNone(find_npc(ocr_results(conf=0.1), "帮派总管"))

    def test_prefers_exact_over_partial(self) -> None:
        results = [
            OcrResult("帮派总管大人", 0, 0, 100, 20, 0.99),
            OcrResult("帮派总管", 200, 200, 80, 20, 0.5),
        ]
        m = find_npc(results, "帮派总管")
        self.assertTrue(m.exact)
        self.assertEqual(m.x, 200)

    def test_find_all_same_name(self) -> None:
        results = [
            OcrResult("帮派总管", 0, 0, 80, 20, 0.9),
            OcrResult("帮派总管", 300, 200, 80, 20, 0.9),
            OcrResult("路人甲", 0, 100, 60, 20, 0.9),
        ]
        self.assertEqual(len(find_all_npcs(results, "帮派总管")), 2)


class CandidateSequenceTest(unittest.TestCase):
    def test_round1_center_then_round2_sides(self) -> None:
        svc, _ = make_service([make_frame()], ocr_results())
        m = find_npc(ocr_results(), "帮派总管")
        pts = svc._candidate_points(m)
        # 第一轮 4 个中央高度档
        self.assertEqual(len(pts), 4 + 4 * 2)
        cx = BOX["x"] + BOX["w"] // 2
        self.assertTrue(all(u == cx for u, _v in pts[:4]))
        # 前 4 个 v 坐标在名字框上方（v < y）
        self.assertTrue(all(v < BOX["y"] for _u, v in pts[:4]))
        # 第二轮含左右偏移
        round2_u = {u for u, _v in pts[4:]}
        self.assertTrue(any(u != cx for u in round2_u))


class FindAndClickFlowTest(unittest.TestCase):
    def test_hover_on_second_candidate_then_click(self) -> None:
        # 帧序列：#0 OCR帧(normal) #1 基线(normal)
        #   候选1 measure #2 normal（不触发）
        #   候选2 measure #3 hover -> confirm #4 hover
        frames = [
            make_frame("normal"),
            make_frame("normal"),
            make_frame("normal"),
            make_frame("hover"),
            make_frame("hover"),
        ]
        svc, client = make_service(frames, ocr_results())
        result = svc.find_and_click("帮派总管", click=True)

        self.assertEqual(result.stage, "clicked")
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        # 候选2 = 中央、高度倍数 1.8：v = 200 - int(1.8*20) = 164
        expected_u, expected_v = 140, 200 - int(1.8 * 20)
        self.assertEqual(result.point_capture, (expected_u, expected_v))
        # B 屏换算：x = 140*1920/400 = 672；y = 164*1080/300 = 590
        self.assertEqual(result.point_screen_b, (672, 590))
        self.assertEqual(len(client.clicks), 1)
        self.assertEqual(client.clicks[0][:2], (672, 590))
        # 第一个动作是移到右下角安全点（margin 作用于截图坐标后再映射）
        safe_bx = round((IMG_W - 8) * B_W / IMG_W)
        safe_by = round((IMG_H - 8) * B_H / IMG_H)
        self.assertEqual(client.moves[0],
                         (safe_bx, safe_by, B_W, B_H))

    def test_hover_requires_second_frame_confirm(self) -> None:
        # measure 帧 hover 但 confirm 帧恢复 normal → 不点击，继续搜索后失败
        frames = [make_frame("normal"), make_frame("normal")]
        # 候选1：hover 后 normal；其后全部 normal
        frames += [make_frame("hover"), make_frame("normal")]
        frames += [make_frame("normal")] * 30
        svc, client = make_service(frames, ocr_results())
        result = svc.find_and_click("帮派总管", click=True)
        self.assertEqual(result.stage, "no_hover")
        self.assertEqual(len(client.clicks), 0)
        self.assertGreater(result.attempts, 1)

    def test_cursor_sized_disturbance_not_triggered(self) -> None:
        # 名字框内仅 1 个像素变化（模拟光标掠过），占比远低于阈值
        baseline = make_frame("normal")
        disturb = make_frame("normal")
        disturb[BOX["y"], BOX["x"]] = HOVER_BGR
        frames = [baseline, baseline] + [disturb] * 30
        svc, client = make_service(frames, ocr_results())
        result = svc.find_and_click("帮派总管", click=True)
        self.assertEqual(result.stage, "no_hover")
        self.assertEqual(len(client.clicks), 0)

    def test_name_not_found(self) -> None:
        frames = [make_frame(None)]
        svc, client = make_service(frames, [])
        result = svc.find_and_click("不存在")
        self.assertEqual(result.stage, "not_found")
        self.assertEqual(client.moves, [])  # 未找到名字不应移动鼠标
        self.assertIsInstance(result, NpcResult)

    def test_hover_mode_does_not_click(self) -> None:
        frames = [
            make_frame("normal"), make_frame("normal"),
            make_frame("normal"),
            make_frame("hover"), make_frame("hover"),
        ]
        svc, client = make_service(frames, ocr_results())
        result = svc.find_and_click("帮派总管", click=False)
        self.assertEqual(result.stage, "located")
        self.assertEqual(len(client.clicks), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
