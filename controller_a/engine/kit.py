"""步骤执行工具包：把截图/识别/动作等服务聚合在一起，供各步骤使用。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from action_client import ActionClient
from coord_mapper import CoordMapper
from engine.regions import RegionBook
from services.npc_service import NpcService, NpcServiceConfig
from vision.capture import ScreenCapture
from vision.ocr_engine import OcrEngine


class StepStopped(RuntimeError):
    """用户请求停止任务。"""


class Kit:
    def __init__(self, capture: ScreenCapture, mapper: CoordMapper,
                 client: ActionClient, ocr: OcrEngine,
                 npc_config: NpcServiceConfig,
                 stop_event: threading.Event,
                 log: Callable[[str], None],
                 region_book: RegionBook | None = None) -> None:
        self.capture = capture
        self.mapper = mapper
        self.client = client
        self.ocr = ocr
        self.npc_config = npc_config
        self.stop_event = stop_event
        self._log = log
        self.region_book = region_book

    def log(self, msg: str) -> None:
        self._log(msg)

    def check_stop(self) -> None:
        if self.stop_event.is_set():
            raise StepStopped("任务已被用户停止")

    def sleep(self, secs: float) -> None:
        """可中断睡眠：停止信号到来时立即抛出 StepStopped。"""
        end = time.monotonic() + max(0.0, secs)
        while True:
            self.check_stop()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            self.stop_event.wait(min(0.05, remaining))

    def new_npc_service(self) -> NpcService:
        return NpcService(
            self.capture, self.mapper, self.client, ocr=self.ocr,
            config=self.npc_config, log=self._log,
        )

    def grab_dims(self) -> tuple[int, int]:
        frame = self.capture.grab()
        return frame, frame.shape[1], frame.shape[0]

    def click_capture_point(self, u: float, v: float,
                            img_w: int, img_h: int) -> tuple[int, int]:
        """截图坐标 → B 屏坐标 → 下发点击。"""
        x, y = self.mapper.to_b(u, v, img_w, img_h)
        self.client.click_at(
            x, y, self.mapper.b_width, self.mapper.b_height)
        return x, y
