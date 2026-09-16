"""A(ActionClient) <-> B(ActionServer) 全链路冒烟测试（无需真实 CH9329）。

用桩模块替代 ch9329，记录收到的动作；覆盖：
正常 move/click、按键校验、坐标越界、执行异常回传且连接存活、断线重连。
"""

from __future__ import annotations

import sys
import threading
import time
import types
import unittest

sys.path.insert(0, "agent_b")
sys.path.insert(0, "controller_a")

# ── 在导入 tcp_server 之前注入 ch9329 桩 ─────────────────────
calls: list[tuple] = []
fail_next = {"flag": False}


def fake_move(com_port, baudrate, target_x, target_y, screen_w, screen_h, log=None):
    if fail_next["flag"]:
        fail_next["flag"] = False
        raise RuntimeError("simulated serial failure")
    calls.append(("move", target_x, target_y, screen_w, screen_h))


def fake_click(com_port, baudrate, x, y, screen_w, screen_h, button="LE", log=None):
    calls.append(("click", x, y, screen_w, screen_h, button))


stub = types.ModuleType("ch9329")
stub.move_to_target_humanlike = fake_move
stub.click_at = fake_click
sys.modules["ch9329"] = stub

from tcp_server import ActionServer  # noqa: E402
from action_client import ActionClient, ActionError  # noqa: E402

PORT = 5099
logs: list[str] = []
states: list[str] = []
server = ActionServer(port=PORT, com_port="COM9",
                      log=lambda m: logs.append(m),
                      status=lambda s: states.append(s))
server.start()
time.sleep(0.3)


class ChainTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        server.stop()

    def test_01_ping_move_click(self) -> None:
        c = ActionClient("127.0.0.1", PORT)
        c.connect()
        self.assertTrue(c.ping())
        c.move_to(960, 540, 1920, 1080)
        c.click_at(100, 200, 1920, 1080, button="RI")
        c.close()
        time.sleep(0.1)
        self.assertIn(("move", 960, 540, 1920, 1080), calls)
        self.assertIn(("click", 100, 200, 1920, 1080, "RI"), calls)

    def test_02_invalid_button_client_side(self) -> None:
        c = ActionClient("127.0.0.1", PORT)
        c.connect()
        with self.assertRaises(ActionError):
            c.click_at(1, 1, 1920, 1080, button="XX")
        c.close()

    def test_03_out_of_range_rejected(self) -> None:
        c = ActionClient("127.0.0.1", PORT)
        c.connect()
        with self.assertRaises(ActionError) as ctx:
            c.move_to(2000, 540, 1920, 1080)
        self.assertIn("out of screen", str(ctx.exception))
        c.close()

    def test_04_execution_error_reported_connection_alive(self) -> None:
        c = ActionClient("127.0.0.1", PORT)
        c.connect()
        fail_next["flag"] = True
        with self.assertRaises(ActionError) as ctx:
            c.move_to(10, 10, 1920, 1080)
        self.assertIn("simulated serial failure", str(ctx.exception))
        # 服务端捕获异常后连接不应中断，后续指令照常
        c.move_to(20, 20, 1920, 1080)
        c.close()
        self.assertIn(("move", 20, 20, 1920, 1080), calls)

    def test_05_reconnect_after_drop(self) -> None:
        c1 = ActionClient("127.0.0.1", PORT)
        c1.connect()
        c1.ping()
        c1.close()
        time.sleep(0.3)
        c2 = ActionClient("127.0.0.1", PORT)
        c2.connect()
        c2.move_to(30, 40, 1920, 1080)
        c2.close()
        self.assertIn(("move", 30, 40, 1920, 1080), calls)

    def test_06_connect_refused(self) -> None:
        c = ActionClient("127.0.0.1", 5999, connect_timeout=1.0)
        with self.assertRaises(ActionError):
            c.connect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
