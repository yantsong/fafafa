"""B 端 ch9329.scroll_at 滚轮报文单测（无需真实 CH9329 硬件）。

覆盖：
1. _absolute_mouse_packet 纯函数：帧头/CMD/LEN、4096 比例坐标、
   wheel 字节（+1 上滚 / 0xFF 下滚 / 0x00 清零）、校验和；
2. scroll_at 端到端：每格 = 事件包 + 清零包，坐标一致，串口最终关闭；
3. 参数校验：direction/ticks 非法抛 ValueError。
"""

from __future__ import annotations

import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent_b"))

# ── 注入硬件相关桩模块 ──────────────────────────────────────
written: list[bytes] = []


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.is_open = True

    def write(self, data: bytes) -> int:
        written.append(bytes(data))
        return len(data)

    def close(self) -> None:
        self.is_open = False


fake_serial = types.ModuleType("serial")
fake_serial.Serial = FakeSerial
fake_serial.ser = None
sys.modules["serial"] = fake_serial


class FakeDataComm:
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.sent = 0

    def send_data_absolute(self, x, y, *args, **kwargs):
        self.sent += 1


fake_mouse = types.ModuleType("ch9329Comm.mouse")
fake_mouse.DataComm = FakeDataComm
fake_comm = types.ModuleType("ch9329Comm")
fake_comm.mouse = fake_mouse
sys.modules["ch9329Comm"] = fake_comm
sys.modules["ch9329Comm.mouse"] = fake_mouse

fake_pyautogui = types.ModuleType("pyautogui")
fake_pyautogui.FAILSAFE = True
fake_pyautogui.position = lambda: (0, 0)
sys.modules["pyautogui"] = fake_pyautogui

fake_windmouse = types.ModuleType("windmouse")
fake_core = types.ModuleType("windmouse.core")


def _fake_wind_mouse(*args, **kwargs):
    # 返回起点+终点两个点即可，_send_trajectory 只负责逐点发送
    return [(args[0], args[1]), (args[2], args[3])]


fake_core.wind_mouse = _fake_wind_mouse
fake_windmouse.core = fake_core
sys.modules["windmouse"] = fake_windmouse
sys.modules["windmouse.core"] = fake_core

import ch9329  # noqa: E402

# 去掉拟人随机等待，测试要快
ch9329.WHEEL_TICK_GAP_RANGE = (0, 0)
ch9329.WHEEL_TICK_INTERVAL_RANGE = (0, 0)
ch9329.CLICK_PRE_DELAY_RANGE = (0, 0)


def _checksum(data: bytes) -> int:
    return (0x57 + 0xAB + 0x00 + 0x04 + 0x07 + sum(data)) & 0xFF


class WheelPacketTest(unittest.TestCase):
    def test_01_packet_header_and_checksum_origin(self) -> None:
        pkt = ch9329._absolute_mouse_packet(0, 0, 1920, 1080, wheel=-1)
        self.assertEqual(len(pkt), 13)
        self.assertEqual(pkt[:5], b"\x57\xAB\x00\x04\x07")
        self.assertEqual(pkt[5], 0x02)          # 鼠标数据标识
        self.assertEqual(pkt[6], 0x00)          # buttons
        self.assertEqual(pkt[7:11], b"\x00\x00\x00\x00")  # x/y=0
        self.assertEqual(pkt[11], 0xFF)         # 下滚 -1 补码
        self.assertEqual(pkt[12], _checksum(pkt[5:12]))

    def test_02_packet_center_coords_and_up_wheel(self) -> None:
        pkt = ch9329._absolute_mouse_packet(960, 540, 1920, 1080,
                                            buttons=1, wheel=1)
        self.assertEqual(pkt[6], 0x01)          # 左键按下
        self.assertEqual(pkt[7:9], b"\x00\x08")  # x=2048
        self.assertEqual(pkt[9:11], b"\x00\x08")  # y=2048
        self.assertEqual(pkt[11], 0x01)         # 上滚 +1
        self.assertEqual(pkt[12], _checksum(pkt[5:12]))

    def test_03_packet_clamps_to_screen_edge(self) -> None:
        # 超屏坐标先夹到屏内：x→1919 换算 4093=0x0FFD，y→1079 换算 4092=0x0FFC
        pkt = ch9329._absolute_mouse_packet(9999, 9999, 1920, 1080)
        self.assertEqual(pkt[7:9], b"\xFD\x0F")
        self.assertEqual(pkt[9:11], b"\xFC\x0F")
        self.assertEqual(pkt[11], 0x00)         # 普通包 wheel=0

    def test_04_scroll_down_emits_event_and_clear_pairs(self) -> None:
        written.clear()
        ch9329.scroll_at(x=960, y=540, screen_w=1920, screen_h=1080,
                         direction="down", ticks=2)
        self.assertEqual(len(written), 4)       # 每格 事件+清零
        for i, pkt in enumerate(written):
            self.assertEqual(pkt[:5], b"\x57\xAB\x00\x04\x07")
            self.assertEqual(pkt[7:9], b"\x00\x08")
            self.assertEqual(pkt[9:11], b"\x00\x08")
        self.assertEqual([p[11] for p in written],
                         [0xFF, 0x00, 0xFF, 0x00])
        self.assertIsNotNone(fake_serial.ser)
        self.assertFalse(fake_serial.ser.is_open)

    def test_05_scroll_up_wheel_byte(self) -> None:
        written.clear()
        ch9329.scroll_at(x=100, y=200, screen_w=1920, screen_h=1080,
                         direction="up", ticks=1)
        self.assertEqual(len(written), 2)
        self.assertEqual(written[0][11], 0x01)
        self.assertEqual(written[1][11], 0x00)

    def test_06_invalid_args_raise(self) -> None:
        with self.assertRaises(ValueError):
            ch9329.scroll_at(direction="left")
        with self.assertRaises(ValueError):
            ch9329.scroll_at(ticks=0)
        with self.assertRaises(ValueError):
            ch9329.scroll_at(ticks=21)


if __name__ == "__main__":
    unittest.main(verbosity=2)
