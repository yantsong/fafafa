"""A/B 两端共享的 TCP 行协议定义（两端各自保留一份相同实现）。

协议：每条消息是一行 UTF-8 JSON，以 \\n 结尾。
请求（A -> B）：
    {"id": 1, "cmd": "ping"}
    {"id": 2, "cmd": "move",  "x": 960, "y": 540, "screen_w": 1920, "screen_h": 1080}
    {"id": 3, "cmd": "click", "x": 960, "y": 540, "screen_w": 1920,
     "screen_h": 1080, "button": "LE"}
    {"id": 4, "cmd": "key", "keys": "alt+2", "times": 1}
响应（B -> A）：
    {"id": 2, "ok": true}
    {"id": 3, "ok": false, "error": "serial unavailable"}
"""

from __future__ import annotations

import json

TERMINATOR = b"\n"
DEFAULT_PORT = 5093

VALID_BUTTONS = ("LE", "RI", "CE")
VALID_CMDS = ("ping", "move", "click", "key")


def encode_message(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def decode_line(line: bytes) -> dict:
    return json.loads(line.decode("utf-8").strip())
