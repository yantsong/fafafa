"""电脑 A 侧的动作客户端：通过 TCP 向 B 机执行器下发鼠标指令。

- 长连接，connect 一次即可；断线时自动尝试重连（可在 connect 时控制）
- 请求/响应按 id 配对；点击动作 B 机执行耗时数百毫秒，超时默认给宽一点
- 所有坐标均为 B 屏绝对坐标，由 coord_mapper 换算后传入
- 线程安全：任务引擎后续会多线程调用，内部加锁
"""

from __future__ import annotations

import socket
import threading
import time

from protocol import VALID_BUTTONS, decode_line, encode_message, DEFAULT_PORT


class ActionError(RuntimeError):
    """B 机返回失败或通信失败。"""


class ActionClient:
    def __init__(self, host: str, port: int = DEFAULT_PORT,
                 timeout: float = 10.0, connect_timeout: float = 3.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self._sock: socket.socket | None = None
        self._recv_buf = b""
        self._next_id = 0
        self._lock = threading.RLock()

    # ── 连接管理 ──────────────────────────────────────────

    def connect(self) -> None:
        with self._lock:
            self._close_locked()
            try:
                sock = socket.create_connection((self.host, self.port),
                                                timeout=self.connect_timeout)
            except OSError as exc:
                raise ActionError(
                    f"无法连接 B 机执行器 {self.host}:{self.port}: {exc}"
                ) from exc
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.timeout)
            self._sock = sock
            self._recv_buf = b""

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def ensure_connected(self, retries: int = 2) -> None:
        """未连接或连接失效时重连。"""
        with self._lock:
            if self._sock is not None:
                return
            last_exc: Exception | None = None
            for _ in range(max(1, retries)):
                try:
                    self.connect()
                    return
                except OSError as exc:
                    last_exc = exc
                    time.sleep(0.3)
            raise ActionError(f"无法连接 B 机执行器 {self.host}:{self.port}: {last_exc}")

    # ── 指令 ──────────────────────────────────────────────

    def ping(self) -> bool:
        self._request({"cmd": "ping"}, timeout=3.0)
        return True

    def move_to(self, x: int, y: int, screen_w: int, screen_h: int) -> None:
        self._request({
            "cmd": "move", "x": int(x), "y": int(y),
            "screen_w": int(screen_w), "screen_h": int(screen_h),
        })

    def click_at(self, x: int, y: int, screen_w: int, screen_h: int,
                 button: str = "LE") -> None:
        if button not in VALID_BUTTONS:
            raise ActionError(f"非法按键: {button!r}，可选 {VALID_BUTTONS}")
        self._request({
            "cmd": "click", "x": int(x), "y": int(y),
            "screen_w": int(screen_w), "screen_h": int(screen_h),
            "button": button,
        })

    def send_keys(self, keys: str, times: int = 1) -> None:
        """发送组合键，keys 如 'alt+2'、'ctrl+tab'、'tab'；times 1~10 连按。

        键位合法性由 B 机校验，这里只做基本防御。
        """
        if not isinstance(keys, str) or not keys.strip():
            raise ActionError("keys 不能为空，例如 'alt+2'")
        times = int(times)
        if not 1 <= times <= 10:
            raise ActionError("times 必须在 1~10 之间")
        self._request({"cmd": "key", "keys": keys.strip(), "times": times})

    def scroll(self, x: int, y: int, screen_w: int, screen_h: int,
               direction: str = "down", ticks: int = 1) -> None:
        """在 (x, y) 处滚动滚轮（B 机会先拟人移动过去再滚）。

        direction: 'down' 下滚 / 'up' 上滚；ticks: 1~20 格。
        用于小地图 NPC 列表等可滚动区域翻页。
        """
        d = str(direction).lower()
        if d not in ("up", "down"):
            raise ActionError("direction 只支持 'up'/'down'")
        ticks = int(ticks)
        if not 1 <= ticks <= 20:
            raise ActionError("ticks 必须在 1~20 之间")
        self._request({
            "cmd": "wheel", "x": int(x), "y": int(y),
            "screen_w": int(screen_w), "screen_h": int(screen_h),
            "direction": d, "ticks": ticks,
        })

    # ── 收发实现 ──────────────────────────────────────────

    def _request(self, payload: dict, timeout: float | None = None) -> dict:
        with self._lock:
            if self._sock is None:
                raise ActionError("尚未连接 B 机执行器")
            self._next_id += 1
            msg_id = self._next_id
            payload = dict(payload, id=msg_id)
            self._sock.settimeout(timeout or self.timeout)
            try:
                self._sock.sendall(encode_message(payload))
                resp = self._read_response_locked(msg_id)
            except (OSError, ActionError):
                self._close_locked()
                raise
            if not resp.get("ok"):
                raise ActionError(resp.get("error", "B 机返回未知错误"))
            return resp

    def _read_response_locked(self, msg_id: int) -> dict:
        assert self._sock is not None
        while True:
            line, self._recv_buf = self._read_line_locked()
            resp = decode_line(line)
            if resp.get("id") != msg_id:
                # 不应出现（严格请求-响应），丢弃错位消息继续等
                continue
            return resp

    def _read_line_locked(self) -> tuple[bytes, bytes]:
        assert self._sock is not None
        while b"\n" not in self._recv_buf:
            data = self._sock.recv(4096)
            if not data:
                raise ActionError("B 机执行器已断开连接")
            self._recv_buf += data
        line, rest = self._recv_buf.split(b"\n", 1)
        return line, rest
