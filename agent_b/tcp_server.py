"""电脑 B 上的 TCP 执行服务。

- 监听 TCP 端口，接受电脑 A 的长连接
- 收到 move/click 指令后调用 ch9329 执行（动作锁串行化，避免并发抢串口）
- 只服务一个控制端；连接断开后继续等待下一个连接
- 本模块不含 UI：通过 status_cb / log_cb 回调向 UI 线程上报状态和日志
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable

import ch9329
from protocol import VALID_BUTTONS, decode_line, encode_message

LogFn = Callable[[str], None]
StatusFn = Callable[[str], None]  # listening / connected / disconnected / stopped / error


class ActionServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5093,
        com_port: str = "COM3",
        baudrate: int = 115200,
        log: LogFn | None = None,
        status: StatusFn | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.com_port = com_port
        self.baudrate = baudrate
        self._log = log
        self._status = status
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._action_lock = threading.Lock()  # 串行化鼠标动作
        self._stopping = threading.Event()

    # ── 生命周期 ──────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        self._report_status("stopped")

    def update_serial(self, com_port: str, baudrate: int) -> None:
        """UI 修改串口参数后热更新（下一条指令生效）。"""
        self.com_port = com_port
        self.baudrate = baudrate

    # ── 内部实现 ──────────────────────────────────────────

    def _emit(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def _report_status(self, state: str) -> None:
        if self._status is not None:
            self._status(state)

    def _serve_loop(self) -> None:
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(0.5)
            self._sock = server
            self._emit(f"[server] listening on {self.host}:{self.port}")
            self._report_status("listening")

            while not self._stopping.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self._emit(f"[server] controller connected: {addr[0]}:{addr[1]}")
                self._report_status("connected")
                try:
                    self._handle_conn(conn)
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
                    if not self._stopping.is_set():
                        self._emit("[server] controller disconnected, waiting...")
                        self._report_status("listening")
        except OSError as exc:
            self._emit(f"[server] socket error: {exc}")
            self._report_status("error")
        finally:
            try:
                if self._sock is not None:
                    self._sock.close()
            except OSError:
                pass
            self._sock = None
            self._emit("[server] stopped")

    def _handle_conn(self, conn: socket.socket) -> None:
        conn.settimeout(None)
        buffer = b""
        while not self._stopping.is_set():
            data = conn.recv(4096)
            if not data:
                return
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                self._dispatch(conn, line)

    def _dispatch(self, conn: socket.socket, raw: bytes) -> None:
        msg_id = None
        try:
            req = decode_line(raw)
            msg_id = req.get("id")
            cmd = req.get("cmd")
            if cmd == "ping":
                self._reply(conn, msg_id, True)
                return
            if cmd == "key":
                keys = req.get("keys")
                if not isinstance(keys, str) or not keys.strip():
                    self._reply(conn, msg_id, False, "key 命令需要非空 keys")
                    return
                try:
                    times = int(req.get("times", 1))
                except (TypeError, ValueError):
                    self._reply(conn, msg_id, False, "times 必须是整数")
                    return
                # 键盘动作同样独占串口串行执行
                with self._action_lock:
                    self._emit(f"[action] key -> {keys!r} ×{times}")
                    ch9329.send_hotkey(
                        com_port=self.com_port, baudrate=self.baudrate,
                        keys=keys, times=times, log=self._log,
                    )
                self._reply(conn, msg_id, True)
                return
            if cmd not in ("move", "click"):
                self._reply(conn, msg_id, False, f"unknown cmd: {cmd!r}")
                return

            x, y, sw, sh = self._parse_coords(req)
            # 鼠标动作串行执行（独占串口）
            with self._action_lock:
                if cmd == "move":
                    self._emit(f"[action] move -> ({x},{y}) screen={sw}x{sh}")
                    ch9329.move_to_target_humanlike(
                        com_port=self.com_port, baudrate=self.baudrate,
                        target_x=x, target_y=y, screen_w=sw, screen_h=sh,
                        log=self._log,
                    )
                else:
                    button = req.get("button", "LE")
                    if button not in VALID_BUTTONS:
                        self._reply(conn, msg_id, False,
                                    f"invalid button: {button!r}")
                        return
                    self._emit(f"[action] click -> ({x},{y}) button={button}")
                    ch9329.click_at(
                        com_port=self.com_port, baudrate=self.baudrate,
                        x=x, y=y, screen_w=sw, screen_h=sh,
                        button=button, log=self._log,
                    )
            self._reply(conn, msg_id, True)
        except Exception as exc:  # 任何执行异常都回给 A 端，连接不中断
            self._emit(f"[action] error: {exc}")
            try:
                self._reply(conn, msg_id, False, str(exc))
            except OSError:
                pass

    @staticmethod
    def _parse_coords(req: dict) -> tuple[int, int, int, int]:
        try:
            x = int(req["x"])
            y = int(req["y"])
            sw = int(req["screen_w"])
            sh = int(req["screen_h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"bad coordinates payload: {exc}") from exc
        if sw <= 0 or sh <= 0:
            raise ValueError(f"screen size must be positive: {sw}x{sh}")
        if not (0 <= x < sw and 0 <= y < sh):
            raise ValueError(f"point ({x},{y}) out of screen {sw}x{sh}")
        return x, y, sw, sh

    def _reply(self, conn: socket.socket, msg_id, ok: bool,
               error: str | None = None) -> None:
        resp: dict = {"id": msg_id, "ok": ok}
        if error is not None:
            resp["error"] = error
        conn.sendall(encode_message(resp))
