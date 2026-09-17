"""电脑 B 执行器 UI（Windows）。

只负责：串口配置 + TCP 服务启停 + 状态/日志展示。
鼠标动作由电脑 A 通过 TCP 下发，本机不做识别和调度。
后台线程不直接碰 Tk 控件，统一通过队列 + after 轮询刷新。
"""

from __future__ import annotations

import json
import os
import queue
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

from tcp_server import ActionServer
from protocol import DEFAULT_PORT

BAUDRATE = 115200      # CH9329 固定波特率
# B 机固定网络 IP（已通过虚拟 IP 绑定）；如需更改直接修改此处
AGENT_B_IP = "10.219.18.34"

BG_COLOR = "#1e1e1e"
PANEL_COLOR = "#2b2b2b"
LABEL_COLOR = "#e8e8e8"
ENTRY_BG = "#ffffff"
ENTRY_FG = "#000000"
BUTTON_BG = "#3a7bd5"
BUTTON_FG = "#ffffff"
BUTTON_STOP_BG = "#c0392b"

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_b.json")

STATUS_TEXT = {
    "stopped": ("已停止", "#888888"),
    "listening": ("等待连接", "#f1c40f"),
    "connected": ("已连接 A", "#2ecc71"),
    "disconnected": ("连接断开", "#e67e22"),
    "error": ("服务异常", "#e74c3c"),
}


def get_font(size: int = 11, bold: bool = False) -> tuple:
    preferred = ["Microsoft YaHei UI", "PingFang SC", "Segoe UI", "Arial"]
    available = set(tkfont.families())
    for family in preferred:
        if family in available:
            return (family, size, "bold") if bold else (family, size)
    return ("TkDefaultFont", size, "bold") if bold else ("TkDefaultFont", size)


def load_config() -> dict:
    defaults = {"com_port": "COM3", "tcp_port": DEFAULT_PORT}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                defaults.update(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
    return defaults


class AgentBApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("B 机执行器 (CH9329 TCP Server)")
        self.root.geometry("460x560")
        self.root.configure(bg=BG_COLOR)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.font = get_font(11)
        self.title_font = get_font(15, bold=True)

        cfg = load_config()
        self.com_var = tk.StringVar(value=cfg["com_port"])
        self.port_var = tk.StringVar(value=str(cfg["tcp_port"]))

        # 后台线程 -> UI 线程的消息队列：("log", text) / ("status", state)
        self.ui_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.server: ActionServer | None = None

        self._build_ui()
        self._poll_queue()

    # ── UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        bar = tk.Frame(self.root, bg=BG_COLOR, padx=24, pady=20)
        bar.pack(fill=tk.X)
        tk.Label(bar, text="B 机执行器", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.title_font).pack(side=tk.LEFT)
        self.status_dot = tk.Canvas(bar, width=14, height=14, bg=BG_COLOR,
                                    highlightthickness=0)
        self.status_dot.pack(side=tk.LEFT, padx=(14, 6))
        self.status_label = tk.Label(bar, text="已停止", bg=BG_COLOR,
                                     fg="#888888", font=self.font)
        self.status_label.pack(side=tk.LEFT)

        body = tk.Frame(self.root, bg=BG_COLOR, padx=24)
        body.pack(fill=tk.BOTH, expand=True)

        self._field(body, "CH9329 串口", self.com_var, "例如 COM3")
        self._field(body, "TCP 监听端口", self.port_var, str(DEFAULT_PORT))

        # B 机固定网络 IP：A 机控制端填这个（小号提示，改 IP 请改 AGENT_B_IP 常量）
        tk.Label(body, text=f"本机网络 IP：{AGENT_B_IP}（A 机控制端填这个）",
                 bg=BG_COLOR, fg="#9a9aa2", font=get_font(9),
                 anchor=tk.W).pack(fill=tk.X, pady=(0, 12))

        self.toggle_btn = tk.Button(
            body, text="启动服务", font=self.font, bg=BUTTON_BG, fg=BUTTON_FG,
            relief=tk.FLAT, padx=10, pady=10, cursor="hand2",
            command=self.on_toggle,
        )
        self.toggle_btn.pack(fill=tk.X, pady=(6, 14))

        tk.Label(body, text="日志", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.font, anchor=tk.W).pack(fill=tk.X)
        log_frame = tk.Frame(body, bg=PANEL_COLOR, padx=2, pady=2)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.log_text = tk.Text(log_frame, height=12, bg=PANEL_COLOR, fg="#d8d8d8",
                                relief=tk.FLAT, bd=0, font=get_font(9),
                                wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _field(self, parent: tk.Frame, label: str,
               var: tk.StringVar, placeholder: str) -> None:
        tk.Label(parent, text=label, bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.font, anchor=tk.W).pack(fill=tk.X, pady=(0, 4))
        entry = tk.Entry(parent, textvariable=var, bg=ENTRY_BG, fg=ENTRY_FG,
                         relief=tk.FLAT, bd=0, font=self.font)
        entry.pack(fill=tk.X, ipady=8, pady=(0, 12))

    # ── 队列轮询（唯一的 UI 更新入口）────────────────────────

    def _poll_queue(self) -> None:
        while True:
            try:
                kind, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(payload)
            elif kind == "status":
                self._set_status(payload)
        self.root.after(100, self._poll_queue)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_status(self, state: str) -> None:
        text, color = STATUS_TEXT.get(state, (state, "#ffffff"))
        self.status_label.configure(text=text, fg=color)
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 12, 12, fill=color, outline=color)

    # ── 事件 ──────────────────────────────────────────────

    def on_toggle(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
            self.toggle_btn.configure(text="启动服务", bg=BUTTON_BG)
            self._set_editable(True)
            self._set_status("stopped")
            return

        try:
            tcp_port = int(self.port_var.get().strip())
        except ValueError:
            messagebox.showwarning("提示", "TCP 端口必须是整数")
            return
        com = self.com_var.get().strip()
        if not com:
            messagebox.showwarning("提示", "请填写串口号")
            return

        self._save_config(com, tcp_port)
        self.server = ActionServer(
            port=tcp_port, com_port=com, baudrate=BAUDRATE,
            log=lambda m: self.ui_queue.put(("log", m)),
            status=lambda s: self.ui_queue.put(("status", s)),
        )
        self.server.start()
        self.toggle_btn.configure(text="停止服务", bg=BUTTON_STOP_BG)
        self._set_editable(False)

    def _set_editable(self, editable: bool) -> None:
        state = tk.NORMAL if editable else tk.DISABLED
        for child in self.root.winfo_children():
            for widget in child.winfo_children():
                if isinstance(widget, tk.Entry):
                    widget.configure(state=state)

    def _save_config(self, com: str, tcp_port: int) -> None:
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({"com_port": com, "tcp_port": tcp_port},
                          f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def on_close(self) -> None:
        if self.server is not None:
            self.server.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    AgentBApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
