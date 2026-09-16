"""电脑 A 控制端控制台（macOS）。

职责：
- 管理与 B 机执行器的 TCP 连接（连接/断开/状态）
- 手动下发移动/点击，用于端到端验证
- 启动校准工具
后续任务引擎也会在这个进程里运行（任务调度只发生在 A）。
所有网络动作在后台线程执行，日志通过 after 回主线程刷新。
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

from action_client import ActionClient, ActionError
from coord_mapper import load_config
from protocol import DEFAULT_PORT

BG_COLOR = "#1d1d22"
PANEL_COLOR = "#2b2b31"
LABEL_COLOR = "#e8e8e8"
ENTRY_BG = "#ffffff"
ENTRY_FG = "#000000"
PRIMARY_BG = "#3a7bd5"
DANGER_BG = "#c0392b"
GHOST_BG = "#3d3d46"


def get_font(size: int = 11, bold: bool = False) -> tuple:
    preferred = ["PingFang SC", "Hiragino Sans GB", "Helvetica Neue", "Arial"]
    available = set(tkfont.families())
    for family in preferred:
        if family in available:
            return (family, size, "bold") if bold else (family, size)
    return ("TkDefaultFont", size, "bold") if bold else ("TkDefaultFont", size)


class ControllerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("A 机控制端")
        self.root.geometry("460x620")
        self.root.configure(bg=BG_COLOR)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.font = get_font(11)
        self.title_font = get_font(16, bold=True)
        self.client: ActionClient | None = None
        self.busy = False

        cfg = load_config()
        self.host_var = tk.StringVar(value=cfg["agent"]["host"])
        self.port_var = tk.StringVar(value=str(cfg["agent"]["port"]))
        self.bw_var = tk.StringVar(value=str(cfg["b_screen"]["width"]))
        self.bh_var = tk.StringVar(value=str(cfg["b_screen"]["height"]))
        self.x_var = tk.StringVar(value=str(cfg["b_screen"]["width"] // 2))
        self.y_var = tk.StringVar(value=str(cfg["b_screen"]["height"] // 2))
        self.status_var = tk.StringVar(value="未连接")

        self._build_ui()

    # ── UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        head = tk.Frame(self.root, bg=BG_COLOR, padx=24, pady=20)
        head.pack(fill=tk.X)
        tk.Label(head, text="A 机控制端", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.title_font).pack(side=tk.LEFT)
        self.status_label = tk.Label(head, textvariable=self.status_var, bg=BG_COLOR,
                                     fg="#9a9aa2", font=self.font)
        self.status_label.pack(side=tk.RIGHT)

        body = tk.Frame(self.root, bg=BG_COLOR, padx=24)
        body.pack(fill=tk.BOTH, expand=True)

        conn = tk.Frame(body, bg=BG_COLOR)
        conn.pack(fill=tk.X)
        self._entry_in(conn, "B 机 IP", self.host_var, 0, width=18)
        self._entry_in(conn, "端口", self.port_var, 1, width=8)
        self.connect_btn = tk.Button(conn, text="连接", font=self.font, bg=PRIMARY_BG,
                                     fg="white", relief=tk.FLAT, padx=14, pady=8,
                                     cursor="hand2", command=self.on_connect)
        self.connect_btn.grid(row=1, column=2, padx=(10, 0), pady=(22, 0), sticky="ew")

        tk.Label(body, text="B 屏幕分辨率（动作坐标范围）", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.font, anchor=tk.W).pack(fill=tk.X, pady=(16, 4))
        res = tk.Frame(body, bg=BG_COLOR)
        res.pack(fill=tk.X)
        self._entry_in(res, "宽", self.bw_var, 0, width=10)
        self._entry_in(res, "高", self.bh_var, 1, width=10)

        tk.Label(body, text="手动测试（B 屏绝对坐标）", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.font, anchor=tk.W).pack(fill=tk.X, pady=(18, 4))
        point = tk.Frame(body, bg=BG_COLOR)
        point.pack(fill=tk.X)
        self._entry_in(point, "X", self.x_var, 0, width=10)
        self._entry_in(point, "Y", self.y_var, 1, width=10)

        actions = tk.Frame(body, bg=BG_COLOR)
        actions.pack(fill=tk.X, pady=(12, 0))
        self.move_btn = tk.Button(actions, text="移动", font=self.font, bg=GHOST_BG,
                                  fg="white", relief=tk.FLAT, pady=9, cursor="hand2",
                                  command=lambda: self.on_action("move"))
        self.move_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.click_btn = tk.Button(actions, text="移动并点击", font=self.font,
                                   bg=PRIMARY_BG, fg="white", relief=tk.FLAT, pady=9,
                                   cursor="hand2",
                                   command=lambda: self.on_action("click"))
        self.click_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        tk.Button(body, text="打开校准工具", font=self.font, bg=GHOST_BG, fg="white",
                  relief=tk.FLAT, pady=9, cursor="hand2",
                  command=self.on_open_calibrator).pack(fill=tk.X, pady=(14, 0))

        tk.Label(body, text="日志", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.font, anchor=tk.W).pack(fill=tk.X, pady=(16, 4))
        log_frame = tk.Frame(body, bg=PANEL_COLOR, padx=2, pady=2)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=10, bg=PANEL_COLOR, fg="#d8d8e0",
                                relief=tk.FLAT, bd=0, font=get_font(9),
                                wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _entry_in(self, parent: tk.Frame, label: str,
                  var: tk.StringVar, col: int, width: int) -> None:
        # 纵向排列：标签在上、输入框在下；整个小框放在父容器的第 col 列
        box = tk.Frame(parent, bg=BG_COLOR)
        box.grid(row=0, column=col, sticky="w", padx=(0, 10))
        tk.Label(box, text=label, bg=BG_COLOR, fg="#9a9aa2",
                 font=get_font(9), anchor=tk.W).pack(anchor=tk.W)
        tk.Entry(box, textvariable=var, bg=ENTRY_BG, fg=ENTRY_FG, relief=tk.FLAT,
                 bd=0, font=self.font, width=width).pack(ipady=7, pady=(2, 0))

    # ── 日志/状态（仅主线程）───────────────────────────────

    def log(self, text: str) -> None:
        def _write() -> None:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, text + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        if threading.current_thread() is threading.main_thread():
            _write()
        else:
            self.root.after(0, _write)

    def set_status(self, text: str, color: str) -> None:
        self.status_var.set(text)
        self.status_label.configure(fg=color)

    # ── 事件 ──────────────────────────────────────────────

    def on_connect(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
            self.set_status("未连接", "#9a9aa2")
            self.connect_btn.configure(text="连接", bg=PRIMARY_BG)
            self.log("已断开与 B 机的连接")
            return

        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            messagebox.showwarning("提示", "端口必须是整数")
            return
        host = self.host_var.get().strip()
        self.connect_btn.configure(state=tk.DISABLED)
        self.set_status("连接中...", "#f1c40f")

        def _work() -> None:
            try:
                client = ActionClient(host, port)
                client.connect()
                client.ping()
                self.client = client
                self.root.after(0, lambda: self._on_connected(host, port))
            except (OSError, ActionError) as exc:
                self.root.after(0, lambda: self._on_connect_failed(exc))

        threading.Thread(target=_work, daemon=True).start()

    def _on_connected(self, host: str, port: int) -> None:
        self.set_status(f"已连接 {host}:{port}", "#2ecc71")
        self.connect_btn.configure(text="断开", bg=DANGER_BG, state=tk.NORMAL)
        self.log(f"已连接 B 机执行器 {host}:{port}")

    def _on_connect_failed(self, exc: Exception) -> None:
        self.set_status("连接失败", "#e74c3c")
        self.connect_btn.configure(state=tk.NORMAL)
        self.log(f"连接失败: {exc}")

    def _parse_action(self) -> tuple[int, int, int, int] | None:
        try:
            x = int(self.x_var.get().strip())
            y = int(self.y_var.get().strip())
            sw = int(self.bw_var.get().strip())
            sh = int(self.bh_var.get().strip())
        except ValueError:
            messagebox.showwarning("提示", "坐标和分辨率必须是整数")
            return None
        if not (0 <= x < sw and 0 <= y < sh):
            messagebox.showwarning("提示", f"坐标 ({x},{y}) 超出屏幕 {sw}x{sh}")
            return None
        return x, y, sw, sh

    def on_action(self, kind: str) -> None:
        if self.busy:
            return
        if self.client is None:
            messagebox.showwarning("提示", "请先连接 B 机执行器")
            return
        parsed = self._parse_action()
        if parsed is None:
            return
        x, y, sw, sh = parsed
        self.busy = True
        self.move_btn.configure(state=tk.DISABLED)
        self.click_btn.configure(state=tk.DISABLED)

        def _work() -> None:
            try:
                assert self.client is not None
                if kind == "move":
                    self.log(f"-> move ({x},{y})")
                    self.client.move_to(x, y, sw, sh)
                    self.log("   完成")
                else:
                    self.log(f"-> click ({x},{y})")
                    self.client.click_at(x, y, sw, sh)
                    self.log("   完成")
            except (OSError, ActionError) as exc:
                self.log(f"   失败: {exc}")
                self.root.after(0, lambda: self.set_status("连接中断", "#e74c3c"))
                self.client = None
                self.root.after(0, lambda: self.connect_btn.configure(
                    text="连接", bg=PRIMARY_BG))
            finally:
                self.root.after(0, self._on_action_done)

        threading.Thread(target=_work, daemon=True).start()

    def _on_action_done(self) -> None:
        self.busy = False
        self.move_btn.configure(state=tk.NORMAL)
        self.click_btn.configure(state=tk.NORMAL)

    def on_open_calibrator(self) -> None:
        # 独立进程启动校准工具，避免两个 Tk root 互相干扰
        subprocess.Popen([sys.executable, "calibrate.py"],
                         cwd=os.path.dirname(os.path.abspath(__file__)))

    def on_close(self) -> None:
        if self.client is not None:
            self.client.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ControllerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
