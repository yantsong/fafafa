import datetime
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Optional

import serial
import ch9329Comm


DEFAULT_COM_PORT = "COM4"
DEFAULT_BAUDRATE = 115200
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080


class CH9329MouseApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CH9329 鼠标控制")
        self.root.geometry("520x480")
        self.root.minsize(480, 420)

        self.serial_conn: Optional[serial.Serial] = None
        self.mouse: Optional[ch9329Comm.mouse.DataComm] = None

        self._build_ui()
        self.log("程序已启动，请先连接 CH9329 设备")

    def _build_ui(self) -> None:
        config_frame = ttk.LabelFrame(self.root, text="设备配置", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=(10, 6))

        ttk.Label(config_frame, text="串口:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.com_var = tk.StringVar(value=DEFAULT_COM_PORT)
        ttk.Entry(config_frame, textvariable=self.com_var, width=10).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(config_frame, text="屏幕宽:").grid(row=0, column=2, sticky=tk.W, padx=(16, 6))
        self.width_var = tk.StringVar(value=str(DEFAULT_SCREEN_WIDTH))
        ttk.Entry(config_frame, textvariable=self.width_var, width=8).grid(row=0, column=3, sticky=tk.W)

        ttk.Label(config_frame, text="屏幕高:").grid(row=0, column=4, sticky=tk.W, padx=(16, 6))
        self.height_var = tk.StringVar(value=str(DEFAULT_SCREEN_HEIGHT))
        ttk.Entry(config_frame, textvariable=self.height_var, width=8).grid(row=0, column=5, sticky=tk.W)

        self.connect_btn = ttk.Button(config_frame, text="连接", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=6, padx=(16, 0))

        coord_frame = ttk.LabelFrame(self.root, text="坐标输入", padding=10)
        coord_frame.pack(fill=tk.X, padx=10, pady=6)

        ttk.Label(coord_frame, text="X:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.x_var = tk.StringVar()
        self.x_entry = ttk.Entry(coord_frame, textvariable=self.x_var, width=12)
        self.x_entry.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(coord_frame, text="Y:").grid(row=0, column=2, sticky=tk.W, padx=(16, 6))
        self.y_var = tk.StringVar()
        self.y_entry = ttk.Entry(coord_frame, textvariable=self.y_var, width=12)
        self.y_entry.grid(row=0, column=3, sticky=tk.W)

        self.confirm_btn = ttk.Button(coord_frame, text="确认", command=self.on_confirm, state=tk.DISABLED)
        self.confirm_btn.grid(row=0, column=4, padx=(16, 0))

        log_frame = ttk.LabelFrame(self.root, text="日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=16, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"

        def append() -> None:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        self.root.after(0, append)

    def toggle_connection(self) -> None:
        if self.serial_conn and self.serial_conn.is_open:
            self.disconnect()
            return
        self.connect()

    def connect(self) -> None:
        com_port = self.com_var.get().strip()
        if not com_port:
            messagebox.showerror("错误", "请输入串口号，例如 COM4")
            return

        try:
            screen_width = int(self.width_var.get().strip())
            screen_height = int(self.height_var.get().strip())
            if screen_width <= 0 or screen_height <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "屏幕宽高必须为正整数")
            return

        try:
            self.serial_conn = serial.Serial(com_port, DEFAULT_BAUDRATE, timeout=0.5)
            serial.ser = self.serial_conn
            self.mouse = ch9329Comm.mouse.DataComm(screen_width, screen_height)
        except serial.SerialException as exc:
            self.serial_conn = None
            self.mouse = None
            self.log(f"连接失败: {exc}")
            messagebox.showerror("连接失败", str(exc))
            return

        self.connect_btn.configure(text="断开")
        self.confirm_btn.configure(state=tk.NORMAL)
        self.log(f"已连接 {com_port}，波特率 {DEFAULT_BAUDRATE}，分辨率 {screen_width}x{screen_height}")

    def disconnect(self) -> None:
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.serial_conn = None
        self.mouse = None

        self.connect_btn.configure(text="连接")
        self.confirm_btn.configure(state=tk.DISABLED)
        self.log("已断开连接")

    def on_confirm(self) -> None:
        if not self.mouse or not self.serial_conn or not self.serial_conn.is_open:
            messagebox.showerror("错误", "请先连接 CH9329 设备")
            return

        try:
            x = int(self.x_var.get().strip())
            y = int(self.y_var.get().strip())
        except ValueError:
            messagebox.showerror("错误", "X 和 Y 必须为整数")
            return

        screen_width = self.mouse.X_MAX
        screen_height = self.mouse.Y_MAX
        if not (0 <= x <= screen_width and 0 <= y <= screen_height):
            messagebox.showerror(
                "错误",
                f"坐标超出屏幕范围 (0~{screen_width}, 0~{screen_height})",
            )
            return

        self.confirm_btn.configure(state=tk.DISABLED)
        self.log(f"开始执行: 移动到 ({x}, {y}) 并单击左键")

        thread = threading.Thread(target=self._move_and_click, args=(x, y), daemon=True)
        thread.start()

    def _move_and_click(self, x: int, y: int) -> None:
        try:
            assert self.mouse is not None
            moved = self.mouse.send_data_absolute(x, y)
            if not moved:
                raise RuntimeError("鼠标移动指令发送失败")

            self.log(f"已移动到 ({x}, {y})")
            self.mouse.click()
            self.log("已单击鼠标左键")
        except Exception as exc:
            self.log(f"执行失败: {exc}")
            self.root.after(0, lambda: messagebox.showerror("执行失败", str(exc)))
        finally:
            self.root.after(0, lambda: self.confirm_btn.configure(state=tk.NORMAL))

    def on_close(self) -> None:
        self.disconnect()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = CH9329MouseApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
