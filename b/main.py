import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

from ch9329 import move_to_target_humanlike


BG_COLOR = "#000000"
LABEL_COLOR = "#ffffff"
ENTRY_BG = "#ffffff"
ENTRY_FG = "#000000"
ENTRY_BORDER = "#cccccc"
BUTTON_BG = "#ffffff"
BUTTON_FG = "#000000"
BUTTON_BORDER = "#cccccc"

COM_PORT = "COM3"


def get_font(size: int = 11, bold: bool = False) -> tuple:
    preferred = [
        "Microsoft YaHei UI",
        "PingFang SC",
        "Helvetica Neue",
        "Arial",
        "DejaVu Sans",
    ]
    available = set(tkfont.families())
    for family in preferred:
        if family in available:
            return (family, size, "bold") if bold else (family, size)
    return ("TkDefaultFont", size, "bold") if bold else ("TkDefaultFont", size)


class CH9329App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CH9329 控制")
        self.root.geometry("380x520")
        self.root.resizable(False, True)
        self.root.configure(bg=BG_COLOR)

        self.font = get_font(11)
        self.title_font = get_font(16, bold=True)
        self.moving = False
        self.move_button = None

        self._build_ui()
        self._center_window()

    def _center_window(self) -> None:
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg=BG_COLOR, padx=40, pady=40)
        container.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            container,
            text="CH9329 控制",
            bg=BG_COLOR,
            fg=LABEL_COLOR,
            font=self.title_font,
        ).pack(anchor=tk.W, pady=(0, 28))

        self.baud_var = tk.StringVar(value="115200")
        self._add_field(container, "波特率", self.baud_var)

        self.x_var = tk.StringVar(value="960")
        self._add_field(container, "X 坐标", self.x_var)

        self.y_var = tk.StringVar(value="540")
        self._add_field(container, "Y 坐标", self.y_var)

        self._add_button(container, "移动到坐标", self.on_submit)
        self._add_log_view(container)

    def _add_field(self, parent: tk.Frame, label: str, variable: tk.StringVar) -> None:
        tk.Label(
            parent,
            text=label,
            bg=BG_COLOR,
            fg=LABEL_COLOR,
            font=self.font,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 6))

        border = tk.Frame(parent, bg=ENTRY_BORDER, padx=2, pady=2)
        border.pack(fill=tk.X, pady=(0, 14))

        entry = tk.Entry(
            border,
            textvariable=variable,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            insertbackground=ENTRY_FG,
            relief=tk.FLAT,
            bd=0,
            font=self.font,
        )
        entry.pack(fill=tk.X, ipady=10)

    def _add_button(self, parent: tk.Frame, text: str, command) -> None:
        border = tk.Frame(parent, bg=BUTTON_BORDER, padx=2, pady=2)
        border.pack(fill=tk.X, pady=(10, 0))

        button = tk.Label(
            border,
            text=text,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            font=self.font,
            padx=12,
            pady=12,
            cursor="hand2",
        )
        button.pack(fill=tk.X)
        button.bind("<Button-1>", lambda _event: command())
        button.bind("<Enter>", lambda _event: button.configure(bg="#eeeeee"))
        button.bind("<Leave>", lambda _event: button.configure(bg=BUTTON_BG))
        self.move_button = button

    def _add_log_view(self, parent: tk.Frame) -> None:
        tk.Label(
            parent,
            text="日志",
            bg=BG_COLOR,
            fg=LABEL_COLOR,
            font=self.font,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(18, 6))

        border = tk.Frame(parent, bg=ENTRY_BORDER, padx=2, pady=2)
        border.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            border,
            height=10,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            relief=tk.FLAT,
            bd=0,
            font=get_font(9),
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def append_log(self, message: str) -> None:
        def _write() -> None:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        if threading.current_thread() is threading.main_thread():
            _write()
        else:
            self.root.after(0, _write)

    def on_submit(self) -> None:
        if self.moving:
            return

        baud_text = self.baud_var.get().strip()
        x_text = self.x_var.get().strip()
        y_text = self.y_var.get().strip()
        if not baud_text:
            messagebox.showwarning("提示", "请输入波特率")
            return
        if not x_text or not y_text:
            messagebox.showwarning("提示", "请输入 X 和 Y 坐标")
            return
        try:
            baudrate = int(baud_text)
            x = int(x_text)
            y = int(y_text)
        except ValueError:
            messagebox.showwarning("提示", "波特率和坐标必须为整数")
            return
        if baudrate <= 0:
            messagebox.showwarning("提示", "波特率必须大于 0")
            return
        if x < 0 or y < 0:
            messagebox.showwarning("提示", "坐标不能为负数")
            return

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self._set_moving(True)
        threading.Thread(
            target=self._move_in_background,
            args=(baudrate, x, y, screen_w, screen_h),
            daemon=True,
        ).start()

    def _set_moving(self, moving: bool) -> None:
        self.moving = moving
        if self.move_button is not None:
            self.move_button.configure(text="移动中..." if moving else "移动到坐标")

    def _move_in_background(
        self, baudrate: int, x: int, y: int, screen_w: int, screen_h: int,
    ) -> None:
        error = None
        try:
            move_to_target_humanlike(
                com_port=COM_PORT,
                baudrate=baudrate,
                target_x=x,
                target_y=y,
                screen_w=screen_w,
                screen_h=screen_h,
                log=self.append_log,
            )
        except Exception as exc:
            error = exc
        self.root.after(0, lambda: self._on_move_done(error, x, y))

    def _on_move_done(self, error: Exception | None, x: int, y: int) -> None:
        self._set_moving(False)
        if error is not None:
            messagebox.showerror("移动失败", f"无法通过 CH9329 移动鼠标：\n{error}")
            return
        messagebox.showinfo("完成", f"鼠标已移动到 ({x}, {y})")


def main() -> None:
    root = tk.Tk()
    CH9329App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
