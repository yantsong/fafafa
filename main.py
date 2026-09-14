import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox


BG_COLOR = "#000000"
LABEL_COLOR = "#ffffff"
ENTRY_BG = "#ffffff"
ENTRY_FG = "#000000"
ENTRY_BORDER = "#cccccc"
BUTTON_BG = "#ffffff"
BUTTON_FG = "#000000"
BUTTON_BORDER = "#cccccc"


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
        self.root.geometry("380x480")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_COLOR)

        self.font = get_font(11)
        self.title_font = get_font(16, bold=True)

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

        self.com_var = tk.StringVar(value="COM4")
        self._add_field(container, "COM 口", self.com_var)

        self.x_var = tk.StringVar()
        self._add_field(container, "X 坐标", self.x_var)

        self.y_var = tk.StringVar()
        self._add_field(container, "Y 坐标", self.y_var)

        self._add_button(container, "确认", self.on_submit)

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

        if sys.platform == "darwin":
            entry.configure(highlightthickness=0)

    def _add_button(self, parent: tk.Frame, text: str, command) -> None:
        border = tk.Frame(parent, bg=BUTTON_BORDER, padx=2, pady=2)
        border.pack(fill=tk.X, pady=(24, 0))

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

    def on_submit(self) -> None:
        com_port = self.com_var.get().strip()
        x_text = self.x_var.get().strip()
        y_text = self.y_var.get().strip()

        if not com_port:
            messagebox.showwarning("提示", "请输入 COM 口")
            return
        if not x_text or not y_text:
            messagebox.showwarning("提示", "请输入 X 和 Y 坐标")
            return

        try:
            x = int(x_text)
            y = int(y_text)
        except ValueError:
            messagebox.showwarning("提示", "X 和 Y 坐标必须为整数")
            return

        messagebox.showinfo(
            "输入内容",
            f"COM 口: {com_port}\nX: {x}\nY: {y}",
        )


def main() -> None:
    root = tk.Tk()
    CH9329App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
