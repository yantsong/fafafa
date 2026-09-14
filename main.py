import tkinter as tk
from tkinter import messagebox


BG_COLOR = "#000000"
LABEL_COLOR = "#ffffff"
ENTRY_BG = "#ffffff"
ENTRY_FG = "#000000"
ENTRY_BORDER = "#ffffff"
BUTTON_BG = "#ffffff"
BUTTON_FG = "#000000"
FONT = ("Microsoft YaHei UI", 11)
TITLE_FONT = ("Microsoft YaHei UI", 16, "bold")


class CH9329App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CH9329 控制")
        self.root.geometry("360x420")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_COLOR)

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
            font=TITLE_FONT,
        ).pack(anchor=tk.W, pady=(0, 30))

        self.com_var = tk.StringVar(value="COM4")
        self._add_field(container, "COM 口", self.com_var)

        self.x_var = tk.StringVar()
        self._add_field(container, "X 坐标", self.x_var)

        self.y_var = tk.StringVar()
        self._add_field(container, "Y 坐标", self.y_var)

        tk.Button(
            container,
            text="确认",
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground="#dddddd",
            activeforeground=BUTTON_FG,
            relief=tk.FLAT,
            font=FONT,
            cursor="hand2",
            command=self.on_submit,
        ).pack(fill=tk.X, pady=(20, 0), ipady=8)

    def _add_field(self, parent: tk.Frame, label: str, variable: tk.StringVar) -> None:
        tk.Label(
            parent,
            text=label,
            bg=BG_COLOR,
            fg=LABEL_COLOR,
            font=FONT,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 6))

        border = tk.Frame(parent, bg=ENTRY_BORDER, padx=2, pady=2)
        border.pack(fill=tk.X, pady=(0, 16))

        entry = tk.Entry(
            border,
            textvariable=variable,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            insertbackground=ENTRY_FG,
            relief=tk.FLAT,
            bd=0,
            font=FONT,
        )
        entry.pack(fill=tk.X, ipady=10)

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
