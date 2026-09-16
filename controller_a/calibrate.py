"""A 机校准工具：框选 uu 画面区域 → 配置 B 分辨率与连接 → 保存 → TCP 远程验证。

使用顺序：
1. B 机启动执行器并显示"等待连接"；A 机能 ping 通 B 的 IP。
2. 在 uu 远程窗口里让电脑 B 显示完整桌面（能看到 B 桌面四边）。
3. 点"框选截图区域"，沿 B 桌面边缘拖一个矩形。
4. 填写 B 的实际分辨率与 B 的 IP/端口。
5. 保存配置；接好 CH9329（在 B 机上）后点"校准验证"：
   程序通过 TCP 让 B 的鼠标依次移到 3 个已知点，帧差定位光标反算坐标，
   报告误差，平均误差 < 5px 即可投入使用。
"""

from __future__ import annotations

import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

import cv2

from action_client import ActionClient, ActionError
from coord_mapper import (
    CoordMapper,
    load_config,
    save_config,
    verify_mapping,
)
from protocol import DEFAULT_PORT
from vision.capture import ScreenCapture, select_region

BG_COLOR = "#1d1d22"
PANEL_COLOR = "#2b2b31"
LABEL_COLOR = "#e8e8e8"
ENTRY_BG = "#ffffff"
ENTRY_FG = "#000000"
BUTTON_BG = "#3a7bd5"
BUTTON_FG = "#ffffff"
PREVIEW_PATH = "capture_preview.png"


def get_font(size: int = 11, bold: bool = False) -> tuple:
    preferred = ["PingFang SC", "Hiragino Sans GB", "Helvetica Neue", "Arial"]
    available = set(tkfont.families())
    for family in preferred:
        if family in available:
            return (family, size, "bold") if bold else (family, size)
    return ("TkDefaultFont", size, "bold") if bold else ("TkDefaultFont", size)


class CalibrateApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("A 机校准工具")
        self.root.geometry("440x700")
        self.root.resizable(False, True)
        self.root.configure(bg=BG_COLOR)

        self.font = get_font(11)
        self.title_font = get_font(16, bold=True)
        self.busy = False

        cfg = load_config()
        self.host_var = tk.StringVar(value=cfg["agent"]["host"])
        self.port_var = tk.StringVar(value=str(cfg["agent"]["port"]))
        self.bw_var = tk.StringVar(value=str(cfg["b_screen"]["width"]))
        self.bh_var = tk.StringVar(value=str(cfg["b_screen"]["height"]))
        self.region: tuple[int, int, int, int] | None = None
        self.region_var = tk.StringVar(value="（未框选）")

        self._build_ui()
        if cfg["capture"]["width"] and cfg["capture"]["height"]:
            self.region = (
                cfg["capture"]["left"], cfg["capture"]["top"],
                cfg["capture"]["width"], cfg["capture"]["height"],
            )
            self.region_var.set(f"{self.region[0]},{self.region[1]} "
                                f"{self.region[2]}x{self.region[3]}（已加载）")

    # ── UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg=BG_COLOR, padx=30, pady=26)
        container.pack(fill=tk.BOTH, expand=True)

        tk.Label(container, text="A 机校准工具", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.title_font).pack(anchor=tk.W, pady=(0, 18))

        self._add_field(container, "B 机 IP 地址", self.host_var)
        self._add_field(container, f"TCP 端口（默认 {DEFAULT_PORT}）", self.port_var)
        self._add_field(container, "B 屏幕宽度", self.bw_var)
        self._add_field(container, "B 屏幕高度", self.bh_var)

        tk.Label(container, text="截图区域", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.font, anchor=tk.W).pack(fill=tk.X, pady=(0, 4))
        tk.Label(container, textvariable=self.region_var, bg=BG_COLOR,
                 fg="#9a9aa2", font=self.font, anchor=tk.W).pack(fill=tk.X, pady=(0, 8))

        self._add_button(container, "① 框选截图区域（沿 B 桌面边缘）", self.on_select_region)
        self._add_button(container, "② 截图预览 + 宽高比检查", self.on_preview)
        self._add_button(container, "③ 保存配置 config.json", self.on_save)
        self._add_button(container, "④ 校准验证（经 TCP 控制 B 机鼠标）", self.on_verify)
        self._add_log_view(container)

    def _add_field(self, parent: tk.Frame, label: str, variable: tk.StringVar) -> None:
        tk.Label(parent, text=label, bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.font, anchor=tk.W).pack(fill=tk.X, pady=(0, 6))
        border = tk.Frame(parent, bg=PANEL_COLOR, padx=2, pady=2)
        border.pack(fill=tk.X, pady=(0, 12))
        tk.Entry(border, textvariable=variable, bg=ENTRY_BG, fg=ENTRY_FG,
                 insertbackground=ENTRY_FG, relief=tk.FLAT, bd=0,
                 font=self.font).pack(fill=tk.X, ipady=9)

    def _add_button(self, parent: tk.Frame, text: str, command) -> None:
        btn = tk.Button(parent, text=text, font=self.font, bg=BUTTON_BG,
                        fg=BUTTON_FG, relief=tk.FLAT, padx=10, pady=9,
                        cursor="hand2", command=command)
        btn.pack(fill=tk.X, pady=(0, 10))

    def _add_log_view(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="日志", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.font, anchor=tk.W).pack(fill=tk.X, pady=(8, 6))
        border = tk.Frame(parent, bg=PANEL_COLOR, padx=2, pady=2)
        border.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(border, height=10, bg=PANEL_COLOR, fg="#d8d8e0",
                                relief=tk.FLAT, bd=0, font=get_font(9),
                                wrap=tk.WORD, state=tk.DISABLED)
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

    # ── 业务 ──────────────────────────────────────────────

    def _parse_basic(self) -> tuple[str, int, CoordMapper] | None:
        try:
            port = int(self.port_var.get().strip())
            mapper = CoordMapper(int(self.bw_var.get().strip()),
                                 int(self.bh_var.get().strip()))
        except ValueError:
            messagebox.showwarning("提示", "TCP 端口与 B 屏幕宽高必须为整数")
            return None
        host = self.host_var.get().strip()
        if not host:
            messagebox.showwarning("提示", "请填写 B 机 IP 地址")
            return None
        return host, port, mapper

    def on_select_region(self) -> None:
        if self.busy:
            return
        self.root.withdraw()
        try:
            region = select_region()
        finally:
            self.root.deiconify()
        if region is None:
            self.append_log("已取消框选")
            return
        self.region = region
        self.region_var.set(f"{region[0]},{region[1]} {region[2]}x{region[3]}")
        self.append_log(f"截图区域已设置: left={region[0]} top={region[1]} "
                        f"w={region[2]} h={region[3]}")

    def on_preview(self) -> None:
        if self.busy or self.region is None:
            messagebox.showwarning("提示", "请先框选截图区域")
            return
        parsed = self._parse_basic()
        if parsed is None:
            return
        _host, _port, mapper = parsed
        self.busy = True

        def _work() -> None:
            try:
                img = ScreenCapture(self.region).grab()
                h, w = img.shape[:2]
                ok = mapper.check_aspect(w, h)
                cv2.imwrite(PREVIEW_PATH, img)
                self.append_log(f"预览已保存 {PREVIEW_PATH}  实际尺寸={w}x{h}")
                self.append_log("宽高比检查: " +
                                ("通过" if ok else
                                 "不通过（框选区域与 B 分辨率比例差异过大，请重新框选）"))
            except Exception as exc:
                self.append_log(f"截图失败: {exc}")
            finally:
                self.root.after(0, lambda: setattr(self, "busy", False))

        threading.Thread(target=_work, daemon=True).start()

    def on_save(self) -> None:
        if self.region is None:
            messagebox.showwarning("提示", "请先框选截图区域")
            return
        parsed = self._parse_basic()
        if parsed is None:
            return
        host, port, mapper = parsed
        save_config({
            "agent": {"host": host, "port": port},
            "b_screen": {"width": mapper.b_width, "height": mapper.b_height},
            "capture": {
                "left": self.region[0], "top": self.region[1],
                "width": self.region[2], "height": self.region[3],
            },
        })
        self.append_log(f"配置已保存: {host}:{port} / {mapper.b_width}x{mapper.b_height}")

    def on_verify(self) -> None:
        if self.busy:
            return
        if self.region is None:
            messagebox.showwarning("提示", "请先框选截图区域")
            return
        parsed = self._parse_basic()
        if parsed is None:
            return
        host, port, mapper = parsed
        self.busy = True
        self.append_log(f"连接 B 机执行器 {host}:{port} ...")

        def _work() -> None:
            client: ActionClient | None = None
            try:
                client = ActionClient(host, port)
                client.connect()
                client.ping()
                self.append_log("TCP 连接正常，开始校准验证（移动 B 鼠标到 3 个点）...")

                capture = ScreenCapture(self.region)

                def move_fn(x: int, y: int) -> None:
                    assert client is not None
                    client.move_to(x, y, mapper.b_width, mapper.b_height)

                results = verify_mapping(capture, mapper, move_fn, log=self.append_log)
                errors = [r["error_px"] for r in results if "error_px" in r]
                if len(errors) == len(results):
                    mean_err = sum(errors) / len(errors)
                    self.append_log(f"验证完成: 平均误差 {mean_err:.1f}px, "
                                    f"最大 {max(errors):.1f}px")
                    self.append_log("映射可靠，可以投入使用" if mean_err < 5 else
                                    "误差偏大：请确认框选区域精确贴合 B 桌面边缘")
                else:
                    self.append_log("部分点位未定位到光标，建议在 B 桌面静止状态下重试")
            except (OSError, ActionError) as exc:
                self.append_log(f"连接/通信失败: {exc}")
            except Exception as exc:
                self.append_log(f"校准验证失败: {exc}")
            finally:
                if client is not None:
                    client.close()
                self.root.after(0, lambda: setattr(self, "busy", False))

        threading.Thread(target=_work, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    CalibrateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
