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
import time
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from tkinter import messagebox

import cv2
import numpy as np

from action_client import ActionClient, ActionError
from coord_mapper import CoordMapper, load_config
from engine.kit import Kit
from engine.loader import TaskConfigError, load_all_tasks
from engine.quest_engine import QuestEngine
from engine.regions import RegionBook, RegionConfigError
from protocol import DEFAULT_PORT
from services.npc_service import NpcService, NpcServiceConfig
from vision.capture import ScreenCapture
from vision.npc_detector import find_npc
from vision.ocr_engine import get_engine

# 诊断：每次「查找」的原始截图存这里，用于分析 OCR 准确率根因
DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
# Retina 2x 截图下正常游戏名字约 20~28px；低于该值视为小字（1x 时代阈值是14）
SMALL_TEXT_PX = 20

BG_COLOR = "#1d1d22"
PANEL_COLOR = "#2b2b31"
LABEL_COLOR = "#e8e8e8"
ENTRY_BG = "#ffffff"
ENTRY_FG = "#000000"
BLUE = "#2196F3"      # 鲜艳蓝：连接/查找/移动
RED = "#F44336"       # 鲜艳红：断开/点击
ORANGE = "#FF9800"    # 橙：悬停
GRAY_BLUE = "#607D8B" # 校准工具次要按钮


def get_font(size: int = 11, bold: bool = False) -> tuple:
    preferred = ["PingFang SC", "Hiragino Sans GB", "Helvetica Neue", "Arial"]
    available = set(tkfont.families())
    for family in preferred:
        if family in available:
            return (family, size, "bold") if bold else (family, size)
    return ("TkDefaultFont", size, "bold") if bold else ("TkDefaultFont", size)


def _darken(hex_color: str, factor: float = 0.75) -> str:
    """把 #RRGGBB 颜色压暗 factor 倍，用于按钮按下反馈。"""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


class ColorButton(tk.Canvas):
    """自绘纯色按钮：macOS Aqua 会忽略 tk.Button 的 bg，必须自绘才能上色。

    对外兼容常用调用：pack/grid（继承 Canvas）、
    configure(text=/bg=/state=)。
    """

    def __init__(self, parent: tk.Widget, text: str, color: str,
                 command, font=None, height: int = 26) -> None:
        # Tk9 Canvas 默认请求宽度 384px，会把同行按钮挤没；
        # 显式 width=1，实际宽度由 pack 的 fill+expand 分配
        super().__init__(parent, width=1, height=height, bg=BG_COLOR,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._text = text
        self._color = color
        self._command = command
        self._font = font
        self._enabled = True
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self) -> None:
        self.delete("all")
        w = max(int(self.winfo_width()), 1)
        h = max(int(self.winfo_height()), 1)
        fill = self._color if self._enabled else "#4a4a52"
        text_fill = "#ffffff" if self._enabled else "#9a9aa2"
        self.create_rectangle(0, 0, w, h, fill=fill, outline=fill, tags="bg")
        self.create_text(w // 2, h // 2, text=self._text,
                         fill=text_fill, font=self._font, tags="label")

    def _on_press(self, _event) -> None:
        if not self._enabled:
            return
        self.itemconfigure("bg", fill=_darken(self._color),
                           outline=_darken(self._color))

    def _on_release(self, _event) -> None:
        if not self._enabled:
            return
        self._draw()
        self._command()

    # ── 兼容调用方使用的 configure 参数 ───────────────────
    def configure(self, cnf=None, **kw):  # noqa: N802 (Tk API 名称)
        if cnf:
            kw.update(cnf)
        if "text" in kw:
            self._text = kw.pop("text")
        if "bg" in kw:
            self._color = kw.pop("bg")
        elif "background" in kw:
            self._color = kw.pop("background")
        if "state" in kw:
            self._enabled = (str(kw.pop("state")) == tk.NORMAL)
        self._draw()
        return super().configure(**kw) if kw else None

    config = configure


class ControllerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("A 机控制端")
        self.root.geometry("440x760")
        self.root.configure(bg=BG_COLOR)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.font = get_font(10)
        self.small_font = get_font(8)
        self.title_font = get_font(14, bold=True)
        self.client: ActionClient | None = None
        self.busy = False
        self.ocr = None  # 延迟初始化（首次 NPC 操作时加载 Vision）

        cfg = load_config()
        self._cfg = cfg
        self.host_var = tk.StringVar(value=cfg["agent"]["host"])
        self.port_var = tk.StringVar(value=str(cfg["agent"]["port"]))
        self.bw_var = tk.StringVar(value=str(cfg["b_screen"]["width"]))
        self.bh_var = tk.StringVar(value=str(cfg["b_screen"]["height"]))
        self.x_var = tk.StringVar(value=str(cfg["b_screen"]["width"] // 2))
        self.y_var = tk.StringVar(value=str(cfg["b_screen"]["height"] // 2))
        self.npc_name_var = tk.StringVar()
        self.npc_mults_var = tk.StringVar(value="2.5,1.8,3.2,1.2")
        self.npc_ratio_var = tk.StringVar(value="0.08")
        self.npc_settle_var = tk.StringVar(value="350")
        self.npc_framegap_var = tk.StringVar(value="120")
        self.status_var = tk.StringVar(value="未连接")

        # 任务调度状态
        self.tasks = []
        self.task_var = tk.StringVar()
        self.task_menu = None
        self.quest_stop_event: threading.Event | None = None
        self.quest_running = False

        self._build_ui()

    # ── UI ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        head = tk.Frame(self.root, bg=BG_COLOR, padx=14, pady=8)
        head.pack(fill=tk.X)
        tk.Label(head, text="A 机控制端", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.title_font).pack(side=tk.LEFT)
        self.status_label = tk.Label(head, textvariable=self.status_var, bg=BG_COLOR,
                                     fg="#9a9aa2", font=self.font)
        self.status_label.pack(side=tk.RIGHT)

        body = tk.Frame(self.root, bg=BG_COLOR, padx=14)
        body.pack(fill=tk.BOTH, expand=True)

        # ── 连接 ─────────────────────────────────────────
        conn = tk.Frame(body, bg=BG_COLOR)
        conn.pack(fill=tk.X, pady=(2, 4))
        conn.columnconfigure(2, weight=1)  # 让连接按钮占满剩余宽度
        self._entry_in(conn, "B 机 IP", self.host_var, 0, width=17)
        self._entry_in(conn, "端口", self.port_var, 1, width=7)
        self.connect_btn = self._make_button(conn, "连接", BLUE, self.on_connect)
        self.connect_btn.grid(row=1, column=2, padx=(8, 0), pady=(12, 0), sticky="ew")

        # ── B 屏幕分辨率 ─────────────────────────────────
        self._section_label(body, "B 屏幕分辨率")
        res = tk.Frame(body, bg=BG_COLOR)
        res.pack(fill=tk.X)
        self._entry_in(res, "宽", self.bw_var, 0, width=9)
        self._entry_in(res, "高", self.bh_var, 1, width=9)

        # ── 手动测试 ─────────────────────────────────────
        self._section_label(body, "手动测试（B 屏绝对坐标）", pady=(6, 1))
        point = tk.Frame(body, bg=BG_COLOR)
        point.pack(fill=tk.X)
        self._entry_in(point, "X", self.x_var, 0, width=9)
        self._entry_in(point, "Y", self.y_var, 1, width=9)

        actions = tk.Frame(body, bg=BG_COLOR)
        actions.pack(fill=tk.X, pady=(4, 0))
        self.move_btn = self._make_button(actions, "移动", BLUE,
                                          lambda: self.on_action("move"))
        self.move_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.click_btn = self._make_button(actions, "移动并点击", RED,
                                           lambda: self.on_action("click"))
        self.click_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # ── NPC 识别测试 ──────────────────────────────────
        self._section_label(body, "NPC 识别（文本匹配 + hover 变色确认）",
                            pady=(8, 1))
        name_row = tk.Frame(body, bg=BG_COLOR)
        name_row.pack(fill=tk.X)
        tk.Label(name_row, text="NPC 名字", bg=BG_COLOR, fg="#9a9aa2",
                 font=self.small_font).pack(side=tk.LEFT)
        self.ocr_engine_var = tk.StringVar(value="rapid")
        engine_box = tk.OptionMenu(name_row, self.ocr_engine_var,
                                   "rapid", "vision",
                                   command=self._on_engine_change)
        engine_box.configure(font=self.small_font, fg="#1d1d22",
                             relief=tk.FLAT, bd=0, highlightthickness=0,
                             cursor="hand2")
        engine_box["menu"].configure(bg=PANEL_COLOR, fg="white",
                                     activebackground=BLUE)
        engine_box.pack(side=tk.RIGHT)
        tk.Label(name_row, text="引擎", bg=BG_COLOR, fg="#9a9aa2",
                 font=self.small_font).pack(side=tk.RIGHT, padx=(6, 2))
        tk.Entry(body, textvariable=self.npc_name_var, bg=ENTRY_BG, fg=ENTRY_FG,
                 relief=tk.FLAT, bd=0, font=self.font).pack(fill=tk.X, ipady=3,
                                                            pady=(1, 3))
        params = tk.Frame(body, bg=BG_COLOR)
        params.pack(fill=tk.X)
        self._entry_in(params, "高度倍数", self.npc_mults_var, 0, width=13)
        self._entry_in(params, "变色阈值", self.npc_ratio_var, 1, width=6)
        self._entry_in(params, "停顿ms", self.npc_settle_var, 2, width=6)
        self._entry_in(params, "确认ms", self.npc_framegap_var, 3, width=6)

        npc_actions = tk.Frame(body, bg=BG_COLOR)
        npc_actions.pack(fill=tk.X, pady=(4, 0))
        self.npc_locate_btn = self._make_button(
            npc_actions, "查找", BLUE, lambda: self.on_npc("locate"))
        self.npc_locate_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self.npc_hover_btn = self._make_button(
            npc_actions, "查找并悬停", ORANGE, lambda: self.on_npc("hover"))
        self.npc_hover_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        self.npc_click_btn = self._make_button(
            npc_actions, "查找并点击", RED, lambda: self.on_npc("click"))
        self.npc_click_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        self._make_button(body, "打开校准工具", GRAY_BLUE,
                          self.on_open_calibrator).pack(fill=tk.X, pady=(6, 0))

        # ── 任务调度 ───────────────────────────────────────
        self._section_label(body, "任务调度（tasks/ 目录 YAML）", pady=(8, 1))
        quest_row = tk.Frame(body, bg=BG_COLOR)
        quest_row.pack(fill=tk.X)
        # 先 pack 定宽元素（停止→下拉），再 pack 弹性按钮，避免 Aqua
        # OptionMenu 按内容请求超大宽度把开始按钮挤出屏幕
        self.quest_stop_btn = self._make_button(
            quest_row, "停止", RED, self.on_stop_quest)
        self.quest_stop_btn.configure(width=64)
        self.quest_stop_btn.pack(side=tk.RIGHT)
        self.task_menu = tk.OptionMenu(quest_row, self.task_var, "")
        # Tk9/macOS 的 OptionMenu 是原生白色弹出按钮，bg 不生效，
        # 文字必须用深色，否则选中项白字白底看不见
        self.task_menu.configure(font=self.small_font, fg="#1d1d22",
                                 width=15, relief=tk.FLAT, bd=0,
                                 highlightthickness=0, cursor="hand2")
        self.task_menu["menu"].configure(bg=PANEL_COLOR, fg="white",
                                         activebackground=BLUE)
        self.task_menu.pack(side=tk.LEFT, padx=(0, 4))
        self.quest_start_btn = self._make_button(
            quest_row, "开始任务", BLUE, self.on_start_quest)
        self.quest_start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._reload_tasks()

        # 队长状态显示行
        leader_row = tk.Frame(body, bg=BG_COLOR)
        leader_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(leader_row, text="当前队长：", bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.small_font, anchor=tk.W).pack(side=tk.LEFT)
        self.leader_var = tk.StringVar(value="（未检测）")
        self.leader_label = tk.Label(
            leader_row, textvariable=self.leader_var, bg=BG_COLOR,
            fg="#ffd93d", font=self.font, anchor=tk.W)
        self.leader_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._section_label(body, "日志", pady=(8, 2))
        log_frame = tk.Frame(body, bg=PANEL_COLOR, padx=2, pady=2)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=14, bg=PANEL_COLOR, fg="#d8d8e0",
                                relief=tk.FLAT, bd=0, font=get_font(9),
                                wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.tag_config("ok", foreground="#2ecc71")
        self.log_text.tag_config("fail", foreground="#ff6b5e")
        self.log_text.tag_config("info", foreground="#7cc4ff")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _section_label(self, parent: tk.Frame, text: str,
                       pady: tuple = (6, 1)) -> None:
        tk.Label(parent, text=text, bg=BG_COLOR, fg=LABEL_COLOR,
                 font=self.font, anchor=tk.W).pack(fill=tk.X, pady=pady)

    def _make_button(self, parent: tk.Frame, text: str, color: str,
                     command) -> ColorButton:
        return ColorButton(parent, text=text, color=color, command=command,
                           font=self.font)

    def _entry_in(self, parent: tk.Frame, label: str,
                  var: tk.StringVar, col: int, width: int) -> None:
        # 纵向排列：标签在上、输入框在下；整个小框放在父容器的第 col 列
        box = tk.Frame(parent, bg=BG_COLOR)
        box.grid(row=0, column=col, sticky="w", padx=(0, 8))
        tk.Label(box, text=label, bg=BG_COLOR, fg="#9a9aa2",
                 font=self.small_font, anchor=tk.W).pack(anchor=tk.W)
        tk.Entry(box, textvariable=var, bg=ENTRY_BG, fg=ENTRY_FG, relief=tk.FLAT,
                 bd=0, font=self.font, width=width).pack(ipady=3, pady=(1, 0))

    # ── 日志/状态（仅主线程）───────────────────────────────

    def log(self, text: str, level: str = "") -> None:
        tag = level if level in ("ok", "fail", "info") else ""

        def _write() -> None:
            self.log_text.configure(state=tk.NORMAL)
            if tag:
                self.log_text.insert(tk.END, text + "\n", tag)
            else:
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
            self.connect_btn.configure(text="连接", bg=BLUE)
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
                # except 块退出时 Python 会删除 exc，lambda 延迟执行会 NameError，
                # 必须先绑定到普通局部变量
                err = exc
                self.root.after(0, lambda: self._on_connect_failed(err))

        threading.Thread(target=_work, daemon=True).start()

    def _on_connected(self, host: str, port: int) -> None:
        self.set_status(f"已连接 {host}:{port}", "#2ecc71")
        self.connect_btn.configure(text="断开", bg=RED, state=tk.NORMAL)
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
        self._set_actions_enabled(False)

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
                    text="连接", bg=BLUE))
            finally:
                self.root.after(0, self._on_action_done)

        threading.Thread(target=_work, daemon=True).start()

    def _on_action_done(self) -> None:
        self.busy = False
        self._set_actions_enabled(True)

    def _set_actions_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in (self.move_btn, self.click_btn,
                    self.npc_locate_btn, self.npc_hover_btn, self.npc_click_btn):
            btn.configure(state=state)

    # ── 任务调度 ─────────────────────────────────────────

    def _reload_tasks(self) -> None:
        """从 tasks/ 重新加载任务列表，填充下拉框。"""
        try:
            self.tasks = load_all_tasks(TASKS_DIR)
        except TaskConfigError as exc:
            self.tasks = []
            self.log(f"任务配置加载失败: {exc}", "fail")
        menu = self.task_menu["menu"]
        menu.delete(0, tk.END)
        if not self.tasks:
            self.task_var.set("（tasks/ 目录无任务）")
            return
        labels = [t.name for t in self.tasks]
        for label in labels:
            menu.add_command(label=label,
                             command=lambda v=label: self.task_var.set(v))
        if not self.task_var.get():
            self.task_var.set(labels[0])

    def _selected_task(self):
        current = self.task_var.get()
        for t in self.tasks:
            if t.name == current:
                return t
        return None

    def on_start_quest(self) -> None:
        if self.quest_running:
            return
        task = self._selected_task()
        if task is None:
            messagebox.showwarning("提示", "没有可运行的任务，请检查 tasks/ 目录")
            return
        if self.client is None:
            messagebox.showwarning("提示", "请先连接 B 机执行器")
            return
        try:
            capture, mapper = self._make_capture_and_mapper()
            npc_cfg = self._make_npc_config()
        except ValueError as exc:
            messagebox.showwarning("提示", str(exc))
            return

        if self.ocr is None:
            engine_name = self.ocr_engine_var.get()
            self.log(f"首次使用，正在初始化 {engine_name} OCR 引擎...", "info")
            try:
                self.ocr = get_engine(engine_name)
            except Exception as exc:
                messagebox.showerror("OCR 初始化失败", str(exc))
                return

        # 加载任务声明的固定区域坐标簿（regions_file 相对任务文件目录）
        region_book = None
        if task.regions_file:
            base_dir = os.path.dirname(task.source_file)
            rp = task.regions_file
            if not os.path.isabs(rp):
                rp = os.path.normpath(os.path.join(base_dir, rp))
            if not os.path.exists(rp):
                messagebox.showwarning("提示", f"固定区域配置文件不存在:\n{rp}")
                return
            try:
                region_book = RegionBook.load(rp)
            except RegionConfigError as exc:
                messagebox.showerror("固定区域配置错误", str(exc))
                return
            self.log(f"已加载固定区域配置: {rp}（游戏窗口 B屏 "
                     f"{region_book.gw}×{region_book.gh}@"
                     f"{region_book.gx},{region_book.gy}）", "info")

        self.quest_running = True
        self.quest_stop_event = threading.Event()
        self.quest_start_btn.configure(state=tk.DISABLED)
        self.log(f"==== 开始任务「{task.name}」（{len(task.steps)} 步）====", "info")

        kit = Kit(capture, mapper, self.client, self.ocr, npc_cfg,
                  self.quest_stop_event, lambda m: self.log(m),
                  region_book=region_book,
                  on_leader=lambda name: self.root.after(
                      0, lambda: self.leader_var.set(name)))
        engine = QuestEngine(task, kit, lambda m, lv="": self.log(m, lv))

        def _done(result) -> None:
            self.root.after(0, lambda: self._on_quest_done(result))

        threading.Thread(target=lambda: _done(engine.run()),
                         daemon=True).start()

    def on_stop_quest(self) -> None:
        if self.quest_stop_event is not None and self.quest_running:
            self.quest_stop_event.set()
            self.log("正在停止任务（当前动作结束后生效）…", "fail")

    def _on_quest_done(self, result) -> None:
        self.quest_running = False
        self.quest_start_btn.configure(state=tk.NORMAL)
        level = "ok" if result.success else "fail"
        suffix = f"｜当前队长：{result.leader}" if result.leader else ""
        self.log(f"==== {result.message}{suffix}｜耗时 {result.elapsed:.1f}s ====",
                 level)
        if result.leader:
            self.leader_var.set(result.leader)

    # ── NPC 识别 ─────────────────────────────────────────

    def _make_capture_and_mapper(self):
        """按当前 UI 分辨率与 config 截图区域构造 capture/mapper。

        校准工具是独立进程，会在控制台运行期间改写 config.json，
        因此这里必须重新读盘，不能用启动时缓存的 self._cfg。
        """
        try:
            sw = int(self.bw_var.get().strip())
            sh = int(self.bh_var.get().strip())
        except ValueError:
            raise ValueError("B 屏幕宽高必须是整数")
        cfg = load_config()
        self._cfg = cfg
        region_cfg = cfg.get("capture", {})
        if not region_cfg.get("width") or not region_cfg.get("height"):
            raise ValueError(
                "尚未配置截图区域：请在「打开校准工具」里框选后点击「③ 保存配置」"
            )
        region = (region_cfg["left"], region_cfg["top"],
                  region_cfg["width"], region_cfg["height"])
        return ScreenCapture(region), CoordMapper(sw, sh)

    def _make_npc_config(self) -> NpcServiceConfig:
        try:
            mults = tuple(float(x) for x in self.npc_mults_var.get().split(",")
                          if x.strip())
            ratio = float(self.npc_ratio_var.get().strip())
            settle = int(self.npc_settle_var.get().strip()) / 1000.0
            frame_gap = int(self.npc_framegap_var.get().strip()) / 1000.0
        except ValueError as exc:
            raise ValueError(f"NPC 参数格式有误: {exc}") from exc
        if not mults or min(ratio, settle, frame_gap) <= 0:
            raise ValueError("NPC 参数必须为正数")
        return NpcServiceConfig(height_mults=mults, changed_ratio=ratio,
                                settle_sec=settle, frame_gap_sec=frame_gap)

    def _on_engine_change(self, choice: str) -> None:
        # 切换后丢弃已加载引擎，下次操作按新选择重新初始化
        self.ocr = None
        self.log(f"已切换 OCR 引擎为 {choice}，下次查找时生效", "info")

    def on_npc(self, mode: str) -> None:
        if self.busy:
            return
        name = self.npc_name_var.get().strip()
        if not name:
            messagebox.showwarning("提示", "请填写 NPC 名字")
            return
        try:
            capture, mapper = self._make_capture_and_mapper()
            npc_cfg = self._make_npc_config()
        except ValueError as exc:
            messagebox.showwarning("提示", str(exc))
            return
        if mode in ("hover", "click") and self.client is None:
            messagebox.showwarning("提示", "请先连接 B 机执行器")
            return

        self.busy = True
        self._set_actions_enabled(False)
        # 点击瞬间立即反馈，避免 OCR 首次加载数秒内看起来"没反应"
        mode_text = {"locate": "查找", "hover": "查找并悬停", "click": "查找并点击"}[mode]
        self.log(f"▶ {mode_text}「{name}」开始...", "info")

        def _work() -> None:
            t0 = time.time()
            try:
                if self.ocr is None:
                    engine_name = self.ocr_engine_var.get()
                    self.log(f"   首次使用，正在初始化 {engine_name} OCR 引擎"
                             f"（约需 1~3 秒）...")
                    t_ocr = time.time()
                    self.ocr = get_engine(engine_name)
                    self.log(f"   OCR 就绪（{time.time() - t_ocr:.1f}s）")
                if mode == "locate":
                    service = NpcService(
                        capture, mapper, self.client, ocr=self.ocr,
                        config=npc_cfg, log=self.log,
                    )
                    self._npc_locate_work(service, name, t0)
                else:
                    assert self.client is not None
                    service = NpcService(
                        capture, mapper, self.client, ocr=self.ocr,
                        config=npc_cfg, log=self.log,
                    )
                    result = service.find_and_click(
                        name, click=(mode == "click")
                    )
                    self._log_npc_result(result)
            except (OSError, ActionError) as exc:
                self.log(f"NPC 操作失败（网络/通信）: {exc}", "fail")
            except Exception as exc:
                self.log(f"NPC 操作异常: {type(exc).__name__}: {exc}", "fail")
            finally:
                self.root.after(0, self._on_action_done)

        threading.Thread(target=_work, daemon=True).start()

    def _npc_locate_work(self, service: NpcService, name: str,
                         t0: float) -> None:
        self.log("   正在截图并 OCR 识别（未命中会自动换帧/放大重试）...")
        frame, results = service.recognize_robust(name, log=self.log)
        h, w = frame.shape[:2]
        self.log(f"   识别完成，画面 {w}x{h}，共 {len(results)} 段文字"
                 f"（总耗时 {time.time() - t0:.1f}s）")

        # 诊断：存原始帧，文件名带时间戳；同时存一份带 OCR 框标注的图
        os.makedirs(DEBUG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(DEBUG_DIR, f"frame_{stamp}.png")
        cv2.imwrite(raw_path, frame)

        # 画标注图：绿框=识别到的文字，用 PIL 写中文（OpenCV 内置字体不支持中文）
        from PIL import Image, ImageDraw, ImageFont

        annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(annotated)
        draw = ImageDraw.Draw(pil_img)
        try:
            font_ann = ImageFont.truetype(
                "/System/Library/Fonts/Hiragino Sans GB.ttc", 14)
        except OSError:
            font_ann = ImageFont.load_default()
        for r in results:
            draw.rectangle([r.x, r.y, r.x + r.w, r.y + r.h],
                           outline=(0, 255, 0), width=1)
            draw.text((r.x, max(0, r.y - 16)), r.text,
                      fill=(0, 255, 0), font=font_ann)
        annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        ann_path = os.path.join(DEBUG_DIR, f"frame_{stamp}_ocr.png")
        cv2.imwrite(ann_path, annotated)

        self.log(f"   OCR 完成，共 {len(results)} 段文字"
                 f"（总耗时 {time.time() - t0:.1f}s）:")
        small_count = 0
        for r in results:
            tag = " ⚠小字" if r.h < SMALL_TEXT_PX else ""
            if r.h < SMALL_TEXT_PX:
                small_count += 1
            self.log(f"     {r.text!r} 高={r.h}px 宽={r.w}px "
                     f"@({r.x},{r.y}) conf={r.confidence:.2f}{tag}")

        heights = sorted(r.h for r in results)
        if heights:
            self.log(f"   文字高度分布: 最小={heights[0]}px "
                     f"中位={heights[len(heights) // 2]}px "
                     f"最大={heights[-1]}px；"
                     f"{small_count}/{len(heights)} 段低于 {SMALL_TEXT_PX}px",
                     "info")
        self.log(f"   原始截图: {raw_path}")
        self.log(f"   OCR标注图: {ann_path}", "info")

        match = find_npc(results, name)
        if match is None:
            self.log(f"✗ 未找到「{name}」：画面中没有匹配的名字。"
                     f"请把上方标注图发我，或检查名字是否有错别字。", "fail")
            return
        cx, cy = match.center
        self.log(
            f"✓ 找到「{name}」！OCR原文={match.text!r} 精确匹配={match.exact} "
            f"名字框中心(截图坐标)=({cx},{cy}) 文字高={match.h}px "
            f"置信度={match.confidence:.2f}",
            "ok",
        )
        self.log("   人物在名字上方；用「查找并悬停」可自动搜索其身体位置"
                 "并验证 hover 变色。", "ok")

    def _log_npc_result(self, result) -> None:
        level = "ok" if result.success else "fail"
        self.log(f"   {result.message}", level)
        if result.point_screen_b is not None:
            self.log(f"   命中点 B 屏坐标={result.point_screen_b} "
                     f"截图坐标={result.point_capture} "
                     f"尝试={result.attempts} 变色占比={result.best_ratio:.2f}", level)

    def on_open_calibrator(self) -> None:
        # 独立进程启动校准工具，避免两个 Tk root 互相干扰
        subprocess.Popen([sys.executable, "calibrate.py"],
                         cwd=os.path.dirname(os.path.abspath(__file__)))

    def on_close(self) -> None:
        if self.quest_stop_event is not None:
            self.quest_stop_event.set()
        if self.client is not None:
            self.client.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ControllerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
