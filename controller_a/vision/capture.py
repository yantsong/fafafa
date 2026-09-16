"""屏幕截图与区域选择（在电脑 A 上截取 uu 远程窗口中的 B 屏画面）。

核心约定：校准时框选的区域必须恰好完整覆盖电脑 B 的整个屏幕内容
（uu 窗口里 B 桌面的四个边缘）。后续所有坐标换算都基于此约定。
"""

from __future__ import annotations

import mss
import numpy as np

# 截图区域 (left, top, width, height)，单位为 A 屏逻辑点（与 tkinter 坐标一致）
Region = tuple[int, int, int, int]


class ScreenCapture:
    """按校准区域定时截取 uu 远程画面，返回 BGR ndarray（可直接给 OpenCV）。"""

    def __init__(self, region: Region) -> None:
        if len(region) != 4:
            raise ValueError(f"region 必须是 (left, top, width, height)，收到: {region}")
        self.left, self.top, self.width, self.height = (int(v) for v in region)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"截图区域尺寸非法: {region}，请先运行校准")

    @property
    def region(self) -> Region:
        return (self.left, self.top, self.width, self.height)

    def grab(self) -> np.ndarray:
        """截取区域，返回 shape=(h, w, 3) 的 BGR 图像。

        Retina 屏上返回像素尺寸可能是区域逻辑尺寸的 2 倍，
        调用方应以返回图像的实际尺寸做归一化换算（见 coord_mapper.CoordMapper）。
        """
        with mss.mss() as sct:
            raw = sct.grab({
                "left": self.left,
                "top": self.top,
                "width": self.width,
                "height": self.height,
            })
        # BGRA -> BGR，copy() 脱离 mss 缓冲区
        return np.asarray(raw)[:, :, :3].copy()


def select_region() -> Region | None:
    """截取全部屏幕的快照并全屏展示，在快照图上拖拽框选区域。

    用快照而不是透明遮罩：macOS 上无边框透明窗口可能不成为焦点窗口，
    收不到鼠标事件；快照方式是普通窗口，事件可靠，且画面清晰便于
    沿 B 桌面边缘对齐。
    支持多显示器（取所有屏幕的并集）。返回 (left, top, width, height)，
    为全局屏幕逻辑坐标；按 Esc 取消返回 None。
    """
    import tkinter as tk

    import cv2

    # 1) 抓取整个虚拟屏幕（monitors[0] = 所有显示器的并集）
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        raw = sct.grab(monitor)
    img = np.asarray(raw)[:, :, :3]  # BGR，像素尺寸（Retina 为逻辑尺寸的 2 倍）
    px_h, px_w = img.shape[:2]

    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    # 2) 缩放到主屏能容纳的尺寸用于展示
    scale = min(screen_w / px_w, screen_h / px_h, 1.0)
    disp_w, disp_h = int(px_w * scale), int(px_h * scale)
    disp = img
    if scale < 1.0:
        disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # 3) BGR -> RGB -> PPM（tkinter 原生支持，无需 Pillow）
    rgb = np.ascontiguousarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    ppm = b"P6\n%d %d\n255\n" % (disp_w, disp_h) + rgb.tobytes()

    result: dict = {"region": None}
    root.title("拖拽框选 B 桌面范围（按 Esc 取消）")
    root.geometry(f"{disp_w}x{disp_h}+0+0")
    root.attributes("-topmost", True)
    root.configure(cursor="crosshair")

    # master 必须显式指定：calibrate 主窗口已是第一个 Tk root，
    # 不指定时 PhotoImage 会绑到错误的解释器（报 image "pyimage1" doesn't exist）
    photo = tk.PhotoImage(master=root, data=ppm)
    canvas = tk.Canvas(root, width=disp_w, height=disp_h,
                       highlightthickness=0, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    canvas.photo = photo  # 防止 PhotoImage 被垃圾回收

    state: dict = {"rect": None, "x0": 0, "y0": 0}

    def on_press(event: tk.Event) -> None:
        state["x0"], state["y0"] = event.x, event.y
        state["rect"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#00ff00", width=2,
        )

    def on_drag(event: tk.Event) -> None:
        if state["rect"] is not None:
            canvas.coords(state["rect"], state["x0"], state["y0"], event.x, event.y)

    def on_release(event: tk.Event) -> None:
        x1, y1 = event.x, event.y
        left, top = min(state["x0"], x1), min(state["y0"], y1)
        width, height = abs(x1 - state["x0"]), abs(y1 - state["y0"])
        if width > 10 and height > 10:
            # 展示坐标 -> 全局逻辑点坐标（k = 每逻辑点对应的展示像素数）
            k = disp_w / monitor["width"]
            gl = monitor["left"] + left / k
            gt = monitor["top"] + top / k
            gw = width / k
            gh = height / k
            result["region"] = (round(gl), round(gt), round(gw), round(gh))
        root.destroy()

    def on_escape(_event: tk.Event) -> None:
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", on_escape)

    root.mainloop()
    return result["region"]
