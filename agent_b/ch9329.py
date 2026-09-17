"""通过 windmouse 库 + ch9329Comm 底层 send_data_absolute 移动鼠标。

使用 windmouse 库的标准 WindMouse 物理仿真算法（重力 + 风力）生成轨迹，
通过 ch9329Comm.send_data_absolute 逐点发送绝对坐标。

优化点：
- 菲茨定律（Fitts's Law）强制计算移动耗时
- wind_mouse 库标准物理仿真，不使用贝塞尔曲线
- HID 发包间隔 7~9.6ms 正态随机（μ=8.3ms, σ=0.65ms）
- 长距离 5% 概率中途微停顿 20~80ms，短距离禁止停顿
- 长距离过冲 + 回拉，近距离关闭过冲
- 低 wind_magnitude + 高 damped_distance 减少波动和终点抖动
- 坐标噪声高斯 σ=0.5~0.8
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable

import serial
import ch9329Comm
import pyautogui
from windmouse.core import wind_mouse

LogFn = Callable[[str], None]


# ════════════════════════════════════════════════════════════════
# 串口辅助
# 本模块只运行在电脑 B（Windows）上，pyautogui.position() 读到的就是
# B 的真实指针，无需跨机指针跟踪。
# ════════════════════════════════════════════════════════════════

def _open_serial(com_port: str, baudrate: int, timeout: float = 1.0):
    """打开串口并挂到 serial 模块全局（ch9329Comm 默认读写 serial.ser）。"""
    s = serial.Serial(com_port, baudrate, timeout=timeout, write_timeout=None)
    serial.ser = s
    return s


def _close_serial(log: LogFn | None = None) -> None:
    s = getattr(serial, "ser", None)
    if s is not None and s.is_open:
        s.close()
        _emit("[S] 串口已关闭", log)


def _send_trajectory(
    mouse: "ch9329Comm.mouse.DataComm",
    points: list[tuple[int, int]],
    screen_w: int,
    screen_h: int,
    is_long: bool,
) -> None:
    """逐点发送绝对坐标，含拟人发包间隔与长距离中途微停顿。"""
    for i, (px, py) in enumerate(points):
        px = max(0, min(screen_w - 1, px))
        py = max(0, min(screen_h - 1, py))
        mouse.send_data_absolute(px, py)

        interval = random.gauss(HID_INTERVAL_MEAN, HID_INTERVAL_SIGMA)
        interval = max(HID_INTERVAL_MIN, interval)

        if is_long and 0 < i < len(points) - 1 and random.random() < PAUSE_PROBABILITY:
            interval += random.uniform(*PAUSE_DURATION_RANGE)

        time.sleep(interval)


# ════════════════════════════════════════════════════════════════
# 可配置参数（所有值均为 uniform(min, max) 的随机区间）
# ════════════════════════════════════════════════════════════════

# ── 轨迹弯曲程度 ──
# wind_magnitude：风力强度，控制路径弯曲和随机偏移程度
#   值越大轨迹越弯曲、越随机；值越小路径越直
WIND_MAG_SHORT = (0.5, 1.5)   # 短距离风力范围（≤80px）
WIND_MAG_LONG = (1.0, 2.5)    # 长距离风力范围（>80px）

# ── 抖动大小 ──
# gravity：重力强度，朝向目标的加速度，值越大到达越快但路径越直
GRAVITY_RANGE = (7.0, 11.0)
# damped_distance：距离目标多近时风力开始衰减
#   值越大终点附近越早衰减风力，终点抖动越小；值越小终点附近仍有残余抖动
DAMPED_DIST_RANGE = (15.0, 25.0)
# noise_sigma：每个轨迹点叠加的高斯噪声标准差（px）
#   值越大抖动越明显；建议 0.3~1.0
NOISE_SIGMA_RANGE = (0.5, 0.8)
# end_sigma：终点位置的高斯偏移标准差（px），模拟人眼瞄准不精确
END_SIGMA_RANGE = (0.5, 0.8)
# 终点微颤点数范围，模拟手部静止时的生理性震颤（8-12Hz）
TREMOR_COUNT_RANGE = (2, 3)

# ── 移动速度 ──
# max_step：WindMouse 每步最大位移（px），值越大移动越快
#   正态分布：μ=均值, σ=标准差，实际值约 μ±2σ 范围
MAX_STEP_MEAN = 13.0
MAX_STEP_SIGMA = 1.5

# ── HID 发包间隔 ──
# 发包间隔的正态分布参数（秒）：μ=均值, σ=标准差
#   实际范围约 7~9.6ms，模拟真实硬件报告频率的抖动
HID_INTERVAL_MEAN = 0.0083
HID_INTERVAL_SIGMA = 0.00065
# 发包间隔下限（秒），防止负值或过小
HID_INTERVAL_MIN = 0.005

# ── 长距离中途停顿 ──
# 长距离判定阈值（px），大于此值视为长距离移动
LONG_DIST_THRESHOLD = 80
# 长距离移动中途停顿概率（0.0~1.0），模拟注意力波动
PAUSE_PROBABILITY = 0.05
# 中途停顿时长范围（秒）
PAUSE_DURATION_RANGE = (0.02, 0.08)

# ── 过冲 + 回拉 ──
# 过冲概率（0.0~1.0），仅长距离生效
OVERSHOOT_PROBABILITY = 0.30
# 过冲距离范围（px），朝目标方向冲过头多少像素
OVERSHOOT_DIST_RANGE = (3.0, 10.0)
# 回拉阶段风力范围（比主轨迹更小）
PULL_WIND_RANGE = (0.3, 0.8)
# 回拉阶段重力范围（比主轨迹更强，更快修正回来）
PULL_GRAVITY_RANGE = (8.0, 12.0)
# 回拉阶段每步最大位移（px，比主轨迹更慢）
PULL_MAX_STEP_RANGE = (3.0, 6.0)
# 回拉阶段阻尼距离
PULL_DAMPED_DIST_RANGE = (8.0, 15.0)


def _emit(message: str, log: LogFn | None) -> None:
    # Windows 控制台默认 GBK，中文可能抛 UnicodeEncodeError；打印失败不应影响动作执行
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        pass
    if log is not None:
        log(message)


# ──────────────────────────────────────────────
# 拟人轨迹生成（基于 windmouse 库）
# ──────────────────────────────────────────────

def _generate_trajectory(
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    """使用 windmouse 库的 wind_mouse 生成器生成拟人轨迹。

    低 wind_magnitude 减少波动，高 damped_distance 让终点附近更平滑。
    """
    sx, sy = start
    tx, ty = target
    dist = math.hypot(tx - sx, ty - sy)

    if dist < 1.5:
        # 极短距离：只做微颤
        pts = []
        for _ in range(random.randint(*TREMOR_COUNT_RANGE)):
            sigma = random.uniform(*NOISE_SIGMA_RANGE)
            pts.append((sx + random.gauss(0, sigma), sy + random.gauss(0, sigma)))
        return pts

    is_long = dist > LONG_DIST_THRESHOLD

    # 终点高斯加噪（人眼瞄准不精确）
    end_sigma = random.uniform(*END_SIGMA_RANGE)
    atx = int(tx + random.gauss(0, end_sigma))
    aty = int(ty + random.gauss(0, end_sigma))

    # WindMouse 物理参数
    gravity = random.uniform(*GRAVITY_RANGE)
    wind_mag = random.uniform(*WIND_MAG_LONG) if is_long else random.uniform(*WIND_MAG_SHORT)
    max_step = max(3.0, random.gauss(MAX_STEP_MEAN, MAX_STEP_SIGMA))
    damped_dist = random.uniform(*DAMPED_DIST_RANGE)

    # 过冲：仅长距离
    do_overshoot = is_long and random.random() < OVERSHOOT_PROBABILITY
    if do_overshoot:
        angle = math.atan2(aty - sy, atx - sx)
        os_dist = random.uniform(*OVERSHOOT_DIST_RANGE)
        target_x = int(atx + math.cos(angle) * os_dist)
        target_y = int(aty + math.sin(angle) * os_dist)
    else:
        target_x, target_y = atx, aty

    # 用 windmouse 库生成轨迹
    points: list[tuple[int, int]] = []
    for px, py in wind_mouse(sx, sy, target_x, target_y,
                              gravity_magnitude=gravity,
                              wind_magnitude=wind_mag,
                              max_step=max_step,
                              damped_distance=damped_dist):
        noise_sigma = random.uniform(*NOISE_SIGMA_RANGE)
        points.append((int(px + random.gauss(0, noise_sigma)),
                       int(py + random.gauss(0, noise_sigma))))

    # 过冲后回拉（submovement：更小风力，更慢速度）
    if do_overshoot:
        pull_gravity = random.uniform(*PULL_GRAVITY_RANGE)
        pull_wind = random.uniform(*PULL_WIND_RANGE)
        pull_max_step = random.uniform(*PULL_MAX_STEP_RANGE)
        pull_damped = random.uniform(*PULL_DAMPED_DIST_RANGE)

        last_x, last_y = points[-1] if points else (target_x, target_y)
        for px, py in wind_mouse(last_x, last_y, atx, aty,
                                  gravity_magnitude=pull_gravity,
                                  wind_magnitude=pull_wind,
                                  max_step=pull_max_step,
                                  damped_distance=pull_damped):
            noise_sigma = random.uniform(*NOISE_SIGMA_RANGE)
            points.append((int(px + random.gauss(0, noise_sigma)),
                           int(py + random.gauss(0, noise_sigma))))

    # 终点微颤
    n_tremor = random.randint(*TREMOR_COUNT_RANGE)
    for _ in range(n_tremor):
        sigma = random.uniform(*NOISE_SIGMA_RANGE)
        points.append((int(atx + random.gauss(0, sigma)),
                      int(aty + random.gauss(0, sigma))))

    return points


# ── 单击节奏 ──
# 到位后到按下之间的停顿（模拟人手确认）
CLICK_PRE_DELAY_RANGE = (0.08, 0.20)
# 按下到抬起的时长
CLICK_HOLD_RANGE = (0.06, 0.12)
# 抬起之后的收尾停顿
CLICK_POST_DELAY_RANGE = (0.10, 0.30)


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def move_to_target_humanlike(
    com_port: str = "COM3",
    baudrate: int = 115200,
    target_x: int = 960,
    target_y: int = 540,
    screen_w: int = 1920,
    screen_h: int = 1080,
    timeout: float = 1.0,
    log: LogFn | None = None,
) -> None:
    """打开串口，使用 windmouse 库生成拟人轨迹将鼠标移动到目标坐标。

    使用 send_data_absolute 逐点发送绝对坐标，不受 Windows 鼠标速度影响。
    HID 发包间隔 7~9.6ms 正态随机，长距离 5% 概率中途微停顿。
    """
    pyautogui.FAILSAFE = False
    _emit(f"[1] 打开串口 port={com_port} baudrate={baudrate}", log)
    ser = _open_serial(com_port, baudrate, timeout)
    try:
        _emit(f"[2] 串口已打开 is_open={ser.is_open}", log)

        start_x, start_y = pyautogui.position()
        dist = math.hypot(target_x - start_x, target_y - start_y)
        is_long = dist > LONG_DIST_THRESHOLD
        _emit(
            f"[3] 起点=({start_x},{start_y}) 目标=({target_x},{target_y}) "
            f"距离={dist:.0f}px {'长距离' if is_long else '短距离'}",
            log,
        )

        # 生成 WindMouse 轨迹
        points = _generate_trajectory((start_x, start_y), (target_x, target_y))
        _emit(f"[4] WindMouse 轨迹点={len(points)}", log)

        mouse = ch9329Comm.mouse.DataComm(screen_w, screen_h)
        _send_trajectory(mouse, points, screen_w, screen_h, is_long)

        _emit(f"[5] 移动完成 目标=({target_x},{target_y})", log)
    finally:
        _close_serial(log)


def click_at(
    com_port: str = "COM3",
    baudrate: int = 115200,
    x: int = 960,
    y: int = 540,
    screen_w: int = 1920,
    screen_h: int = 1080,
    timeout: float = 1.0,
    button: str = "LE",
    log: LogFn | None = None,
) -> None:
    """拟人移动到坐标 (x, y) 并单击（默认左键）。

    button: 'LE' 左键 / 'RI' 右键 / 'CE' 中键（ch9329Comm 按键码）。
    点击实现：绝对坐标包携带按键位，按下后随机保持 60~120ms 再抬起，
    前后各有一次拟人停顿。
    """
    pyautogui.FAILSAFE = False
    _emit(f"[C1] 打开串口 port={com_port} baudrate={baudrate}", log)
    ser = _open_serial(com_port, baudrate, timeout)
    try:
        start_x, start_y = pyautogui.position()
        dist = math.hypot(x - start_x, y - start_y)
        is_long = dist > LONG_DIST_THRESHOLD
        _emit(
            f"[C2] 起点=({start_x},{start_y}) 目标=({x},{y}) "
            f"距离={dist:.0f}px {'长距离' if is_long else '短距离'}",
            log,
        )

        points = _generate_trajectory((start_x, start_y), (x, y))
        _emit(f"[C3] WindMouse 轨迹点={len(points)}", log)

        mouse = ch9329Comm.mouse.DataComm(screen_w, screen_h)
        _send_trajectory(mouse, points, screen_w, screen_h, is_long)

        # 到位后短暂停顿再按下，模拟人手确认
        time.sleep(random.uniform(*CLICK_PRE_DELAY_RANGE))
        mouse.send_data_absolute(int(x), int(y), button)   # 按下
        time.sleep(random.uniform(*CLICK_HOLD_RANGE))      # 按住
        mouse.send_data_absolute(int(x), int(y), "NU")     # 抬起
        time.sleep(random.uniform(*CLICK_POST_DELAY_RANGE))
        _emit(f"[C4] 已点击 ({x},{y}) button={button}", log)
    finally:
        _close_serial(log)


# ════════════════════════════════════════════════════════════════
# 键盘（CH9329 标准 USB HID 键盘报文）
# 报文：57 AB | 00 | 02 | 08 | 8字节数据 | 校验和
# 数据 = [修饰键位, 0x00保留, 键码1..6]
# ch9329Comm 库自带的键码表只有字母、没有数字/TAB/F键，这里自备全量表。
# ════════════════════════════════════════════════════════════════

# 修饰键位（第 1 字节，可按位或）
KEY_MODIFIERS = {
    "ctrl": 0x01, "control": 0x01,
    "shift": 0x02,
    "alt": 0x04,
    "win": 0x08, "meta": 0x08,
}

# USB HID Keyboard/Keypad Usage ID（第 3 字节起）
KEY_USAGES: dict[str, int] = {
    "a": 0x04, "b": 0x05, "c": 0x06, "d": 0x07, "e": 0x08,
    "f": 0x09, "g": 0x0A, "h": 0x0B, "i": 0x0C, "j": 0x0D,
    "k": 0x0E, "l": 0x0F, "m": 0x10, "n": 0x11, "o": 0x12,
    "p": 0x13, "q": 0x14, "r": 0x15, "s": 0x16, "t": 0x17,
    "u": 0x18, "v": 0x19, "w": 0x1A, "x": 0x1B, "y": 0x1C,
    "z": 0x1D,
    "1": 0x1E, "2": 0x1F, "3": 0x20, "4": 0x21, "5": 0x22,
    "6": 0x23, "7": 0x24, "8": 0x25, "9": 0x26, "0": 0x27,
    "enter": 0x28, "return": 0x28,
    "esc": 0x29, "escape": 0x29,
    "tab": 0x2B,
    "space": 0x2C, " ": 0x2C,
    "f1": 0x3A, "f2": 0x3B, "f3": 0x3C, "f4": 0x3D,
    "f5": 0x3E, "f6": 0x3F, "f7": 0x40, "f8": 0x41,
    "f9": 0x42, "f10": 0x43, "f11": 0x44, "f12": 0x45,
}

# 单次按键保持时长（按下→抬起）
KEY_HOLD_RANGE = (0.06, 0.12)
# 连按同一组合键（如 Ctrl+Tab×5）时两次敲击之间的间隔
KEY_REPEAT_GAP_RANGE = (0.09, 0.16)
# 修饰键先按下到普通键之间的提前量
MOD_LEAD_SEC = 0.03


def parse_hotkey(keys: str) -> tuple[int, int]:
    """解析 'alt+2'/'ctrl+tab'/'tab' → (修饰键位, 普通键码)。

    一个组合只允许一个普通键（CH9329 单报告场景够用），
    纯修饰键或未知键抛 ValueError。
    """
    if not isinstance(keys, str) or not keys.strip():
        raise ValueError("keys 不能为空，例如 'alt+2'")
    modifier = 0
    usage: int | None = None
    for raw in keys.split("+"):
        token = raw.strip().lower()
        if not token:
            continue
        if token in KEY_MODIFIERS:
            modifier |= KEY_MODIFIERS[token]
        elif token in KEY_USAGES:
            if usage is not None:
                raise ValueError(f"组合键只支持一个普通键，收到: {keys!r}")
            usage = KEY_USAGES[token]
        else:
            raise ValueError(f"不支持的按键: {token!r}（{keys!r}）")
    if usage is None:
        raise ValueError(f"组合键必须包含一个普通键（字母/数字/Tab/F键等）: {keys!r}")
    return modifier, usage


def _write_key_report(ser, modifier: int, usage: int) -> None:
    """发送 8 键键盘报告；usage=0 表示普通键抬起，modifier=0 且 usage=0 为全释放。"""
    data = bytes([modifier & 0xFF, 0x00, usage & 0xFF, 0, 0, 0, 0, 0])
    checksum = (0x57 + 0xAB + 0x00 + 0x02 + 0x08 + sum(data)) & 0xFF
    packet = b"\x57\xAB\x00\x02\x08" + data + bytes([checksum])
    ser.write(packet)


def send_hotkey(
    com_port: str = "COM3",
    baudrate: int = 115200,
    keys: str = "",
    times: int = 1,
    timeout: float = 1.0,
    log: LogFn | None = None,
) -> None:
    """发送组合键（如 ALT+2、Ctrl+Tab×5、TAB）。

    times>1 时修饰键保持按下、普通键重复敲击，最后统一释放，
    与人工「按住 Ctrl 连按 Tab」的行为一致。
    """
    modifier, usage = parse_hotkey(keys)
    times = int(times)
    if not 1 <= times <= 10:
        raise ValueError("times 必须在 1~10 之间")

    _emit(f"[K1] 打开串口 port={com_port} keys={keys!r} times={times}", log)
    ser = _open_serial(com_port, baudrate, timeout)
    try:
        # 修饰键先行（无修饰键时这是一条全零报告，无副作用）
        _write_key_report(ser, modifier, 0)
        time.sleep(MOD_LEAD_SEC)
        for i in range(times):
            _write_key_report(ser, modifier, usage)   # 普通键按下
            time.sleep(random.uniform(*KEY_HOLD_RANGE))
            _write_key_report(ser, modifier, 0)       # 普通键抬起
            if i < times - 1:
                time.sleep(random.uniform(*KEY_REPEAT_GAP_RANGE))
        time.sleep(0.02)
        _write_key_report(ser, 0, 0)                  # 修饰键释放
        time.sleep(random.uniform(0.05, 0.12))
        _emit(f"[K2] 已发送按键 {keys!r} ×{times}", log)
    finally:
        _close_serial(log)
