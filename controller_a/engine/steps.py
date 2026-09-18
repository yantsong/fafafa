"""任务步骤原语。

每个原语签名：handler(step: dict, ctx: QuestContext, kit: Kit) -> None
成功返回 None；失败抛 StepFailed；超时/停止由 poll 助手抛 StepTimeout/StepStopped。

支持的 do 类型：
  log              打印日志（支持 ${var}）
  sleep            固定等待
  click_npc        找 NPC → hover 变色确认 → 点击（复用 NpcService）
  wait_dialog      等待对话区出现指定文字（不给 match 则等待任意文字）
  wait_text        wait_dialog 的语义别名（可用于等任意屏幕文字）
  choose_option    在画面中找到选项文字并点击其中心
  read_route       OCR 当前对话 → 查路线表 → 把目标点/场景/NPC 存入上下文
  click_point      点击 B 屏绝对坐标或归一化点（小地图寻路用）
  wait_arrive      轮询直到场景名出现（自动寻路到达）
  dialog_until     连续点对话，直到出现目标文字
  hotkey           发送组合键（alt+2 / ctrl+tab ×N / tab / f8 等，由 B 端 HID 执行）
  click_region     点击 regions 文件里的命名固定区域，框内随机取点，支持左右键
  transfer_leader  切换队长：右键队友槽位→左键给与队长菜单，times=1给与/times=5给回
  map_click        查 MAP 点大地图地名（level=big）或小地图寻路点（level=small）
  wait_npc         到达判定：游戏窗口内轮询 OCR 名字 → 模板匹配
  wait_arrive_pos  到达判定：HUD 场景名+坐标「先移动、再静止 stable_secs 秒」
                   （map 可选：不传时 NPC 确认只走 OCR，无模板/盲点）
  click_quest_npc  点目标 NPC：搜索窗口内轮询 ① OCR 名字 ② 模板匹配，超时后 npc_box 盲点
                   （map 可选：不传时只有 OCR，识别不到直接失败）
  click_text       命名区域内 OCR 找文字并点击（世界地图场景名/「寻」按钮等通用）
  minimap_npc_pick 小地图右侧 NPC 树：必要时点「普通NPC」展开 → OCR 列表找 NPC，
                   未命中则在列表区滚轮下滚继续 OCR，触底（内容不再变化）即失败
  dialog_choose    对话框选选项：绿色文字=可点选项，在选项行内随机点击，optional 可跳过
  check_quest_accepted  接取成功确认：对话框+任务追踪栏同时出现红色文字
  check_leader      切换队长后 OCR 提示区，匹配「现在由XX担任队长」提取当前队长
  switch_tab        切换标签：OCR 队伍标签栏，按当前队长名匹配并点击对应标签
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import cv2

from engine.context import ContextError, QuestContext
from engine.hud import coord_close, parse_hud, scene_matches
from engine.kit import Kit, StepStopped
from engine.regions import RegionConfigError, load_map_file
from vision.color_text import has_green_text, has_red_text
from vision.dialog_reader import (
    crop_roi,
    find_text,
    join_text,
    recognize_in_roi,
    text_matches,
    validate_roi,
)
from vision.npc_detector import find_npc, normalize
from vision.template_matcher import load_template, match_template


class StepFailed(RuntimeError):
    """步骤执行失败（已耗尽重试或不支持重试）。"""


class StepTimeout(StepFailed):
    """步骤在限定时间内未满足条件。"""


# ── 公共助手 ────────────────────────────────────────────

def _get(step: dict, key: str, default: Any = None, required: bool = False) -> Any:
    if key in step and step[key] is not None:
        return step[key]
    if required:
        raise StepFailed(f"步骤缺少必填参数: {key}")
    return default


def _text_fuzzy(text: str | None, target: str) -> bool:
    """宽松文字匹配：去空白标点后双向包含。"""
    from vision.dialog_reader import text_matches
    return text_matches(text or "", target)


def poll(kit: Kit, timeout: float, interval: float,
         predicate: Callable[[], Any], what: str = "条件") -> Any:
    """轮询 predicate 直到真值（返回该值）、超时或停止。"""
    deadline = time.monotonic() + float(timeout)
    attempt = 0
    while True:
        kit.check_stop()
        try:
            value = predicate()
        except StepStopped:
            raise
        except Exception:
            value = None  # 单轮 OCR/截图异常不致命，下一轮重试
        if value:
            return value
        if time.monotonic() >= deadline:
            raise StepTimeout(f"等待{what}超时（{timeout:g}s）")
        attempt += 1
        kit.sleep(interval)


def _read_texts(kit: Kit, roi):
    frame = kit.capture.grab()
    return frame, recognize_in_roi(kit.ocr, frame, roi)


def _click_text(kit: Kit, match, roi, use_regex: bool,
                settle: float) -> tuple[int, int]:
    frame = kit.capture.grab()
    results = recognize_in_roi(kit.ocr, frame, roi)
    hit = find_text(results, match, use_regex=use_regex)
    if hit is None:
        raise StepFailed(f"画面中找不到可点击文字: {match!r}")
    h, w = frame.shape[:2]
    bx, by = kit.click_capture_point(hit.x + hit.w // 2,
                                     hit.y + hit.h // 2, w, h)
    kit.sleep(settle)
    return bx, by


# ── 步骤实现 ────────────────────────────────────────────

def step_log(step, ctx: QuestContext, kit: Kit) -> None:
    kit.log(ctx.interpolate(_get(step, "text", required=True)))


def step_sleep(step, ctx: QuestContext, kit: Kit) -> None:
    kit.sleep(float(_get(step, "secs", 0.5)))


def step_click_npc(step, ctx: QuestContext, kit: Kit) -> None:
    npc = ctx.interpolate(_get(step, "npc", required=True))
    kit.log(f"点击 NPC「{npc}」…")
    result = kit.new_npc_service().find_and_click(npc, click=True)
    if not result.success:
        raise StepFailed(result.message)
    kit.log(f"  {result.message}")
    kit.sleep(float(_get(step, "settle", 0.6)))


def step_wait_dialog(step, ctx: QuestContext, kit: Kit) -> None:
    _wait_text_impl(step, ctx, kit, label="对话")


def step_wait_text(step, ctx: QuestContext, kit: Kit) -> None:
    _wait_text_impl(step, ctx, kit, label="文字")


def _wait_text_impl(step, ctx: QuestContext, kit: Kit, label: str) -> None:
    match = _get(step, "match")
    if match is not None:
        match = ctx.interpolate(match)
    roi = validate_roi(_get(step, "roi"))
    timeout = float(_get(step, "timeout", 10))
    interval = float(_get(step, "interval", 0.6))
    use_regex = bool(_get(step, "regex", False))

    def _check():
        frame, results = _read_texts(kit, roi)
        if not results:
            return None
        if match is None:
            return join_text(results)
        hit = find_text(results, match, use_regex=use_regex)
        return join_text(results) if hit else None

    text = poll(kit, timeout, interval, _check, what=label)
    ctx.set("dialog", text)
    kit.log(f"  已等到{label}：{text[:40]}")


def step_choose_option(step, ctx: QuestContext, kit: Kit) -> None:
    match = ctx.interpolate(_get(step, "match", required=True))
    roi = validate_roi(_get(step, "roi"))
    timeout = float(_get(step, "timeout", 8))
    interval = float(_get(step, "interval", 0.6))
    settle = float(_get(step, "settle", 0.5))

    def _find():
        frame = kit.capture.grab()
        results = recognize_in_roi(kit.ocr, frame, roi)
        hit = find_text(results, match)
        return (frame, hit) if hit else None

    frame, hit = poll(kit, timeout, interval, _find, what=f"选项「{match}」")
    h, w = frame.shape[:2]
    bx, by = kit.click_capture_point(hit.x + hit.w // 2,
                                     hit.y + hit.h // 2, w, h)
    kit.log(f"  点击选项「{match}」B屏=({bx},{by})")
    kit.sleep(settle)


def step_read_route(step, ctx: QuestContext, kit: Kit) -> None:
    """读当前对话 → 按关键词查路线表 → 存入上下文。

    路线表 JSON：{ "关键词": {"target": [x,y], "npc": "...", "scene": "..."} }
    取最长匹配的关键词。target 为小地图点击点（B 屏绝对坐标）。
    """
    map_rel = _get(step, "route_map", required=True)
    base_dir = str(ctx.get("_task_dir", ""))
    map_path = map_rel if os.path.isabs(map_rel) else os.path.join(base_dir, map_rel)
    if not os.path.exists(map_path):
        raise StepFailed(f"路线表不存在: {map_path}")
    with open(map_path, "r", encoding="utf-8") as f:
        route_map: dict = json.load(f)

    roi = validate_roi(_get(step, "roi"))
    store_as = _get(step, "store_as", "route")

    frame = kit.capture.grab()
    results = recognize_in_roi(kit.ocr, frame, roi)
    text = join_text(results)
    ctx.set("dialog", text)
    if not text:
        raise StepFailed("当前没有可读取的对话文字")

    for keyword in sorted(route_map, key=len, reverse=True):
        from vision.dialog_reader import text_matches
        if text_matches(text, keyword):
            ctx.set(store_as, route_map[keyword])
            ctx.set(f"{store_as}.keyword", keyword)
            kit.log(f"  路线命中关键词「{keyword}」→ {route_map[keyword]}")
            return
    raise StepFailed(f"对话内容未命中任何路线关键词：{text[:40]}")


def step_click_point(step, ctx: QuestContext, kit: Kit) -> None:
    """点击小地图/固定点。

    point:    B 屏绝对坐标 [x, y]，支持 ${var}
    point_frac: B 屏归一化坐标 [fx, fy]（0~1）
    mode:     click（默认）/ move（只移动）
    """
    mode = _get(step, "mode", "click")
    settle = float(_get(step, "settle", 0.8))
    if "point" in step and step["point"] is not None:
        raw = ctx.resolve_point(step["point"])
        x, y = int(raw[0]), int(raw[1])
    elif "point_frac" in step and step["point_frac"] is not None:
        fx, fy = (float(v) for v in step["point_frac"])
        x, y = int(fx * kit.mapper.b_width), int(fy * kit.mapper.b_height)
    else:
        raise StepFailed("click_point 需要 point 或 point_frac")

    if mode == "move":
        kit.client.move_to(x, y, kit.mapper.b_width, kit.mapper.b_height)
        kit.log(f"  移动到 B屏({x},{y})")
    else:
        kit.client.click_at(x, y, kit.mapper.b_width, kit.mapper.b_height)
        kit.log(f"  点击 B屏({x},{y})（小地图寻路）")
    kit.sleep(settle)


def step_wait_arrive(step, ctx: QuestContext, kit: Kit) -> None:
    """等待自动寻路到达：轮询场景名 ROI 出现目标场景文字。"""
    scene = ctx.interpolate(_get(step, "scene", required=True))
    roi = validate_roi(_get(step, "roi"))
    timeout = float(_get(step, "timeout", 60))
    interval = float(_get(step, "interval", 1.5)

                      )
    poll(kit, timeout, interval, lambda: _arrived(kit, roi, scene),
         what=f"到达「{scene}」")
    kit.log(f"  已到达「{scene}」")


def _arrived(kit: Kit, roi, scene: str) -> bool:
    frame = kit.capture.grab()
    results = recognize_in_roi(kit.ocr, frame, roi)
    return find_text(results, scene) is not None


def step_dialog_until(step, ctx: QuestContext, kit: Kit) -> None:
    """反复推进对话，直到出现目标文字。

    continue_option: 每轮点击的选项/按钮文字（如「继续」），
                     找不到时回退点 fallback_frac（对话区下方）；
    max_rounds/timeout 双保险。
    """
    match = ctx.interpolate(_get(step, "match", required=True))
    roi = validate_roi(_get(step, "roi"))
    continue_option = _get(step, "continue_option")
    fallback_frac = _get(step, "fallback_frac", [0.5, 0.82])
    max_rounds = int(_get(step, "max_rounds", 12))
    timeout = float(_get(step, "timeout", 30))
    settle = float(_get(step, "settle", 0.7))
    deadline = time.monotonic() + timeout

    for round_idx in range(1, max_rounds + 1):
        kit.check_stop()
        frame = kit.capture.grab()
        h, w = frame.shape[:2]
        results = recognize_in_roi(kit.ocr, frame, roi)
        text = join_text(results)
        ctx.set("dialog", text)
        if find_text(results, match):
            kit.log(f"  对话第 {round_idx} 轮出现「{match}」，完成")
            return

        if time.monotonic() >= deadline:
            raise StepTimeout(f"对话推进超时，仍未出现「{match}」：{text[:30]}")

        clicked = False
        if continue_option:
            hit = find_text(results, continue_option)
            if hit:
                kit.click_capture_point(hit.x + hit.w // 2,
                                        hit.y + hit.h // 2, w, h)
                clicked = True
        if not clicked:
            fx, fy = (float(v) for v in fallback_frac)
            kit.client.click_at(int(fx * kit.mapper.b_width),
                                int(fy * kit.mapper.b_height),
                                kit.mapper.b_width, kit.mapper.b_height)
        kit.log(f"  对话第 {round_idx} 轮，继续推进…")
        kit.sleep(settle)

    raise StepFailed(f"对话 {max_rounds} 轮后仍未出现「{match}」")


# ════════════════════════════════════════════════════════════════
# 保镖任务：组合键 / 固定区域 / 两级地图寻路 / NPC 视觉识别
# ════════════════════════════════════════════════════════════════

_CONTROLLER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(_CONTROLLER_DIR, "assets")
DEBUG_DIR = os.path.join(_CONTROLLER_DIR, "debug")
# 大地图地名/小地图寻路点是精确 UI 热区，默认只允许 ±3px 抖动，
# 避免点到相邻项；可在 MAP JSON 的 defaults.click_jitter_px 覆盖
DEFAULT_MAP_POINT_JITTER_PX = 3.0
MAP_POINT_JITTER_PX = DEFAULT_MAP_POINT_JITTER_PX  # 向后兼容别名


def _map_jitter_px(data: dict) -> float:
    """读取 MAP defaults.click_jitter_px（地图点击 ±N px 拟人抖动）。"""
    d = data.get("defaults")
    if isinstance(d, dict):
        try:
            return max(0.0, float(d.get(
                "click_jitter_px", DEFAULT_MAP_POINT_JITTER_PX)))
        except (TypeError, ValueError):
            pass
    return DEFAULT_MAP_POINT_JITTER_PX


_BUTTON_MAP = {"left": "LE", "right": "RI", "le": "LE", "ri": "RI"}

# 任务栏红字正则：例「阳关的秦溪山和长安的典韦」→ place1,npc1,place2,npc2
_QUEST_PATTERN = re.compile(r'(.+?)的(.+?)和(.+?)的(.+)')
_LEADER_PATTERN = re.compile(r'现在由(.+?)担任(?:队伍)?队长')


def _book(kit: Kit):
    if kit.region_book is None:
        raise StepFailed("该步骤需要固定区域配置，但任务未声明 regions_file")
    return kit.region_book


def _resolve_under_task(ctx: QuestContext, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(
        str(ctx.get("_task_dir", ".")), rel)


def _load_map(ctx: QuestContext, map_rel: str) -> tuple[dict, str]:
    path = _resolve_under_task(ctx, map_rel)
    if not os.path.exists(path):
        raise StepFailed(f"MAP 文件不存在: {path}")
    try:
        return load_map_file(path), path
    except (OSError, json.JSONDecodeError) as exc:
        raise StepFailed(f"MAP 文件无法解析 {path}: {exc}") from exc


def _map_place(data: dict, place: str) -> dict:
    places = data.get("places")
    if not isinstance(places, dict) or place not in places:
        raise StepFailed(
            f"MAP 中找不到地名「{place}」，现有: {list((places or {}).keys())}")
    return places[place]


def _map_npc_def(place_def: dict, place: str, npc: str) -> dict:
    npcs = place_def.get("npcs", {})
    if npc not in npcs:
        raise StepFailed(
            f"MAP「{place}」下找不到 NPC「{npc}」，现有: {list(npcs)}")
    return npcs[npc]


def _valid_box(value, label: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise StepFailed(f"{label} 必须是 [x,y,w,h] 四个数")
    box = [int(v) for v in value]
    if any(v < 0 for v in box) or box[2] <= 0 or box[3] <= 0:
        raise StepFailed(f"{label} 非法或仍是占位值: {value}")
    return box


def _optional_npc_def(ctx: QuestContext, map_rel: str | None,
                      place: str, npc: str) -> dict | None:
    """map 给了就查 NPC 定义（模板/npc_box 兜底）；不给返回 None（纯 OCR）。"""
    if not map_rel:
        return None
    data, _ = _load_map(ctx, map_rel)
    return _map_npc_def(_map_place(data, place), place, npc)


def _interp_match(value, ctx: QuestContext):
    """match 参数支持字符串或字符串列表，元素里的 ${var} 做插值。"""
    if isinstance(value, str):
        return ctx.interpolate(value)
    if isinstance(value, (list, tuple)):
        return [ctx.interpolate(str(v)) for v in value]
    return value


def _click_ocr_hit(kit: Kit, frame, hit, jitter_px: float = 3.0
                   ) -> tuple[int, int]:
    """点 OCR 文字块中心（叠加 ±jitter_px 抖动），返回 B 屏坐标。"""
    h, w = frame.shape[:2]
    j = max(0, int(jitter_px))
    cx = hit.x + hit.w // 2 + (random.randint(-j, j) if j else 0)
    cy = hit.y + hit.h // 2 + (random.randint(-j, j) if j else 0)
    cx = min(max(cx, 0), w - 1)
    cy = min(max(cy, 0), h - 1)
    bx, by = kit.mapper.to_b(cx, cy, w, h)
    kit.client.click_at(bx, by, kit.mapper.b_width, kit.mapper.b_height)
    return bx, by


# ── hotkey ──────────────────────────────────────────────

def step_hotkey(step, ctx: QuestContext, kit: Kit) -> None:
    keys = ctx.interpolate(_get(step, "keys", required=True))
    times = int(_get(step, "times", 1))
    if not 1 <= times <= 10:
        raise StepFailed("times 必须在 1~10 之间")
    kit.log(f"  ⌨ 发送组合键 {keys.upper()} ×{times}")
    kit.client.send_keys(keys, times)
    kit.sleep(float(_get(step, "settle", 0.3)))


# ── click_region ────────────────────────────────────────

def step_click_region(step, ctx: QuestContext, kit: Kit) -> None:
    name = _get(step, "region", required=True)
    button = _BUTTON_MAP.get(str(_get(step, "button", "left")).lower())
    if button is None:
        raise StepFailed("button 只支持 left/right")
    try:
        bx, by = _book(kit).region_click_b(name)
    except RegionConfigError as exc:
        raise StepFailed(str(exc)) from exc
    side = "右键" if button == "RI" else "左键"
    kit.log(f"  🖱 {side}点击固定区域「{name}」→ B屏({bx},{by})")
    kit.client.click_at(bx, by, kit.mapper.b_width, kit.mapper.b_height,
                        button)
    kit.sleep(float(_get(step, "settle", 0.4)))


# ── transfer_leader ──────────────────────────────────────

def step_transfer_leader(step, ctx: QuestContext, kit: Kit) -> None:
    """切换队长：右键点队友槽位 → 左键点「给与队长/给回队长」菜单，重复 N 次。

    两种行为：
      - 给与队长（单次）：times=1，把队长交给队友
      - 给回队长（5次）：times=5，自己收回队长（每点一次队长在队伍里轮一圈）

    菜单坐标固定，不需要 OCR，直接点固定区域即可。
    """
    slot_region = _get(step, "slot_region", "team_slot")
    confirm_region = _get(step, "confirm_region", "give_leader_confirm")
    times = int(_get(step, "times", 1))
    settle_between = float(_get(step, "settle", 0.6))
    if not 1 <= times <= 10:
        raise StepFailed("times 必须在 1~10 之间")

    book = _book(kit)
    # 预先解析两个区域坐标，避免每轮重复读
    try:
        slot_b = book.region_click_b(slot_region)
        confirm_b = book.region_click_b(confirm_region)
    except RegionConfigError as exc:
        raise StepFailed(str(exc)) from exc

    label = "给与队长" if times == 1 else f"给回队长（×{times}）"
    kit.log(f"  🔄 {label}：右键「{slot_region}」→ 左键「{confirm_region}」")

    for i in range(1, times + 1):
        kit.check_stop()
        kit.log(f"    第 {i}/{times} 次：右键({slot_b[0]},{slot_b[1]}) "
                f"→ 左键({confirm_b[0]},{confirm_b[1]})")
        kit.client.click_at(slot_b[0], slot_b[1],
                            kit.mapper.b_width, kit.mapper.b_height, "RI")
        kit.sleep(0.25)  # 等右键菜单弹出
        kit.client.click_at(confirm_b[0], confirm_b[1],
                            kit.mapper.b_width, kit.mapper.b_height, "LE")
        if i < times:
            kit.sleep(settle_between)


# ── map_click ───────────────────────────────────────────

def step_map_click(step, ctx: QuestContext, kit: Kit) -> None:
    map_rel = _get(step, "map", required=True)
    level = _get(step, "level", "big")
    place = ctx.interpolate(_get(step, "place", required=True))
    data, _ = _load_map(ctx, map_rel)
    place_def = _map_place(data, place)
    book = _book(kit)

    if level == "big":
        xy = place_def.get("bigmap_click")
        label = f"大地图地名「{place}」"
    elif level == "small":
        npc = ctx.interpolate(_get(step, "npc", required=True))
        npc_def = _map_npc_def(place_def, place, npc)
        xy = npc_def.get("minimap_click")
        label = f"小地图寻路点「{place}/{npc}」"
    else:
        raise StepFailed("level 只支持 big（大地图地名）/ small（小地图寻路点）")

    if not isinstance(xy, (list, tuple)) or len(xy) != 2 or \
            any(int(v) < 0 for v in xy):
        raise StepFailed(f"{label} 的坐标未配置或仍是占位值: {xy}")
    bx, by = book.game_point_to_b(float(xy[0]), float(xy[1]),
                                  _map_jitter_px(data))
    kit.log(f"  🗺 点击{label} → B屏({bx},{by})")
    kit.client.click_at(bx, by, kit.mapper.b_width, kit.mapper.b_height)
    kit.sleep(float(_get(step, "settle", 0.6)))


# ── NPC 视觉识别（OCR 名字 → 模板匹配）──────────────────

def _detect_npc(kit: Kit, npc: str, npc_def: dict | None = None,
               warn_missing_template: bool = True):
    """在游戏窗口 ROI 内检测目标 NPC。

    返回 (method, u, v, frame, info)；未命中返回 None。
    method='ocr'     info={"conf": 置信度}
    method='template' info={"score": 匹配分, "rect": (x,y,w,h)}（坐标相对全帧）
    """
    book = _book(kit)
    roi = book.game_roi(kit.mapper.b_width, kit.mapper.b_height)
    frame = kit.capture.grab()

    results = recognize_in_roi(kit.ocr, frame, roi)
    hit = find_npc(results, npc)
    if hit is not None:
        # 与 NpcService hover 同一套身体点公式：名字框中线上方
        # height_mults[0]（默认 2.5 倍名字高度）处 = 人物身体，
        # 不直接点名字文字（点文字游戏可能不响应）。
        mults = kit.npc_config.height_mults
        body_mult = float(mults[0]) if mults else 2.5
        fh, fw = frame.shape[:2]
        cx = hit.x + hit.w // 2
        body_y = int(hit.y - body_mult * hit.h)
        # 小幅随机抖动，避免每次点击同一像素
        jx = max(2, int(hit.w * 0.08))
        jy = max(2, int(hit.h * 0.12))
        cu = min(max(cx + random.randint(-jx, jx), 0), fw - 1)
        cv = min(max(body_y + random.randint(-jy, jy), 0), fh - 1)
        return "ocr", cu, cv, frame, {
            "conf": hit.confidence, "body_mult": body_mult}

    templ_name = (npc_def or {}).get("template") if npc_def else None
    if templ_name:
        # template 支持单字符串或字符串数组（多视角/多姿态模板）
        if isinstance(templ_name, str):
            templ_paths = [templ_name]
        else:
            templ_paths = [t for t in templ_name if t]
        sub, ox, oy = crop_roi(frame, roi)
        best_hit = None
        best_path = ""
        missing = []
        for tpath in templ_paths:
            templ = load_template(os.path.join(ASSETS_DIR, tpath))
            if templ is None:
                missing.append(tpath)
                continue
            found = match_template(sub, templ)
            if found is not None:
                if best_hit is None or found.score > best_hit.score:
                    best_hit = found
                    best_path = tpath
        if best_hit is not None:
            rect = (best_hit.x + ox, best_hit.y + oy,
                    best_hit.w, best_hit.h)
            cu = best_hit.x + ox + best_hit.w // 2
            cv = best_hit.y + oy + best_hit.h // 2
            info = {"score": best_hit.score, "scale": best_hit.scale,
                    "rect": rect, "template": best_path}
            if missing and warn_missing_template:
                kit.log(f"  ⚠ 部分模板图读取失败：{missing}"
                        f"（命中 {best_path}）")
            return "template", cu, cv, frame, info
        if warn_missing_template and missing:
            kit.log(f"  ⚠ 模板图读取失败（请检查 assets/）：{missing}")
    return None


def _save_npc_debug(frame, npc: str, method: str, rect=None) -> str | None:
    """命中后存一张带框的全帧图到 debug/，方便核对点得对不对。"""
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%H%M%S")
        img = frame.copy()
        if rect is not None:
            x, y, w, h = rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        path = os.path.join(DEBUG_DIR, f"npc_{method}_{stamp}.png")
        cv2.imwrite(path, img)
        return path
    except Exception:
        return None


def step_wait_npc(step, ctx: QuestContext, kit: Kit) -> None:
    map_rel = _get(step, "map", required=True)
    place = ctx.interpolate(_get(step, "place", required=True))
    npc = ctx.interpolate(_get(step, "npc", required=True))
    timeout = float(_get(step, "timeout", 90))
    interval = float(_get(step, "interval", 1.2))

    data, _ = _load_map(ctx, map_rel)
    npc_def = _map_npc_def(_map_place(data, place), place, npc)
    state = {"round": 0, "warned": False}

    def _check():
        state["round"] += 1
        warn = not state["warned"]
        found = _detect_npc(kit, npc, npc_def,
                            warn_missing_template=warn)
        if found is None and warn:
            state["warned"] = True
        if found is None and state["round"] % 5 == 0:
            kit.log(f"  寻路中…还没见到「{npc}」（第 {state['round']} 轮）")
        return found

    method, _u, _v, _frame, info = poll(
        kit, timeout, interval, _check, what=f"见到 NPC「{npc}」")

    if method == "ocr":
        kit.log(f"  ✓ 见到「{npc}」｜识别方式：OCR 文字"
                f"（置信度 {info['conf']:.2f}，身体点倍数 "
                f"{info.get('body_mult', 2.5):g}）")
    else:
        kit.log(f"  ✓ 见到「{npc}」｜识别方式：模板匹配"
                f"（相似度 {info['score']:.2f}，尺度 {info['scale']:.2f}）")
        path = _save_npc_debug(_frame, npc, method, info["rect"])
        if path:
            kit.log(f"  命中调试图: {path}")
    ctx.set("found_npc.method", method)


def step_click_quest_npc(step, ctx: QuestContext, kit: Kit) -> None:
    """点目标 NPC：轮询搜索窗口内识别 → 命中即点；超时后 npc_box 盲点兜底。

    识别方式优先级：① OCR 名字 → ② 游戏窗口内模板匹配。
    NPC 可能在活动区域内随机移动，单次检测可能因 NPC 暂时走出画面而失败。
    因此进入一个「搜索窗口」：在 search_secs 秒内反复识别（间隔 search_interval），
    NPC 走回画面即可命中点击；窗口结束仍未命中才用 npc_box 盲点兜底。

    map 可选：不传 map 时只做 OCR 识别，没有模板和 npc_box，
    搜索窗口结束仍未命中直接判失败（交给上层 retry）。
    """
    map_rel = _get(step, "map")
    place = ctx.interpolate(_get(step, "place", required=True))
    npc = ctx.interpolate(_get(step, "npc", required=True))
    search_secs = float(_get(step, "search_secs", 8))
    search_interval = float(_get(step, "search_interval", 0.5))

    npc_def = _optional_npc_def(ctx, map_rel, place, npc)
    book = _book(kit)

    kit.log(f"  搜索 NPC「{npc}」（窗口 {search_secs:g}s，"
            f"间隔 {search_interval:g}s"
            f"{'，纯 OCR 无兜底' if npc_def is None else ''}）…")
    deadline = time.monotonic() + search_secs
    found = None
    last_log = 0.0
    while True:
        kit.check_stop()
        found = _detect_npc(kit, npc, npc_def, warn_missing_template=False)
        if found is not None:
            break
        if time.monotonic() >= deadline:
            break
        now = time.monotonic()
        if now - last_log >= 1.5:
            last_log = now
            kit.log(f"  搜索中… 还未识别到「{npc}」（NPC 可能在画面外，"
                    f"继续等待其走回画面）")
        kit.sleep(search_interval)

    if found is not None:
        method, u, v, frame, info = found
        h, w = frame.shape[:2]
        bx, by = kit.mapper.to_b(u, v, w, h)
        if method == "template":
            _save_npc_debug(frame, npc, method, info["rect"])
            how = f"模板匹配（相似度 {info['score']:.2f}，点模型中心）"
        else:
            how = (f"OCR 文字（置信度 {info['conf']:.2f}，"
                   f"点名字上方身体点 ×{info.get('body_mult', 2.5):g}）")
        kit.log(f"  点击 NPC「{npc}」｜识别方式：{how} → B屏({bx},{by})")
    else:
        if npc_def is None:
            raise StepFailed(
                f"搜索窗口 {search_secs:g}s 内始终未识别到 NPC「{npc}」，"
                f"且本步骤未配置 map/npc_box，无法盲点兜底")
        box = _valid_box(npc_def.get("npc_box"),
                         f"NPC「{npc}」的 npc_box 保底点击框")
        bx, by = book.game_box_click_b(box)
        kit.log(f"  点击 NPC「{npc}」｜搜索窗口 {search_secs:g}s 内均未识别到，"
                f"使用 npc_box 保底盲点 → B屏({bx},{by})")

    kit.client.click_at(bx, by, kit.mapper.b_width, kit.mapper.b_height)
    kit.sleep(float(_get(step, "settle", 0.8)))


def step_wait_arrive_pos(step, ctx: QuestContext, kit: Kit) -> None:
    """按 HUD 场景名+坐标 + NPC 可见性判定寻路到达（先动后静双条件）。

    判定规则：
      阶段1 观察到「移动」——坐标相对起点变化超过 coord_tol，
             或读到的场景名一度不是目标场景（发生过场景切换）；
      阶段2 进入目标场景后，坐标相对静止锚点连续稳定 ≥ stable_secs
             （相邻读数差 ≤ coord_tol，吸收 OCR 个位抖动与待机微移）；
      阶段3 静止窗口结束时做一次目标 NPC 识别确认（OCR 名字 → 模板）：
             · 识别到 → 到达成立，立即结束；
             · 没识别到 → 本次静止作废，重新锚定再等一个窗口；
             · 连续 confirm_checks 个窗口（默认 4，合计约 10s）都没
               识别到，也兜底判到达（后续 click_quest_npc 自己再找）。
    OCR 读不到 HUD（过图加载等）时本轮不计入，静止计时重新锚定。
    """
    place = ctx.interpolate(_get(step, "place", required=True))
    npc = ctx.interpolate(_get(step, "npc", required=True))
    map_rel = _get(step, "map")
    region_name = _get(step, "region", "hud_status")
    timeout = float(_get(step, "timeout", 90))
    interval = float(_get(step, "interval", 0.5))
    stable_secs = float(_get(step, "stable_secs", 2.5))
    coord_tol = int(_get(step, "coord_tol", 4))
    confirm_checks = int(_get(step, "confirm_checks", 4))
    if confirm_checks < 1:
        raise StepFailed("confirm_checks 必须 ≥ 1")

    # map 可选：纯 OCR 寻路方案不查 MAP，NPC 确认只靠 OCR 名字
    npc_def = _optional_npc_def(ctx, map_rel, place, npc)
    if npc_def is None:
        kit.log("  未配置 map：到达后的 NPC 确认仅使用 OCR（无模板匹配）")

    book = _book(kit)
    try:
        roi = book.region_roi(region_name,
                              kit.mapper.b_width, kit.mapper.b_height)
    except RegionConfigError as exc:
        raise StepFailed(str(exc)) from exc

    kit.log(f"  等待寻路到达「{place}/{npc}」：先移动 → 静止 {stable_secs:g}s"
            f" → 连续确认 NPC（最多 {confirm_checks} 轮，约 "
            f"{confirm_checks * interval:g}s 兜底）")

    deadline = time.monotonic() + timeout
    start_xy: tuple[int, int] | None = None
    moved = False
    anchor: tuple[int, int] | None = None
    anchor_at = 0.0
    round_idx = 0
    bad_reads = 0
    checks_done = 0
    last_beat = 0.0

    while True:
        kit.check_stop()
        round_idx += 1
        frame = kit.capture.grab()
        results = recognize_in_roi(kit.ocr, frame, roi)
        state = parse_hud(join_text(results))

        if state is None:
            # 过图加载/遮挡：本轮无效，静止锚点作废重来
            bad_reads += 1
            anchor = None
            if bad_reads % 4 == 1:
                kit.log("  HUD 暂时读不到（可能过图加载），等待…")
        else:
            bad_reads = 0
            xy = (state.x, state.y)
            if start_xy is None:
                start_xy = xy
                kit.log(f"  起点坐标 {start_xy}（场景：{state.scene}）")
            # 阶段1：移动判定（坐标走够远 或 曾不在目标场景）
            if not moved:
                if not coord_close(xy, start_xy, coord_tol):
                    moved = True
                    kit.log(f"  检测到移动：{start_xy} → {xy}")
                elif not scene_matches(state.scene, place):
                    moved = True
                    kit.log(f"  检测到场景切换：当前「{state.scene}」"
                            f"（目标「{place}」），判定已出发")
            # 阶段2：目标场景内的静止锚定
            if moved and scene_matches(state.scene, place):
                now = time.monotonic()
                if anchor is None or not coord_close(xy, anchor, coord_tol):
                    anchor = xy
                    anchor_at = now
                else:
                    stable_for = now - anchor_at
                    if stable_for >= stable_secs:
                        # 阶段3：静止窗口成立，做一次 NPC 确认
                        checks_done += 1
                        kit.log(f"  坐标已静止 {stable_for:.1f}s，"
                                f"第 {checks_done}/{confirm_checks} 次"
                                f"确认 NPC「{npc}」…")
                        found = _detect_npc(kit, npc, npc_def)
                        if found is not None:
                            method, _u, _v, f2, info = found
                            if method == "template":
                                _save_npc_debug(f2, npc, method,
                                                info["rect"])
                                how = (f"模板匹配（相似度 "
                                       f"{info['score']:.2f}）")
                            else:
                                how = f"OCR 文字（置信度 {info['conf']:.2f}）"
                            kit.log(f"  ✓ 已到达「{place}」{xy}，"
                                    f"NPC 确认成功：{how}")
                            ctx.set("arrive.scene", state.scene)
                            ctx.set("arrive.x", state.x)
                            ctx.set("arrive.y", state.y)
                            ctx.set("arrive.method", method)
                            return
                        if checks_done >= confirm_checks:
                            kit.log(f"  ⚠ 已连续 {confirm_checks} 次未确认到"
                                    f" NPC「{npc}」（约 "
                                    f"{confirm_checks * interval:g}s 连续搜索），"
                                    f"按兜底规则判定到达（交给点击步骤继续找）")
                            ctx.set("arrive.scene", state.scene)
                            ctx.set("arrive.x", state.x)
                            ctx.set("arrive.y", state.y)
                            ctx.set("arrive.method", "timeout_fallback")
                            return
                        # 坐标仍静止，不清零锚点；下一轮直接再确认 NPC
                        # （NPC 可能在画面边缘/外游走，连续轮询等其走回画面）
                        kit.log(f"  ✗ 第 {checks_done} 次未确认到「{npc}」，"
                                f"坐标仍静止，继续轮询…")
                    elif now - last_beat >= 1.0:
                        last_beat = now
                        kit.log(f"  已到「{state.scene}」{xy}，"
                                f"静止确认 {stable_for:.1f}/{stable_secs:g}s")
            elif moved and round_idx % 6 == 0:
                kit.log(f"  寻路中…当前「{state.scene}」{xy}")

        if time.monotonic() >= deadline:
            raise StepTimeout(
                f"等待到达「{place}」超时（{timeout:g}s）："
                f"moved={moved}，起点={start_xy}，锚点={anchor}，"
                f"NPC 确认 {checks_done}/{confirm_checks}")
        kit.sleep(interval)


def step_dialog_choose(step, ctx: QuestContext, kit: Kit) -> None:
    """对话框选选项：绿色文字 = 可点选项，白色 = 描述。

    流程：轮询 OCR region 区域 → 找到 match 文字且该处像素为绿色
    → 在该选项所在行的随机位置点击（不点文字固定中心，防检测）。

    optional=true 时超时→日志跳过继续；false 时超时→失败。
    点击坐标 x 覆盖整个对话框宽度（"那一行"），y 在文字上下范围内随机。
    """
    region_name = _get(step, "region", "dialog_area")
    match_text = ctx.interpolate(_get(step, "match", required=True))
    timeout = float(_get(step, "timeout", 5))
    interval = float(_get(step, "interval", 0.5))
    settle = float(_get(step, "settle", 1.0))
    optional = bool(_get(step, "optional", False))

    book = _book(kit)
    try:
        roi = book.region_roi(region_name,
                              kit.mapper.b_width, kit.mapper.b_height)
    except RegionConfigError as exc:
        raise StepFailed(str(exc)) from exc

    tag = "可选" if optional else "必须"
    kit.log(f"  对话选项「{match_text}」（{tag}）轮询中…")

    deadline = time.monotonic() + timeout
    while True:
        kit.check_stop()
        frame = kit.capture.grab()
        h, w = frame.shape[:2]
        results = recognize_in_roi(kit.ocr, frame, roi)
        # 遍历所有 OCR 结果：文字匹配 + 绿色验证
        for r in results:
            if match_text not in (r.text or "") and not _text_fuzzy(r.text, match_text):
                continue
            crop = frame[r.y:r.y + r.h, r.x:r.x + r.w]
            if not has_green_text(crop):
                continue   # 白色描述文字，跳过
            # 绿色选项命中：在整行随机点击
            roi_x2 = int(roi[2] * w)
            margin = 5
            # 向左最多 5px，向右可到对话框右边缘
            click_x = random.randint(max(0, r.x - margin),
                                     min(roi_x2 - margin, w - 1))
            # 上下各不超过 2px
            click_y = random.randint(
                max(0, r.y - 2),
                min(h - 1, r.y + r.h + 2))
            bx, by = kit.mapper.to_b(click_x, click_y, w, h)
            kit.client.click_at(bx, by, kit.mapper.b_width,
                                kit.mapper.b_height, button="LE")
            kit.log(f"  ✓ 点击对话选项「{match_text}」（绿色确认）"
                    f"行内随机点 → B屏({bx},{by})")
            kit.sleep(settle)
            return

        if time.monotonic() >= deadline:
            if optional:
                kit.log(f"  可选对话选项「{match_text}」未出现，跳过")
                return
            raise StepTimeout(
                f"对话选项「{match_text}」超时未出现（{timeout:g}s）")
        kit.sleep(interval)


def step_check_quest_accepted(step, ctx: QuestContext, kit: Kit) -> None:
    """接取成功确认：对话框红色文字 + 任务追踪栏文案匹配。

    成功条件（两者同时满足）：
      ① 对话框区域有红色像素（HSV 检测，快）
      ② 任务追踪栏 OCR 文字命中正则「(.+?)的(.+?)和(.+?)的(.+)」
         → 提取两组 place/npc 存入上下文供后续寻路用
    失败（超时）→ 按 ESC 关闭可能残留的对话框 → 抛 StepTimeout。
    """
    dialog_region = _get(step, "dialog_region", "dialog_area")
    quest_region = _get(step, "quest_region", "quest_board")
    timeout = float(_get(step, "timeout", 5))
    interval = float(_get(step, "interval", 0.5))
    threshold = int(_get(step, "red_threshold", 5))

    book = _book(kit)
    try:
        dialog_roi = book.region_roi(
            dialog_region, kit.mapper.b_width, kit.mapper.b_height)
        quest_roi = book.region_roi(
            quest_region, kit.mapper.b_width, kit.mapper.b_height)
    except RegionConfigError as exc:
        raise StepFailed(str(exc)) from exc

    kit.log("  等待接取确认（对话框红色 + 任务板文案匹配）…")
    deadline = time.monotonic() + timeout
    last_beat = 0.0

    while True:
        kit.check_stop()
        frame = kit.capture.grab()
        # ① 快速红色检测（对话框）
        dialog_crop = crop_roi(frame, dialog_roi)[0]
        d_red = has_red_text(dialog_crop, threshold)
        if not d_red:
            now = time.monotonic()
            if now - last_beat >= 1.0:
                last_beat = now
                kit.log("  等待中… 对话框红色:否")
            if now >= deadline:
                kit.log("  ✗ 接取失败：对话框无红色文字，按 ESC 关闭")
                kit.client.send_keys("esc")
                raise StepTimeout("接取确认超时：对话框无红色文字")
            kit.sleep(interval)
            continue
        # ② 对话框已红 → OCR 任务栏 → 正则匹配
        quest_results = recognize_in_roi(kit.ocr, frame, quest_roi)
        quest_text = join_text(quest_results).replace(" ", "").replace("\u3000", "")
        m = _QUEST_PATTERN.search(quest_text)
        if m:
            t1_place, t1_npc, t2_place, t2_npc = m.groups()
            kit.log("  ✓ 接取成功：对话框红色 + 任务板匹配")
            kit.log(f"    任务1：{t1_place}的{t1_npc}")
            kit.log(f"    任务2：{t2_place}的{t2_npc}")
            ctx.set("quest_accepted", True)
            ctx.set("quest.t1.place", t1_place)
            ctx.set("quest.t1.npc", t1_npc)
            ctx.set("quest.t2.place", t2_place)
            ctx.set("quest.t2.npc", t2_npc)
            return
        now = time.monotonic()
        if now - last_beat >= 1.0:
            last_beat = now
            kit.log(f"  等待中… 对话框红色:是 任务板:未匹配"
                    f"（{quest_text[:20]}…）")
        if now >= deadline:
            kit.log("  ✗ 接取失败：任务板未匹配任务文案，按 ESC 关闭")
            kit.client.send_keys("esc")
            raise StepTimeout(
                f"接取确认超时：任务板文案未匹配，原文={quest_text[:40]}")
        kit.sleep(interval)


def step_check_leader(step, ctx: QuestContext, kit: Kit) -> None:
    """切换队长后，OCR 提示区，匹配「现在由XX担任队长」提取当前队长。

    流程：轮询 OCR 提示区域 → 正则「现在由(.+?)担任(?:队伍)?队长」
         → 命中：记录 ctx.leader.name，打印当前队长
         → 超时：抛 StepTimeout（不阻断，可配合 retry/optional 使用）

    调试：设 debug_save: true 会把第一轮裁剪出的提示区存到 debug/{region}.png，
         方便确认截图区域是否正确（如果图是黑的/错位，说明区域坐标或
         capture 框选范围不对）。
    """
    region_name = _get(step, "region", "leader_hint")
    timeout = float(_get(step, "timeout", 10))
    interval = float(_get(step, "interval", 0.5))
    debug_save = bool(_get(step, "debug_save", False))
    settle = float(_get(step, "settle", 0.5))  # 切换队长后等提示弹出

    book = _book(kit)
    try:
        roi = book.region_roi(region_name,
                              kit.mapper.b_width, kit.mapper.b_height)
    except RegionConfigError as exc:
        raise StepFailed(str(exc)) from exc

    if settle > 0:
        kit.log(f"  等 {settle:g}s 让队长切换提示弹出…")
        kit.sleep(settle)

    kit.log(f"  OCR 区域「{region_name}」等待队长提示… "
            f"(ROI=({roi[0]:.3f},{roi[1]:.3f},{roi[2]:.3f},{roi[3]:.3f}))")
    deadline = time.monotonic() + timeout
    last_log = 0.0
    first_iter = True

    while True:
        kit.check_stop()
        frame = kit.capture.grab()
        sub, ox, oy = crop_roi(frame, roi)
        if first_iter and debug_save:
            first_iter = False
            os.makedirs("debug", exist_ok=True)
            debug_path = f"debug/{region_name}.png"
            cv2.imwrite(debug_path, sub)
            kit.log(f"  🖼 已保存 OCR 裁剪图到 {debug_path} "
                    f"(尺寸 {sub.shape[1]}x{sub.shape[0]})，请人工确认是否截到提示文字")
        results = kit.ocr.recognize(sub)
        # OCR 结果坐标是裁剪子图内的，加偏移回全图（仅供日志，不影响匹配）
        text = " ".join(r.text for r in results).replace(" ", "").replace("\u3000", "")
        m = _LEADER_PATTERN.search(text)
        if m:
            leader = m.group(1)
            ctx.set("leader.name", leader)
            kit.notify_leader(leader)
            kit.log(f"  ✓ 当前队长：{leader}")
            return
        now = time.monotonic()
        if now - last_log >= 1.5:
            last_log = now
            snippet = text[:30] + "…" if len(text) > 30 else text
            kit.log(f"  等待中… 提示区暂未匹配（OCR 原文={snippet!r}）")
        if now >= deadline:
            raise StepTimeout(
                f"等待队长切换提示超时（{timeout:g}s）："
                f"提示区未匹配「现在由…担任队长」，原文={text[:40]!r}")
        kit.sleep(interval)


def step_switch_tab(step, ctx: QuestContext, kit: Kit) -> None:
    """切换标签：OCR 队伍标签栏，按当前队长名匹配并点击对应标签。

    用于切换队长后，把视角切回当前队长对应的队员标签。
    当前队长名从 ctx.leader.name 读取（由 check_leader 步骤写入）。
    """
    region_name = _get(step, "region", "team_tabs")
    leader_key = _get(step, "leader_key", "leader.name")
    timeout = float(_get(step, "timeout", 8))
    interval = float(_get(step, "interval", 0.4))
    settle = float(_get(step, "settle", 0.4))

    leader = ctx.get(leader_key)
    if not leader:
        raise StepFailed(
            f"上下文中没有「{leader_key}」，请先执行 check_leader 步骤检测当前队长")

    book = _book(kit)
    try:
        roi = book.region_roi(region_name,
                              kit.mapper.b_width, kit.mapper.b_height)
    except RegionConfigError as exc:
        raise StepFailed(str(exc)) from exc

    kit.log(f"  切换标签：在「{region_name}」中查找队长「{leader}」并点击")
    deadline = time.monotonic() + timeout
    last_log = 0.0

    while True:
        kit.check_stop()
        frame = kit.capture.grab()
        results = recognize_in_roi(kit.ocr, frame, roi)
        hit = find_text(results, leader)
        if hit is not None:
            h, w = frame.shape[:2]
            bx, by = kit.click_capture_point(
                hit.x + hit.w // 2, hit.y + hit.h // 2, w, h)
            kit.log(f"  ✓ 命中标签「{leader}」（置信度 {hit.confidence:.2f}）"
                    f"→ 点击 B屏({bx},{by})")
            kit.sleep(settle)
            return
        now = time.monotonic()
        if now - last_log >= 1.5:
            last_log = now
            snippet = join_text(results)[:30]
            kit.log(f"  标签栏未找到「{leader}」（OCR={snippet!r}）")
        if now >= deadline:
            raise StepTimeout(
                f"切换标签超时（{timeout:g}s）：标签栏未找到「{leader}」")
        kit.sleep(interval)


# ── OCR 地图寻路方案：区域找字点击 / NPC 树滚动选择 ──────

def step_click_text(step, ctx: QuestContext, kit: Kit) -> None:
    """在 regions 命名区域内 OCR 找文字并点击（通用）。

    用于：世界地图点场景名进小地图、点「普通NPC」节点、点「寻」按钮等。
    match 支持单个字符串或候选列表（任一命中即可），支持 ${var} 插值。
    命中后点文字块中心，叠加 ±jitter_px 随机偏移（默认 3px）。
    """
    region_name = _get(step, "region", required=True)
    match = _interp_match(_get(step, "match", required=True), ctx)
    timeout = float(_get(step, "timeout", 6))
    interval = float(_get(step, "interval", 0.5))
    settle = float(_get(step, "settle", 0.5))
    jitter_px = float(_get(step, "jitter_px", 3))

    book = _book(kit)
    try:
        roi = book.region_roi(region_name,
                              kit.mapper.b_width, kit.mapper.b_height)
    except RegionConfigError as exc:
        raise StepFailed(str(exc)) from exc

    kit.log(f"  在区域「{region_name}」内 OCR 查找文字 {match!r} …")

    def _find():
        frame = kit.capture.grab()
        results = recognize_in_roi(kit.ocr, frame, roi)
        hit = find_text(results, match)
        return (frame, hit) if hit else None

    frame, hit = poll(kit, timeout, interval, _find,
                      what=f"区域「{region_name}」内文字 {match!r}")
    bx, by = _click_ocr_hit(kit, frame, hit, jitter_px)
    ctx.set("click_text.hit", hit.text)
    kit.log(f"  ✓ 点击文字「{hit.text}」（置信度 {hit.confidence:.2f}）"
            f"→ B屏({bx},{by})")
    kit.sleep(settle)


def step_minimap_npc_pick(step, ctx: QuestContext, kit: Kit) -> None:
    """小地图右侧 NPC 树里选中目标 NPC：展开目录 → OCR → 滚轮翻页。

    流程（每轮都先 OCR 列表区）：
      1. 已能看到目标 NPC 名 → 直接点击（重入安全：列表已展开时绝不再点
         「普通NPC」，防止把树点折叠）；
      2. 看不到目标、且列表里只有「普通NPC」节点没有任何其它名字
         → 点一次节点展开列表（只点一次）；
      3. 列表已展开但没目标 → 指针移进列表区下发滚轮下滚指令，
         settle 后重新 OCR；
      4. 触底判定：连续 bottom_same 次滚动后 OCR 文字指纹不变 → 列表已到底，
         失败；另有 max_scrolls 与 timeout 双保险；OCR 为空不计触底
         （可能面板还在加载）。
    """
    npc = ctx.interpolate(_get(step, "npc", required=True))
    region_name = _get(step, "region", "minimap_npc_panel")
    folder_match = _interp_match(_get(step, "folder_match", "普通NPC"), ctx)
    timeout = float(_get(step, "timeout", 20))
    settle = float(_get(step, "settle", 0.45))
    scroll_ticks = int(_get(step, "scroll_ticks", 2))
    max_scrolls = int(_get(step, "max_scrolls", 15))
    bottom_same = int(_get(step, "bottom_same", 2))
    jitter_px = float(_get(step, "jitter_px", 3))
    if scroll_ticks < 1 or max_scrolls < 1 or bottom_same < 1:
        raise StepFailed("scroll_ticks/max_scrolls/bottom_same 必须 ≥ 1")

    book = _book(kit)
    try:
        roi = book.region_roi(region_name,
                              kit.mapper.b_width, kit.mapper.b_height)
    except RegionConfigError as exc:
        raise StepFailed(str(exc)) from exc

    kit.log(f"  在 NPC 列表「{region_name}」中查找「{npc}」"
            f"（单次下滚 {scroll_ticks} 格，最多 {max_scrolls} 次）…")

    deadline = time.monotonic() + timeout
    folder_clicked = False
    scrolls = 0
    prev_sig: str | None = None
    same = 0
    waited_folder = 0.0

    while True:
        kit.check_stop()
        frame = kit.capture.grab()
        results = recognize_in_roi(kit.ocr, frame, roi)
        hit = find_text(results, npc)
        if hit is not None:
            bx, by = _click_ocr_hit(kit, frame, hit, jitter_px)
            ctx.set("minimap_npc.found", hit.text)
            kit.log(f"  ✓ NPC 列表命中「{hit.text}」（置信度 "
                    f"{hit.confidence:.2f}，滚动 {scrolls} 次后）"
                    f"→ B屏({bx},{by})")
            kit.sleep(settle)
            return

        if time.monotonic() >= deadline:
            raise StepTimeout(
                f"NPC 列表查找「{npc}」超时（{timeout:g}s，已滚动 "
                f"{scrolls} 次）")

        # 目录展开判定：除节点文字外还有别的名字 = 已展开，绝不再点节点
        others = [r for r in results
                  if not text_matches(r.text or "", folder_match)]
        expanded = bool(others)

        if not folder_clicked and not expanded:
            folder = find_text(results, folder_match)
            if folder is None:
                # 面板可能还在加载：等一等，不滚动（折叠状态滚动无意义）
                kit.log("  列表区暂未出现「普通NPC」节点，等待面板展开…")
                kit.sleep(max(settle, 0.2))
                waited_folder += max(settle, 0.2)
                if waited_folder >= timeout:
                    raise StepTimeout(
                        f"等待「普通NPC」节点超时（{timeout:g}s），"
                        f"请检查区域「{region_name}」是否框对")
                continue
            bx, by = _click_ocr_hit(kit, frame, folder, jitter_px)
            folder_clicked = True
            kit.log(f"  点击「{folder.text}」展开 NPC 目录 → B屏({bx},{by})")
            kit.sleep(settle)
            prev_sig = None   # 展开后列表内容会变，清掉触底指纹
            same = 0
            continue
        folder_clicked = True  # 进来时就已展开，后续不再管目录

        # 已展开但没目标 → 在列表区内下滚
        bx, by = book.region_click_b(region_name)
        kit.client.scroll(bx, by, kit.mapper.b_width, kit.mapper.b_height,
                          direction="down", ticks=scroll_ticks)
        scrolls += 1
        names = [r.text.strip() for r in results
                 if (r.text or "").strip()]
        kit.log(f"  第 {scrolls} 次下滚（当前可见 {len(names)} 项："
                f"{'、'.join(names)[:60]}）")

        sig = normalize(" ".join(names))
        if sig:
            same = same + 1 if sig == prev_sig else 0
            prev_sig = sig
            if same >= bottom_same:
                raise StepFailed(
                    f"连续 {bottom_same} 次滚动后列表内容不再变化"
                    f"（已滚到底），NPC「{npc}」不在列表中；"
                    f"请确认场景名/NPC 名是否正确")
        else:
            same = 0
            prev_sig = None

        if scrolls >= max_scrolls:
            raise StepFailed(
                f"已滚动 {max_scrolls} 次仍未在列表中找到 NPC「{npc}」")
        kit.sleep(settle)


HANDLERS: dict[str, Callable] = {
    "log": step_log,
    "sleep": step_sleep,
    "click_npc": step_click_npc,
    "wait_dialog": step_wait_dialog,
    "wait_text": step_wait_text,
    "choose_option": step_choose_option,
    "read_route": step_read_route,
    "click_point": step_click_point,
    "wait_arrive": step_wait_arrive,
    "dialog_until": step_dialog_until,
    "hotkey": step_hotkey,
    "click_region": step_click_region,
    "transfer_leader": step_transfer_leader,
    "map_click": step_map_click,
    "wait_npc": step_wait_npc,
    "wait_arrive_pos": step_wait_arrive_pos,
    "click_quest_npc": step_click_quest_npc,
    "dialog_choose": step_dialog_choose,
    "check_quest_accepted": step_check_quest_accepted,
    "check_leader": step_check_leader,
    "switch_tab": step_switch_tab,
    "click_text": step_click_text,
    "minimap_npc_pick": step_minimap_npc_pick,
}
