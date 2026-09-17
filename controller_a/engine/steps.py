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
  map_click        查 MAP 点大地图地名（level=big）或小地图寻路点（level=small）
  wait_npc         到达判定：游戏窗口内轮询 OCR 名字 → 模板匹配
  click_quest_npc  点目标 NPC：① OCR 名字 ② 游戏窗口内模板匹配 ③ npc_box 盲点
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import cv2

from engine.context import ContextError, QuestContext
from engine.kit import Kit, StepStopped
from engine.regions import RegionConfigError, load_map_file
from vision.dialog_reader import (
    crop_roi,
    find_text,
    join_text,
    recognize_in_roi,
    validate_roi,
)
from vision.npc_detector import find_npc
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
# 大地图地名/小地图寻路点是精确 UI 热区，只允许 ±3px 抖动，避免点到相邻项
MAP_POINT_JITTER_PX = 3.0
_BUTTON_MAP = {"left": "LE", "right": "RI", "le": "LE", "ri": "RI"}


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
                                  MAP_POINT_JITTER_PX)
    kit.log(f"  🗺 点击{label} → B屏({bx},{by})")
    kit.client.click_at(bx, by, kit.mapper.b_width, kit.mapper.b_height)
    kit.sleep(float(_get(step, "settle", 0.6)))


# ── NPC 视觉识别（OCR 名字 → 模板匹配）──────────────────

def _detect_npc(kit: Kit, npc: str, npc_def: dict,
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
        cu, cv = hit.center
        return "ocr", cu, cv, frame, {"conf": hit.confidence}

    templ_name = (npc_def or {}).get("template")
    if templ_name:
        templ = load_template(os.path.join(ASSETS_DIR, templ_name))
        if templ is None:
            if warn_missing_template:
                kit.log(f"  ⚠ 模板图读取失败（请检查 assets/{templ_name}）")
        else:
            sub, ox, oy = crop_roi(frame, roi)
            found = match_template(sub, templ)
            if found is not None:
                rect = (found.x + ox, found.y + oy, found.w, found.h)
                cu, cv = found.x + ox + found.w // 2, \
                    found.y + oy + found.h // 2
                return "template", cu, cv, frame, {
                    "score": found.score, "scale": found.scale, "rect": rect}
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
        kit.log(f"  ✓ OCR 见到「{npc}」conf={info['conf']:.2f}")
    else:
        kit.log(f"  ✓ 模板匹配见到「{npc}」score={info['score']:.2f} "
                f"scale={info['scale']:.2f}")
        path = _save_npc_debug(_frame, npc, method, info["rect"])
        if path:
            kit.log(f"  命中调试图: {path}")
    ctx.set("found_npc.method", method)


def step_click_quest_npc(step, ctx: QuestContext, kit: Kit) -> None:
    """点目标 NPC：① OCR 名字 → ② 游戏窗口内模板匹配 → ③ npc_box 盲点。"""
    map_rel = _get(step, "map", required=True)
    place = ctx.interpolate(_get(step, "place", required=True))
    npc = ctx.interpolate(_get(step, "npc", required=True))

    data, _ = _load_map(ctx, map_rel)
    npc_def = _map_npc_def(_map_place(data, place), place, npc)
    book = _book(kit)

    found = _detect_npc(kit, npc, npc_def)
    if found is not None:
        method, u, v, frame, info = found
        h, w = frame.shape[:2]
        bx, by = kit.mapper.to_b(u, v, w, h)
        if method == "template":
            _save_npc_debug(frame, npc, method, info["rect"])
        kit.log(f"  点击 NPC「{npc}」（{method} 命中，"
                f"{'conf' if method == 'ocr' else 'score'}="
                f"{info.get('conf', info.get('score')):.2f}）→ B屏({bx},{by})")
    else:
        box = _valid_box(npc_def.get("npc_box"),
                         f"NPC「{npc}」的 npc_box 保底点击框")
        bx, by = book.game_box_click_b(box)
        kit.log(f"  ⚠ OCR/模板均未命中「{npc}」，使用 npc_box 保底盲点 "
                f"→ B屏({bx},{by})")

    kit.client.click_at(bx, by, kit.mapper.b_width, kit.mapper.b_height)
    kit.sleep(float(_get(step, "settle", 0.8)))


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
    "map_click": step_map_click,
    "wait_npc": step_wait_npc,
    "click_quest_npc": step_click_quest_npc,
}
