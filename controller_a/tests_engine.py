"""任务引擎测试：YAML 加载、上下文插值、步骤执行、重试/重启/停止。

OCR/截图/动作全部用内存假对象，不需要屏幕、网络与真实模型。
"""

import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.context import ContextError, QuestContext  # noqa: E402
from engine.kit import Kit, StepStopped  # noqa: E402
from engine.loader import TaskConfigError, load_all_tasks, load_task_file  # noqa: E402
from engine.quest_engine import QuestEngine  # noqa: E402
from engine import steps  # noqa: E402
from services.npc_service import NpcServiceConfig  # noqa: E402
from vision.ocr_engine import OcrResult  # noqa: E402

TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")


# ── 假对象 ──────────────────────────────────────────────

class FakeCapture:
    def __init__(self, frame=None):
        self.frame = frame if frame is not None else np.zeros((100, 200, 3), np.uint8)
        self.grabs = 0

    def grab(self):
        self.grabs += 1
        return self.frame


class FakeMapper:
    b_width, b_height = 1000, 500

    def to_b(self, u, v, w, h):
        return int(u / w * 1000), int(v / h * 500)


class FakeClient:
    def __init__(self):
        self.clicks = []
        self.moves = []
        self.keys = []
        self.scrolls = []

    def click_at(self, x, y, screen_w=None, screen_h=None, button="LE"):
        self.clicks.append((x, y, button))

    def move_to(self, x, y, w, h):
        self.moves.append((x, y))

    def send_keys(self, keys, times=1):
        self.keys.append((keys, times))

    def scroll(self, x, y, w, h, direction="down", ticks=1):
        self.scrolls.append((x, y, direction, ticks))


class FakeOcr:
    """按调用次数返回不同结果，模拟对话逐轮变化。"""

    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def recognize(self, _img):
        r = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return list(r)

    def recognize_upscaled(self, _img, scale=1.5):
        return self.recognize(_img)


def make_kit(ocr, client=None, stop=None, logs=None, region_book=None,
             capture=None):
    log_fn = logs.append if logs is not None else (lambda s: None)
    return Kit(capture or FakeCapture(), FakeMapper(),
               client or FakeClient(), ocr,
               NpcServiceConfig(), stop or threading.Event(), log_fn,
               region_book=region_book)


def ocr_text(text, x=10, y=10, w=60, h=20, conf=0.9):
    return OcrResult(text, x, y, w, h, conf)


# ── 1. 上下文 ───────────────────────────────────────────

def test_context_interpolation():
    ctx = QuestContext({"name": "赛天成"})
    ctx.set("route.target", [1750, 165])
    assert ctx.interpolate("去找${name}") == "去找赛天成"
    assert ctx.resolve_point("${route.target}") == [1750, 165]
    assert ctx.resolve_point([1, 2]) == [1, 2]
    try:
        ctx.interpolate("${missing}")
        assert False, "应抛 ContextError"
    except ContextError:
        pass
    print("test_context_interpolation OK")


# ── 2. YAML 加载与校验 ─────────────────────────────────

def test_loader_example():
    tasks = load_all_tasks(TASKS_DIR)
    assert any(t.key == "example_escort" for t in tasks)
    escort = next(t for t in tasks if t.key == "example_escort")
    assert escort.on_fail == "notify"
    assert escort.steps[0]["do"] == "click_npc"
    assert escort.steps[0]["npc"] == "赛天成"
    print("test_loader_example OK")


def test_loader_rejects_bad(tmp_path="/tmp/bad_task.yaml"):
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("name: bad\nsteps:\n  - do: not_exist\n    bogus: 1\n")
    try:
        load_task_file(tmp_path)
        assert False
    except TaskConfigError:
        pass
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("name: bad\nsteps: []\n")
    try:
        load_task_file(tmp_path)
        assert False
    except TaskConfigError:
        pass
    print("test_loader_rejects_bad OK")


# ── 3. 对话/选项步骤 ───────────────────────────────────

def run_step(handler, step, ocr, client=None, ctx=None, logs=None, stop=None):
    logs = logs if logs is not None else []
    kit = make_kit(ocr, client, stop, logs)
    handler(step, ctx or QuestContext(), kit)
    return kit, logs


def test_choose_option_click_center():
    ocr = FakeOcr([[ocr_text("接受任务", x=40, y=80, w=80, h=20)]])
    client = FakeClient()
    step = {"do": "choose_option", "match": "接受", "timeout": 3,
            "settle": 0}
    kit, _ = run_step(steps.step_choose_option, step, ocr, client)
    # 文字中心 (80,90)/200x100 → B(400,450)
    assert client.clicks == [(400, 450, "LE")], client.clicks
    print("test_choose_option_click_center OK")


def test_wait_text_polling_then_found():
    ocr = FakeOcr([[], [ocr_text("长安", x=2, y=6, w=20, h=10)]])
    step = {"match": "长安", "timeout": 3, "interval": 0.01,
            "roi": [0.0, 0.0, 0.2, 0.2]}
    ctx = QuestContext()
    run_step(steps.step_wait_text, step, ocr, ctx=ctx)
    assert "长安" in ctx.get("dialog")
    print("test_wait_text_polling_then_found OK")


def test_wait_text_timeout():
    ocr = FakeOcr([[]])
    step = {"match": "不存在", "timeout": 0.2, "interval": 0.02}
    try:
        run_step(steps.step_wait_text, step, ocr)
        assert False
    except steps.StepTimeout:
        pass
    print("test_wait_text_timeout OK")


def test_read_route_and_click_point(tmp_path="/tmp/rt.json"):
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write('{"洛阳": {"target": [800, 200], "scene": "洛阳", "npc": "会长"}}')
    ocr = FakeOcr([[ocr_text("请把镖银送到洛阳商会")]])
    ctx = QuestContext({"_task_dir": "/tmp"})
    kit, _ = run_step(steps.step_read_route,
                      {"route_map": "rt.json", "store_as": "route"},
                      ocr, ctx=ctx)
    assert ctx.get("route.target") == [800, 200]
    assert ctx.get("route.scene") == "洛阳"

    client = FakeClient()
    kit.client = client
    steps.step_click_point(
        {"point": "${route.target}", "settle": 0}, ctx, kit)
    assert client.clicks == [(800, 200, "LE")]

    steps.step_click_point(
        {"point_frac": [0.5, 0.5], "settle": 0, "mode": "move"}, ctx, kit)
    assert client.moves == [(500, 250)]
    print("test_read_route_and_click_point OK")


def test_dialog_until_rounds():
    ocr = FakeOcr([
        [ocr_text("你好，有何贵干")],
        [ocr_text("这是镖银")],
        [ocr_text("任务完成，赏银拿去")],
    ])
    client = FakeClient()
    ctx = QuestContext()
    # 无 continue_option：每轮回退点 (0.5,0.82) → B(500,410)
    run_step(steps.step_dialog_until,
             {"match": "完成", "max_rounds": 5, "timeout": 5,
              "settle": 0, "fallback_frac": [0.5, 0.82]},
             ocr, client, ctx)
    assert len(client.clicks) == 2 and client.clicks[0][:2] == (500, 410)
    print("test_dialog_until_rounds OK")


def test_dialog_until_exhausted():
    ocr = FakeOcr([[ocr_text("无关对话")]])
    try:
        run_step(steps.step_dialog_until,
                 {"match": "完成", "max_rounds": 2, "timeout": 5,
                  "settle": 0}, ocr)
        assert False
    except steps.StepFailed:
        pass
    print("test_dialog_until_exhausted OK")


# ── 4. 引擎调度：重试 / 重启 / 停止 ────────────────────

class FlakyHandler:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, step, ctx, kit):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise steps.StepFailed("模拟失败")
        kit.log("成功")


def run_engine(task_steps, on_fail="abort", max_restarts=0,
               extra_handlers=None, stop=None):
    from engine.loader import TaskDef
    handlers = dict(steps.HANDLERS)
    if extra_handlers:
        handlers.update(extra_handlers)
    old = steps.HANDLERS.copy()
    steps.HANDLERS.clear()
    steps.HANDLERS.update(handlers)
    try:
        task = TaskDef(key="t", name="t", steps=task_steps,
                       on_fail=on_fail, max_restarts=max_restarts,
                       source_file="/tmp/t.yaml")
        logs = []
        ocr = FakeOcr([[]])
        kit = make_kit(ocr, logs=logs, stop=stop)
        return QuestEngine(task, kit, lambda m, lv="": logs.append(m)).run(), logs
    finally:
        steps.HANDLERS.clear()
        steps.HANDLERS.update(old)


def test_engine_step_retry_then_success():
    flaky = FlakyHandler(fail_times=2)
    result, _ = run_engine(
        [{"do": "flaky", "retry": 2}],
        extra_handlers={"flaky": flaky})
    assert result.success and flaky.calls == 3
    print("test_engine_step_retry_then_success OK")


def test_engine_retry_exhausted_abort():
    flaky = FlakyHandler(fail_times=99)
    result, logs = run_engine(
        [{"do": "flaky", "retry": 1}],
        extra_handlers={"flaky": flaky})
    assert not result.success and result.failed_step == 1
    assert flaky.calls == 2
    print("test_engine_retry_exhausted_abort OK")


def test_engine_restart_whole_task():
    calls = {"n": 0}

    def h(step, ctx, kit):
        calls["n"] += 1
        if calls["n"] < 3:
            raise steps.StepFailed("先失败两次")

    result, _ = run_engine(
        [{"do": "h", "retry": 0}],
        on_fail="restart", max_restarts=2,
        extra_handlers={"h": h})
    assert result.success and result.restarts == 2 and calls["n"] == 3
    print("test_engine_restart_whole_task OK")


def test_engine_stop():
    stop = threading.Event()
    stop.set()

    def h(step, ctx, kit):
        raise AssertionError("停止后不应执行任何步骤")

    result, _ = run_engine([{"do": "h"}], stop=stop,
                           extra_handlers={"h": h})
    assert not result.success and result.stopped
    print("test_engine_stop OK")


def test_engine_unknown_step():
    result, _ = run_engine([{"do": "nope"}])
    assert not result.success and "未知步骤" in result.message
    print("test_engine_unknown_step OK")


# ── 5. 保镖链路：组合键 / 区域簿 / 两级地图 / NPC 视觉 ─────

import json as _json  # noqa: E402

import cv2  # noqa: E402

REGIONS_FIXTURE = {
    "game_window": {"x": 0, "y": 50, "w": 400, "h": 300},
    "defaults": {"click_jitter_ratio": 0.6},
    "regions": {
        "r1": {"x": 100, "y": 100, "w": 40, "h": 40},
        "hud_status": {"x": 38, "y": 15, "w": 90, "h": 40},
        "dialog_area": {"x": 50, "y": 50, "w": 200, "h": 100},
        "quest_board": {"x": 300, "y": 20, "w": 50, "h": 200},
        # OCR 寻路方案用的三个区域（坐标都在 game_window 400x300 内）
        "worldmap_scene_area": {"x": 40, "y": 120, "w": 300, "h": 120},
        "minimap_npc_panel": {"x": 250, "y": 60, "w": 120, "h": 180},
        "minimap_xun_area": {"x": 330, "y": 40, "w": 40, "h": 20},
    },
}


def _write_fixtures(template_pos=None):
    """生成 regions + map + assets 夹具，返回 (book, ctx, assets_dir)。"""
    from engine.regions import RegionBook
    from vision.template_matcher import clear_template_cache
    clear_template_cache()  # 每次重建 assets 目录，必须清模板缓存
    with open("/tmp/te_regions.json", "w", encoding="utf-8") as f:
        _json.dump(REGIONS_FIXTURE, f)
    assets = "/tmp/te_assets"
    os.makedirs(assets, exist_ok=True)
    templ_path = os.path.join(assets, "t.png")
    if template_pos is not None:
        # 50x60 的彩色模板（非纯色，带随机纹理才有区分度）
        templ = np.random.randint(0, 255, (60, 50, 3), np.uint8)
        cv2.imwrite(templ_path, templ)
    mp = {
        "defaults": {"click_jitter_px": 3},
        "places": {
            "阳关": {
                "bigmap_click": [100, 100],
                "npcs": {
                    "秦溪山": {
                        "minimap_click": [200, 200],
                        "npc_box": [-1, -1, -1, -1],
                        "template": "t.png" if template_pos is not None else "",
                    },
                    "典韦": {
                        "minimap_click": [210, 210],
                        "npc_box": [10, 20, 40, 50],
                        "template": "",
                    },
                },
            },
            "长安": {
                "bigmap_click": [100, 200],
                "npcs": {
                    "典韦": {
                        "minimap_click": [220, 220],
                        "npc_box": [10, 20, 40, 50],
                        "template": "",
                    },
                },
            },
        }
    }
    with open("/tmp/te_map.json", "w", encoding="utf-8") as f:
        _json.dump(mp, f)
    ctx = QuestContext({"_task_dir": "/tmp"})
    return RegionBook.load("/tmp/te_regions.json"), ctx, assets


def test_region_book_geometry():
    from engine.regions import RegionBook
    book, _, _ = _write_fixtures()
    roi = book.game_roi(1920, 1080)
    assert roi == (0 / 1920, 50 / 1080, 400 / 1920, 350 / 1080)
    for _ in range(50):
        x, y = book.region_click_b("r1")
        # 框内 jitter 0.6：点击点必须落在区域内，且不贴边
        assert 100 <= x <= 140 and 150 <= y <= 190
    try:
        RegionBook.load("/tmp/te_regions.json").region_click_b("不存在")
        assert False
    except Exception:
        pass
    print("test_region_book_geometry OK")


def test_step_hotkey():
    client = FakeClient()
    kit = make_kit(FakeOcr([[]]), client)
    steps.step_hotkey({"keys": "alt+2", "times": 1, "settle": 0},
                      QuestContext(), kit)
    assert client.keys == [("alt+2", 1)]
    print("test_step_hotkey OK")


def test_step_map_click_big_and_small():
    book, ctx, _ = _write_fixtures()
    client = FakeClient()
    kit = make_kit(FakeOcr([[]]), client, region_book=book)
    steps.step_map_click(
        {"map": "te_map.json", "level": "big", "place": "阳关", "settle": 0},
        ctx, kit)
    x, y, btn = client.clicks[-1]
    assert btn == "LE" and abs(x - 100) <= 3 and abs(y - 150) <= 3  # +gy50
    steps.step_map_click(
        {"map": "te_map.json", "level": "small", "place": "阳关",
         "npc": "秦溪山", "settle": 0}, ctx, kit)
    x, y, _ = client.clicks[-1]
    assert abs(x - 200) <= 3 and abs(y - 250) <= 3
    print("test_step_map_click_big_and_small OK")


def test_step_click_region_right_button():
    book, _, _ = _write_fixtures()
    client = FakeClient()
    kit = make_kit(FakeOcr([[]]), client, region_book=book)
    steps.step_click_region(
        {"region": "r1", "button": "right", "settle": 0},
        QuestContext(), kit)
    assert client.clicks[-1][2] == "RI"
    print("test_step_click_region_right_button OK")


def test_transfer_leader_give_once():
    """给与队长（times=1）：1 次右键 + 1 次左键。"""
    book, _, _ = _write_fixtures()
    # 在 regions fixture 里加 team_slot 和 give_leader_confirm
    import json as _j
    with open("/tmp/te_regions.json", "r") as f:
        data = _j.load(f)
    data["regions"]["team_slot"] = {"x": 315, "y": 15, "w": 40, "h": 40}
    data["regions"]["give_leader_confirm"] = {"x": 340, "y": 80, "w": 1, "h": 1}
    with open("/tmp/te_regions.json", "w") as f:
        _j.dump(data, f)
    from engine.regions import RegionBook
    book = RegionBook.load("/tmp/te_regions.json")
    client = FakeClient()
    kit = make_kit(FakeOcr([[]]), client, region_book=book)
    steps.step_transfer_leader(
        {"times": 1, "settle": 0}, QuestContext(), kit)
    # 应该有 2 次点击：RI 然后 LE
    assert len(client.clicks) == 2, f"期望 2 次点击，实际 {len(client.clicks)}"
    assert client.clicks[0][2] == "RI"
    assert client.clicks[1][2] == "LE"
    print("test_transfer_leader_give_once OK")


def test_transfer_leader_return_five_times():
    """给回队长（times=5）：5 次右键 + 5 次左键 = 10 次点击。"""
    book, _, _ = _write_fixtures()
    import json as _j
    with open("/tmp/te_regions.json", "r") as f:
        data = _j.load(f)
    data["regions"]["team_slot"] = {"x": 315, "y": 15, "w": 40, "h": 40}
    data["regions"]["give_leader_confirm"] = {"x": 340, "y": 80, "w": 1, "h": 1}
    with open("/tmp/te_regions.json", "w") as f:
        _j.dump(data, f)
    from engine.regions import RegionBook
    book = RegionBook.load("/tmp/te_regions.json")
    client = FakeClient()
    kit = make_kit(FakeOcr([[]]), client, region_book=book)
    steps.step_transfer_leader(
        {"times": 5, "settle": 0}, QuestContext(), kit)
    assert len(client.clicks) == 10, f"期望 10 次点击，实际 {len(client.clicks)}"
    # 奇偶交替：RI, LE, RI, LE, ...
    for i, (_, _, btn) in enumerate(client.clicks):
        assert btn == ("RI" if i % 2 == 0 else "LE"), f"第 {i} 次按钮不对"
    print("test_transfer_leader_return_five_times OK")


def _frame_with_template(assets, pos):
    """构造 400x225 帧，把 assets/t.png 贴到游戏窗口 ROI 内的 pos 处。

    game_window 在 FakeMapper(1000x500) 下 ROI=(0,0.1,0.4,0.7)，
    对应帧裁剪区 x[0:160] y[22:157]。
    """
    frame = np.full((225, 400, 3), 25, np.uint8)
    templ = cv2.imread(os.path.join(assets, "t.png"))
    th, tw = templ.shape[:2]
    u, v = pos  # 相对游戏窗口子图的坐标
    ox, oy = 0, int(0.1 * 225)
    frame[oy + v:oy + v + th, ox + u:ox + u + tw] = templ
    return frame


def test_wait_and_click_npc_by_template():
    book, ctx, assets = _write_fixtures(template_pos=True)
    frame = _frame_with_template(assets, (40, 30))
    capture = FakeCapture(frame)
    client = FakeClient()
    kit = Kit(capture, FakeMapper(), client, FakeOcr([[]]),
              NpcServiceConfig(), threading.Event(), lambda s: None,
              region_book=book)
    step = {"map": "te_map.json", "place": "阳关", "npc": "秦溪山",
            "timeout": 3, "interval": 0.01}
    old_assets = steps.ASSETS_DIR
    steps.ASSETS_DIR = assets
    try:
        steps.step_wait_npc(step, ctx, kit)
        assert ctx.get("found_npc.method") == "template"
        steps.step_click_quest_npc(dict(step, settle=0), ctx, kit)
    finally:
        steps.ASSETS_DIR = old_assets
    assert len(client.clicks) == 1
    print("test_wait_and_click_npc_by_template OK")


def test_click_quest_npc_multi_template_array():
    """template 字段是数组时，循环匹配取最高分。"""
    book, ctx, assets = _write_fixtures(template_pos=True)
    # 造两张不同模板：t.png（与帧匹配）和 t2.png（随机噪声，匹配分低）
    t2 = np.random.randint(0, 10, (60, 50, 3), np.uint8)
    cv2.imwrite(os.path.join(assets, "t2.png"), t2)
    frame = _frame_with_template(assets, (40, 30))
    old_assets = steps.ASSETS_DIR
    steps.ASSETS_DIR = assets
    try:
        # 单独写一份多模板 MAP，避免被其他测试覆盖
        mp = {"defaults": {"click_jitter_px": 3}, "places": {
            "阳关": {"bigmap_click": [100, 100], "npcs": {
                "秦溪山": {"minimap_click": [200, 200],
                           "npc_box": [-1, -1, -1, -1],
                           "template": ["t.png", "t2.png"]}}}}}
        _json.dump(mp, open(os.path.join(assets, "multi_map.json"), "w"))
        client = FakeClient()
        kit = Kit(FakeCapture(frame), FakeMapper(), client, FakeOcr([[]]),
                  NpcServiceConfig(), threading.Event(), lambda s: None,
                  region_book=book)
        ctx2 = QuestContext({"_task_dir": assets})
        steps.step_click_quest_npc(
            {"map": "multi_map.json", "place": "阳关", "npc": "秦溪山",
             "settle": 0, "search_secs": 2, "search_interval": 0.02},
            ctx2, kit)
        assert len(client.clicks) == 1
        print("test_click_quest_npc_multi_template_array OK")
    finally:
        steps.ASSETS_DIR = old_assets


def test_click_npc_ocr_first_then_box_fallback():
    book, ctx, _ = _write_fixtures()
    # ① OCR 命中：OCR 在游戏 ROI 内返回名字（recognize_in_roi 会加 ROI 偏移）
    # FakeCapture 默认帧 200x100；ROI=(0,0.1,0.4,0.7) → y 偏移 10，
    # 名字全帧坐标 y=70,h=16；身体点 = 70-2.5*16=30 → B屏 y≈150
    # （不是名字中心；名字中心 y=78 → B屏 390）
    client = FakeClient()
    ocr = FakeOcr([[OcrResult("秦溪山", 60, 60, 40, 16, 0.95)]])
    kit = make_kit(ocr, client, region_book=book)
    steps.step_click_quest_npc(
        {"map": "te_map.json", "place": "阳关", "npc": "秦溪山", "settle": 0},
        ctx, kit)
    assert len(client.clicks) == 1
    bx, by, btn = client.clicks[0]
    assert btn == "LE"
    assert 385 <= bx <= 415 and 135 <= by <= 165, (bx, by)

    # ③ OCR/模板全空 → npc_box 盲点（典韦 [10,20,40,50] → B: x100区, y 70~120）
    client2 = FakeClient()
    kit2 = make_kit(FakeOcr([[]]), client2, region_book=book)
    steps.step_click_quest_npc(
        {"map": "te_map.json", "place": "阳关", "npc": "典韦", "settle": 0},
        ctx, kit2)
    x, y, _ = client2.clicks[-1]
    assert 10 <= x <= 50 and 70 <= y <= 120, (x, y)
    print("test_click_npc_ocr_first_then_box_fallback OK")


def test_click_quest_npc_polls_until_npc_returns():
    """NPC 前两轮不在画面（OCR 空），第 3 轮走回画面被识别 → 点 OCR 身体点。"""
    book, ctx, _ = _write_fixtures()
    client = FakeClient()
    # 前 2 次 OCR 空（NPC 在画面外），第 3 次返回名字
    ocr = FakeOcr([[], [], [OcrResult("秦溪山", 60, 60, 40, 16, 0.95)]])
    kit = make_kit(ocr, client, region_book=book)
    steps.step_click_quest_npc(
        {"map": "te_map.json", "place": "阳关", "npc": "秦溪山",
         "settle": 0, "search_secs": 2, "search_interval": 0.02}, ctx, kit)
    # 命中后点击 OCR 身体点（非 npc_box 盲点）
    bx, by, btn = client.clicks[0]
    assert btn == "LE"
    assert 385 <= bx <= 415 and 135 <= by <= 165, (bx, by)
    print("test_click_quest_npc_polls_until_npc_returns OK")


def test_click_quest_npc_search_timeout_blind_fallback():
    """搜索窗口内始终识别不到 NPC → 超时后 npc_box 盲点兜底。"""
    book, ctx, _ = _write_fixtures()
    client = FakeClient()
    ocr = FakeOcr([[]])  # 永远识别不到
    kit = make_kit(ocr, client, region_book=book)
    steps.step_click_quest_npc(
        {"map": "te_map.json", "place": "阳关", "npc": "典韦",
         "settle": 0, "search_secs": 0.2, "search_interval": 0.02}, ctx, kit)
    # 典韦 npc_box [10,20,40,50] → B屏 x[10,50] y[70,120]
    x, y, _ = client.clicks[0]
    assert 10 <= x <= 50 and 70 <= y <= 120, (x, y)
    print("test_click_quest_npc_search_timeout_blind_fallback OK")


def test_wait_npc_timeout():
    book, ctx, assets = _write_fixtures(template_pos=True)
    # 帧里只有低强度噪声，与随机纹理模板不可能相关到阈值以上
    frame = np.random.randint(0, 10, (225, 400, 3), np.uint8)
    kit = Kit(FakeCapture(frame), FakeMapper(), FakeClient(), FakeOcr([[]]),
              NpcServiceConfig(), threading.Event(), lambda s: None,
              region_book=book)
    old_assets = steps.ASSETS_DIR
    steps.ASSETS_DIR = assets
    try:
        try:
            steps.step_wait_npc(
                {"map": "te_map.json", "place": "阳关", "npc": "秦溪山",
                 "timeout": 0.3, "interval": 0.02}, ctx, kit)
            assert False
        except steps.StepTimeout:
            pass
    finally:
        steps.ASSETS_DIR = old_assets
    print("test_wait_npc_timeout OK")


# ── 6. HUD 到达判定：坐标解析 + 先动后静 + NPC 确认闸门 ────

def test_hud_parser():
    from engine.hud import coord_close, parse_coords, parse_hud, parse_scene, scene_matches
    assert parse_coords("X:286  Y:291") == (286, 291)
    assert parse_coords("x：286   y：291") == (286, 291)
    assert parse_coords("坐标 286 291 附近") == (286, 291)  # X/Y 被漏识兜底
    assert parse_coords("无数字") is None
    assert parse_scene("X:286 Y:291 长安") == "长安"
    st = parse_hud("X:286  Y:291  长安城")
    assert st is not None and (st.x, st.y) == (286, 291) and st.scene == "长安城"
    assert parse_hud("长安") is None          # 缺坐标
    assert parse_hud("X:1 Y:2") is None       # 缺场景
    assert scene_matches("长安城", "长安")
    assert not scene_matches("洛阳", "长安")
    assert coord_close((286, 291), (288, 288), 4)
    assert not coord_close((286, 291), (300, 291), 4)
    print("test_hud_parser OK")


class RoutingOcr:
    """按裁剪图宽度区分：HUD 小区域帧 vs 游戏窗口 NPC 识别帧。"""

    def __init__(self, hud_frames, npc_frames):
        self.hud = FakeOcr(hud_frames)
        self.npc = FakeOcr(npc_frames)
        self.npc_calls = 0

    def recognize(self, img):
        if img.shape[1] <= 30:   # HUD 裁剪子图很窄（约18px）
            return self.hud.recognize(img)
        self.npc_calls += 1
        return self.npc.recognize(img)

    def recognize_upscaled(self, img, scale=1.5):
        return self.recognize(img)


def _hud_frames(seq):
    """seq: [(x,y,scene)] → HUD OCR 帧（坐标行+场景行两个结果）。"""
    return [
        [ocr_text(f"X:{x}  Y:{y}", 0, 0, 40, 10),
         ocr_text(scene, 0, 12, 30, 10)]
        for x, y, scene in seq
    ]


def _arrive_kit(book, hud_seq, npc_frames=None):
    ocr = RoutingOcr(_hud_frames(hud_seq),
                     npc_frames if npc_frames is not None else [[]])
    return make_kit(ocr, FakeClient(), region_book=book), ocr


def _arrive_step(place, npc, stable=0.3, checks=4, timeout=5):
    return {"map": "te_map.json", "place": place, "npc": npc,
            "stable_secs": stable, "coord_tol": 4,
            "confirm_checks": checks, "timeout": timeout, "interval": 0.01}


def test_wait_arrive_scene_switch_npc_confirmed():
    """跨场景 + 静止后 OCR 确认到 NPC → 到达。"""
    book, ctx, _ = _write_fixtures()
    seq = [(286, 291, "长安")] + [(150, 80, "阳关")] * 60
    kit, ocr = _arrive_kit(book, seq, [[ocr_text("秦溪山", 60, 50, 40, 16)]])
    steps.step_wait_arrive_pos(_arrive_step("阳关", "秦溪山"), ctx, kit)
    assert ctx.get("arrive.scene") == "阳关"
    assert ctx.get("arrive.method") == "ocr"
    assert ocr.npc_calls == 1   # 首次确认即命中，不重置
    print("test_wait_arrive_scene_switch_npc_confirmed OK")


def test_wait_arrive_coord_move_npc_confirmed():
    """同场景坐标移动 → 静止 → NPC 确认到达。"""
    book, ctx, _ = _write_fixtures()
    seq = [(10, 10, "阳关"), (10, 10, "阳关"),
           (200, 200, "阳关")] + [(200, 200, "阳关")] * 60
    kit, ocr = _arrive_kit(book, seq, [[ocr_text("秦溪山", 60, 50, 40, 16)]])
    steps.step_wait_arrive_pos(_arrive_step("阳关", "秦溪山"), ctx, kit)
    assert (ctx.get("arrive.x"), ctx.get("arrive.y")) == (200, 200)
    assert ocr.npc_calls == 1
    print("test_wait_arrive_coord_move_npc_confirmed OK")


def test_wait_arrive_never_moved_times_out():
    """关键防误判：没动过，静止再久也不做 NPC 确认、不算到达。"""
    book, ctx, _ = _write_fixtures()
    seq = [(10, 10, "阳关")] * 40
    kit, ocr = _arrive_kit(book, seq, [[ocr_text("秦溪山", 60, 50, 40, 16)]])
    try:
        steps.step_wait_arrive_pos(
            _arrive_step("阳关", "秦溪山", timeout=0.3), ctx, kit)
        assert False
    except steps.StepTimeout:
        pass
    assert ctx.get("arrive.scene") is None
    assert ocr.npc_calls == 0
    print("test_wait_arrive_never_moved_times_out OK")


def test_wait_arrive_ocr_jitter_within_tol_counts_stable():
    """OCR 个位抖动 ±3（≤ coord_tol=4）不打断静止，NPC 确认到达。"""
    book, ctx, _ = _write_fixtures()
    jitter = [(286, 291, "长安"), (283, 294, "长安"), (288, 289, "长安"),
              (285, 292, "长安")] * 20
    seq = [(100, 100, "长安"), (100, 100, "长安")] + jitter
    kit, ocr = _arrive_kit(book, seq, [[ocr_text("典韦", 60, 50, 30, 16)]])
    steps.step_wait_arrive_pos(_arrive_step("长安", "典韦"), ctx, kit)
    assert ocr.npc_calls == 1
    print("test_wait_arrive_ocr_jitter_within_tol_counts_stable OK")


def test_wait_arrive_npc_miss_twice_then_confirmed():
    """前两次静止窗口没见到 NPC → 重置；第 3 次确认成功 → 到达。"""
    book, ctx, _ = _write_fixtures()
    seq = [(10, 10, "阳关"), (200, 200, "阳关")] + [(200, 200, "阳关")] * 300
    npc_frames = [[], [], [ocr_text("秦溪山", 60, 50, 40, 16)]]
    kit, ocr = _arrive_kit(book, seq, npc_frames)
    steps.step_wait_arrive_pos(
        _arrive_step("阳关", "秦溪山", stable=0.05), ctx, kit)
    assert ctx.get("arrive.method") == "ocr"
    assert ocr.npc_calls == 3
    print("test_wait_arrive_npc_miss_twice_then_confirmed OK")


def test_wait_arrive_four_misses_force_arrive():
    """连续 4 个静止窗口（约 4×stable_secs）都没 NPC → 兜底判到达。"""
    book, ctx, _ = _write_fixtures()
    seq = [(10, 10, "阳关"), (200, 200, "阳关")] + [(200, 200, "阳关")] * 400
    kit, ocr = _arrive_kit(book, seq, [[]])   # 永远识别不到
    steps.step_wait_arrive_pos(
        _arrive_step("阳关", "秦溪山", stable=0.05, checks=4), ctx, kit)
    assert ctx.get("arrive.method") == "timeout_fallback"
    assert ocr.npc_calls == 4
    print("test_wait_arrive_four_misses_force_arrive OK")


def test_map_click_jitter_configurable():
    """MAP defaults.click_jitter_px 控制地图点击随机偏移。"""
    book, ctx, _ = _write_fixtures()
    # 自定义一份 jitter=8 的地图
    with open("/tmp/te_map8.json", "w", encoding="utf-8") as f:
        _json.dump({"defaults": {"click_jitter_px": 8}, "places": {
            "阳关": {"bigmap_click": [100, 100], "npcs": {}}}}, f)
    client = FakeClient()
    kit = make_kit(FakeOcr([[]]), client, region_book=book)
    for _ in range(40):
        steps.step_map_click(
            {"map": "te_map8.json", "level": "big", "place": "阳关",
             "settle": 0}, ctx, kit)
        x, y, _ = client.clicks[-1]
        assert abs(x - 100) <= 8 and abs(y - 150) <= 8
    # 至少有一次偏出旧的 ±3，证明新配置确实生效
    assert any(abs(c[0] - 100) > 3 or abs(c[1] - 150) > 3 for c in client.clicks)
    print("test_map_click_jitter_configurable OK")


# ── 7. 对话框选选项 + 接取确认 ────────────────────────────

# FakeMapper 是 1000x500，默认 FakeCapture 帧是 100x200（比例一致 2:1）
# dialog_area game(50,50,200,100)+gw(0,50,400,300)
# region_roi 正则化 by 1000x500 → (0.05, 0.2, 0.25, 0.3)
# 在 100x200 帧中 → crop (y:20-30, x:10-50) = 10×40 子图

def _green_frame(x, y, w, h, fw=200, fh=100):
    """造一帧：指定位置有绿色文字像素 BGR(72,244,12)。"""
    import numpy as _np
    frame = _np.zeros((fh, fw, 3), _np.uint8)
    frame[y:y+h, x:x+w] = (72, 244, 12)
    return frame


def _red_frame(regions, fw=200, fh=100):
    """造一帧：多个区域有红色像素 BGR(0,0,255)。regions=[(x,y,w,h), ...]。"""
    import numpy as _np
    frame = _np.zeros((fh, fw, 3), _np.uint8)
    for x, y, w, h in regions:
        frame[y:y+h, x:x+w] = (0, 0, 255)
    return frame


def test_dialog_choose_green_option_click():
    """绿色选项被识别并点击，点击在对话框行宽范围内。"""
    book, ctx, _ = _write_fixtures()
    # crop-local (5,3,30,6) + offset(10,20) → frame (15,23,30,6)
    frame = _green_frame(15, 23, 30, 6)
    ocr = FakeOcr([[OcrResult("小队运镖", 5, 3, 30, 6, 0.95)]])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book, capture=FakeCapture(frame))
    steps.step_dialog_choose(
        {"region": "dialog_area", "match": "小队运镖",
         "timeout": 1, "settle": 0, "interval": 0.01}, ctx, kit)
    assert len(client.clicks) == 1
    x, y, _ = client.clicks[0]
    # 对话框 frame x=10~50 → B 屏 50~250
    assert 45 <= x <= 255, f"点击 x={x} 不在对话框行宽内"
    print("test_dialog_choose_green_option_click OK")


def test_dialog_choose_white_text_skipped():
    """白色文字匹配但不是绿色 → 跳过，最终超时失败。"""
    book, ctx, _ = _write_fixtures()
    import numpy as _np
    frame = _np.zeros((100, 200, 3), _np.uint8)  # 全黑（无绿色）
    ocr = FakeOcr([[OcrResult("小队运镖", 5, 3, 30, 6, 0.95)]])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book, capture=FakeCapture(frame))
    try:
        steps.step_dialog_choose(
            {"region": "dialog_area", "match": "小队运镖",
             "timeout": 0.3, "interval": 0.01, "settle": 0}, ctx, kit)
        assert False
    except steps.StepTimeout:
        pass
    assert len(client.clicks) == 0
    print("test_dialog_choose_white_text_skipped OK")


def test_dialog_choose_optional_timeout_skips():
    """optional=true 时超时不报错，正常返回。"""
    book, ctx, _ = _write_fixtures()
    import numpy as _np
    frame = _np.zeros((100, 200, 3), _np.uint8)
    ocr = FakeOcr([[]])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book, capture=FakeCapture(frame))
    steps.step_dialog_choose(
        {"region": "dialog_area", "match": "确认",
         "optional": True, "timeout": 0.2, "interval": 0.01,
         "settle": 0}, ctx, kit)
    assert len(client.clicks) == 0
    print("test_dialog_choose_optional_timeout_skips OK")


def test_dialog_choose_line_random_not_center():
    """连续点击同一选项，x 坐标不应每次都相同。"""
    book, ctx, _ = _write_fixtures()
    frame = _green_frame(15, 23, 30, 6)
    ocr = FakeOcr([[OcrResult("小队运镖", 5, 3, 30, 6, 0.95)]] * 20)
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book, capture=FakeCapture(frame))
    for _ in range(10):
        steps.step_dialog_choose(
            {"region": "dialog_area", "match": "小队运镖",
             "timeout": 1, "settle": 0, "interval": 0.01}, ctx, kit)
    xs = [c[0] for c in client.clicks]
    assert len(set(xs)) > 1, f"所有点击 x 相同: {xs}"
    print("test_dialog_choose_line_random_not_center OK")


class QuestOcr:
    """按裁剪图宽度区分：dialog 红色检测不需要 OCR，quest_board OCR 返回任务文案。"""

    def __init__(self, quest_text=None):
        self.quest_text = quest_text or ""
        self.calls = 0

    def recognize(self, img):
        self.calls += 1
        if self.quest_text:
            return [OcrResult(self.quest_text, 0, 0, 80, 12, 0.95)]
        return []

    def recognize_upscaled(self, img, scale=1.5):
        return self.recognize(img)


def test_check_quest_accepted_pattern_match():
    """对话框红色 + 任务板 OCR 匹配「xx的yy和mm的nn」→ 成功，提取两组 place/npc。"""
    book, ctx, _ = _write_fixtures()
    # dialog_area ROI → frame (10-50, 20-30)，放红色像素
    # quest_board ROI → frame (60-70, 14-50)，OCR 返回任务文案
    frame = _red_frame([(12, 22, 5, 5)])
    ocr = QuestOcr("阳关的秦溪山和长安的典韦")
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book, capture=FakeCapture(frame))
    steps.step_check_quest_accepted(
        {"dialog_region": "dialog_area", "quest_region": "quest_board",
         "timeout": 1, "interval": 0.01}, ctx, kit)
    assert ctx.get("quest_accepted") is True
    assert ctx.get("quest.t1.place") == "阳关"
    assert ctx.get("quest.t1.npc") == "秦溪山"
    assert ctx.get("quest.t2.place") == "长安"
    assert ctx.get("quest.t2.npc") == "典韦"
    print("test_check_quest_accepted_pattern_match OK")


def test_check_quest_accepted_no_red_esc():
    """对话框无红色 → 超时 → 按 ESC → 失败。"""
    import numpy as _np
    book, ctx, _ = _write_fixtures()
    frame = _np.zeros((100, 200, 3), _np.uint8)  # 无红色
    ocr = QuestOcr("阳关的秦溪山和长安的典韦")
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book, capture=FakeCapture(frame))
    try:
        steps.step_check_quest_accepted(
            {"dialog_region": "dialog_area",
             "quest_region": "quest_board",
             "timeout": 0.2, "interval": 0.01}, ctx, kit)
        assert False
    except steps.StepTimeout:
        pass
    assert ("esc", 1) in client.keys  # ESC 被按下
    print("test_check_quest_accepted_no_red_esc OK")


def test_check_quest_accepted_red_but_no_pattern_esc():
    """对话框有红色但任务板 OCR 不匹配 → 超时 → 按 ESC → 失败。"""
    book, ctx, _ = _write_fixtures()
    frame = _red_frame([(12, 22, 5, 5)])  # 对话框有红色
    ocr = QuestOcr("随便什么文字")  # 不匹配正则
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book, capture=FakeCapture(frame))
    try:
        steps.step_check_quest_accepted(
            {"dialog_region": "dialog_area",
             "quest_region": "quest_board",
             "timeout": 0.2, "interval": 0.01}, ctx, kit)
        assert False
    except steps.StepTimeout:
        pass
    assert ("esc", 1) in client.keys
    print("test_check_quest_accepted_red_but_no_pattern_esc OK")


def test_check_leader_extracts_name():
    """OCR 聊天区命中「现在由1606@吴青峰担任队长」→ 提取用户名存入 ctx.leader.name。"""
    book, ctx, _ = _write_fixtures()
    # 在 regions fixture 加 chat_window 区域
    import json as _j
    with open("/tmp/te_regions.json", "r") as f:
        data = _j.load(f)
    data["regions"]["chat_window"] = {"x": 815, "y": 25, "w": 250, "h": 600}
    data["regions"]["leader_hint"] = {"x": 251, "y": 255, "w": 300, "h": 50}
    with open("/tmp/te_regions.json", "w") as f:
        _j.dump(data, f)
    from engine.regions import RegionBook
    book = RegionBook.load("/tmp/te_regions.json")
    # FakeOcr 返回聊天文本
    ocr = FakeOcr([[OcrResult("现在由1606@吴青峰担任队长", 10, 10, 200, 20, 0.95)]])
    kit = make_kit(ocr, region_book=book)
    steps.step_check_leader(
        {"timeout": 1, "interval": 0.01, "settle": 0}, ctx, kit)
    assert ctx.get("leader.name") == "1606@吴青峰"
    print("test_check_leader_extracts_name OK")


def test_check_leader_timeout_no_match():
    """聊天区没有队长提示 → 超时抛 StepTimeout。"""
    book, ctx, _ = _write_fixtures()
    import json as _j
    with open("/tmp/te_regions.json", "r") as f:
        data = _j.load(f)
    data["regions"]["chat_window"] = {"x": 815, "y": 25, "w": 250, "h": 600}
    data["regions"]["leader_hint"] = {"x": 251, "y": 255, "w": 300, "h": 50}
    with open("/tmp/te_regions.json", "w") as f:
        _j.dump(data, f)
    from engine.regions import RegionBook
    book = RegionBook.load("/tmp/te_regions.json")
    ocr = FakeOcr([[OcrResult("世界频道：你好", 10, 10, 100, 10, 0.9)]])
    kit = make_kit(ocr, region_book=book)
    try:
        steps.step_check_leader(
            {"timeout": 0.2, "interval": 0.02, "settle": 0}, ctx, kit)
        assert False
    except steps.StepTimeout:
        pass
    assert ctx.get("leader.name") is None
    print("test_check_leader_timeout_no_match OK")


def test_switch_tab_clicks_leader_label():
    """switch_tab 从 ctx.leader.name 取队长名，OCR team_tabs 匹配后点击。"""
    book, ctx, _ = _write_fixtures()
    import json as _j
    with open("/tmp/te_regions.json", "r") as f:
        data = _j.load(f)
    # team_tabs 是屏幕绝对坐标（screen=true）
    data["regions"]["team_tabs"] = {"x": 15, "y": 30, "w": 790, "h": 20,
                                     "screen": True}
    with open("/tmp/te_regions.json", "w") as f:
        _j.dump(data, f)
    from engine.regions import RegionBook
    book = RegionBook.load("/tmp/te_regions.json")
    ctx.set("leader.name", "陈粒")
    # OCR 返回队长名在标签栏
    ocr = FakeOcr([[OcrResult("陈粒", 100, 2, 40, 16, 0.95)]])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book)
    steps.step_switch_tab(
        {"timeout": 1, "interval": 0.01, "settle": 0}, ctx, kit)
    # 点击了命中文字中心
    assert len(client.clicks) == 1
    print("test_switch_tab_clicks_leader_label OK")


def test_switch_tab_no_leader_raises():
    """ctx 里没有 leader.name 时抛 StepFailed。"""
    book, ctx, _ = _write_fixtures()
    kit = make_kit(FakeOcr([[]]), region_book=book)
    try:
        steps.step_switch_tab(
            {"timeout": 0.2, "interval": 0.02}, ctx, kit)
        assert False
    except steps.StepFailed:
        pass
    print("test_switch_tab_no_leader_raises OK")


# ── 8. OCR 地图寻路方案：click_text / minimap_npc_pick / map 可选 ──

def test_loader_nav_ocr():
    """nav_ocr.yaml 能被加载且新字段全部在白名单内。"""
    tasks = load_all_tasks(TASKS_DIR)
    task = next((t for t in tasks if t.key == "nav_ocr"), None)
    assert task is not None, "nav_ocr 任务未被加载"
    assert task.params == {"place": "长安", "npc": "典韦"}
    actions = [s["do"] for s in task.steps]
    assert "click_text" in actions and "minimap_npc_pick" in actions
    print("test_loader_nav_ocr OK")


def test_loader_nav_ocr_test():
    """实测任务 nav_ocr_chuiyunsou（大唐东/垂云叟）能被加载。"""
    tasks = load_all_tasks(TASKS_DIR)
    task = next((t for t in tasks if t.key == "nav_ocr_chuiyunsou"), None)
    assert task is not None, "nav_ocr_chuiyunsou 任务未被加载"
    assert task.params == {"place": "大唐东", "npc": "垂云叟"}
    assert any(s["do"] == "minimap_npc_pick"
               and s.get("npc") == "${npc}" for s in task.steps)
    print("test_loader_nav_ocr_test OK")


def test_click_text_hits():
    """命名区域 OCR 命中 → 点文字中心（±3px → B屏±15）。

    假 OCR 坐标是裁剪子图内坐标，recognize_in_roi 会加回 ROI 偏移：
    world 区帧内裁剪偏移 (8,34)，局部(12,6,30,10) → 全帧中心(35,45)
    → B屏(175,225)。
    """
    book, ctx, _ = _write_fixtures()
    ocr = FakeOcr([[OcrResult("长安", 12, 6, 30, 10, 0.95)]])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book)
    steps.step_click_text(
        {"region": "worldmap_scene_area", "match": "长安",
         "timeout": 1, "settle": 0}, ctx, kit)
    assert len(client.clicks) == 1
    bx, by, btn = client.clicks[0]
    assert btn == "LE" and 160 <= bx <= 190 and 210 <= by <= 240, (bx, by)
    assert ctx.get("click_text.hit") == "长安"
    # match 列表：任一命中即可 + ${var} 插值
    ocr2 = FakeOcr([[OcrResult("长安城", 12, 6, 30, 10, 0.95)]])
    client2 = FakeClient()
    kit2 = make_kit(ocr2, client2, region_book=book)
    ctx2 = QuestContext({"place": "长安"})
    steps.step_click_text(
        {"region": "worldmap_scene_area",
         "match": ["洛阳", "${place}"], "timeout": 1, "settle": 0},
        ctx2, kit2)
    assert len(client2.clicks) == 1
    print("test_click_text_hits OK")


def test_click_text_timeout():
    book, ctx, _ = _write_fixtures()
    kit = make_kit(FakeOcr([[]]), region_book=book)
    try:
        steps.step_click_text(
            {"region": "worldmap_scene_area", "match": "长安",
             "timeout": 0.2, "interval": 0.02, "settle": 0}, ctx, kit)
        assert False
    except steps.StepTimeout:
        pass
    print("test_click_text_timeout OK")


def _pick_step(**over):
    step = {"region": "minimap_npc_panel", "npc": "典韦",
            "timeout": 5, "settle": 0, "interval": 0.01,
            "scroll_ticks": 2, "max_scrolls": 10, "bottom_same": 2}
    step.update(over)
    return step


# NPC 列表区在帧内裁剪偏移 (50,22)（panel 24x36 子图）：
#   目标 L(2,18,20,10)  → 全帧中心(62,45) → B(310,225)
#   目录 F(1,2,22,8)    → 全帧中心(62,28) → B(310,140)
#   他人 O(2,10,20,8)   → 全帧中心(62,36) → B(310,180)

def test_minimap_pick_immediate():
    """列表已展开且目标直接可见 → 直接点，不展开目录也不滚动。"""
    book, ctx, _ = _write_fixtures()
    ocr = FakeOcr([[OcrResult("典韦", 2, 18, 20, 10, 0.95)]])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book)
    steps.step_minimap_npc_pick(_pick_step(), ctx, kit)
    assert len(client.clicks) == 1 and client.scrolls == []
    bx, by, _ = client.clicks[0]
    assert 295 <= bx <= 325 and 210 <= by <= 240, (bx, by)
    assert ctx.get("minimap_npc.found") == "典韦"
    print("test_minimap_pick_immediate OK")


def test_minimap_pick_expand_then_pick():
    """先只见「普通NPC」节点 → 点一次展开 → 下一轮点 NPC。"""
    book, ctx, _ = _write_fixtures()
    ocr = FakeOcr([
        [OcrResult("普通NPC", 1, 2, 22, 8, 0.9)],
        [OcrResult("典韦", 2, 18, 20, 10, 0.95)],
    ])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book)
    steps.step_minimap_npc_pick(_pick_step(), ctx, kit)
    assert client.scrolls == []
    assert len(client.clicks) == 2
    assert 295 <= client.clicks[1][0] <= 325
    print("test_minimap_pick_expand_then_pick OK")


def test_minimap_pick_wait_folder_then_expand():
    """面板还没加载（OCR 空）→ 等待 → 节点出现 → 展开 → 选中。"""
    book, ctx, _ = _write_fixtures()
    ocr = FakeOcr([
        [],
        [OcrResult("普通NPC", 1, 2, 22, 8, 0.9)],
        [OcrResult("典韦", 2, 18, 20, 10, 0.95)],
    ])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book)
    steps.step_minimap_npc_pick(_pick_step(timeout=5), ctx, kit)
    assert client.scrolls == [] and len(client.clicks) == 2
    print("test_minimap_pick_wait_folder_then_expand OK")


def test_minimap_pick_scroll_until_found():
    """已展开、目标不可见 → 下滚两次后出现 → 只点 NPC。"""
    book, ctx, _ = _write_fixtures()
    ocr = FakeOcr([
        [OcrResult("张三", 2, 10, 20, 8, 0.95)],
        [OcrResult("李四", 2, 12, 20, 8, 0.95)],
        [OcrResult("典韦", 2, 18, 20, 10, 0.95)],
    ])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book)
    steps.step_minimap_npc_pick(_pick_step(), ctx, kit)
    assert len(client.scrolls) == 2
    assert all(s[2] == "down" and s[3] == 2 for s in client.scrolls)
    # 滚动点必须落在 minimap_npc_panel 框内（B屏：x274~346 y146~254）
    for x, y, _, _ in client.scrolls:
        assert 274 <= x <= 346 and 146 <= y <= 254, (x, y)
    assert len(client.clicks) == 1
    print("test_minimap_pick_scroll_until_found OK")


def test_minimap_pick_already_expanded_no_folder_click():
    """节点和其它 NPC 同时可见 = 已展开，禁止再点节点（防折叠）。"""
    book, ctx, _ = _write_fixtures()
    ocr = FakeOcr([
        [OcrResult("普通NPC", 1, 2, 22, 8, 0.9),
         OcrResult("张三", 2, 10, 20, 8, 0.95)],
        [OcrResult("典韦", 2, 18, 20, 10, 0.95)],
    ])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book)
    steps.step_minimap_npc_pick(_pick_step(), ctx, kit)
    assert len(client.clicks) == 1 and len(client.scrolls) == 1
    print("test_minimap_pick_already_expanded_no_folder_click OK")


def test_minimap_pick_bottom_detected():
    """列表内容连续 bottom_same 次不变 → 判滚到底，失败且不再空转。"""
    book, ctx, _ = _write_fixtures()
    ocr = FakeOcr([[OcrResult("张三", 2, 10, 20, 8, 0.95)]])
    client = FakeClient()
    kit = make_kit(ocr, client, region_book=book)
    try:
        steps.step_minimap_npc_pick(
            _pick_step(bottom_same=2, timeout=10), ctx, kit)
        assert False
    except steps.StepFailed as exc:
        assert "到底" in str(exc)
    # 第 1 滚建立指纹，第 2 滚 same=1，第 3 滚 same=2 触发
    assert client.clicks == [] and len(client.scrolls) == 3
    print("test_minimap_pick_bottom_detected OK")


class CyclingOcr:
    """列表内容每轮都变（不同名字循环），用于绕过触底判定测 max_scrolls。"""

    def __init__(self, texts):
        self.texts = texts
        self.i = 0

    def recognize(self, _img):
        t = self.texts[self.i % len(self.texts)]
        self.i += 1
        return [OcrResult(t, 2, 10, 20, 8, 0.95)]

    def recognize_upscaled(self, _img, scale=1.5):
        return self.recognize(_img)


def test_minimap_pick_max_scrolls():
    book, ctx, _ = _write_fixtures()
    client = FakeClient()
    kit = make_kit(CyclingOcr(["张三", "李四", "王五", "赵六"]),
                   client, region_book=book)
    try:
        steps.step_minimap_npc_pick(
            _pick_step(max_scrolls=3, timeout=10), ctx, kit)
        assert False
    except steps.StepFailed as exc:
        assert "滚动 3 次" in str(exc)
    assert client.clicks == [] and len(client.scrolls) == 3
    print("test_minimap_pick_max_scrolls OK")


def test_click_quest_npc_no_map_ocr():
    """不传 map：OCR 命中照常点击（纯 OCR 模式）。"""
    book, ctx, _ = _write_fixtures()
    client = FakeClient()
    ocr = FakeOcr([[OcrResult("典韦", 60, 60, 40, 16, 0.95)]])
    kit = make_kit(ocr, client, region_book=book)
    steps.step_click_quest_npc(
        {"place": "长安", "npc": "典韦", "settle": 0}, ctx, kit)
    assert len(client.clicks) == 1
    bx, by, btn = client.clicks[0]
    # game ROI 裁剪偏移 oy=10：名字全帧(80,70)，身体点 y=70-40=30 → B(400,150)
    assert btn == "LE" and 385 <= bx <= 415 and 135 <= by <= 165, (bx, by)
    print("test_click_quest_npc_no_map_ocr OK")


def test_click_quest_npc_no_map_miss_fails():
    """不传 map 且 OCR 始终失败 → 直接 StepFailed（没有 npc_box 可盲点）。"""
    book, ctx, _ = _write_fixtures()
    client = FakeClient()
    kit = make_kit(FakeOcr([[]]), client, region_book=book)
    try:
        steps.step_click_quest_npc(
            {"place": "长安", "npc": "典韦", "settle": 0,
             "search_secs": 0.15, "search_interval": 0.02}, ctx, kit)
        assert False
    except steps.StepFailed as exc:
        assert "兜底" in str(exc)
    assert client.clicks == []
    print("test_click_quest_npc_no_map_miss_fails OK")


def test_wait_arrive_no_map_ocr_confirmed():
    """不传 map 的到达判定：场景切换 + 静止 + OCR 确认 NPC → 到达。"""
    book, ctx, _ = _write_fixtures()
    seq = [(286, 291, "长安")] + [(150, 80, "阳关")] * 60
    kit, ocr = _arrive_kit(book, seq, [[ocr_text("秦溪山", 60, 50, 40, 16)]])
    step = {"place": "阳关", "npc": "秦溪山", "stable_secs": 0.3,
            "coord_tol": 4, "confirm_checks": 4,
            "timeout": 5, "interval": 0.01}
    steps.step_wait_arrive_pos(step, ctx, kit)
    assert ctx.get("arrive.scene") == "阳关"
    assert ctx.get("arrive.method") == "ocr"
    assert ocr.npc_calls == 1
    print("test_wait_arrive_no_map_ocr_confirmed OK")


if __name__ == "__main__":
    test_context_interpolation()
    test_loader_example()
    test_loader_rejects_bad()
    test_choose_option_click_center()
    test_wait_text_polling_then_found()
    test_wait_text_timeout()
    test_read_route_and_click_point()
    test_dialog_until_rounds()
    test_dialog_until_exhausted()
    test_engine_step_retry_then_success()
    test_engine_retry_exhausted_abort()
    test_engine_restart_whole_task()
    test_engine_stop()
    test_engine_unknown_step()
    test_region_book_geometry()
    test_step_hotkey()
    test_step_map_click_big_and_small()
    test_step_click_region_right_button()
    test_transfer_leader_give_once()
    test_transfer_leader_return_five_times()
    test_wait_and_click_npc_by_template()
    test_click_quest_npc_multi_template_array()
    test_click_npc_ocr_first_then_box_fallback()
    test_click_quest_npc_polls_until_npc_returns()
    test_click_quest_npc_search_timeout_blind_fallback()
    test_wait_npc_timeout()
    test_hud_parser()
    test_wait_arrive_scene_switch_npc_confirmed()
    test_wait_arrive_coord_move_npc_confirmed()
    test_wait_arrive_never_moved_times_out()
    test_wait_arrive_ocr_jitter_within_tol_counts_stable()
    test_wait_arrive_npc_miss_twice_then_confirmed()
    test_wait_arrive_four_misses_force_arrive()
    test_map_click_jitter_configurable()
    test_dialog_choose_green_option_click()
    test_dialog_choose_white_text_skipped()
    test_dialog_choose_optional_timeout_skips()
    test_dialog_choose_line_random_not_center()
    test_check_quest_accepted_pattern_match()
    test_check_quest_accepted_no_red_esc()
    test_check_quest_accepted_red_but_no_pattern_esc()
    test_check_leader_extracts_name()
    test_check_leader_timeout_no_match()
    test_switch_tab_clicks_leader_label()
    test_switch_tab_no_leader_raises()
    test_loader_nav_ocr()
    test_loader_nav_ocr_test()
    test_click_text_hits()
    test_click_text_timeout()
    test_minimap_pick_immediate()
    test_minimap_pick_expand_then_pick()
    test_minimap_pick_wait_folder_then_expand()
    test_minimap_pick_scroll_until_found()
    test_minimap_pick_already_expanded_no_folder_click()
    test_minimap_pick_bottom_detected()
    test_minimap_pick_max_scrolls()
    test_click_quest_npc_no_map_ocr()
    test_click_quest_npc_no_map_miss_fails()
    test_wait_arrive_no_map_ocr_confirmed()
    print("\n全部 60 项引擎测试通过")
