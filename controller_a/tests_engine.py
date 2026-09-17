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

    def click_at(self, x, y, w, h, button="LE"):
        self.clicks.append((x, y, button))

    def move_to(self, x, y, w, h):
        self.moves.append((x, y))

    def send_keys(self, keys, times=1):
        self.keys.append((keys, times))


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


def make_kit(ocr, client=None, stop=None, logs=None, region_book=None):
    log_fn = logs.append if logs is not None else (lambda s: None)
    return Kit(FakeCapture(), FakeMapper(), client or FakeClient(), ocr,
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
    },
}


def _write_fixtures(template_pos=None):
    """生成 regions + map + assets 夹具，返回 (book, ctx, assets_dir)。"""
    from engine.regions import RegionBook
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
            }
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


def test_click_npc_ocr_first_then_box_fallback():
    book, ctx, _ = _write_fixtures()
    # ① OCR 命中：OCR 在游戏 ROI 内返回名字（子图坐标会被 recognize_in_roi 平移）
    client = FakeClient()
    ocr = FakeOcr([[OcrResult("秦溪山", 60, 60, 40, 16, 0.95)]])
    kit = make_kit(ocr, client, region_book=book)
    steps.step_click_quest_npc(
        {"map": "te_map.json", "place": "阳关", "npc": "秦溪山", "settle": 0},
        ctx, kit)
    assert len(client.clicks) == 1  # 点了 OCR 名字，未走兜底

    # ③ OCR/模板全空 → npc_box 盲点（典韦 [10,20,40,50] → B: x100区, y 70~120）
    client2 = FakeClient()
    kit2 = make_kit(FakeOcr([[]]), client2, region_book=book)
    steps.step_click_quest_npc(
        {"map": "te_map.json", "place": "阳关", "npc": "典韦", "settle": 0},
        ctx, kit2)
    x, y, _ = client2.clicks[-1]
    assert 10 <= x <= 50 and 70 <= y <= 120, (x, y)
    print("test_click_npc_ocr_first_then_box_fallback OK")


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
    test_wait_and_click_npc_by_template()
    test_click_npc_ocr_first_then_box_fallback()
    test_wait_npc_timeout()
    print("\n全部 21 项引擎测试通过")
