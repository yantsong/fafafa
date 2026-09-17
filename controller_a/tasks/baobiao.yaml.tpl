# ============================================================================
# 保镖任务流程骨架（规划稿，尚不能运行）
#
# 注意：本文件后缀是 .yaml.tpl，控制台不会加载它（避免未知步骤导致任务列表
# 报错）。等 B 端键盘/右键 + A 端新原子实现后，改名为 baobiao.yaml 即上线。
#
# 配套数据文件（你正在测量）：
#   regions/baobiao.json        固定区域坐标（相对游戏窗口原点）
#   tasks/maps/baobiao_map.json 地名→大地图坐标 / NPC→小地图寻路坐标 / NPC模型框
#
# 待实现的新 do 原子（本骨架用到，代码还没写）：
#   hotkey          发组合键，keys 如 "ctrl+tab"、"tab"，times=按键次数
#   click_region    点击 regions 文件里的命名区域，button=left/right，框内随机点
#   read_quest      OCR 任务栏区域 → 正则解析「xx的yy和 mm的nn」→ 存 t1/t2
#   map_click       查 MAP 点大地图(level=big)或小地图寻路坐标(level=small)
#   wait_npc        到达判定：在游戏窗口区域内轮询 OCR 名字 / 模板匹配
#   click_quest_npc 点目的地NPC，三级兜底：① OCR名字 ② 游戏窗口内模板匹配
#                                   ③ npc_box 框内随机盲点（纯坐标）
# 已有的原子：log / sleep / choose_option / wait_dialog
# 新原子会让 wait_dialog/choose_option 支持 region: 区域名（替代手写 roi）。
# ============================================================================

# regions_file: regions/baobiao.json   # 实现后由引擎自动加载

tasks:
  baobiao:
    name: "保镖任务"
    on_fail: notify
    max_restarts: 0

    steps:
      # ── 阶段一：账号A 在镖局接任务 ──────────────────────────

      # 1) 点固定站位的赛天威
      - do: click_region
        region: npc_saitianwei
        button: left
        settle: 0.8

      # 2) 三轮对话：每轮等到指定文案出现，再点对话区翻页
      #    【待你提供】把每轮对话框里实际出现的一句稳定文字填进 match
      - do: wait_dialog
        region: dialog_area
        match: "第1轮对话的稳定文案"
        timeout: 8
      - do: click_region
        region: dialog_continue
        settle: 0.6

      - do: wait_dialog
        region: dialog_area
        match: "第2轮对话的稳定文案"
        timeout: 8
      - do: click_region
        region: dialog_continue
        settle: 0.6

      - do: wait_dialog
        region: dialog_area
        match: "第3轮对话的稳定文案"
        timeout: 8
      - do: click_region
        region: dialog_continue
        settle: 1.0

      # 3) 读右侧任务栏红字，例：「阳关的秦溪山和长安的典韦」
      #    解析后上下文得到：
      #      quest.t1.place=阳关  quest.t1.npc=秦溪山
      #      quest.t2.place=长安  quest.t2.npc=典韦
      - do: read_quest
        region: quest_board
        map: maps/baobiao_map.json
        store_as: quest
        timeout: 6

      # ====================================================================
      # 阶段二：第 1 个目标 ${quest.t1.npc}（${quest.t1.place}）
      # 当前：活动标签=账号A，队长=A
      # ====================================================================

      # 4a) 右键自己头像 → 点 B 的槽位 = 给与队长
      - do: click_region
        region: portrait_self
        button: right
        settle: 0.5
      - do: click_region
        region: give_leader_slot_b
        settle: 0.5

      # 4b) Ctrl+TAB 切到账号B（1 次）
      - do: hotkey
        keys: "ctrl+tab"
        times: 1
        settle: 1.0

      # 5a) TAB 唤出大地图 → 点地名进入小地图
      - do: hotkey
        keys: "tab"
        settle: 0.8
      - do: map_click
        map: maps/baobiao_map.json
        level: big
        place: "${quest.t1.place}"
        settle: 1.0

      # 5b) 小地图上点该 NPC 的寻路坐标（代替输入名字+点“寻”）
      - do: map_click
        map: maps/baobiao_map.json
        level: small
        place: "${quest.t1.place}"
        npc: "${quest.t1.npc}"
        settle: 0.5

      # 5c) 等待到达：见到 NPC（OCR 名字 → 模板 → 模型框），90s 兜底
      - do: wait_npc
        map: maps/baobiao_map.json
        place: "${quest.t1.place}"
        npc: "${quest.t1.npc}"
        timeout: 90
        interval: 1.0

      # 5d) 账号B 先点一次 NPC
      - do: click_quest_npc
        map: maps/baobiao_map.json
        place: "${quest.t1.place}"
        npc: "${quest.t1.npc}"
        settle: 0.8

      # 6a) Ctrl+TAB 切回账号A（5 次，次数你实测可改）
      - do: hotkey
        keys: "ctrl+tab"
        times: 5
        settle: 1.0

      # 6b) 账号A 点同一个 NPC 弹出物品框
      - do: click_quest_npc
        map: maps/baobiao_map.json
        place: "${quest.t1.place}"
        npc: "${quest.t1.npc}"
        settle: 0.8

      # 6c) 点任务物品 icon → 点第一个格子 → 点「给与」按钮(文字匹配)
      - do: click_region
        region: item_icon
        settle: 0.4
      - do: click_region
        region: item_slot_first
        settle: 0.4
      - do: choose_option
        match: "给与"
        timeout: 6
        settle: 1.0

      # ====================================================================
      # 阶段三：第 2 个目标 ${quest.t2.npc}（${quest.t2.place}）
      # 复制阶段二，把 t1 全部改成 t2。
      # 【游戏内确认】此时队长是否已是 B：
      #   若是 B，本段开头的“给与队长”两步可删；若队长已回到 A，保留。
      # ====================================================================

      - do: click_region
        region: portrait_self
        button: right
        settle: 0.5
      - do: click_region
        region: give_leader_slot_b
        settle: 0.5
      - do: hotkey
        keys: "ctrl+tab"
        times: 1
        settle: 1.0

      - do: hotkey
        keys: "tab"
        settle: 0.8
      - do: map_click
        map: maps/baobiao_map.json
        level: big
        place: "${quest.t2.place}"
        settle: 1.0
      - do: map_click
        map: maps/baobiao_map.json
        level: small
        place: "${quest.t2.place}"
        npc: "${quest.t2.npc}"
        settle: 0.5
      - do: wait_npc
        map: maps/baobiao_map.json
        place: "${quest.t2.place}"
        npc: "${quest.t2.npc}"
        timeout: 90
        interval: 1.0
      - do: click_quest_npc
        map: maps/baobiao_map.json
        place: "${quest.t2.place}"
        npc: "${quest.t2.npc}"
        settle: 0.8

      - do: hotkey
        keys: "ctrl+tab"
        times: 5
        settle: 1.0
      - do: click_quest_npc
        map: maps/baobiao_map.json
        place: "${quest.t2.place}"
        npc: "${quest.t2.npc}"
        settle: 0.8
      - do: click_region
        region: item_icon
        settle: 0.4
      - do: click_region
        region: item_slot_first
        settle: 0.4
      - do: choose_option
        match: "给与"
        timeout: 6
        settle: 1.0

      # ====================================================================
      # 阶段四：回到初始账号 A，飞行旗回镖局
      # ====================================================================

      # 8a) 切回账号A（同步骤6 的 5 次；如当前已在 A 则删此步）
      - do: hotkey
        keys: "ctrl+tab"
        times: 5
        settle: 1.0

      # 8b) 使用飞行旗【待你给实际快捷键】，再点镖局落点
      - do: hotkey
        keys: "F8"            # ← 占位，替换为游戏内飞行旗快捷键
        settle: 0.8
      - do: click_region
        region: fly_flag_target
        settle: 1.0

      - do: log
        text: "保镖任务结束"
