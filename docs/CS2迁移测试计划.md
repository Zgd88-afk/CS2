# 在 CS2 上测试行为克隆智能体 —— 迁移测试计划

> 配套文档：[新手入门指南.md](新手入门指南.md)（原项目原理与 CSGO 用法）
> 本文回答三个问题：
> ① 原 CSGO 项目各组件在 CS2 下还能不能用（兼容性分析）；
> ② GitHub 上有哪些 CS2 相关开源项目可以借用（生态调研）；
> ③ 怎样系统地在 CS2 上测试这套智能体（分阶段测试计划，含通过/失败标准）。
>
> 配套代码（本文档同批产出，均位于仓库根目录，**未实机验证**，按 Phase 1 步骤先行校验）：
> - `cs2_config.py` —— CS2 适配配置（窗口标题、裁剪偏移、鼠标缩放系数）
> - `dm_run_agent_cs2.py` —— CS2 版智能体运行脚本
> - `tools_screen_calibrate_cs2.py` —— 截屏裁剪校准工具
> - `tools_demo_stats.py` —— demo 解析成绩统计脚本

---

## 目录

1. [总体思路](#1-总体思路)
2. [兼容性分析：CSGO 组件在 CS2 下的状态](#2-兼容性分析csgo-组件在-cs2-下的状态)
3. [GitHub CS2 开源生态调研](#3-github-cs2-开源生态调研)
4. [Phase 0：环境准备](#4-phase-0环境准备)
5. [Phase 1：最小移植冒烟测试（约半天）](#5-phase-1最小移植冒烟测试约半天)
6. [Phase 2：零样本基线测试（1~2 天）](#6-phase-2零样本基线测试12-天)
7. [Phase 3：定量评估（2~3 天）](#7-phase-3定量评估23-天)
8. [Phase 4：域差距诊断与改进路线](#8-phase-4域差距诊断与改进路线)
9. [风险与注意事项](#9-风险与注意事项)
10. [时间与工作量估算](#10-时间与工作量估算)
11. [附录](#11-附录)

---

## 1. 总体思路

原项目的智能体本质是一个"看屏幕 → 按键鼠"的纯外部循环（外部截屏 + 模拟键鼠），
**不依赖游戏引擎接口**。这决定了迁移到 CS2 的难度远低于深度改引擎的项目：

- **智能体运行路径**（`dm_run_agent.py`）：只用「截屏 + SendInput 键鼠」，理论上 CS2 可直接复用，
  唯二要动的是**窗口标题**和**画面裁剪区域**（HUD 位置变了）。
- **数据采集路径**（录制脚本）：重度依赖 **内存读取（pymem + hazedumper 偏移）** 和 **GSI**，
  内存偏移在 CS2 全部失效 —— 这是迁移的最大断点，但**不影响直接运行预训练模型做测试**。
- **成绩统计**：原代码靠 GSI 数击杀；CS2 的 GSI 字段被删减，所以改用
  **主方案：录制 demo + demoparser2 离线解析**（最可靠），**备选：GSI 实测可用字段**。

测试遵循"先冒烟、再基线、再定量、最后诊断改进"的阶梯，
每个 Phase 有明确的**通过/失败标准**，失败即可停在当前阶段做归因，避免带着坏管道往下测。

**测试假设（开始前请确认）：**

- [ ] Windows 10/11 机器，有 CUDA 显卡（前向传播需在 ~62ms 内完成，即 16fps）
- [ ] 已按[新手入门指南](新手入门指南.md) 4.2 节建好 conda 环境（Python 3.7 + TF 2.3.0）
- [ ] 已下载预训练模型 `ak47_sub_55k_drop_d4_dmexpert_28_stateful.json/.h5` 并放入 `./model/`
- [ ] Steam 账号拥有 CS2（测试**只做离线 bot 局**，见第 9 节风险说明）

---

## 2. 兼容性分析：CSGO 组件在 CS2 下的状态

| # | 组件 | 原文件 | CS2 下状态 | 说明与对策 |
|---|---|---|---|---|
| 1 | 窗口截屏（win32 BitBlt） | `screen_input.py` | ⚠️ **需小改** | 机制不变；但窗口标题变为 **`Counter-Strike 2`**，且 CS2 的 HUD/雷达默认位置与 CSGO 不同，裁剪偏移（上下 135px、左右 100px）需重新校准 → 用 `tools_screen_calibrate_cs2.py` |
| 2 | 模拟键鼠（SendInput/DirectInput） | `key_output.py` / `key_input.py` | ✅ **直接可用** | Windows 层面的输入注入，与游戏引擎无关 |
| 3 | 内存读取（pymem + hazedumper） | `meta_utils.py` / `dm_hazedumper_offsets.py` | ❌ **完全失效** | CS2 换了 Source 2 引擎，结构体/偏移全变，hazedumper 偏移表不适用。**影响范围**：仅录制脚本、地图覆盖分析（玩家坐标）。运行智能体本身**不需要**它。若确需坐标 → 用 [cs2-dumper](https://github.com/a2x/cs2-dumper) 重新生成偏移后移植，或改用 demo 解析拿坐标 |
| 4 | GSI 游戏状态集成 | `meta_utils.py` | ⚠️ **部分可用，需实测** | CS2 保留了 GSI 但**删减了部分字段**；cfg 放到 `.../game/csgo/cfg/`。所需字段（`player.match_stats.kills/deaths`、`player.state.health`、`map.phase`）是否还在，按附录 B 步骤实测。失败则走 demo 解析主方案 |
| 5 | 控制台命令（`sv_cheats`、`give weapon_ak47`、`bot_add`、`mp_restartgame`、`sv_infinite_ammo`、`fps_max`） | `console_csgo_setup.txt` / `dm_run_agent.py` 自动打字 | ✅ **基本可用** | CS2 离线 bot 局中这些命令依然有效（`bot_add` 拆成 `bot_add_t`/`bot_add_ct`，原 `bot_add ct` 语法也仍被接受）。`dm_run_agent_cs2.py` 的自动重启/发枪逻辑沿用 |
| 6 | 训练管线（TF 2.3 + ConvLSTM） | `dm_train_model.py` 等 | ✅ **与游戏无关，不动** | 训练在数据上，与哪个版本游戏无关。将来用 CS2 数据微调时直接复用 |
| 7 | 鼠标灵敏度映射 | config（灵敏度 2.5 + Raw Input **关**） | ⚠️ **存在域差距** | **CS2 移除了 `m_rawinput`，强制 Raw Input 开启**。原模型在 raw off（经 Windows 指针加速管线）下训练，同一位移对应的转身角度在 CS2 中不同 → Phase 2 做转角校准，用 `cs2_config.MOUSE_SCALE` 补偿 |
| 8 | 游戏画面分布（域差距） | — | ⚠️ **固有差异** | Source 2 渲染：体积烟雾、光影/色调、准星渲染、HUD/计分板样式都变了。模型未必失灵（地图几何、武器模型大体一致），但可能出现"看不懂烟雾"等失败模式 → Phase 4 诊断 |
| 9 | 游戏内设置项（分辨率/准星/灵敏度） | README 附录 E | ✅ **可复刻** | 1024×768 4:3 窗口化、灵敏度 2.5、准星共享代码在 CS2 中均可用。部分 viewmodel cvar（如 `cl_bobcycle`）在 CS2 已移除——只影响画面观感，次要 |

**一句话结论：** 直接运行预训练模型做 CS2 测试 → 只需 Phase 1 的最小适配（改标题 + 校准裁剪）；
做严谨评估 → 加上 demo 解析统计；要让模型"重获新生" → 需要 CS2 数据微调（Phase 4 路线）。

---

## 3. GitHub CS2 开源生态调研

调研结论：**目前没有**直接"从画面玩 CS2"的成熟开源智能体项目（反作弊与 ToS 使这类项目难公开持续维护），
与本项目同源/相关的研究线如下。测试所需的游戏侧工具链则全部有开源实现：

### 3.1 测试工具链（本次会用到）

| 项目 | 语言/形式 | 在本测试计划中的角色 |
|---|---|---|
| [a2x/cs2-dumper](https://github.com/a2x/cs2-dumper) | Rust 工具，输出 C++/Rust/Python 头文件 | 社区标准 CS2 偏移转储器（持续跟进游戏更新）。仅当需要移植内存读取（如玩家坐标做地图覆盖分析）时使用 |
| [LaihoE/demoparser](https://github.com/LaihoE/demoparser)（PyPI 包名 `demoparser2`） | Python（Rust 内核） | ⭐ **成绩统计主方案**：解析 CS2 demo 文件，提取击杀事件、位置、视角等。`pip install demoparser2`，配 `tools_demo_stats.py` 使用 |
| [roflmuffin/CounterStrikeSharp](https://github.com/roflmuffin/CounterStrikeSharp) | C# / .NET 8 服务端插件框架（需 MetaMod:Source） | 备选方案：写插件实时记录 K/D 到文件。功能最强但要搭本地专用服务器 + MetaMod，工程量大，本计划仅列为备选 |
| [cs-gamestate](https://pypi.org/project/cs-gamestate/) / cs2_gsi(Rust) 等社区 GSI 库 | Python / Rust | 备选方案：GSI 服务端库。CS2 的 GSI 字段可用性需先按附录 B 验证 |
| [markus-wa/demoinfocs-golang](https://github.com/markus-wa/demoinfocs-golang) | Go | 另一个成熟 CS2 demo 解析库（备选，若需 Go 侧工具） |
| [pnxenopoulos/awpy](https://github.com/pnxenopoulos/awpy) | Python | CS 数据分析库（基于 demo 解析，做位置/图表分析可参考） |

### 3.2 相关研究项目（了解/引用用）

| 项目 | 说明 |
|---|---|
| 本项目作者的后续工作 **MLMove**（论文 *Learning to Move Like Professional Counter-Strike Players*, [arXiv 2408.13934](https://arxiv.org/html/2408.13934v1)） | 用职业选手 demo 数据学移动行为的后继研究，方法可迁移到 CS2 |
| [eloialonso/diamond](https://github.com/eloialonso/diamond) | 扩散世界模型（DIAMOND）训练的 CS 智能体，在神经网络环境里玩类 CS 游戏，思路可借鉴做"CS2 域差距"研究 |
| [mwydmuch/ViZDoom](https://github.com/mwydmuch/ViZDoom) | 经典"从像素玩 FPS"研究平台，CS2 不可行时的替代实验环境 |
| [datamllab/awesome-game-ai](https://github.com/datamllab/awesome-game-ai) | Game AI 资源汇总，跟进新工作 |

> ⚠️ 调研中出现的外挂类项目（自瞄/透视）一律**不采用**：本项目是学术研究，
> 使用此类代码既偏离研究目的，也会带来账号与法律风险。

---

## 4. Phase 0：环境准备

**目标：** 一台按训练配置复刻了设置的 CS2 + 可运行模型推理的 Python 环境。
**预计耗时：** 1~2 小时。**通过标准：** 0.4 的自检清单全部打勾。

### 0.1 Python / GPU 环境

```bash
conda activate e2e          # Python 3.7 + TF 2.3.0（见新手入门指南 4.2 节）
python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"   # 期望 True
```

模型文件就位（同[新手入门指南路线 A](新手入门指南.md)）：

```
D:\ZCode\CSGO\v1\model\
├── ak47_sub_55k_drop_d4_dmexpert_28_stateful.json
└── ak47_sub_55k_drop_d4_dmexpert_28_stateful.h5
```

### 0.2 CS2 安装与设置复刻 checklist

启动 CS2 → Play → **Practice（With Bots）** → 选 Dust II（测试用离线局）。
先在设置里完成以下复刻（对照原训练配置）：

| 设置项 | 目标值 | CS2 现状 |
|---|---|---|
| 开发者控制台 | 启用（设置→游戏→启用开发者控制台） | ✅ |
| 显示模式 | **窗口化**（不是全屏/无边框） | ✅，截屏模块依赖窗口句柄 |
| 分辨率 | **1024×768，4:3** | ✅ |
| 鼠标灵敏度 | **2.50** | ✅ |
| Raw Input | **无法关闭**（CS2 已移除该选项，强制开启） | ⚠️ 固有域差距，Phase 2 校准 |
| 准星 | 代码 `CSGO-UKcZG-QN8eW-WQMvd-NX6xr-RPqRP`（经典静态绿） | ✅ 共享代码机制仍可用 |
| 画质 | 全最低（减少截屏耗时与画面差异） | ✅ |
| HUD | 尽量调大（CS2 的 HUD 缩放/位置设置与 CSGO 不同，调到接近 CSGO 观感即可） | ⚠️ 具体以 Phase 1 校准为准 |

### 0.3 首次进局验证控制台命令

进局后按 `~` 打开控制台，逐条验证（详见附录 A）：

```
sv_cheats 1
mp_limitteams 0; mp_autoteambalance 0
bot_kick
bot_add_t easy; bot_add_ct easy   （各加 5~6 个）
sv_infinite_ammo 1
mp_roundtime 6000
mp_restartgame 1
give weapon_ak47
fps_max 64
```

**通过标准：** bot 出现、AK47 到手、无限弹药生效、`cl_showpos 1` 能在屏幕上看到角度数值
（Phase 2 校准要用）。

---

## 5. Phase 1：最小移植冒烟测试（约半天）

**目标：** 验证「截屏 → 动作输出 → 键鼠注入」这条管道在 CS2 下畅通（不评估模型质量）。
**通过标准：** 1.1~1.4 四项全部达标。

### 1.1 截屏校准（对应工具 `tools_screen_calibrate_cs2.py`）

模型吃的是 150×280 的小图，由 1024×768 窗口裁掉上下 135px、左右 100px 再缩放得到。
CS2 的 HUD/雷达位置与 CSGO 不同，必须重新校准裁剪框：

```bash
python tools_screen_calibrate_cs2.py
```

- 保持 CS2 窗口化在 1024×768 并**处于对局中**，运行工具后会弹出一个实时预览窗口
- 调整滑条改变裁剪区域：**目标 = 裁出"准星居中、含 HUD 血量/弹药、去掉窗口标题栏"的画面**，
  且缩小到 150×280 后人眼可辨认敌我轮廓
- 按 `s` 保存为 `cs2_crop_offsets.json`（`dm_run_agent_cs2.py` 会自动读取），按 `q` 退出
- 再运行一次工具确认加载了保存值

**达标标准：** 预览的 150×280 图里，准星位于画面中心附近、HUD 数字完整、无标题栏黑边。

### 1.2 窗口发现

```bash
python dm_run_agent_cs2.py --check-window
```

脚本按标题查找 CS2 窗口（`Counter-Strike 2`，做模糊匹配以防后缀变化）。
**达标标准：** 打印出窗口句柄且非 0。

### 1.3 硬编码动作管道测试（最重要）

`dm_run_agent_cs2.py --smoke` 会忽略模型，按固定脚本发动作，每段 5 秒并在控制台提示：

1. 鼠标匀速右移（应看到**准星持续右转**）→ 验证鼠标注入
2. 按住 W 前进 → 验证键盘注入
3. 左键开枪（弹道/声音）→ 验证点击注入
4. 空格跳、R 换弹

**达标标准：** 四类动作全部在游戏中有可见效果。
**任一失败 → 停在此处排查**（管理员权限运行？焦点在游戏窗口？杀毒软件拦截 SendInput？），
不要继续 1.4。

### 1.4 截屏时延测量

工具模式 `--fps-test` 连续抓 1000 帧统计平均耗时。
**达标标准：** 平均单帧 < 40ms（留 22ms 给推理，凑满 62ms/帧 的 16fps 预算）。
若不达标：确认画质全低、关闭其他占 GPU 程序；仍不达标则考虑把 `loop_fps` 降到 12
（注意：这会造成与训练数据 16fps 的时序差异，评估时需记录）。

---

## 6. Phase 2：零样本基线测试（1~2 天）

**目标：** 把预训练模型原封不动放到 CS2 里跑，回答「CSGO 模型在 CS2 能不能玩」，
并完成鼠标映射校准。
**通过标准：** 2.1、2.2 完成且拿到至少 3 局有效的 demo 统计数据（无论成绩好坏——差也是重要结论）。

### 2.1 鼠标转角校准（应对 Raw Input 强制开启的域差距）

原理：CS2 里 SendInput 位移 X 与视角转角 θ 的比例，可能与训练环境（CSGO raw off）不同。
校准方法用游戏自带的 `cl_showpos 1`：

1. 控制台 `cl_showpos 1`，屏幕出现角度读数（vel/pitch/yaw）
2. 运行 `python dm_run_agent_cs2.py --calibrate`：每 5 秒发一段恒定位移
   （`mouse_x = +30` 三档：+10/+30/+60），**人工记录**每段起止的 yaw 读数
3. 计算每档「度/帧」；对照训练数据统计（可用 `tools_view_save_egs.py` 看原数据集
   `mouse_x` 档位对应的视角变化，或直接参考：理论值 `deg = Δcounts × 0.022 × 2.5`）
4. 得出补偿系数 → 写入 `cs2_config.py` 的 `MOUSE_SCALE`（默认 1.0）
5. 复测：校准后再跑一次第 3 步，误差应在 ±10% 内

> demo 解析也可以自动化这一步（录 demo 后用 demoparser2 读每 tick 视角差），
> 但人工读 `cl_showpos` 对 3 档 × 5 秒的实验更快，先手动做。

### 2.2 正式运行（零样本）

```bash
python dm_run_agent_cs2.py --demo
```

- `--demo` 等价原脚本的 `IS_DEMO=True`：弹出 AI 视野窗口（鼠标向量、开枪概率、按键状态叠加）
- 每局 10 分钟（脚本默认），结束后自动暂停并释放按键；**按 Q 可随时安全退出**
- 对局开始前在控制台输入 `record cs2_zero_01`（附录 A），结束后 `stop` —— demo 文件落盘用于统计

**观察记录表（每局填写，Phase 4 诊断用）：**

| 观察项 | 记录 |
|---|---|
| 是否主动移动/卡墙/原地转圈 | |
| 是否朝敌人方向转（说明视觉理解仍有效） | |
| 开枪时机（对枪才开 / 乱扫 / 永不开枪） | |
| 对烟雾弹的反应（CS2 体积烟雾视觉差异大） | |
| 卡死频率（inaction 计数、LSTM 重置打印） | |
| 前向传播耗时（fwd 字段）是否稳定 ≤ 62ms | |

### 2.3 成绩统计（主方案：demo + demoparser2）

```bash
pip install demoparser2 pandas          # 装在任意 >=3.8 环境（不必是 e2e 环境）
python tools_demo_stats.py *.dem        # 解析并输出 demo_stats_summary.csv
```

输出每个 demo 的：总击杀 / 死亡 / K/D / 每分钟击杀 / 对局时长 / 地图名。
**注意：** demo 里的"玩家名"是 agent 控制的那个账号，统计脚本会列出所有玩家的对照数据方便核对。

### 2.4 （备选）GSI 实测

若想恢复实时 K/D 显示：先按**附录 B** 验证 CS2 GSI 字段。可用 → `dm_run_agent_cs2.py --gsi`；
不可用 → 放弃 GSI，统一走 demo 统计（本计划主方案）。

---

## 7. Phase 3：定量评估（2~3 天）

**目标：** 产出可比较的数字，回答「在 CS2 上表现到底如何」。
**通过标准：** 每个配置 ≥5 局有效数据 + 一份汇总表。

### 7.1 实验矩阵

| 配置 | 内容 |
|---|---|
| E1（核心） | `ak47_sub_55k_drop_d4_dmexpert_28`，CS2，dust2，6 easy bots，无限弹药，每轮 10 分钟 × **N=5 局** |
| E2（对照，推荐） | 同上配置在 **CS:GO legacy**（Steam → CS2 属性 → Betas → `csgo_legacy`）重跑 × N=5 局 → 直接量化"CS2 迁移损失" |
| E3（难度梯度，可选） | easy → fair bot 各 5 局，观察性能曲线 |

控制变量要点：bot 数量与难度固定；每局前 `mp_restartgame 1` 重开；同一时间段连跑（避免
服务器/内存状态漂移）；agent 参数（`IS_SPLIT_MOUSE`、概率动作开关）全程不动。

### 7.2 指标定义

| 指标 | 来源 | 说明 |
|---|---|---|
| K/D per 1000 frames | demo 解析（主）/ GSI（备） | 与论文/原代码打印口径一致（`ks`、`ds` 字段） |
| K/D per minute | demo 解析 | 跨帧率可比 |
| 每局存活时间占比 | demo 解析（死亡事件间隔） | 反映"卡死/送死"程度 |
| 前向传播耗时 p50/p95 | 运行日志（fwd 字段） | 排除性能干扰 |
| inaction 触发次数 | 运行日志（`reset states` 打印计数） | 反映模型输出是否坍缩 |
| （可选）地图活动范围 EMD | demo 解析出坐标 + `tools_map_coverage_analysis.py` 思路 | 与训练数据分布对比，量化行为偏移 |

### 7.3 结果记录模板

```
| 局号 | 日期 | 模型 | 平台 | bot配置 | 时长 | 击杀 | 死亡 | K/D | K/min | 前向p95 | 异常事件 |
|-----|------|------|------|--------|------|------|------|-----|-------|---------|----------|
| 01  |      | dmexpert_28 | CS2 | 6 easy | 10min |      |      |     |       |         |          |
```

全部跑完后计算均值±标准差；E1 vs E2 做简单显著性对照（样本小，报均值差异即可，别硬上 t 检验）。

---

## 8. Phase 4：域差距诊断与改进路线

**目标：** 解释 Phase 2/3 的结果，给出改进优先级。
**方法：** 用 Phase 2 的观察记录表 + `--demo` 视野窗口回放，对照下表归因：

| 失败模式 | 可能原因 | 验证方法 | 改进选项（按性价比排序） |
|---|---|---|---|
| AI 完全不动 / 只会原地小抖 | 截屏裁剪没对准（模型看到的画面构图不对） | `--demo` 看 AI 视野是否构图正常 | 重新跑 1.1 校准；检查 `MOUSE_SCALE` |
| 会动但不朝人转、枪法全无 | 画面域差距（Source 2 光影/材质）超出泛化能力 | 用 `tools_view_save_egs.py` 对比训练数据帧与 CS2 实帧 | P1：① CS2 数据微调（见下）；P2：图像风格化预处理（颜色直方图匹配到 CSGO 数据分布） |
| 对烟雾时行为异常 | CS2 体积烟雾与 CSGO 烟雾视觉完全不同 | 观察 AI 过烟点位的动作 | 微调数据中加入 CS2 烟雾场景 |
| 转身幅度普遍过大/过小 | Raw Input 强制开启导致的映射差 | 2.1 校准实验 | 调 `MOUSE_SCALE` |
| 频繁卡死、输出坍缩 | 输入分布偏移导致概率输出塌缩 | 日志 inaction 计数、reset 次数 | 同上；运行时已有概率采样自救机制 |
| 一切正常但成绩略降 | 预期内的迁移损失 | E1 vs E2 对比 | 记录结论即可 |

**CS2 数据微调路线（若诊断指向画面域差距）：**

1. 用 **`dm_record_data_me_wasd.py` 思路**在 CS2 自录数据——原脚本依赖内存偏移，
   迁移时砍掉 RAM/GSI 元数据，只存 `(截图, 键鼠动作)`：动作直接来自 `key_input` 的键盘/鼠标状态，
   **完全绕开内存读取**（这正是它设计 TFGHUM 替代键位的原因）
2. 补一个简单的动作推断/对齐步骤 → 走 `dm_pretrain_process.py` → `dm_train_model.py`
   从 `ak47_sub_55k_drop_d4` checkpoint 微调（配置参照[新手入门指南路线 B](新手入门指南.md)）
3. 数据量参考：专家微调集 190 文件 ≈ 24GB ≈ 3 小时游戏即有明显效果，CS2 微调可先按此量级规划

> 若走深研究路线，可参考 MLMove（职业选手动作生成）与 DIAMOND（世界模型）的做法，
> 以及用 cs2-dumper + demoparser2 从职业 demo 构造大规模 CS2 训练数据。

---

## 9. 风险与注意事项

1. **只做离线 bot 局。** `sv_cheats 1`、`give` 等命令只在本地服务器有效，不会触发 VAC；
   但**截屏注入与模拟键鼠在带反作弊的官方服务器上可能被检测**，绝不要把本测试流程用于
   官方匹配/社区对战服务器。
2. **账号策略：** 用小号测试。原项目 README 明确免责：模拟输入与内存解析可能被 Valve
   识别并引发作弊嫌疑，风险自担。
3. **ToS / 研究伦理：** 本测试仅限个人研究用途（原仓库许可证禁止商用）；
   不使用任何外挂类第三方代码。
4. **demo 文件位置：** CS2 录制的 demo 在 `.../Counter-Strike Global Offensive/game/csgo/*.dem`
   （注意与 CSGO 旧路径不同，`game/` 子目录是 CS2 新结构）。
5. **脚本未经实机验证：** 配套 CS2 脚本基于原代码逻辑改写 + 语法校验，但作者环境无法运行
   CS2 实测；请严格按 Phase 1 冒烟流程先行验证，遇到问题对照第 2 节兼容性表定位。

---

## 10. 时间与工作量估算

| 阶段 | 内容 | 预估 | 前置 |
|---|---|---|---|
| Phase 0 | 环境 + 设置复刻 | 1~2 小时 | 模型已下载 |
| Phase 1 | 冒烟测试（校准+管道验证） | 半天 | Phase 0 |
| Phase 2 | 零样本基线 + 鼠标校准 | 1~2 天 | Phase 1 全通过 |
| Phase 3 | 定量评估（E1 必做，E2 推荐加 0.5 天装 legacy 版） | 2~3 天 | Phase 2 |
| Phase 4 | 诊断 + 改进（可选微调另计：数据采集 1~2 天 + 训练 1~2 天） | 1 天起步 | Phase 3 |

---

## 11. 附录

### 附录 A：CS2 测试局控制台命令块（可直接整段粘贴）

```
// —— 准备（进局后一次性）——
sv_cheats 1;
mp_limitteams 0; mp_autoteambalance 0;
bot_kick;
mp_roundtime 6000;
sv_infinite_ammo 1;
fps_max 64;
mp_warmup_end;

// —— 加 bot（easy 6 个 / 换 fair 做难度梯度）——
bot_add_t easy; bot_add_t easy; bot_add_t easy;
bot_add_ct easy; bot_add_ct easy; bot_add_ct easy;

// —— 重开并发枪（每次重开局用）——
mp_restartgame 1;
give weapon_ak47;

// —— 录 demo（每局开始前）——
record cs2_zero_01;     // 结束时 stop
// demo 落盘于 .../Counter-Strike Global Offensive/game/csgo/

// —— 调试辅助 ——
cl_showpos 1;           // 屏幕显示视角角度（鼠标校准用）
```

### 附录 B：CS2 GSI 字段验证步骤（备选方案，~30 分钟）

1. 在 CS2 安装目录 `.../Counter-Strike Global Offensive/game/csgo/cfg/` 下新建
   `gamestate_integration_test.cfg`（注意 CS2 的 `game/csgo/cfg` 新路径）：

   ```
   "Counter-Strike 2"
   {
       "uri" "http://localhost:3000"
       "timeout" "5.0"
       "auth"   { "token" "MYTOKENHERE" }
       "data"
       {
           "provider"       "1"
           "map"            "1"
           "round"          "1"
           "player_id"      "1"
           "player_state"   "1"
           "player_weapons" "1"
           "match_stats"    "1"
       }
   }
   ```

2. 临时运行一个监听器看数据（不必动原代码）：

   ```python
   # gsi_probe.py —— 打印 CS2 GSI 推送来的原始 JSON
   from http.server import BaseHTTPRequestHandler, HTTPServer
   import json
   class H(BaseHTTPRequestHandler):
       def do_POST(self):
           body = self.rfile.read(int(self.headers['Content-Length']))
           print(json.dumps(json.loads(body), indent=2, ensure_ascii=False))
           self.send_response(200); self.send_header('Content-type','text/html')
           self.end_headers()
       def log_message(self, *a): pass
   HTTPServer(('localhost', 3000), H).serve_forever()
   ```

3. 启动 `python gsi_probe.py` → 进局击杀一个 bot → 观察输出：
   - 有 `player.match_stats.kills/deaths`、`player.state.health` → GSI 可用，
     `dm_run_agent_cs2.py --gsi` 可开（token 改一致）
   - 缺字段或完全无推送 → 放弃 GSI，走 demo 统计主方案
   - 顶层节点名若是 `"Counter-Strike 2"` 而非 CSGO 旧名，`meta_utils.py` 的
     认证匹配不受影响（它只查 `auth.token`），但保险起见两种写法都可试

### 附录 C：demoparser2 解析示例

```python
# 最小示例：列出 demo 中的击杀事件
from demoparser2 import DemoParser

p = DemoParser("cs2_zero_01.dem")
kills = p.parse_event("player_death")
print(kills.columns.tolist())          # 先看有哪些列（版本间略有差异）
print(kills[["attacker_name", "user_name"]])   # 击杀者 / 被击杀者
```

日常统计直接用配套的 `tools_demo_stats.py`（自动兼容列名差异，输出汇总 CSV）。

### 附录 D：配套脚本速查

| 命令 | 作用 |
|---|---|
| `python tools_screen_calibrate_cs2.py` | 校准 CS2 截屏裁剪区域 → 保存 `cs2_crop_offsets.json` |
| `python dm_run_agent_cs2.py --check-window` | 检查能否找到 CS2 窗口 |
| `python dm_run_agent_cs2.py --smoke` | 硬编码动作管道测试（不开模型） |
| `python dm_run_agent_cs2.py --fps-test` | 截屏时延测量 |
| `python dm_run_agent_cs2.py --calibrate` | 鼠标转角校准（配合 cl_showpos 1） |
| `python dm_run_agent_cs2.py --demo` | 开 AI 视野可视化运行 |
| `python dm_run_agent_cs2.py --gsi` | 启用 GSI 实时统计（附录 B 验证通过后） |
| `python tools_demo_stats.py *.dem` | demo 成绩解析 → CSV |

### 附录 E：参考链接

- Valve Developer Wiki — Game State Integration（已覆盖 CS2）：https://developer.valvesoftware.com/wiki/Game_State_Integration
- a2x/cs2-dumper：https://github.com/a2x/cs2-dumper
- LaihoE/demoparser（demoparser2）：https://github.com/LaihoE/demoparser
- roflmuffin/CounterStrikeSharp：https://github.com/roflmuffin/CounterStrikeSharp
- markus-wa/demoinfocs-golang：https://github.com/markus-wa/demoinfocs-golang
- MLMove 论文：https://arxiv.org/html/2408.13934v1
- DIAMOND：https://github.com/eloialonso/diamond
- Raw Input 在 CS2 中强制开启的讨论：https://www.reddit.com/r/GlobalOffensive/comments/16z6oas/
- 原项目 README / 新手入门指南.md

---

*文档生成于 2026-10-05。CS2 持续更新，具体设置项/命令以当前游戏版本实测为准。*
