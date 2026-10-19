# CS2 Behavioural Cloning — CS:GO 行为克隆研究的 CS2 迁移

> **关于本仓库（署名 / Attribution）**
> 本仓库基于 [TeaPearce/CounterStrike_Behavioural_Cloning](https://github.com/TeaPearce/CounterStrike_Behavioural_Cloning) 二次开发——原项目为 Tim Pearce 与 Jun Zhu 的 IEEE CoG 2022 最佳论文项目（[arXiv:2104.04258](https://arxiv.org/abs/2104.04258)）。
> 原始代码、论文、预训练模型与数据集版权归原作者所有；本仓库记录的是**我对 CS2 环境的迁移与改进工作**，仅供个人学习与研究。
> 仓库中 2021–2024 年的原始提交历史来自上游项目，予以保留以完整追溯。

## 我做了什么（My Work）

在原项目（面向 CS:GO）基础上，完成面向 CS2 的迁移与配套数据工程：

### 1. 采集链路迁移
- **屏幕捕获适配**：适配 CS2 窗口与渲染方式变化，修复陈旧帧捕获问题，增加焦点守卫（`tools_screen_calibrate_cs2.py`）
- **录制器 v2**：改用 `WM_INPUT` 原始鼠标输入录制，修正 24 字节 `RAWMOUSE` 结构布局解析（`dm_record_data_cs2.py`）
- **配置参数化**：新增 `cs2_config.py`，裁剪偏移等参数独立管理

### 2. 数据质量工程
- **增量录制校验**：新增 `tools_validate_recording.py`，对长时间录制会话做增量校验，尽早发现坏数据
- **分布差异验证**：用直方图匹配检验 CS:GO 旧数据到 CS2 的可迁移性；Stage 1 结论是视觉分布差异过大，决定重新采集 CS2 数据而非复用旧数据（决策过程见提交记录）

### 3. 模型微调
- `dm_finetune_cs2.py`：开火片段过采样微调（`--fire-oversample`，默认 3 倍），缓解开火帧占比过低的类别不均衡问题
- 训练环境：TensorFlow 2.10.1 + RTX 4070 Ti SUPER（Windows 下采集与推理）

### 4. 过程文档
- `docs/`：CS2 迁移入门指南、迁移测试计划、录制检查清单（操作规范与环境要求）

## 阶段记录（Progress Log）

| 阶段 | 内容 | 结果 |
| --- | --- | --- |
| Phase 0 | 训练/推理环境搭建（TF 2.10.1 GPU） | ✅ 完成 |
| Phase 1 | CS2 采集冒烟测试（捕获 + 焦点守卫） | ✅ 通过 |
| Stage 1 | 旧数据直方图匹配迁移验证 | ❌ 差异过大 → 转为重采 CS2 数据 |

## 上游项目资源（Upstream）

- 论文：[arXiv:2104.04258](https://arxiv.org/abs/2104.04258)（IEEE CoG 2022 **Best Paper**；另发表于 NeurIPS 2021 Offline RL Workshop）
- 预训练模型与数据集：[HuggingFace: TeaPearce/CounterStrike_Deathmatch](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch)
- 完整的脚本说明、环境配置与故障排查：见[上游 README](https://github.com/TeaPearce/CounterStrike_Behavioural_Cloning#readme)

## 免责声明（Disclaimer）

- 本项目**仅供个人学习与研究**（行为克隆 / 模仿学习工程实践），遵循上游许可，不用于商业用途。
- 使用游戏自动化相关代码请自行了解并遵守游戏服务条款，请勿做破坏其他玩家体验的用途。
