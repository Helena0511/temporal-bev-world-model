# ResWorld 原生 PyTorch 复现记录

> 当前阶段已选定常见城市驾驶，最新配置、数据选择与实验见 `docs/CITY_PHASE1.md`。
> `scripts/train_full.sh`默认配置为 `configs/bev_attention_city60.json`；后续段落记录各历史阶段。

> 后续更新：BEVFormer-inspired感知与物理范围推导见 `docs/BEV_DESIGN.md`。
> 下文的原生模型测试和单epoch精度是旧配置基线。当前训练脚本默认候选已改为
> `configs/bev_attention_road60.json`，按明确的道路/运动假设推导范围；
> `configs/full_model.json`、60m `wide_*` 仍作为历史对照，不能混用不同范围的checkpoint。
> 正式长训练尚未启动，最终目标运行条件仍需明确。

当前已经实现和验证从真实 nuScenes 多相机图像到规划轨迹的完整训练链路。
这是便于独立运行的轻量重实现，**不是原版权重兼容实现，也不是论文精度复现完成**。
本地原版参考位于 `../ResWorld`；旧的教学示例保留不变。

## 文件与入口

| 文件 | 用途 |
| --- | --- |
| `resworld/driving_data.py` | 原始 JSON 读取、场景划分、三帧六相机、标定、位姿、未来轨迹、掩码、稀疏深度 |
| `resworld/driving_model.py` | 相机到 BEV、状态条件、多帧融合、场景与残差 Token、动作条件、未来 BEV、FGTR |
| `scripts/train_driving.py` | smoke、固定批次过拟合、训练、评估、恢复训练 |
| `scripts/inspect_driving_data.py` | 文件、标签、划分检查，支持 trainval |
| `scripts/visualize_driving.py` | 验证集图片与轨迹导出 |
| `configs/nuscenes_splits.json` | 官方 trainval 场景名单及来源 |
| `configs/full_model.json` | 较大原生模型配置；不代表原版网络配置 |
| `scripts/train_full.sh` | 后续正式 trainval 训练入口，尚未启动长训练 |

依赖仅为 PyTorch、NumPy、Pillow；可视化额外依赖 matplotlib。
本次测试环境为 PyTorch 2.14.0+cu130，`torch.cuda.is_available()` 为 False，
`nvidia-smi` 无法连接驱动，因此所有记录均为 CPU 实测。

## 数据流与约定

1. 输入按 `[t-2,t-1,t]` 排列，每帧六个相机，采用 nuScenes sample 关键帧。
2. 图片直接缩放，同时缩放内参；相机各自的曝光位姿用于转换到该帧 CAM_FRONT 参考自车坐标。
3. 图像卷积特征预测类别深度分布，用内参逆矩阵与外参将射线提升到三维，再 splat 到米制 BEV。
4. 训练时将同一关键帧 LiDAR 投影到相机，按特征像素取最近可见表面，监督类别深度。
   LiDAR 不作为模型输入；没有 LiDAR 文件的位置使用 -1 深度监督掩码。
5. 状态条件使用历史位姿差分得到的速度和角速度，不读取未来状态。
6. 两帧历史分别对齐到当前帧，三帧拼接经卷积生成场景特征。
7. 当前与最近历史用同一组空间权重提取残差 Token。权重只在历史覆盖区域归一化。
   `t-2` 参与场景融合；残差定义为当前减去最近历史，不将两组跨帧差分简单平均。
8. 场景 Token 解码三种初始轨迹；导航指令选择的初始**逐步位移**作为动作条件。
9. 动作调制后的残差 Token 经 Transformer、TokenFuser 重建一个预测 BEV，并通过可学习采样偏移
   和权重提取初始轨迹附近的上下文，解码位移修正。
10. 两个轨迹头均输出逐步位移，累计后得到当前固定自车坐标系中的位置。

坐标为 x 向前、y 向左、单位米；BEV row 0 为最大 x、col 0 为最大 y。
变换 `current2history = inverse(E_history) @ E_current` 是 grid_sample 的目标到源采样方向。
越界历史不作为场景变化；无历史覆盖时残差权重为零。

未来标签为 6 个约 0.5 秒的关键帧，缺失尾帧不计损失。导航标签由未来 3 秒末位置的横向位移
以 ±2 米阈值生成；这是离线开放环路的 oracle route 指令，不是独立导航系统的预测。
未来不足 3 秒时采用 straight 回退。评估时仍需要给模型一个 route 指令，但不输入未来轨迹或深度标签。

损失为初始和最终逐步位移的 masked L1 之和，加 0.1 倍稀疏深度交叉熵。
ADE 按所有有效位置的欧氏误差汇总；1/2/3 秒 L2 为对应时刻误差，要求该时段标签完整；
FDE 仅统计 6 步完整轨迹。所有指标都是米。它们不是原版的完整 UniAD/VAD 评估器，尚无碰撞率指标。

mini 使用 8 个训练场景、2 个验证场景；trainval 使用 700/150 场景名单。
来源：[nuScenes devkit splits.py](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/utils/splits.py)。
相邻窗口可重叠，但同一场景不跨训练与验证。元数据使用流式 JSON 解码，并共享索引，避免对 >1GB
sample_data 表重复全量反序列化；缺少图片的序列会跳过并报告数量。

## 与本地原版的主要差异

| 模块 | 本次原生实现 | 本地原版参考 |
| --- | --- | --- |
| 图像编码 | 三层卷积，从零训练 | ResNet50 + FPN |
| 视角变换 | 可学习类别深度 lift/splat | RCSample |
| 状态信息 | 从历史位姿计算 3 维运动量 | 18 维 CAN bus |
| 位置与 Token | 线性米制位置编码、LayerNorm TokenLearner、简化 TokenFuser | LearnedPositionalEncoding、GroupNorm TokenLearner、原版 TokenFuser |
| 动作与注意力 | FiLM、单头组可变形采样、位移修正 | MLN、MMCV 多头可变形注意力、独立最终位移解码 |
| 栅格与容量 | 默认 32 通道、32×48 BEV、16 深度；较大配置 64 通道、100×100、68 深度 | 256 通道 BEV；原版感知、深度、优化配置 |

因此，本阶段结论是“核心思路和真实数据训练闭环已跑通”。要追求论文数字，仍须对齐原版感知、
CAN bus、Token/注意力细节、增强、预训练、损失权重和官方评估协议；不能仅延长当前配置的训练就宣称复现论文。

## 已执行的验证（2026-10-14）

- `python -m unittest test_driving test_fgtr test_mini_resworld -q`：23 项通过。
  覆盖解析相机射线到米制栅格、四元数、稀疏深度最近表面、缺失未来帧损失/指标、
  相同历史零残差、无重叠掩码、元数据跨块解码，以及原有 FGTR/模型测试。
- mini 真实序列固定批次（2 条）100 次优化：规划损失 3.8621 → 0.2660，ADE 13.1120 → 0.2532 米。
  图像、深度、融合、场景 Token、残差 Token、动作条件、TokenFuser、采样器、修正头梯度均非零且有限。
  这是过拟合测试，不能用于报告泛化精度。记录：`runs/mini_overfit/metrics.jsonl`。
- mini 全部 299 条训练序列，单个 epoch（150 次更新），随后评估独立的 75 条验证序列。
  CPU 单 epoch 约 91 秒，包括验证。记录与权重：`runs/mini_pipeline_check/`。
- checkpoint 重新加载的所有轨迹指标与保存前一致；恢复训练从第 151 次更新继续。
  记录：`runs/mini_reload_eval/`、`runs/mini_resume_check/`。
- trainval：检查 26,030 个训练序列与 5,569 个验证序列，所需相机文件无缺失；
  完整 3 秒标签分别 22,530 与 4,819 条。记录：`runs/trainval_data_audit.json`。
  trainval 小批真实图片/标定/深度/轨迹也执行了前向、反向和验证集抽样评估。
- 默认和较大配置均执行 smoke；正式 smoke 为三次更新，避免零初始化修正头在首次更新前
  暂时阻断世界分支梯度。记录：`runs/trainval_smoke_final/`、`runs/larger_config_smoke_final/`。
- 较大配置（64 通道、100×100 BEV、128×352 图片）额外执行单条真实序列 60 步过拟合：
  规划损失 3.8208 → 0.3007，最终轨迹 ADE 12.9877 → 0.5430 米，世界/采样分支梯度恢复到
  可观测的非零量级。初期微小梯度来自零初始化修正头及优化过程，不是永久断开的计算图。
  记录：`runs/larger_config_overfit/`。该单批最终轨迹仍弱于初始轨迹，不能据此认定 FGTR 提升精度。

mini 单 epoch 独立验证结果：

| 方法 | ADE | 1s L2 | 2s L2 | 3s L2 / FDE |
| --- | ---: | ---: | ---: | ---: |
| 初始轨迹 | 3.7893 | 1.8068 | 4.3299 | 7.6049 |
| FGTR 最终轨迹 | 3.7629 | 1.8633 | 4.3057 | 7.4206 |
| 仅用过去位姿的匀速基线 | 1.9571 | 0.6890 | 2.1862 | 4.4620 |
| 静止基线 | 8.9040 | 5.2821 | 10.4498 | 15.4550 |

当前模型弱于匀速基线。单 epoch 和极小网络仅验证链路；mini 只有两个验证场景，不能代表 trainval。
特别是按照当前路线定义，mini 训练计数为左 22、直行 277、右 0，而验证计数为左 3、直行 55、右 17；
**mini 训练中没有右转样本，不能用它评价右转模式的学习能力。** 不应把验证场景加入训练来解决这一问题。
该 checkpoint 的分组验证 ADE 为左转 2.0060、直行 2.7390、右转 7.0845 米；
样本分布缺口是解释当前误差的重要证据，但并不排除网络容量、感知与优化不足。
FGTR 在 ADE/3 秒改善不大，1 秒略差，不能据此宣称模块已带来稳定收益。
可视化：`runs/mini_predictions/trajectories.png`，对应 token 和轨迹保存在同目录 `predictions.json`。

## 复跑命令

```bash
python -m unittest discover -s tests -t . -q
python -m scripts.inspect_driving_data --version v1.0-trainval --output runs/trainval_data_audit.json
python -m scripts.train_driving --mode smoke --limit-val 4 --output runs/mini_smoke
python -m scripts.train_driving --mode overfit --steps 100 --output runs/mini_overfit_repeat
python -m scripts.train_driving --mode train --epochs 1 --output runs/mini_pipeline_repeat
python -m scripts.train_driving --mode eval --resume runs/mini_pipeline_check/best.pt --output runs/eval_repeat
python -m scripts.visualize_driving --checkpoint runs/mini_pipeline_check/best.pt --output runs/predictions_repeat
```

训练日志为 JSONL；权重同时保存模型、优化器、随机数状态、参数与步数。每轮保留 last.pt，
按验证 ADE 保存 best.pt。config.json 保存场景名单、模型配置与源代码 SHA256。

正式训练入口已准备，按用户约定先讨论精度方案，再启动：

```bash
# 需要可用的 CUDA；当前会话尚不可用。
bash scripts/train_full.sh

# 可明确选择 CPU；较大配置的全量训练会显著慢于 mini 测试。
RESWORLD_DEVICE=cpu bash scripts/train_full.sh

# 恢复训练；图像尺寸、数据版本与 split-file 需要与 checkpoint 匹配。
bash scripts/train_full.sh --resume runs/trainval_native/last.pt
```

较大配置为原生模型 64 通道、3 层、16 Token、68 深度、100×100 BEV，图片 128×352，
batch size 2，12 epochs，AdamW 和 cosine 学习率，不含原版预训练感知网络。
当前未启动该长训练，也未把它标为完成。

## 精度讨论的建议顺序

1. 先确定目标是验证核心算法收益，还是严格复现原版数字。后者优先对齐上表架构/协议，
   不是同时修改多项设计后比较论文结果。
2. 先跑原生 trainval 基线，解决 mini 缺失右转训练样本的问题。评估代码已输出按场景/导航指令
   分组的误差（`final_by_group`），再做去历史、去世界模型、去 FGTR 的对照；
   匀速基线是必须保留的参照。分组记录：`runs/mini_grouped_eval/metrics.jsonl`。
3. 转弯样本稀少：trainval 路线计数为左 1,259、直行 23,197、右 1,574；需要按指令报告误差，
   再决定采样平衡策略。轨迹图中当前右转拟合尤其不足。
4. 最大感知提升候选是预训练编码器、图像分辨率与深度/BEV 分辨率。提高这些项后需监控 GPU
   显存；原生 lift/splat 在高分辨率下有较大的深度体积开销。
5. 用过去速度构造运动学轨迹先验是另一个可做的消融，但会改变当前模型定义；应单独命名和记录。
6. 避免仅用训练损失衡量“精度”；最终还需要与原版相同的数据处理及碰撞率评估。
