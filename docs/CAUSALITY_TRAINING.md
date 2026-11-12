# 感知、自车状态与分阶段融合

用户提出的目标是先去掉自车状态重训一个baseline，再分别训练感知和自车状态，最后训练二者配合。
本轮将它作为独立研究变体实现；原城市模型与运动学先验checkpoint继续保留。

## 论文核对

[Exploring the Causality of End-to-End Autonomous Driving](https://arxiv.org/html/2407.06546v2)
的§3.3、§4.2讨论输入消融、输入编辑及训练时输入dropout，实验在CARLA闭环中进行。
论文也指出当前速度的重要性；不能把它概括成“去掉状态会提高性能”。
本项目的分支预训练、冻结后融合、联合微调是待验证的适配方案，不是该论文训练流程的复现。
分阶段训练本身已有先例：[UniAD官方说明](https://github.com/OpenDriveLab/UniAD#results-and-pre-trained-models)
先训练track/map感知模块，再联合优化全部任务。这里的单独ego分支和状态dropout融合仍是本项目的设计。

## 三种信息隔离

1. **只去掉显式状态**：`ModelConfig.use_ego_state=False`结构性绕过`state_condition`，
   同时禁止`kinematic_prior=True`。三帧图像、历史位姿对齐、相机曝光时刻补偿保留。
   因此它仍使用几何位姿，名称必须明确，不能称为完全无自车运动信息。
2. **严格当前帧视觉**：再设`temporal_context=single_frame`、`use_world_delta=False`。
   只编码当前六相机图像，采用`camera_to_ego_static`固定安装标定；不读取过去图像、
   `motion`、`history_dt`、`current2history`或带曝光补偿的外参。nuScenes六相机并非严格同步，
   取消曝光位姿补偿会引入额外几何误差；单帧也不能唯一确定瞬时速度。
3. **仅自车状态**：`EgoTrajectoryBranch`只读取过去运动、相邻过去位姿与时间间隔，
   初值为当前坐标系匀速轨迹，再学习三种指令各自的位移修正。不读取图像。

所有规划对照共享左/直/右指令；目前指令由未来轨迹生成，是离线oracle导航，仍非部署条件。
图像置零的推理测试和上述独立重训回答不同问题；二者不能互相替代，也不能据此宣称识别了真实因果结构。

## 感知监督与训练顺序

感知单训应有环境真值，不能只根据轨迹ADE判断是否看懂环境。目前接入：

- 稀疏LiDAR投影的相机深度分类监督与量化深度MAE。
- 当前nuScenes三维标注框在BEV的平面投影，类别为背景、vehicle、human。
  `size`按nuScenes的宽/长/高解析，框旋转与当前ego坐标转换后栅格化；很小的框至少覆盖中心单元。
  其他标注类别所在单元忽略，类别重叠按确定顺序处理。分辨率变化会改变小框的离散化误差。
  此指标叫**标注框投影IoU**，并非nuScenes检测mAP、真实可行驶空间或完整碰撞占据评测。

这些标签只参与训练损失/评估，不作为相机推理输入。尚无车道/道路边界、交通灯等任务，
不能称为完整环境理解。BEV范围沿用城市工程包络，不根据验证误差缩小。

`scripts/train_staged_driving.py --stage ...`各阶段：

| 阶段 | 训练参数 | 主要目标 |
| --- | --- | --- |
| perception | 相机编码器、BEV空间注意力、感知头 | 深度+标注框投影；无规划损失 |
| ego | 自车状态MLP与轨迹头 | 过去状态→未来位移 |
| visual | 无显式状态的视觉规划器、感知头 | 视觉轨迹baseline+感知监督 |
| fusion | 视觉规划器、融合头；固定相机编码器、感知头和ego分支 | 学习运动初值与环境特征的配合 |
| joint | 联合解冻；相机和ego学习率默认为其他参数的0.1 | 小步联合微调 |

融合头读取视觉规划特征、视觉轨迹、自车embedding及状态轨迹，输出状态轨迹的位移修正。
零初始化时等于已训练的ego分支。训练中按序列以0.3概率同时屏蔽ego embedding和ego轨迹，
此时基准轨迹改用视觉分支；视觉辅助轨迹损失权重0.5，感知监督继续保留。
位姿对齐仍在视觉分支中，这个dropout针对的是显式状态分支。
联合微调默认是`3e-4`规划/融合学习率、`3e-5`相机/ego学习率。

## 可复跑实验

纯输入消融使用原训练入口、原损失、相同48/24条数据与两轮预算：

```bash
python -m scripts.train_driving --version v1.0-trainval --mode train \
  --model-config configs/bev_attention_city60_pilot_no_ego.json \
  --selection-file runs/city_phase1/selection.json \
  --epochs 2 --output runs/city_phase1/no_ego_repeat
```

分支预训练和融合流程另行运行，输出目录必须不存在：

```bash
python -m scripts.run_city_causality_pilot --output runs/city_phase1/causality_repeat
```

该脚本依次运行ego 2轮、perception 2轮、visual 2轮、严格当前帧visual 2轮、fusion 1轮、joint 1轮，
以及从随机初始化直接联合训练的2轮对照。`visual`对照有额外BEV监督，区别于上面的纯输入消融。
无状态轨迹baseline本轮从随机初始化训练；perception checkpoint用于冻结后融合。
分阶段方案使用更多预训练更新，不能与从零联合训练2轮的结果做等计算量优劣结论；
正式实验需要匹配训练预算、多个随机种子，并评估更大验证集。
所有小试验都采用同一米制范围、图像尺寸、划分和48/24清单，禁止跨split迁移训练样本。

checkpoint保存阶段、配置、数据清单SHA256、源代码SHA256、训练参数和来源checkpoint SHA256。
视觉和整体迁移核对模型几何、图像尺寸、版本、split及数据清单。独立ego分支不读取图像，
其迁移核对channels、future_steps、版本、split及数据清单，允许更换图像主干与尺寸。
`--init-checkpoint`是权重初始化，
从新优化器和第0轮开始；不是中断恢复。`--perception-checkpoint`和`--ego-checkpoint`只导入对应分支，
不会用感知checkpoint中未训练的ego参数覆盖已训练的ego分支。

## 已完成的CPU小试验

48条训练、24条独立场景且路线均衡的验证，seed=0。原模型重新加载评估仍为2.634879m，
确认输入隔离改造没有改变旧checkpoint的默认推理结果。

| 模型 | 验证ADE（m） |
| --- | ---: |
| 原城市模型，原损失 | 2.6349 |
| 只关闭显式状态，原损失重训 | 3.4601 |
| 无显式状态，增加BEV感知监督 | 3.5073 |
| 严格当前帧视觉，增加BEV感知监督 | 3.6744 |
| 独立ego分支 | 1.7884 |
| 两路预训练后冻结编码器，训练融合 | 1.7601 |
| 再联合微调 | 1.7010 |
| 随机初始化直接联合训练 | 1.7281 |

以上分阶段与从零联合的训练预算不同，差异也小，尚未证明分阶段更优。
现有运动学先验变体为1.6096m，本轮没有超过它；不能把新流程称为新的最佳精度。
原始结果和预算见 `runs/city_phase1/causality_pilot/{comparison,summary}.json`。

感知单训2轮末，车辆/行人框投影前景mIoU为0.809%，量化深度MAE 10.663m，明显未收敛。
感知best checkpoint按mIoU选为第1轮（0.835%），其深度MAE为18.444m；融合使用该best，
不能把第2轮的深度指标写成融合所用checkpoint的指标。
可查看 `runs/city_phase1/causality_pilot/perception_last.png` 的固定昼/夜验证样本。

推理敏感性：原损失无状态模型，正常图像ADE 3.4601m，图像置零3.4495m；
联合微调后，正常图像1.7010m，图像置零1.7100m，轨迹平均变化仅0.0905m。
这些结果提示当前视觉贡献仍弱，不能把融合的小幅收益归因于已经学会环境感知。
详见 `runs/city_phase1/no_ego_control_diagnosis.json` 与
`runs/city_phase1/causality_pilot/joint_sensitivity.json`。

冻结阶段的真实checkpoint逐张量核对，38个感知编码器张量、6个感知头张量、6个ego张量均保持完全一致；
联合解冻后，感知编码器38个和ego 6个张量均发生更新。对应检查保存为
`fusion_freeze_check.json`、`joint_unfreeze_check.json`。当前49项测试通过，包含输入隔离、梯度、
冻结/迁移、标注框尺寸与方向，以及state/current模式避免多余图像读取。

单独评估示例：

```bash
python -m scripts.train_staged_driving --stage joint --eval-only \
  --selection-file runs/city_phase1/selection.json \
  --init-checkpoint runs/city_phase1/causality_pilot/joint/best.pt \
  --output runs/city_phase1/joint_eval_repeat
```

初始流程试验在CPU完成。用户随后授权最多使用一张GPU，已通过conda执行入口验证CUDA，
固定物理GPU 1的UUID `GPU-81073757-da40-048b-bff4-9e85a426129c`。
`scripts/train_staged_driving.py`在CUDA模式要求可见设备数恰好为1，并记录设备UUID环境变量、型号和峰值显存。
感知阶段只解码当前六相机，ego阶段不解码图像；两者仍用相同历史状态与标签定义。
`scripts/train_perception_single_gpu.sh`在完整22,480/4,819条工况筛选清单上运行感知预训练：
1m配置、128×352、batch 16、4轮。它是完整数据的感知预训练，不是0.5m配置的完整端到端训练。
启动命令应通过可访问GPU的conda入口，并在外层设置`RESWORLD_GPU`为上述单个UUID；
launcher与训练程序各自检查单卡限制，所有权重和日志独立保存。
batch 16单卡试验峰值已分配显存为5,084MiB；batch 4为1,290MiB。

完整数据训练已全部完成，流水线结束于北京时间2026-10-15 01:39，四项任务exit code均为0。
以下为当时使用的tmux会话与日志位置；任务结束后会话自动退出：

- `resworld_perception_gpu1`：`runs/city_phase1/perception_full_gpu1.log`，GPU 1，4轮。
- `resworld_ego_full`：`runs/city_phase1/ego_full_cpu.log`，CPU、batch 64，4轮。

状态分支已正常完成4轮（exit code 0），在4,819条完整工况筛选验证序列上ADE为1.02843m，
纵向0.94996m、横向0.21883m、3秒终点2.20288m。这是仅状态基线，不是视觉或联合模型精度；
它与24条路线均衡小试验的分布不同，不能直接对比数值。
`resworld_staged_pipeline`会话中的 `scripts/continue_staged_training.py`已在两路正常结束后，
于**同一个GPU UUID上顺序完成**1轮`fusion_full_gpu1`、1轮`joint_full_gpu1`，batch 4。
该脚本在任何上游非零退出时停止；本轮没有同时启动两个GPU训练任务。
`runs/city_phase1/staged_full_pipeline_status.json`现在为`complete`。

## 完整城市工况清单的首轮结果

汇总文件为 `runs/city_phase1/full_staged_results.json`。检查best checkpoint、退出码及manifest，
四项采用同一个22,480/4,819数据清单；三项CUDA任务均只可见物理GPU 1，自车状态任务使用CPU。

感知单训4轮、5,620次更新，按前景mIoU选择的best为第3轮：

| 独立感知指标 | 验证结果 |
| --- | ---: |
| vehicle框投影IoU | 6.349% |
| human框投影IoU | 0.208% |
| 前景mIoU | 3.279% |
| 稀疏量化深度MAE | 8.786m |

这些是辅助感知指标，仍偏低。第4轮前景mIoU为3.239%，没有继续提高；
当前结果不足以说明感知已经学好，也不能只凭低指标就认定追加轮数可以解决。
后续应检查图像编码器语义能力、监督/可见性与类别分布，并做可视化和独立对照。

| 同一4,819条验证序列上的规划指标（m） | 匀速 | 独立ego | 冻结融合 | 联合微调 |
| --- | ---: | ---: | ---: | ---: |
| ADE | 1.2246 | 1.0284 | 1.0125 | 0.9644 |
| 纵向平均绝对误差 | 0.9951 | 0.9500 | 0.9470 | 0.9020 |
| 横向平均绝对误差 | 0.4186 | 0.2188 | 0.1983 | 0.1929 |
| 3秒终点误差 | 2.6089 | 2.2029 | 2.1697 | 2.0759 |

联合模型的独立感知前景mIoU为3.174%，比感知best的3.279%略低；
规划ADE的改善不能直接归因于环境感知增强。全清单checkpoint尚未进行图像干预与等预算多seed对照。
本轮仍为1m/C32配置，不能当作0.5m/C64配置的正式训练结果。

后续已针对感知做图像干预、小样本拟合及目标函数对照，详见 `docs/BEV_PERCEPTION_DIAGNOSIS.md`。
预训练ResNet50/FPN＋CE/Dice的独立感知重训已在单张GPU 4正常完成4轮、11,240次更新，
best为第4轮。同一4,819条验证序列的vehicle/human IoU为26.222%/12.041%，前景mIoU为19.131%。
对应栅格召回率仅35.912%/19.229%，误报减少但漏检仍多，因此尚未衔接规划。
新旧主干、预训练、损失、图像尺寸和更新步数不同，不能把结果视作单因素或等预算对照。
汇总见 `runs/city_phase1/perception_resnet50_comparison.json`；上表规划指标仍属于旧Tiny流水线。

旧Tiny视觉任务使用完整工况筛选清单与128×352尺寸，新ResNet50使用相同清单与256×448尺寸；
两者结果均不能与上面的48/24均衡试验直接比较。
每轮保存`last.pt`与按独立验证指标选择的`best.pt`，launcher退出时写入`.exit_code`。
相机只接收当前帧、ego不读取图像，后续融合仍可读取三帧；迁移检查保证相同数据清单和米制几何。
后续重点是感知IoU/深度、纵横向误差及交互场景表现，观察融合是否利用环境约束。
仅有nuScenes开环轨迹误差，不能替代CARLA闭环的碰撞、违规与路线完成率。

## ResNet50后续融合实验（2026-10-15启动）

用户要求继续后续步骤，观察是否改善规划结果。新增
`scripts/run_resnet_staged_training.py`，从已完成的ResNet50感知best和原ego best初始化，
在单张物理GPU 7（UUID `GPU-c80c9bc4-c793-56cb-99a2-8cb6f9d7d32a`）上顺序训练1轮fusion、1轮joint。实验目录为
`runs/city_phase1/resnet50_staged_v2/`；`status.json`记录当前阶段，
`fusion.log`和`joint.log`记录训练，两阶段完成后自动生成`comparison.json`。
任何阶段非零退出都会停止，不会继续下一阶段或覆盖已有实验。

保持22,480/4,819清单、官方划分、1m BEV、256×448图像、CE/Dice、seed 0和既定学习率。
三帧ResNet50联合反向的真实数据试运行在batch 2时峰值已分配显存6,196MiB；
固定单卡且考虑当前剩余显存，正式两阶段均使用batch 2，各11,240次更新。
旧Tiny两阶段均为batch 4、各5,620次更新，因此这次比较并非等更新次数或等计算量实验。

启动前55项测试通过；真实训练样本上各试跑2次fusion/joint更新，确认新感知与旧ego可严格加载、
融合初值精确等于ego预测、fusion冻结的366个张量不变，joint中78个感知编码器张量和6个ego张量更新。
这些是运行检查，不能作为验证精度。待完整验证完成后，比较ADE、纵横向误差、FDE及感知IoU/召回率。
对照旧冻结融合ADE 1.01249m、旧联合模型0.96439m、仅ego 1.02843m；目前尚无新版规划结果。

第一次启动`resnet50_staged_v1`使用GPU 4，但其他用户的vLLM任务随后占满显存，
首个训练更新前即CUDA OOM退出，无新checkpoint。原始失败日志和退出码保留。
随后在GPU 7上再次通过真实数据融合/联合检查（联合峰值6,094MiB），以新目录v2重新启动，
启用可扩展显存分配；每个阶段启动前检查单卡剩余显存至少8,500MiB，不足时等待。
