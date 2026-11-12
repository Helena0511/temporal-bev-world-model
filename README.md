# TemporalBEV-World-Model

## 基于时序 BEV 世界模型的多视角感知与轨迹规划

**Multi-View Perception and Trajectory Planning Based on a Temporal BEV World Model**

TemporalBEV-World-Model 是一个基于原生 PyTorch 的 nuScenes 多相机驾驶世界模型研究项目，包含数据读取、相机到 BEV 感知、时序融合、未来 BEV 预测、轨迹规划、分阶段训练及评估工具。

> 本项目是轻量研究实现，不兼容原论文权重，也不代表已完成论文精度复现。已有实验结果和当前限制见 [复现记录](docs/REPRODUCTION.md) 与 [精度状态](docs/ACCURACY_STATUS.md)。

## 项目结构

```text
resworld/           核心数据、模型、感知目标与损失
scripts/            训练、评估、诊断和实验编排入口
scripts/legacy/     不再接入主流程的早期独立原型，仅供追溯
configs/            模型配置与 nuScenes 官方场景划分
tests/              单元测试和几何、训练契约测试
docs/               设计、诊断和复现文档
runs/               本地实验日志、指标和 checkpoint（不作为源码发布）
weights/            本地预训练权重（不作为源码发布）
```

主训练链路使用 `resworld/` 与 `scripts/` 下的非 `legacy` 代码。`runs/` 中已有结果未被整理过程修改或删除。

## 环境与安装

- Python 3.10+
- PyTorch 2.1+
- NumPy、Pillow、Matplotlib
- nuScenes v1.0-mini 或 v1.0-trainval 数据集；正式训练建议 CUDA GPU

请先按 PyTorch 官方说明安装与机器 CUDA 匹配的版本，再安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

也可以使用已有 Conda 环境。项目根目录下的 `requirements.txt` 可用于仅安装 Python 依赖。

## 数据与快速检查

数据目录应包含 `v1.0-mini/` 或 `v1.0-trainval/` 元数据及对应 `samples/`、`sweeps/` 文件。通过 `--root` 或 `RESWORLD_DATA_ROOT` 指定数据位置。

```bash
# 无需数据集的单元测试
python -m unittest discover -s tests -t . -v

# 检查 mini 数据与标签
python -m scripts.inspect_driving_data \
  --root /path/to/nuscenes --version v1.0-mini \
  --output runs/mini_data_audit.json

# 小批量前向、反向和评估检查
python -m scripts.train_driving \
  --root /path/to/nuscenes --version v1.0-mini \
  --mode smoke --limit-val 4 --output runs/mini_smoke

# 固定批次过拟合检查
python -m scripts.train_driving \
  --root /path/to/nuscenes --version v1.0-mini \
  --mode overfit --steps 100 --output runs/mini_overfit_repeat
```

训练输出包括 `config.json`、`metrics.jsonl`、`last.pt`，验证 ADE 改善时还会生成 `best.pt`。恢复训练使用 `--resume <checkpoint>`；数据划分、模型配置或源码哈希不匹配时应先检查兼容性，不要直接覆盖旧实验。

## 正式训练入口

```bash
# 端到端训练，默认使用 configs/bev_attention_city60.json
RESWORLD_DATA_ROOT=/path/to/nuscenes RESWORLD_DEVICE=cuda \
  bash scripts/train_full.sh

# 分阶段感知训练（需本地 ImageNet ResNet50 权重）
RESWORLD_DATA_ROOT=/path/to/nuscenes RESWORLD_GPU=0 \
  bash scripts/train_perception_resnet_single_gpu.sh
```

分阶段训练、因果性对照及感知消融有独立的数据和 checkpoint 前置条件，运行前请阅读对应文档，不要在没有数据或 GPU 的环境中直接启动长任务。

## 文档导航

- [复现步骤与数据约定](docs/REPRODUCTION.md)
- [BEV 设计](docs/BEV_DESIGN.md)
- [城市驾驶阶段一](docs/CITY_PHASE1.md)
- [因果性训练](docs/CAUSALITY_TRAINING.md)
- [感知改进与消融](docs/PERCEPTION_IMPROVEMENTS.md)
- [当前精度状态](docs/ACCURACY_STATUS.md)
- [BEV 指标审计](docs/BEV_METRIC_AUDIT.md)

所有命令默认从项目根目录执行。早期原型仅用于历史对照，不属于当前受测试支持的主流程。

## GitHub 发布

建议 GitHub 仓库名为 `TemporalBEV-World-Model`。上传前检查、提交和推送步骤见 [GitHub 发布指南](docs/GITHUB_UPLOAD.md)。内部 Python 包仍名为 `resworld`，以保持现有导入和实验代码兼容。
