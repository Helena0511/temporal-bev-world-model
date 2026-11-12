# GitHub 发布指南

建议仓库名：`TemporalBEV-World-Model`。

## 发布内容

仓库应包含核心源码、训练与诊断脚本、配置、测试和文档。以下内容已被 `.gitignore` 排除，不应直接上传到 GitHub：

- `runs/`：实验日志、指标和 checkpoint，当前约 6.9 GB。
- `weights/`：ResNet50 等预训练权重，当前约 98 MB。
- Python 缓存、虚拟环境、构建目录和本地工具配置。

大型实验结果可选择性整理后通过 GitHub Releases、对象存储或实验跟踪平台发布，并在文档中记录来源和 SHA256。不要将整个历史实验目录直接提交到 Git。

## 公开前检查

1. 根据实际授权选择并添加 `LICENSE`；当前项目未默认指定开源许可证。
2. 确认 nuScenes 数据、官方场景划分和 ImageNet 权重的使用与再分发符合各自条款。数据集和权重文件不要随源码提交。
3. 检查文档中的本机路径、实验状态和结论，公开实验应保留其适用条件，不能表述为已完成论文精度复现。
4. 执行 `python -m unittest discover -s tests -t . -v`，并确认 GitHub Actions 测试通过。
5. 使用 `git status --short` 和 `git diff --cached --stat` 检查待提交内容，确保没有密钥、数据集、缓存或大型二进制文件。

## 初始化并上传

在项目根目录执行以下命令，将 `<YOUR_GITHUB_USER>` 替换为实际 GitHub 用户名或组织名：

```bash
git init
git add .
git status --short
git diff --cached --stat
git commit -m "Initial release of TemporalBEV-World-Model"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USER>/TemporalBEV-World-Model.git
git push -u origin main
```

如果已经初始化仓库或配置远程地址，不要重复执行对应命令。推送前先在 GitHub 创建空的 `TemporalBEV-World-Model` 仓库，不要同时生成会与本仓库冲突的 README、`.gitignore` 或许可证。
