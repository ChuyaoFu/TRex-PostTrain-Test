# 交付检查（2026-10-04）

已通过：

- 从公开 GitHub fetch 固定 commit 并 checkout，应用 overlay；首次和重复入口 dry-run 均成功。
- 入口和全部 shell 脚本 `bash -n`，所有 Python 源文件 AST 解析成功。
- dry-run 生成的完整训练参数交给实际 train.py 的 argparse 定义，解析成功；检查 8 卡、bs16、100 epochs、无 step cap、完整分支、严格 midtrain 加载。
- 按 Linux x86_64 / manylinux_2_28 / Python 3.10，只接受 binary wheels，成功解析训练及数据依赖锁定清单；PyTorch 固定 2.6.0、torchvision 0.21.0、CUDA runtime 12.4。
- 注入准备阶段 exit 37，完整 pipeline 返回 37，stderr 和失败原因写入前台 tee 日志，后续下载/训练不执行。
- 固定 HF snapshot 的 meta/info.json SHA256 与当前完整 PPU 数据相同，200 episodes / 208581 frames。
- CPU 使用 Pinocchio 4.0.0 + NumPy 2.2.6，在真实原始数据前 2 episodes / 2844 frames 上重新做 FK、chunk、统计转换；独立 SE(3) 校验 24 个 chunk steps，最大误差 1.309e-7，手部 target / force 完全保留，episode 边界和统计通过。
- 前述完整 PPU 120 steps 训练和独立 checkpoint reload 的修复代码已原样纳入。

验证边界：

- 上述实际 CPU FK 检查使用现有 Python 3.12 interpreter，并切到 NumPy 2.2.6；Python 3.10 依赖是平台解析检查，没有宣称已在新的完整 Python 3.10 CUDA 环境跑过训练。
- H100 / NVIDIA CUDA 运行时、NCCL、DeepSpeed 8 卡启动、实机显存和性能尚未验证，因为当前只有 PPU 硬件。
- 目标服务器的 prepare / 10-step probe 会检查真实环境导入、8 卡 BF16、数据 batch、模型加载、反向传播和 policy 导出。probe 成功后再执行默认 100 epochs。
- 下载依赖目标机器的网络访问；HF/uv/Git 下载错误会中止，并可以用原工作目录重试。
