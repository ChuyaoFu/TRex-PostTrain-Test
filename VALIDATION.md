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

本次 JSON/PNG 更新已通过：

- 两个完整 episodes / 2844 frames / 36972 PNG 的顺序解帧；12个包含边界的配对样本，真实 processor 17项张量逐位一致。
- DataLoader 2/4 workers、两轮倒序配对测速，6种配置；每配置共16个测量batches。
- 2 PPU × bs16，PNG、本地MP4、共享盘MP4三组各20updates；稳定更新PNG3.619s，对原共享盘视频5.195s，耗时减少30.3%。全部loss分支保留，无OOM。
- 正式导出器复用已有图片并实际重新提取每种view的第0帧；JSON/统计完整输出，17项tensor相同，episode划分相同；完成缓存复用及不匹配输入拒绝通过。
- 新JSON入口按episode划分，2 PPU完成6updates，训练loss1.800→0.721；验证action0.568→0.396、tactile0.906→0.490；实际run_config与scheduler horizon为6。
- 使用全任务200个episodes的真实长度和实际SftDataset划分函数，核对199135train/9446val frames和8×16、100epochs的155600updates。
- 修改后的H100入口dry-run及真实train.py argparse检查通过；PPU dry-run输出默认JSON、episode split、8×16、100epochs、无限step cap。
- 推理入口严格加载120-step policy：missing0/unexpected0，VQ原始权重精确恢复FP32；离线slow_and_fast→fast→slow→fast返回[16,62]有限动作。不启动socket或机器人。
- 全量图片提取已在ali-dev本地盘启动（输出`/tmp/trex_tong_transfer_json_20260820`）；此处只声称进程运行和持续产出，未声称全量完成。

验证边界：

- 上述实际 CPU FK 检查使用现有 Python 3.12 interpreter，并切到 NumPy 2.2.6；Python 3.10 依赖是平台解析检查，没有宣称已在新的完整 Python 3.10 CUDA 环境跑过训练。
- H100 / NVIDIA CUDA 运行时、NCCL、DeepSpeed 8 卡启动、实机显存和性能尚未验证，因为当前只有 PPU 硬件。
- 目标服务器的 prepare / 10-step probe 会检查真实环境导入、8 卡 BF16、数据 batch、模型加载、反向传播和 policy 导出。probe 成功后再执行默认 100 epochs。
- 下载依赖目标机器的网络访问；HF/uv/Git 下载错误会中止，并可以用原工作目录重试。
- 固定上游README的state/与training_state.json声明未被其源码实现，本包不支持精确训练状态续训。部署硬件/闭环性能尚待实验室验证；完整差异见UPSTREAM_README_AUDIT.md。
