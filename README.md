# T-Rex tong-transfer：8 × H100 交付包

这个仓库是用于交接 task posttrain 的小型启动包。目标服务器配置：NVIDIA driver **580.95.05**、系统 CUDA **12.8**、**8 × H100**。

数据、公开权重和官方源码由训练服务器自动下载，无需从 PPU 服务器拷贝。仓库包含可直接运行的入口及 overlay；同步更新的压缩交付包也保存在 [`dist/trex_h100_handoff_20261004.tar.gz`](dist/trex_h100_handoff_20261004.tar.gz)。
机器要求：Linux x86_64（glibc ≥ 2.28）、8 张 H100、已安装的 NVIDIA 驱动、能够访问 GitHub / PyPI / Hugging Face；命令 git、curl、python3、nvidia-smi 可用。
建议主机内存至少 256 GiB，本地 SSD 空闲至少 200 GiB（建议 250 GiB）。默认现在离线提取 PNG，预计约 110 GB 图片、5 GB JSON、271 万个文件，再加公开输入、环境和 checkpoint。脚本不安装或更换驱动和系统 CUDA，也不需要 root。

## 一条命令启动完整流程

先在 H100 服务器的本地 SSD 目录 clone 本仓库，然后执行：

```bash
git clone git@github.com:ChuyaoFu/TRex-PostTrain-Test.git
cd TRex-PostTrain-Test
bash run_h100.sh train
```

这会在当前仓库下创建 `T-Rex_h100/` 工作目录。入口依次安装环境、下载数据和权重、官方 FK 转换、离线视频解帧为 PNG＋JSON、配对校验，再启动 8 卡训练。
需要把工作目录放到其他磁盘时，指定一个新的绝对路径：

```bash
TREX_WORKDIR=/your/local_ssd/T-Rex_h100 bash run_h100.sh train
```

使用已有驱动，不改动系统 CUDA 12.8。训练环境使用官方 PyTorch 2.6.0 的 cu124 wheel，由环境提供 CUDA 12.4 runtime，兼容目标 580 驱动。

**默认完整训练：每卡 batch 16，全局 batch 128，LR 3e-5，100 epochs，无训练 step 上限。** 10 步 probe 需要显式设置 `MAX_TRAIN_STEPS=10`。

脚本在前台运行，Ctrl+C 可中断；安装、下载、转换、验证和训练的 stdout/stderr 都写到 `T-Rex_h100/logs/<RUN_NAME>.pipeline.log`。
训练指标另外记录在 `outputs/tong_transfer_h100/<RUN_NAME>/metrics.jsonl`。
建议始终设置固定 `TREX_WORKDIR`，重复执行可复用环境、HF 下载和完成的数据转换。
失败会返回非零退出码，tee 不会隐藏训练失败。

如果想先检查准备流程、再进行 10 步 H100 实机验证：

```bash
export TREX_WORKDIR=/your/local_ssd/T-Rex_h100
bash run_h100.sh prepare
SKIP_SETUP=1 SKIP_DOWNLOAD=1 MAX_TRAIN_STEPS=10 RUN_NAME=h100_probe bash run_h100.sh train
SKIP_SETUP=1 SKIP_DOWNLOAD=1 RUN_NAME=h100_full bash run_h100.sh train
```

10 步只检查链路/显存/导出，不足以证明任务已收敛。正式命令自动恢复 100 epochs，无 step 上限。
只查看最终训练命令：`bash run_h100.sh dry-run`；首次仍会下载官方源码。
源码已准备好时也可直接执行 `bash scripts/posttrain_h100.sh dry-run`。

## 自动完成的步骤

1. 下载官方源码到新工作目录，固定 commit `f88e10c61da123c68bf0927cf4860bc97a0381f3`，应用本包里的训练/loader 修复和脚本。Vega-1 URDF / mesh 从官方仓库获得。
2. 必要时下载 uv，创建独立 Python 3.10 训练环境和 CPU 数据转换环境；核心及其传递依赖使用 Linux x86_64 / Python 3.10 的锁定清单；安装 PyTorch 2.6.0 cu124 / torchvision 0.21.0，保留官方 transformers 4.57.3、accelerate 1.8.1、DeepSpeed 0.15.4。
3. 检查 8 张 H100、BF16 matmul、训练模块导入。使用 BF16 + ZeRO-2，无 CPU offload。
4. HF 下载下面的固定 snapshot，并校验两份大权重 SHA256。可以通过正常 `hf auth login` 提供令牌；不得将令牌放进交付包。
5. CPU 使用官方 Vega-1 FK，把全部 200 episodes / 208581 frames 的关节 58 维转换成 T-Rex EEF 62 维，生成 16 步动作 chunk 和任务 q01/q99 统计；检查 SE(3)、手部动作、触觉值、episode 边界、真实视频解码及 batch。
6. `utils/export_lerobot_to_json_images.py` 顺序解码每个视频一次，导出官方 JSON loader 使用的无损 PNG＋JSONL＋统计文件。每视频记录完成状态、原子写文件，支持中断重试；完成前训练不启动。真实 release processor 配对验证 17 项张量和 episode 验证划分，动作/触觉/FLARE/归一化完全保留。
7. 8 卡训练，冻结 encoder 的 BN 统计，保留 VQ-VAE FP32，MSE 用 FP32；保存实际 `run_config.json`。完成后验证 checkpoint、输出曲线和 summary，并自动做离线 slow/fast 推理 smoke，不连接机器人。

LeRobot 0.4.0 使用 `--no-deps` 安装，仅启用此项目实际使用的 dataset loader / PyAV 路径。
其整个机器人控制、GUI、其他 policy 依赖没有安装，而且其包元数据要求不同版本的 accelerate / wandb / HF Hub。
因此全环境 `pip check` 会报告 LeRobot 的这些依赖差异；本脚本用实际 dataset 导入、视频读取、batch 和训练验证这条路径，避免自动把官方训练依赖升级。

## 哪些文件需要传输

训练方 clone 本仓库即可；也可下载并解压 `dist/` 的原始交付包，执行同一入口。以下大文件不在 Git 仓库中，由目标机器下载：

| 输入 | HF repo | 固定 revision | 下载范围 |
|---|---|---|---|
| 原始 task 数据 | miniFranka/trex_gateway_tong_transfer_sf_norawtac_20260820 | 9e20b13d042c708e1546138adda25c13ee6aa4e7 | 完整原始数据 |
| T-Rex midtrain | miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6 | 62efb3bcb45a3df0e088c8909d759b582cfb98af | model.pt、config、training_args、stats、processor |
| Qwen 初始化 | Qwen/Qwen3-VL-2B-Instruct | 89644892e4d85e24eaac8bacfd4f463576704203 | JSON / tokenizer / safetensors |

沿用你提供的原始下载命令，仅增加固定 revision；目标目录默认是工作目录下的 `data/`：

```bash
huggingface-cli download miniFranka/trex_gateway_tong_transfer_sf_norawtac_20260820 \
  --repo-type dataset \
  --revision 9e20b13d042c708e1546138adda25c13ee6aa4e7 \
  --local-dir ./data/trex_gateway_tong_transfer_sf_norawtac_20260820
```

midtrain 的 model.pt 是 8.505 GB，Qwen model.safetensors 是 4.255 GB；两者合计约 12.76 GB（十进制）。
当前构造函数仍用完整 Qwen 初始化再加载 midtrain，所以保留这一公开下载步骤。
midtrain 已内嵌 deform encoder 和 VQ-VAE，**无需另外下载/传输这两份 checkpoint**。
FK 转换后的数值数据约 373 MB，视频仍是原始软链接。随后在训练机器本地提取 PNG；无需额外传输图片。JSON 内使用绝对图片路径，缓存不能直接移动到另一台机器；应在目标机器重新生成。保留原始视频及 EEF62 目录以便校验/重试。
HF 下载可断点复用；`SKIP_DOWNLOAD=1` 只适用于全部输入已经完整存在，并仍会检查权重 hash。

## 默认训练参数及可覆盖项

默认保持此前确认的 PPU 训练参数：8 GPUs、每卡 batch 16、累积 1、全局 batch 128、LR 3e-5、warmup 5%、cosine 最小 LR 比例 0.1。
**训练长度保持官方后训练脚本的 100 epochs，`MAX_TRAIN_STEPS=0`。**当前数据划分为 train 190 episodes / 199135 frames，val 10 episodes / 9446 frames；当前 loader / Accelerate 配置每 epoch 1556 optimizer updates，总计 155600 updates。
保留 action + tactile + VQ codes + FLARE 全部分支，image 384×288，action chunk 16。
训练读取 worker 默认每进程 4；验证固定抽样，每 500 步验证 4 batches/GPU。每 500 步保存，最多保留 2 份 policy；保存峰值需容纳第三份写入。

官方 `scripts/train.sh` 的学习率配置是 LR=1e-4、warmup=0、min_lr_ratio=0、max_val_batches=30；若训练方要完全采用这几个官方值：

```bash
LR=1e-4 WARMUP_RATES=0 MIN_LR_RATIO=0 MAX_VAL_BATCHES=30 \
  TREX_WORKDIR=/your/local_ssd/T-Rex_h100 bash run_h100.sh train
```

`TRAIN_BSZ`、`GRAD_ACCUM`、`NUM_WORKERS`、`SAVE_STEPS`、`VAL_FREQ`、`MAX_CKPTS`、`MASTER_PORT`、`RUN_NAME` 均可通过环境变量设置。
`RAW_ROOT`、`LEROBOT_ROOT`、`WEIGHTS_ROOT`、`OUTPUT_DIR`、`LOG_DIR`、`TRAIN_VENV`、`DATA_VENV` 支持指定绝对路径。
`DATA_FORMAT=json` 为默认；`DATA_FORMAT=lerobot` 保留旧视频训练路径。`IMAGE_ROOT` 默认工作目录下 `training_data/tong_transfer_json`；`IMAGE_WORKERS=4`，新提取至少需要 `MIN_IMAGE_FREE_GIB=140` GiB 空闲。JSON 默认显式使用 `--val_split_by_episode 1`，保持现有 190/10 episodes 划分。LR schedule 已按 Accelerate 实际 padding 后的更新数计算，100 epochs 对应 155600 updates。
同一个已有 metrics 的 RUN_NAME 会被拒绝，避免混淆不同训练。
改变 batch 或数据划分后，update 数会改变；不应通过写死 155600 替代官方 100 epochs。

## 为什么默认离线图片

官方 README 的默认 JSON 路径先离线解视频为 PNG。我们原先选择了官方支持的 LeRobot opt-in 路径；不是官方没有这一步。配对实测见 [`IO_BENCHMARK.md`](IO_BENCHMARK.md)：dev 2 PPU、每卡 bs16、全部 loss 分支开启，原共享盘视频 5.19 s/update，JSON＋本地 PNG 3.62 s/update，训练耗时减少约 30%，吞吐提高约 44%；两组模型计算约 3.6 s，差异主要是等待数据。DataLoader 单独测速快约 5.7 倍，不能把这个倍数直接用于训练 ETA。

这些是 dev 两卡、两条完整 episodes、warm-cache 的结果；8 卡 PPU 和 H100 尚需实测。PNG/JSON 容量按样本外推，提取一次之后可跨全部 100 epochs 复用。

## 已有 PPU 环境的准备与启动

服务器 `/mnt/world_foundational_model/chuyao/T-Rex` 的 PPU 启动脚本也默认使用同一图片流程，复用现有 vendor torch 环境：

```bash
cd /mnt/world_foundational_model/chuyao/T-Rex
# 只做提取与校验，不启动 GPU 训练；dev 上已启动这一流程。
PREPARE_ONLY=1 bash scripts/posttrain_ppu_8gpu.sh
# 图片完成后，8 PPU 正式训练仍以前台运行并正常 tee 日志。
RUN_NAME=tong_transfer_ppu_images bash scripts/posttrain_ppu_8gpu.sh
```

PPU 的 `IMAGE_ROOT` 默认 `/tmp/trex_tong_transfer_json_20260820`，应按需改到有容量的持久本地 SSD。多节点必须每个节点都准备相同绝对路径的本地缓存；dev 的 `/tmp` 不与其他节点共享。`DATA_FORMAT=lerobot` 可继续旧流程。已运行的旧训练不会自动切换数据或被重启。

## 训练结果交回及部署输入

完成后查看 `outputs/tong_transfer_h100/<RUN_NAME>/latest_policy.txt`，它指向最新且通过参数验证的 checkpoint 目录。
交回该目录的 **model.pt、config.json、training_args.json、stats_data.json、processor/**，再加该 run 的 `summary.json` / `loss_curve.png` / `checkpoint_verification.json`。
部署必须携带 task 的 stats_data.json，不能换成 midtrain 数据统计。
源代码复用这份交付包；无需将训练机器的环境、HF 缓存、原始/转换数据、其他 checkpoint 复制到部署机器。
当前 checkpoint 不包含 optimizer、scheduler、RNG 状态，不能用于精确断点续训。

机器人按已确认的 Dexmate Vega-1，锁定官方默认 torso `[0.9,1.57,0.1]`、head `[0.28,0,0]`。
原始数据没有这些关节；FK 得到的是记录 target joints 的末端位姿。如果现场锁定姿态有差异，需要重新生成转换数据和 task 统计。

推理入口使用已修复的双臂 10 路触觉 warm-up，严格校验 checkpoint 并从原始权重恢复 VQ-VAE FP32，避免 BF16 往返损失。进入生成的源码目录、指定已配置的训练/推理 Python 后，可先离线检查再开服务：

```bash
INFERENCE_PYTHON=/path/to/environment/bin/python bash scripts/serve_policy.sh /path/to/checkpoint --smoke_only 1
INFERENCE_PYTHON=/path/to/environment/bin/python PORT=5678 bash scripts/serve_policy.sh /path/to/checkpoint
```

这只验证模型与 slow/fast 协议；现场机器人控制、传感器标定、时序和成功率尚未验证。官方 README 的逐项审查与所有差异见 [`UPSTREAM_README_AUDIT.md`](UPSTREAM_README_AUDIT.md)。

## 已验证范围

原始完整数据 + 全量 midtrain 已在 PPU 完成训练、导出和独立 reload 验证；120 步训练均值下降约 54%，固定验证 action / tactile loss 都下降；bs16 已在 PPU 跑通。
本 H100 包复用这些修复。新增脚本、依赖和 CPU 数据链路的检查结果见 `VALIDATION.md`。
**尚无 H100 硬件上的实机训练结果**；建议训练方先执行上述 10 步 probe，之后正式 100 epochs。H100 的完成时间应根据目标服务器实际稳定 step time 估算。

兼容性依据：
- [PyTorch 官方版本安装说明](https://docs.pytorch.org/get-started/previous-versions/)（2.6.0 cu124 安装命令）
- [NVIDIA CUDA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)（较新驱动对 CUDA 12 运行时的向后兼容）
