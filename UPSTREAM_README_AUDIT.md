# 官方 README 后训练流程审查（2026-10-04）

依据固定版本 [`f88e10c61da123c68bf0927cf4860bc97a0381f3`](https://github.com/ZhuoyangLiu2005/T-Rex/tree/f88e10c61da123c68bf0927cf4860bc97a0381f3) 的 README、train.sh/train.py、test.sh/test.py、JSON/LeRobot converter 和模型代码。以下区分官方文档、实际源码、已复现范围；不能宣称 H100 或实机部署已全部验证。

| 项目 | 官方要求/实际行为 | 本任务状态与证据 |
|---|---|---|
| Release 范围 | main 仅 posttrain + inference；pretrain/midtrain 在另一分支 | 使用公开 midtrain；无需重新跑前两阶段。未复现论文完整训练/全部任务成功率。 |
| 环境 | Python 3.10、torch 2.6 cu124、transformers 4.57.3、Accelerate 1.8.1、DS 0.15.4 | H100 锁定这些版本；8 H100 实机仍待 probe。PPU 必须用厂商 Python 3.12/torch，不伪称相同二进制。 |
| 自有数据入口 | raw success/episode_N，HDF5 + MP4；JSON 默认、LeRobot opt-in | HF 本任务是 joint58 LeRobot；先官方 Vega-1 FK 转 EEF62，再导出相同 JSON/PNG 合约。原始 raw HDF5 converter 不能直接吃这个 HF 目录。 |
| 默认图片流程 | gen_json converter 顺序解 RGB 为 PNG，并导出 deform PNG 和统计 | 之前交付默认用支持的 LeRobot 视频路径，未做默认图片性能对照。本次已补齐离线解帧，H100/PPU 均默认 JSON＋PNG。 |
| 位姿/动作 | EEF62，16 步 delta-base chunk，两臂各 9D EEF + 22D hand | 官方 URDF/FK 与 pose helper；完整数据校验。缺失 torso/head 按已确认的固定姿态补齐；记录 target joints 的 FK，不等同于缺失的 pre-IK teleop pose。 |
| 图像 | head crop；right/left wrist；384×288 LANCZOS | 固定 crop `[0,300,140,540]`，保持 right/left 顺序；17 项 paired batch 张量逐位一致。 |
| 触觉/VQ-VAE | F6 `[10,6]`、10 路 deform、16-frame raw history；release 内嵌 VQ，默认在线编码 | 保留全部分支，不要求 rawtac，也不另行下载 VQ/deform checkpoint；不做 legacy code pre-baking。 |
| Cascade/FLARE | stage2，10/6 split，dropout 0.1；8未来帧×4 tokens，stride4，loss weight0.5 | 参数保留；完整 20-step 配对训练，包含全部 loss 分支。模型源文件保持官方原样。 |
| Encoder 冻结/精度 | frozen ViT/deform/VQ；README 声称 VQ 编码等价 | 补 `.eval()` 避免冻结 BN 仍更新统计；VQ 保留原始 FP32，MSE 用 FP32。120-step checkpoint 参数与冻结权重验证通过。 |
| 验证划分 | 原 JSON 源码按帧；LeRobot 源码按 episode | 明确保留当前任务 episode 划分；JSON 新增 `--val_split_by_episode 1`，190/10 episodes，199135/9446 frames。原默认帧划分仍可用，未悄悄换验证集。 |
| 长度/批量 | train.sh 100 epochs、8卡×16、accum1 | 相同；无 step cap，155600 updates。原 scheduler 在 sharding 前整除会得到155575，本包按实际 loader 长度修正为155600。 |
| 优化器/LR | AdamW、1e-4、warmup0、minratio0、weightdecay0 | AdamW/weightdecay同；保留此前确认的3e-5、warmup5%、minratio0.1，README 提供官方这些数值的覆盖命令。属于明确任务配方差异。 |
| 验证/保存 | 官方 val500、最多30 batches，save每50epochs；离线W&B | 仍val500，但当前约定4 batches/GPU、固定抽样和初始验证；save每500updates、留2份。显式差异；JSONL/run_config额外记录，不改变模型分支。 |
| 原有加速 | BF16、ZeRO2、SDPA、partial-flow KV复用、冻结模块no_grad | 均保留；官方原来就没有启用compile、gradient checkpointing或通信overlap。PPU实际SDPA是否走厂商fused kernel未测，不把API名当成证据。 |
| Checkpoint 交付 | model、config、processor、training_args、task stats | 全部导出并独立 reload。增加数据格式/图像尺寸/划分记录、严格模型校验。 |
| 精确断点状态 | README列state/和training_state.json，但固定源码save_checkpoint没save_state/load_state | 本包同样不支持 optimizer/scheduler/RNG 精确续训；README明确限制。这是上游文档/源码不符，不能当作已复现。已有checkpoint是policy权重，不是完整训练状态。 |
| 推理重建 | 从config+model+processor+task stats重建，内嵌VQ；slow/fast协议 | 修复BF16往返导致的VQ精度损失；恢复原始FP32权重并逐位检查。严格加载120-step policy missing=0/unexpected=0。 |
| 双臂warm-up | 原test.py给5路F6/deform，但双臂窗口要求10路 | 已修复为10路；离线slow_and_fast→fast→slow→fast通过，动作[16,62]且有限，不启动socket或机器人。 |
| 部署 | ZMQ server + Vega-1客户端/硬件配置 | 交付serve_policy入口及自动离线smoke；真实机器人、camera crop/torso姿态现场一致性、触觉频率、闭环成功率仍待实验室验证。未启动机器人。 |
| 多节点 | 官方按8GPU/node；当前PPU可能4×2节点 | H100本任务单机8卡；PPU脚本支持1/2/4/8节点总8卡。图片缓存须每节点相同绝对路径、本地准备。 |
| 可选工具 | code预烘焙、外部VQ merge、dataset replay、全硬件采集 | midtrain已内嵌VQ，因此不是本任务训练前置条件；没有宣称这些可选/硬件功能已逐项运行。 |

防止再次混淆：启动脚本显式选择格式并打印参数；图片输出有输入 fingerprint、完成标记和排他锁；训练前对照真实 processor 检查所有 batch 张量与验证划分；训练记录实际 run_config、数据规模、schedule horizon；导出后检查参数及离线推理。公开输入和官方源码都固定 revision。

实测、能力边界和正在运行的提取进程记录分别见 `IO_BENCHMARK.md`、`VALIDATION.md`。截至本次交付更新，全量图片仍在提取；准备阶段必须完成后方可正式训练。
