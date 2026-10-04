# 交付检查（2026-10-04）

最新完整复核见 [AUDIT_20261004.md](AUDIT_20261004.md)。后续已完成全量图片与JSON导出，并在新Python3.10环境执行数据转换、图片/视频张量对照及CPU checkpoint检查；下文早期记录的环境和配方按历史背景阅读。

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
- 早期全量提取记录为进行中；最新复核已完成200episodes/208581frames、2711553张图片和JSON，新增全量文件库存检查通过，详见最新审查报告。

验证边界：

- 早期CPU FK检查使用Python3.12；最新复核已用新Python3.10.18环境重新转换2844帧，17项真实processor张量配对通过。仍没有在完整Python3.10 CUDA/DeepSpeed环境跑过H100训练。
- H100 / NVIDIA CUDA 运行时、NCCL、DeepSpeed 8 卡启动、实机显存和性能尚未验证，因为当前只有 PPU 硬件。
- 目标服务器的 prepare / 10-step probe 会检查真实环境导入、8 卡 BF16、数据 batch、模型加载、反向传播和 policy 导出。probe 成功后再执行默认 100 epochs。
- 下载依赖目标机器的网络访问；HF/uv/Git 下载错误会中止，并可以用原工作目录重试。
- 固定上游README的state/与training_state.json声明未被其源码实现，本包不支持精确训练状态续训。部署硬件/闭环性能尚待实验室验证；完整差异见UPSTREAM_README_AUDIT.md。

## 当前默认配方更新

PPU/H100与交付overlay同步改为官方LR1e-4、warmup0、minratio0；JSON按帧划分5%，val500/30batches，无固定噪声或额外初始/最终验证；每50epochs及结束保存，save_steps0、max_ckpts10。旧120步、PNG/视频benchmark和上文episode划分结果保留为历史验证，不代表当前默认配方。新默认JSON计划154900updates，LeRobot可选路径155600updates。此次不启动完整训练。

当前配方已通过四个入口（主目录及overlay的PPU/H100）dry-run及真实argparse检查；执行实际JSON划分函数检查全量帧覆盖与无交集，198152/10429；执行实际保存条件确认仅epoch49/99保存，并检查不重置验证随机种子的分支。详见RECIPE_VALIDATION.json。未启动GPU训练。

## W&B在线监控更新

PPU/H100默认online，凭据从仓库外私密文件或环境读取；认证/网络预检在训练前执行。项目/空间可通过WANDB_PROJECT/WANDB_ENTITY设置。SDK仅升级到0.22.3，其余H100锁定版本保留；PPU实机使用--no-deps仅升级W&B包，10项依赖兼容检查和真实训练模块导入通过。

训练和验证指标以optimizer_step为显式横轴，避免同一个W&B内部step提交两次造成指标丢失。5项自动检查覆盖凭据读取、环境优先级、缺少凭据阻止online、显式offline，以及同一optimizer_step训练/验证均被记录。真实W&B小型run完成指标上传和回读，无模型训练、无机器人操作；结果见WANDB_VALIDATION.json。H100完整训练仍未执行。
