# dev 上的 JSON/PNG 与 LeRobot/PyAV 实测

2026-10-04，ali-dev，2×PPU-ZW810E（各96GiB）。两条完整 episodes 共2844 frames；真实release processor、同一组EEF62动作/全任务统计、相同随机样本顺序，每卡batch16。当前3RGB、8FLARE未来图、10deform和16步F6历史全部开启。12个含边界的样本，17项batch张量逐位相同。独立正式导出器在两个episodes上再次通过17项配对检查及episode划分检查。

三组分别为共享盘MP4、相同字节MP4复制到本地盘、同一视频像素无损导出为本地PNG＋JSON；区分磁盘位置与读取流程。数据测试两轮倒序复测，各预热4批、测8批。缓存为warm，没有清除主机page cache。平均值包含DataLoader等待的突发性，不能只看接近0的median。

| DataLoader，16 samples/batch | 2 workers，s/batch | 4 workers，s/batch |
|---|---:|---:|
| LeRobot/PyAV，共享盘MP4 | 5.290 | 2.658 |
| LeRobot/PyAV，本地MP4 | 5.229 | 2.654 |
| JSON＋本地PNG | 0.924 | 0.462 |

提前解帧的数据吞吐快约5.7倍；MP4移本地盘只有很小差别。JSON语法本身不是主要收益，离线解帧避开训练时随机seek/视频解码。

实际训练每组20个optimizer updates，从同一完整midtrain开始，BF16＋ZeRO2，固定相同sampler，2 workers/rank。关闭验证和保存；使用updates5–19的15个稳定更新。对Accelerate iterator/训练body做对称CUDA同步计时，不改正式训练同步策略。body包括前反向、optimizer、通信及原有日志；iterator包含等待、预取和传输。

| 流程 | 等batch/传输，s | 训练body，s | 实际s/update |
|---|---:|---:|---:|
| 共享盘MP4 | 1.620 | 3.575 | 5.195 |
| 本地MP4 | 1.547 | 3.580 | 5.128 |
| 本地PNG＋JSON | 0.014 | 3.605 | 3.619 |

相对原共享盘视频：每步耗时减少30.3%，吞吐提高43.5%；相对本地MP4：耗时减少29.4%，吞吐提高41.7%。计算量不变，主要消掉数据等待。三组均正常完成，首步loss同为1.65658，最后约0.291；训练期间最大loss差约0.0057，未宣称并行BF16训练逐步bitwise相同。显存峰值allocated64.51GiB/reserved89.76GiB。

两个episodes导出36972张PNG，1.506GB；JSON68.1MB；首次实际视频解帧约53.3秒。全量208581frames线性外推：2711553张PNG约110.46GB，JSON约5GB。仅是容量估算，全量提取正在运行，不把小样本估算当作全量完成数据。

2 PPU结果不能直接替代8 PPU/H100的ETA，尤其H100默认4workers，计算/数据重叠比例会改变。完整原始数值见 `benchmarks/20261004/summary.json` 和 `loader_results.json`；dev复现工具及日志保存在本地任务的 `artifacts/trex_io_benchmark_20261004`，远端临时配对数据位于 `/tmp/trex_io_benchmark_20261004`。
