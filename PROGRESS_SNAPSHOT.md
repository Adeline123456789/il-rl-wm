> 2026-10-19 补评完成：用户明确批准300秒后，106/106正式条件完成。U2均值/随机各0/100，旧预算中断记录保留；未增加训练或恢复自动化。最新表、补评账本和研究限制见[FINAL_BATCH_REPORT.md](FINAL_BATCH_REPORT.md)。下方旧104/106或未授权描述为此前历史。

# 2026-10-19 增量进度

> 2026-10-19 终态更新：正式测试104/106条件完成，U2均值预算中断、随机未启动。详见[FINAL_BATCH_REPORT.md](FINAL_BATCH_REPORT.md)。共享总账23.647652小时；原最终cap已用完，未安排重试或新训练。下文开发记录/早期状态保留为历史，不能当作仍在运行。

U2完整开发阶段已验收为有限负结果，最新入口为ACT_U2_FINAL_REVIEW.md、ACT_TRANSFER_CAUSAL_GAPS.md和NEXT_RESEARCH_BATCH.md。当前106条件正式评测003仍在后台，尚非最终批次结项。新R0/R1/R3/预热实验未获GPU授权。阶段上传回执见artifacts/github-act-stage-upload-*，以最新完成且回读验证的回执为准。

以下2026-10-16快照原文作为历史保留，其“尚未完成/测试未访问”表述仅适用于当时。

---

# 2026-10-16 当前进度快照

本次应用户要求提前上传当前代码与进度，**不是首批结项，也不是最终测试报告**。后台单卡训练继续运行。精确捕获时间、正在运行的进程和已结束作业清单见 [snapshot.json](artifacts/github-snapshot-20261016-001/snapshot.json)；文件在短时间窗口内逐项读取，运行中的日志只是部分记录。

## 已真实完成

- ManiSkill 官方 BC/SAC 主线、冻结 IL 与 BC→SAC 初始化、固定 50% expert 混合、两步动作条件 WM、独立纯任务价值头、双来源候选统一评分、20% 实际随机 SAC 探索、四方法完整回合验证，以及无 IL 推理导出。
- PickCube 开发 seed0 的 BC/A/B/C、完整 U0、同父 C 续训；U1 的 C-LN、WM/价值预热、完整规划段及 C-LN 续训。U1 唯一机制变化是 SAC critic 隐藏层 LayerNorm，见 [U1_CHANGE.md](U1_CHANGE.md)。
- 三任务真实环境检查和官方演示处理；StackCube 全量回放可用 769 条，Peg 修复版可用 126 条、88 个不同来源组。失败轨迹保留，训练/验证按来源组隔离。
- 隔离 JAX/rfcl RLPD 的真实 4096 交互 smoke、回放审计、便携 actor 和独立评测 smoke；尚未完成完整 RLPD 基线。
- 四次真实 F1 诊断、固定输入价值漂移、实际行为分布、执行一致性和成本分析；完整 CPU 单元测试 29 项通过。

## 当前证据能说明什么

所有下表数字均为 **开发 seed0 的固定最终模型、16 回合验证**，不是最终测试或多训练种子结论。

| 方法 | 成功数 / 16 |
|---|---:|
| 完整 U0 裸 SAC / 含 IL 规划 / 无 IL 规划 | 0 / 0 / 0 |
| 同父无规划 C 续训 | 16 |
| 完整 U1 裸 SAC / 含 IL 规划 / 无 IL 规划 | 16 / 13 / 15 |
| C-LN 续训 | 16 |

U0 后期出现行为退化与价值数值失稳，全部失败结果保留。U1 在这个开发运行中未复现崩溃，但规划器尚未优于稳定 SAC。相关竞争解释、共同输入数值和解释边界见 [U1_seed0](diagnostics/U1_seed0.md) 与 [F-U0-001](diagnostics/F-U0-001.md)。

四次 F1 共 496320 额外诊断交互，未用于学习；两种续接目标均没有超过预登记容限的可排序候选对。不能由零 regret 宣称评分可靠。本批已停止前缀搜索，见 [F1_batch_diagnosis](diagnostics/F1_batch_diagnosis.md)。

U0/U1 可训练参数分别 599351/602423；规划训练整卡采样峰值均 4703MiB，隔离 RLPD smoke 峰值 4893MiB。训练动作选择 batch16 平均约 2.043/2.074ms，只是稀疏测量描述，非 batch1 部署基准；[原始成本汇总](artifacts/seed0-runtime-costs-001/result.json)区分推理范围与含仿真的评测墙钟。

## 正在执行与未完成

独立 PickCube seed1 已完成 BC 和 SAC A：分别最终验证 0/40、0/16。快照捕获时的实际当前阶段写在 snapshot.json，后续依赖链继续 BC 初始化 B、C、WM、价值、U0 与 C 续训，再执行其余已登记种子和任务。

Pick seeds1–4 的完整 U0/U1 独立确认、Stack/Peg 完整开发训练、RLPD 完整基线，以及 52 模型/阶段、88 方法条件各 100 回合的最终测试尚未全部完成。最终测试 seed20000 在此次快照时未访问。方法已固定，评测与 CPU 汇总等待前置训练成功后执行；训练保留 2 小时评测额度，总上限仍为 24 GPU 作业小时，不按成绩删条件。详见 [NEXT_STEPS.md](NEXT_STEPS.md) 和 [注册表](configs/final_evaluation_plan.json)。

## 如何复核与交付边界

- [REPRODUCING.md](REPRODUCING.md)：环境锁、官方数据来源、训练/诊断/评测命令；[UPSTREAMS.md](UPSTREAMS.md)：锁定 commit、许可及适配范围；[RESEARCH_LOG.md](RESEARCH_LOG.md)：失败与修复经过。保留原始 PROMPT/Charter。
- `artifacts/` 包含逐回合/逐步指标、配置、命令、退出码、资源采样、检查点哈希、真实诊断数组和测试记录；`diagnostics/`、`literature/` 与图形目录保存解释和依据。运行中 `manifest.json` 的 ended_at/exit_code 为 null，不能当已完成作业。
- `releases/` 中已有可携带分析权重、U0/U1 无 IL 推理包、RLPD smoke actor；它们不含大回放和优化器，**不能恢复完整训练**。完整训练 checkpoint、原始演示、回放、环境和缓存仍只保留本机，本次未上传；公开数据的重建来源/版本与本地 checkpoint 哈希均有记录。
- [源码历史归档](releases/source-history-20261016-001/README.md)保留实验所引用的原始本地 Git commit。GitHub 连接器发布 commit 的 SHA 可不同，但发布树须与本地上传快照的 Git tree 完全相同。历史未提交绘图文件有一处已知缺口，见 source-history-preflight-001；不声称所有历史目录字节已恢复。

本次上传不改变奖励、学习算法、现有工作文件或训练进程，也不合并 main。完整批次结束后仍需按 PROMPT.md 的十项要求补齐最终报告与全部可完成条件。
